

import torch
import torch.nn.functional as F

import numpy as np
import cv2
import copy


from .utils import draw_circle_ndarray, draw_shadow_ndarray, transform_image
from .utils_decode import get_circle_and_score, transform_circle_hw
from .default_config import Config as cfg


def decode_localization(out_local, device, input_size, original_size):
    # local info
    assert len(out_local) == 5
    heatmaps = out_local[0]
    inner_det_radii, inner_det_offset = out_local[1], out_local[2]
    outer_det_radii, outer_det_offset = out_local[3], out_local[4]

    det_inner_batch_info = get_circle_and_score(heatmaps, 0, inner_det_radii, inner_det_offset, 0.01, device)
    det_outer_batch_info = get_circle_and_score(heatmaps, 1, outer_det_radii, outer_det_offset, 0.01, device)
    det_inner_circle, scores1 = det_inner_batch_info[0]
    det_outer_circle, scores2 = det_outer_batch_info[0]

    det_circle = np.append(det_inner_circle, det_outer_circle).astype("int")
    out_circle = transform_circle_hw(input_size, det_circle, original_size)

    return out_circle


def decode_logics(mask_logits, original_size):
    image_h, image_w = original_size

    det_seg = transform_image(mask_logits, (image_h, image_w))
    log_prob = F.softmax(det_seg, dim=1).data.cpu().numpy()  # b c h w

    out_seg = np.argmax(log_prob, axis=1)  # b h w, class index:0 is background, so the vale is 0 or 1.
    out_seg = out_seg[0]  # (h, w)

    return out_seg


def decode_model_output(model_outputs, input_size, original_size, device):
    # decode detected info
    out_local, mask_logics = model_outputs

    detected_circle = decode_localization(out_local, device, input_size, original_size)
    predicted_mask = decode_logics(mask_logics, original_size)

    return detected_circle, predicted_mask


def before_input(image_cv2, img_size=320):

    image_h, image_w, c = image_cv2.shape
    image = copy.deepcopy(image_cv2)

    if image_h == 1080 or image_w == 1920:
        new_h, new_w = int(image_h / 5), int(image_w / 5)
        image = cv2.resize(image, (new_w, new_h))
        print("image size is  [1080, 1920]")
        return image
    elif image_h >= img_size and image_w >= img_size:
        return image
    else:
        ratio = 256 / min(image_h, image_w)
        ratio = int(np.ceil(ratio))
        new_h, new_w = int(image_h * ratio), int(image_w * ratio)
        image = cv2.resize(image, (new_w, new_h))
        print("=> info: image hw < 256, resize to > 256..")
        return image


def scale_coord(org_circle, new_size, orig_size, img_size=320):
    org_h, org_w = orig_size
    img_h, img_w = new_size
    circle = org_circle.copy()

    if org_h >= img_size and org_w >= img_size:
        return circle
    else:
        ratio_h, ratio_w = img_h / org_h, img_w / org_w
        circle[0], circle[3] = circle[0] / ratio_w, circle[3] / ratio_w
        circle[1], circle[4] = circle[1] / ratio_h, circle[4] / ratio_h

        print("=> info: make scale of predicted circle as orig size.", ratio_h, ratio_w)
        circle[2], circle[5] = circle[2] / ratio_h, circle[5] / ratio_h
        return circle


def after_output(predicted_circle, predicted_mask, new_size, original_size):
    assert len(predicted_mask.shape) == 2
    org_height, org_width = original_size

    mask = cv2.resize(predicted_mask.astype("uint8"), (org_width, org_height))
    circle = scale_coord(predicted_circle, new_size, original_size)

    return circle, mask


def detect_image_iris(ndarray, model, device, vis=False):
    """
    :param ndarray: [h, w, c], RGB
    :param model:
    :param device:
    :return:
    """
    image_h, image_w = ndarray.shape[:2]

    original_size = (image_h, image_w)
    image = before_input(ndarray)  # deep copy
    af_size = image.shape[:2]  # (h, w)

    # [h, w, c] to [c, h, w]
    image_input = np.transpose(image, (2, 0, 1))
    image_input = torch.from_numpy(image_input).type(torch.FloatTensor)
    # original image shape to input size
    image_input = transform_image(image_input, cfg.input_size)
    # [c, h, w] to [1, c, h, w]
    image_input = image_input.unsqueeze(0)
    if device:
        image_input = image_input.to(device)
    with torch.no_grad():
        model_output = model(image_input)
    detected_circle, predicted_mask = decode_model_output(model_output, cfg.input_size, af_size, device)

    # detected_circle = check_detected_circle(detected_circle, (320, 448))
    # for debug and vis after aa ndarray's size.
    # if vis:
    #     # from PIL import Image
    #     # show = Image.fromarray(vision_image.astype("uint8"))
    #     # show.show()
    #     circle, mask = after_output(detected_circle, predicted_mask, af_size, original_size)
    #     circle = check_detected_circle(circle, original_size)
    #     vision_image = cv2.resize(image.astype("uint8"), (image_w, image_h))
    #
    #     vision_image = draw_circle_ndarray(vision_image, circle[:3], color=(0, 255, 0), thickness=1)
    #     vision_image = draw_circle_ndarray(vision_image, circle[3:], color=(255, 0, 0), thickness=1)
    #     vision_image = draw_shadow_ndarray(vision_image, mask, shadow_color=(100, 50, 255))
    #
    #     return circle, mask, vision_image
    if vis:
        image = draw_circle_ndarray(image, detected_circle[:3], color=(0, 255, 0), thickness=2)
        image = draw_circle_ndarray(image, detected_circle[3:], color=(255, 0, 0), thickness=2)
        image = draw_shadow_ndarray(image, predicted_mask, shadow_color=(100, 50, 255))
        vision_image = cv2.resize(image.astype("uint8"), (image_w, image_h))
        # from PIL import Image
        # show = Image.fromarray(vision_image.astype("uint8"))
        # show.show()
        circle, mask = after_output(detected_circle, predicted_mask, af_size, original_size)
        return circle, mask, vision_image
    else:
        circle, mask = after_output(detected_circle, predicted_mask, af_size, original_size)
        return circle, mask


def check_detected_circle(det_circle, org_size):
    def is_ring(circle):
        iris_l, iris_t = circle[3] - circle[5], circle[4] - circle[5]
        iris_r, iris_b = circle[3] + circle[5], circle[4] + circle[5]

        pupil_l, pupil_t = circle[0] - circle[2], circle[1] - circle[2]
        pupil_r, pupil_b = circle[0] + circle[2], circle[1] + circle[2]

        if pupil_l < iris_l or pupil_t < iris_t or pupil_r > iris_r or pupil_b > iris_b:
            return False
        else:
            return True

    def is_middle(circle, size):
        # im_h, im_w = size
        # cx, cy = im_w // 2, im_h // 2
        # # x1, y1, r1 = circle[0], circle[1], circle[2]
        # x2, y2, r2 = circle[3], circle[4], circle[5]
        # dist = np.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        #
        # return False if dist > (im_w / 3) else True
        return True

    def is_out(circle, size):
        im_h, im_w = size
        x2, y2, r2 = circle[3], circle[4], circle[5]
        l, r, t, b = x2 - r2, x2 + r2, y2 - r2, y2 + r2
        if l <= 0 or t <= 0 or r >= im_w or b >= im_h:
            return True
        else:
            return False

    assert len(det_circle) == 6
    circle = det_circle.copy()
    image_h, image_w = org_size

    ring = is_ring(circle)
    mid = is_middle(circle, org_size)

    print("------------------------------------------ check detected circle")
    if is_out(circle, org_size):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>out of boundary")
        return np.array([0, 0, 0, 0, 0, 0])
    else:
        if ring and mid:
            print("--------------------------------------------------circle is ok")
            circle = circle
        elif ring is False and mid is True:
            circle[0], circle[1], circle[2] = image_w / 2, image_h / 2, circle[5] / 2
        elif ring is True and mid is False:
            circle[0], circle[1] = image_w / 2, image_h / 2
            circle[3], circle[4] = image_w / 2, image_h / 2
        else:
            circle[0], circle[1], circle[2] = image_w / 2, image_h / 2, image_w / 15
            circle[3], circle[4], circle[5] = image_w / 2, image_h / 2, image_w / 6
        return circle


# if __name__ == "__main__":
#     # main()
#     print("---")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using {} device ...".format(device.type))
#     model = create_model("model.pth", device)
#
#     image_path = "2.jpg"
#     org_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     org_cv2 = cv2.cvtColor(org_cv2, cv2.COLOR_BGR2RGB)
#     # cv2.imwrite("org_cv2.jpg", org_cv2)
#
#     circle, mask, vision = detect_image_iris(org_cv2, model, device, vis=True)
#     vision = cv2.cvtColor(vision, cv2.COLOR_RGB2BGR)
#     cv2.imwrite("t-out.jpg", vision)
#
#     print("-" * 20)
#
