

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import os


from utils_a.decode_logits import get_circle_and_score
from model.IrisCenterNet import CenterNet
from config import Config as cfg


def load_model_and_weights(weight_path):
    print(weight_path)

    assert os.path.exists(weight_path)

    model = CenterNet(cfg)
    model.load_state_dict(torch.load(weight_path))
    return model


# local utils
def transform_image(image, size):
    """
    resize image, mask to given size, and modify center point (x,y)
    :param image: PIL, 'RGB'
    :return: PIL, PIL,
    """
    transform = T.Compose([
        T.CenterCrop(size=size)
    ])
    trans_image = transform(image)

    return trans_image


def transform_circle(im_size, circle, to_size):
    """
    transform circle location to fit new size
    :param im_size: old image size, which formard is PIL.size, note: w, h = pil.size
    :param to_size: new image size, which formard is also PIL.size
    :param circle:  old circle, array: x1,y1,r1,x2,y2,r2
    :return:        new circle
    """
    iw, ih = im_size
    ia_w, ia_h = to_size
    circle_trans = circle.copy()
    # new center_x center_y
    sc_w, sc_h = (ia_w - iw)//2, (ia_h - ih)//2
    # radius is not change
    circle_trans[0] += sc_w + 1
    circle_trans[1] += sc_h + 1
    circle_trans[3] += sc_w + 1
    circle_trans[4] += sc_h + 1

    # circle_trans[[0, 3]] = np.clip(circle_trans[[0, 3]], 0, ia_w - 1)
    # circle_trans[[1, 4]] = np.clip(circle_trans[[1, 4]], 0, ia_h - 1)

    return circle_trans


# seg utils
def get_outer_coordinate(circle):
    assert len(circle) == 6 or len(circle) == 3

    if len(circle) == 6 and circle[2] < circle[5]:
        outer_circle_center, radius = (circle[3], circle[4]), circle[5]
    else:
        outer_circle_center, radius = (circle[0], circle[1]), circle[2]

    return outer_circle_center, radius


def paste_cropped_mask(mask_logits, roi_coords, mask_size):

    batch_size = mask_logits.shape[0]

    assert batch_size == len(roi_coords)
    h, w = mask_size

    batch_masks = torch.zeros(batch_size, 2, h, w)
    for i in range(batch_size):
        top, bottom, left, right = roi_coords[i]
        mask = mask_logits[i]
        batch_masks[i, :, top:bottom, left:right] = mask

    return batch_masks


# vision utils
def draw_circle(image_pil, circle, color=(255, 0, 0), thickness=1):
    """
    draw circle on image or mask image
    :param image: ndarray, shape is H W C
    :param circle: ndarray, shanpe is (6,0) or (3,0)
    :return: image
    """
    import cv2

    assert len(circle) == 3 or len(circle) == 6

    image = np.array(image_pil)
    image = image.astype(np.int32)
    circle = circle.astype(np.int32)

    if len(circle) == 3:
        inner_center, inner_radius = (circle[0], circle[1]), circle[2]
        img = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
    else:
        if circle[2] < circle[5]:
            inner_center, inner_radius = (circle[0], circle[1]), circle[2]
            outer_center, outer_radius = (circle[3], circle[4]), circle[5]
        else:
            inner_center, inner_radius = (circle[3], circle[4]), circle[5]
            outer_center, outer_radius = (circle[0], circle[1]), circle[2]
        img = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
        img = cv2.circle(img, outer_center, outer_radius, color=color, thickness=thickness)

    img_pil = Image.fromarray(img.astype("uint8"))
    return img_pil


# draw shadow on RGB image
def draw_shadow(image_pil, mask, shadow_color=(255, 255, 0)):

    image = np.array(image_pil)
    image = image.astype(np.int32)

    assert len(image.shape) == 3 and len(mask.shape) == 2

    if np.max(mask) == 255:
        mask /= 255

    image = image.astype(np.int32)
    seglap = image.copy()
    segout = image.copy()

    mask_img = np.array(mask, dtype="int32")
    mask_t = mask_img > 0
    for i in range(3):
        seglap[mask_t, i] = shadow_color[i]
    alpha = 0.5
    cv2.addWeighted(seglap, alpha, segout, 1 - alpha, 0, segout)

    segout = Image.fromarray(segout.astype("uint8"))
    return segout


if __name__ == "__main__":

    model = load_model_and_weights("./checkpoints/" + cfg.checkpoints + "/" + "model.pth")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device.type))

    model.to(device)
    model.eval()

    root = "test_images"
    image_list = os.listdir(root)
    for idx in range(len(image_list)):
        image_path = root + "/" + image_list[idx]
        assert os.path.basename(image_path).split(".")[-1] == "jpg"

        image_pil = Image.open(image_path).convert('RGB')
        image_w, image_h = image_pil.size
        # image_pil.show()

        # localization and segmentation
        image = transform_image(image_pil, cfg.input_size)
        # image.show()
        img = np.array(image, dtype="float")
        img = np.transpose(img, (2, 0, 1))
        input = torch.from_numpy(img).type(torch.FloatTensor)
        input = input.unsqueeze(0)
        if device:
            input = input.to(device)
        with torch.no_grad():
            output = model(input)

        # decode detected info
        out_local, mask_logits, roi_coords = output[0], output[1], output[2]

        # local info
        heatmaps = out_local[0]
        inner_det_wh, inner_det_offset = out_local[1]["wh"], out_local[1]["offset"]
        outer_det_wh, outer_det_offset = out_local[2]["wh"], out_local[2]["offset"]

        det_inner_batch_info = get_circle_and_score(heatmaps, 0, inner_det_wh, inner_det_offset, 0.01, device)
        det_outer_batch_info = get_circle_and_score(heatmaps, 1, outer_det_wh, outer_det_offset, 0.01, device)
        det_inner_circle, scores1 = det_inner_batch_info[0]
        det_outer_circle, scores2 = det_outer_batch_info[0]

        det_circle = np.append(det_inner_circle, det_outer_circle).astype("int")
        out_circle = transform_circle(image.size, det_circle, image_pil.size)

        # mask info
        det_seg = paste_cropped_mask(mask_logits, roi_coords, cfg.input_size)
        det_seg = transform_image(det_seg, (image_h, image_w))
        log_prob = F.softmax(det_seg, dim=1).data.cpu().numpy()  # b c h w
        out_seg = np.argmax(log_prob, axis=1)  # b h w, class index:0 is background, so the vale is 0 or 1.
        out_seg = out_seg[0]  # h, w

        # # # draw detected circle
        image_pr = draw_circle(image_pil, out_circle[:3], color=(0, 255, 0))
        image_pr = draw_circle(image_pr, out_circle[3:], color=(255, 0, 0))
        image_pr.show()

        # # # draw detected mask
        image_pr2 = draw_shadow(image_pr, out_seg, shadow_color=(200, 0, 0))
        image_pr2.show()

        save_path = "out_images/" + str(idx).zfill(4) + ".jpg"
        image_pr2.save(save_path)
