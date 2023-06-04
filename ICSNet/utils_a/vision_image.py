

__all__ = ['draw_circle', 'convert_image', 'vision_image']


import numpy as np


def draw_circle(image, circle, color=(255, 0, 0), thickness=1):
    """
    draw circle on image or mask image
    :param image: ndarray, shape is H W C
    :param circle: ndarray, shanpe is (6,0) or (3,0)
    :return: image
    """
    import cv2

    assert len(circle) == 3 or len(circle) == 6

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

    return img


def convert_image(batch_image):
    """
    convert torch.tensor (c h w) on GPU to np.array (h w c) on CPU
    :param batch_image:
    :return:
    """
    # h w c to c h w
    image = batch_image.cpu().numpy()
    # print(image.shape)  # (3,512,512)
    # image = image[::-1, :, :].transpose(1, 2, 0)
    image = image.transpose(1, 2, 0)
    image = np.ascontiguousarray(image)

    return image


def vision_image(image, index, iou):
    import matplotlib.pyplot as plt

    title = "index:" + str(index) + ", bbox iou: " + str(iou)
    plt.title(title)
    plt.axis("off")
    plt.imshow(image)

    image_name = "./output/" + str(index).zfill(4) + ".jpg"
    plt.savefig(image_name)


def vision_mask(logits, targets):
    import matplotlib.pyplot as plt

    log_prob = F.softmax(logits, dim=1).data.cpu().numpy()  # b c h w to b h w
    pred = np.argmax(log_prob, axis=1)  # b h w, class index:0 is background, so the vale is 0 or 1.
    gt = targets.data.cpu().numpy()  # b h w

    # (b h w)
    assert pred.shape == gt.shape and pred.shape[0] == 1

    plt.imshow(pred[0], cmap="gray")
    plt.show()

    plt.imshow(gt[0])
    plt.show()

    return pred, gt
