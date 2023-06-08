

__all__ = ['transform_image', 'draw_circle_ndarray', 'draw_shadow_ndarray']


import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T


def draw_circle_ndarray(image, circle, color=(255, 0, 0), thickness=1):
    """
    draw circle on image or mask image
    :param image: ndarray, shape is H W C
    :param circle: ndarray, shanpe is (6,0) or (3,0)
    :return: image
    """
    import cv2

    assert len(circle) == 3 or len(circle) == 6

    image = image.astype("int")
    circle = circle.astype("int")

    if len(circle) == 3:
        inner_center, inner_radius = (circle[0], circle[1]), circle[2]
        image = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
    else:
        if circle[2] < circle[5]:
            inner_center, inner_radius = (circle[0], circle[1]), circle[2]
            outer_center, outer_radius = (circle[3], circle[4]), circle[5]
        else:
            inner_center, inner_radius = (circle[3], circle[4]), circle[5]
            outer_center, outer_radius = (circle[0], circle[1]), circle[2]
        image = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
        image = cv2.circle(image, outer_center, outer_radius, color=color, thickness=thickness)

    return image


def draw_shadow_ndarray(image, mask, shadow_color=(255, 255, 0)):

    image = image.astype('int')
    mask = mask.astype("int")

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

    # segout = Image.fromarray(segout.astype("uint8"))

    return segout


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
