

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    y1 = y-top if y-top > 0 else 0
    y2 = y + bottom if y+bottom < height else height
    x1 = x - left if x - left > 0 else 0
    x2 = x + right if x + right < width else width
    masked_heatmap = heatmap[y1:y2, x1:x2]
    # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def transform_circle(im_size, size, circle):
    """
    transform circle location to fit new size
    :param im_size: old image size, which formard is PIL.size, note: w, h = pil.size
    :param size:    new image size, which formard is also PIL.size
    :param circle:  old circle
    :return:        new circle
    """
    iw, ih = im_size
    ia_w, ia_h = size
    circle_trans = circle.copy()
    # new center_x center_y
    sc_w, sc_h = (ia_w - iw)//2, (ia_h - ih)//2
    # radius is not change
    circle_trans[0] += sc_w
    circle_trans[1] += sc_h
    circle_trans[3] += sc_w
    circle_trans[4] += sc_h
    return circle_trans


def transform(image, mask, circle, size):
    """
    resize image, mask to given size, and modify center point (x,y)
    :param image: PIL, 'RGB'
    :param mask:  PIL, 'L'
    :param circle: array: x1,y1,r1,x2,y2,r2
    :return: PIL, PIL, ndarray: new image, new mask, new center location
    """
    transform = T.Compose([
        T.CenterCrop(size=size)
    ])
    trans_image = transform(image)
    trans_mask = transform(mask)
    trans_circle = transform_circle(image.size, trans_image.size, circle)

    return trans_image, trans_mask, trans_circle


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def draw_circle(image, circle, color = (255, 0, 0)):
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
        img = cv2.circle(image, inner_center, inner_radius, color=color, thickness=1)
    else:
        if circle[2] < circle[5]:
            inner_center, inner_radius = (circle[0], circle[1]), circle[2]
            outer_center, outer_radius = (circle[3], circle[4]), circle[5]
        else:
            inner_center, inner_radius = (circle[3], circle[4]), circle[5]
            outer_center, outer_radius = (circle[0], circle[1]), circle[2]
        img = cv2.circle(image, inner_center, inner_radius, color=color, thickness=1)
        img = cv2.circle(img, outer_center, outer_radius, color=color, thickness=1)

    return img


if __name__ == "__main__":

    image = Image.open("./1.jpg").convert('RGB')
    mask = Image.open("./1.png").convert('L')
    circle = np.load("./1.npy")
    print(circle)
    print("image_size", image.size)

    img_trans, mask_trans, circle_trans = transform(image, mask, circle, (300, 400))  # h 300, w 400

    print("---")
    print(circle_trans)

    print("img_size",img_trans.size)

    img_for_view = np.array(img_trans, np.float32)
    print(img_for_view.shape)

    image_with_circle = draw_circle(img_for_view, circle)

    img_pil = Image.fromarray(image_with_circle.astype(np.uint8))
    img_pil.show()