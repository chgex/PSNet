

import numpy as np


def color_image(image_pil, pred_mask, gt_mask):

    image = np.array(image_pil)

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    assert image.shape[:2] == pred_mask.shape == gt_mask.shape

    image_h, image_w = image.shape[:2]
    for row in range(image_h):
        for col in range(image_w):
            # true positive, bule
            if gt_mask[row, col] == 1. and pred_mask[row, col] == 1.:
                image[row, col] = [0, 0, 255]
            # false positive, green
            if gt_mask[row, col] == 0 and pred_mask[row, col] == 1:
                image[row, col] = [0, 255, 0]
            # false negative, red
            if gt_mask[row, col] == 1 and pred_mask[row, col] == 0:
                image[row, col] = [255, 0, 0]
    return image