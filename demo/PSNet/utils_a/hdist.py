

import torch
import numpy as np
import cv2
from PIL import Image


def get_outer_coordinate(circle):
    circle = np.array(circle, dtype=int)
    assert len(circle) == 6 or len(circle) == 3

    if len(circle) == 6 and circle[2] < circle[5]:
        outer_circle_center, radius = (circle[3], circle[4]), circle[5]
    else:
        outer_circle_center, radius = (circle[0], circle[1]), circle[2]

    return outer_circle_center, radius


def get_inner_coordinate(circle):
    circle = np.array(circle, dtype=int)
    assert len(circle) == 6 or len(circle) == 3

    if len(circle) == 6 and circle[2] > circle[5]:
        inner_circle_center, radius = (circle[3], circle[4]), circle[5]
    else:
        inner_circle_center, radius = (circle[0], circle[1]), circle[2]

    return inner_circle_center, radius


def get_edg_mask(det_circle, gt_circle, size, sign):

    assert sign in ["inner", "outer"]

    if sign == "outer":
        center1, radius1 = get_outer_coordinate(gt_circle)
        center2, radius2 = get_outer_coordinate(det_circle)
    else:
        center1, radius1 = get_inner_coordinate(gt_circle)
        center2, radius2 = get_inner_coordinate(det_circle)

    gt_edg_mask = np.zeros(size, dtype="uint8")
    gt_edg_mask = cv2.circle(gt_edg_mask, center1, radius1, color=(255), thickness=1)

    det_edg_mask = np.zeros(size, dtype="uint8")
    det_edg_mask = cv2.circle(det_edg_mask, center2, radius2, color=(255), thickness=1)

    return det_edg_mask, gt_edg_mask


def get_H(points_A, points_B):
    """
    compute H(A, B)
    :param points_A: np.ndarray, shape is [N,2]
    :param points_B: np.ndarray, shape is [N,2]
    :return: H(A, B)
    """

    # if points_B.shape[1] == 2:
    #     points_B = np.transpose(points_B)
    # if points_A.shape[0] == 2:
    #     points_A = np.transpose(points_A)

    assert points_A.shape[1] == points_B.shape[1] == 2

    l_min_list = []
    for point in points_A:
        l2_dists = np.sqrt(np.sum(np.power((points_B - point), 2), axis=1))
        l_min = np.min(l2_dists)
        # print(l_min)
        l_min_list.append(l_min)
    l_max = max(l_min_list)
    # print(l_max)
    return l_max


def hausdorff(A, B, size, use_Norm=True):
    """
    :param A: shape is (h, w), mask image
    :param B: shape is (h, w), mask image
    :return: h(a, b)
    """
    A = np.array(A, dtype="int")
    B = np.array(B, dtype="int")

    if np.max(A) == 0 or np.max(B) == 0:
        if np.max(A) == 0 and np.max(B) == 0:
            return 0
        else:
            return 1
    # 1 h w to h w
    if len(A.shape) == 3:
        A = A.squeeze(0)
    if len(B.shape) == 3:
        B = B.squeeze(0)

    assert len(A.shape) == len(B.shape) == 2
    im_h, im_w = size

    # get circle contour coordinates from mask image
    rows, cols = np.nonzero(A)
    if use_Norm:
        rows, cols = rows / im_h, cols / im_w
    points_A = np.concatenate([rows.reshape(-1, 1), cols.reshape(-1, 1)], axis=-1)

    rows, cols = np.nonzero(B)
    if use_Norm:
        rows, cols = rows / im_h, cols / im_w
    points_B = np.concatenate([rows.reshape(-1, 1), cols.reshape(-1, 1)], axis=-1)

    # compute H(A,B) and H(B,A)
    h_ab = get_H(points_A, points_B)
    h_ba = get_H(points_B, points_A)

    hsdf_dist = max(h_ab, h_ba)

    return hsdf_dist


from utils_a import get_iou


if __name__ == "__main__":

    # a = torch.ones(128, 128)
    # b = a
    # h = hausdorff(a, b)
    #
    # print("h: ", h)

    a = np.array([130, 60, 6, 130, 60, 25])
    b = np.array([130, 66, 6, 132, 66, 26])

    det_edg_mask, gt_edg_mask = get_edg_mask(a, b, (444, 270), "outer")
    print(det_edg_mask.shape)
    print(gt_edg_mask.shape)

    image1 = Image.fromarray(det_edg_mask)
    image1.show()
    image2 = Image.fromarray(gt_edg_mask)
    image2.show()

    h = hausdorff(det_edg_mask, gt_edg_mask, (444, 270))
    print(h)

    # a = np.ones((2, 4))
    # rs, cs = np.nonzero(a)
    # print(a)
    # print(rs)
    # print(cs)



