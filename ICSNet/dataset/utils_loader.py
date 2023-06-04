

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as F


def get_box_data(circle, num_classes = 2):
        """
        convert [x,y,r,x,y,r] to [[x,y,r,c],[x,y,r,c]]
        :param circle: ndarray (6,)
        :return:       (2,4)
        """
        # le = len(circle)
        assert len(circle) == 6
        if circle[2] < circle[5]:
            inner_center, inner_radius = (circle[0], circle[1]), circle[2]
            outer_center, outer_radius = (circle[3], circle[4]), circle[5]
        else:
            inner_center, inner_radius = (circle[3], circle[4]), circle[5]
            outer_center, outer_radius = (circle[0], circle[1]), circle[2]
        box_data = np.zeros((num_classes, 4)) # (2,4)
        # 0: inner_circle, 1:outer_circle
        # x,y,r,class * num_box
        assert num_classes == 2
        # inner class:0
        box_data[0,0],box_data[0,1] = inner_center
        box_data[0,2],box_data[0,3] = inner_radius,0
        # outer class:1
        box_data[1, 0], box_data[1, 1] = outer_center
        box_data[1, 2], box_data[1, 3] = outer_radius,1

        return box_data


def to_tensor(image):
        """
        convert ndarray image h*w*c to torch.FloatTensor of shape c*h*h
        :param image: h w c in the range (0,255)
        :return:      c h w in the range (0.0,1.0)
        """
        return F.to_tensor(image)


def draw_circle_on_mask(mask, circle, color, thickness=-1):
    import cv2
    assert len(circle) == 3

    mask = np.array(mask, dtype=np.int32)
    circle = np.array(circle, dtype=np.int32)

    center, radius = (circle[0], circle[1]), circle[2]
    mask_circle = cv2.circle(mask, center, radius, color=color, thickness=thickness)

    return mask_circle


import os
import glob
import pandas as pd


def load_images_labels_circles(root_dataset_folder, dataName, signal):
    assert signal in ["train", "test"]

    imgs_pth = os.path.join(root_dataset_folder, signal, dataName, "images", '*.jpg')
    all_source_imgs = sorted(glob.glob(imgs_pth))

    label_pth = os.path.join(root_dataset_folder, signal, dataName, 'labels', '*.png')
    all_target_imgs = sorted(glob.glob(label_pth))

    circle_pth = os.path.join(root_dataset_folder, signal, dataName, 'circles', '*.npy')
    all_source_circles = sorted(glob.glob(circle_pth))

    data = pd.DataFrame({'image_path': all_source_imgs, 'mask_path': all_target_imgs, 'circle_path': all_source_circles})

    print("data_path: ", root_dataset_folder, "/", signal, "/", dataName)
    print('total: ', len(all_source_imgs))

    assert len(all_source_imgs) == len(all_target_imgs) == len(all_source_circles)

    return data

