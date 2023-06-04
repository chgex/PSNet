

import math
from random import shuffle

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

from dataset.utils import draw_gaussian, gaussian_radius
from dataset.utils_loader import get_box_data, to_tensor, draw_circle_on_mask
from dataset.utils_loader import load_images_labels_circles
from dataset.utils import transform as custom_transform


class CustomDataset(Dataset):
    def __init__(self, df, transform, input_size, scale_size, num_classes):
        self.df = df
        self.transform = transform
        self.input_size = input_size
        self.output_size = (int(input_size[0] / scale_size), int(input_size[1] / scale_size))
        self.num_classes = num_classes
        self.net_branch = 2


    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        mask_path = self.df.iloc[item]['mask_path']
        circle_pth = self.df.iloc[item]['circle_path']

        image = Image.open(image_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        circle = np.load(circle_pth)
        # print("pil size: ", image.size)

        img_trans, mask_trans, circle_trans = self.transform(image, mask_pil, circle, self.input_size)

        mask = np.array(mask_trans, dtype=np.float32)
        mask = np.ascontiguousarray(mask)
        mask[mask == 255.] = 1
        mask_target = torch.from_numpy(mask).type(torch.LongTensor)
        # cls0_mask = mask.clone()
        # cls0_mask[mask == 0] = 1
        # cls0_mask[mask == 1] = 0
        # mask_target = torch.stack([cls0_mask, mask], dim=0)

        img, y = np.array(img_trans, np.float32), get_box_data(circle_trans)

        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_size[0], self.output_size[1], 1, self.net_branch), dtype=np.float32)
        batch_reg = np.zeros((self.output_size[0], self.output_size[1], 2, self.net_branch), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)

        boxes = np.array(y[:, :3], dtype=np.float32)
        # x,y,r
        boxes[:, 0] = boxes[:, 0] / self.input_size[1] * self.output_size[1]
        boxes[:, 1] = boxes[:, 1] / self.input_size[0] * self.output_size[0]
        boxes[:, 2] = boxes[:, 2] / self.input_size[1] * self.output_size[1]

        for i in range(len(y)):
            bbox = boxes[i].copy()
            bbox = np.array(bbox)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size[1] - 1)
            bbox[[1, 2]] = np.clip(bbox[[1, 2]], 0, self.output_size[0] - 1)

            # class and r
            cls_id = int(y[i, -1])
            r = bbox[2]
            assert r > 0

            radius = gaussian_radius((math.ceil(r), math.ceil(r)), min_overlap=0.7)
            radius = max(0, int(radius))

            ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
            batch_wh[ct_int[1], ct_int[0], :, cls_id] = 1. * r
            batch_reg[ct_int[1], ct_int[0], :, cls_id] = ct - ct_int
            batch_reg_mask[ct_int[1], ct_int[0], cls_id] = 1

        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)

        batch_hm = np.transpose(batch_hm, (2, 0, 1))
        batch_hm = np.ascontiguousarray(batch_hm)

        batch_wh = np.transpose(batch_wh, (3, 2, 0, 1))
        batch_wh = np.ascontiguousarray(batch_wh)

        batch_reg = np.transpose(batch_reg, (3, 2, 0, 1))
        batch_reg = np.ascontiguousarray(batch_reg)

        batch_reg_mask = np.transpose(batch_reg_mask, (2, 0, 1))
        batch_reg_mask = np.ascontiguousarray(batch_reg_mask)

        return img, mask_target, batch_hm, \
               batch_wh[0], batch_reg[0], batch_reg_mask[0],\
               batch_wh[1], batch_reg[1], batch_reg_mask[1]


def collate_fn(batch):
    imgs, masks, batch_hms,\
    inner_batch_whs, inner_batch_regs, inner_batch_reg_masks, \
    outer_batch_whs, outer_batch_regs, outer_batch_reg_masks \
        = [], [], [], [], [], [], [], [], []

    for img, mask, batch_hm, \
        inner_batch_wh, inner_batch_reg, inner_batch_reg_mask, \
        outer_batch_wh, outer_batch_reg, outer_batch_reg_mask \
            in batch:
        imgs.append(img)
        masks.append(mask)
        batch_hms.append(batch_hm)

        inner_batch_whs.append(inner_batch_wh)
        inner_batch_regs.append(inner_batch_reg)
        inner_batch_reg_masks.append(inner_batch_reg_mask)

        outer_batch_whs.append(outer_batch_wh)
        outer_batch_regs.append(outer_batch_reg)
        outer_batch_reg_masks.append(outer_batch_reg_mask)

    imgs = np.array(imgs)
    masks = torch.stack(masks)
    batch_hms = np.array(batch_hms)

    inner_batch_whs = np.array(inner_batch_whs)
    inner_batch_regs = np.array(inner_batch_regs)
    inner_batch_reg_masks = np.array(inner_batch_reg_masks)

    outer_batch_whs = np.array(outer_batch_whs)
    outer_batch_regs = np.array(outer_batch_regs)
    outer_batch_reg_masks = np.array(outer_batch_reg_masks)

    return [imgs, batch_hms,
           inner_batch_whs, inner_batch_regs, inner_batch_reg_masks,
           outer_batch_whs, outer_batch_regs, outer_batch_reg_masks, masks]


def load_data(cfg):
    # cfg = cfgs()
    root_dataset_folder = cfg.root
    dataName = cfg.dataName

    train_data = load_images_labels_circles(root_dataset_folder, dataName, "train")
    val_data = load_images_labels_circles(root_dataset_folder, dataName, "test")

    input_size = cfg.input_size
    scale_size = cfg.scale_size
    num_class = 2
    datasets = {
        'train': CustomDataset(train_data, custom_transform, input_size, scale_size, num_class),
        'val': CustomDataset(val_data, custom_transform, input_size, scale_size, num_class),
    }
    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, num_workers=0,
                            shuffle=True, pin_memory=False, drop_last=True, collate_fn=collate_fn),
        "val": DataLoader(datasets["val"], batch_size=cfg.batch_size, num_workers=0,
                            shuffle=False, pin_memory=False, drop_last=True, collate_fn=collate_fn)
    }
    return dataloaders
