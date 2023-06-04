

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
    def __init__(self, df, transform, input_size, scale_size, num_classes, alphas, is_pad):
        self.df = df
        self.transform = transform
        self.input_size = input_size
        self.output_size = (int(input_size[0] / scale_size), int(input_size[1] / scale_size))
        self.num_classes = num_classes
        self.net_branch = 2
        self.alphas = alphas
        self.is_pad = is_pad

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        mask_path = self.df.iloc[item]['mask_path']
        circle_pth = self.df.iloc[item]['circle_path']

        image = Image.open(image_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        circle = np.load(circle_pth)

        img_trans, mask_trans, circle_trans = self.transform(image, mask_pil, circle, self.input_size)

        mask = np.array(mask_trans, dtype=np.float32)
        mask = np.ascontiguousarray(mask)
        mask[mask == 255.] = 1
        mask_target = torch.from_numpy(mask).type(torch.LongTensor)

        img, y = np.array(img_trans, np.float32), get_box_data(circle_trans)

        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_size[0], self.output_size[1], 1, self.net_branch), dtype=np.float32)
        batch_reg = np.zeros((self.output_size[0], self.output_size[1], 2, self.net_branch), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)

        boxes = np.array(y[:, :3], dtype=np.float32)   # x,y,r,c 中，取前3个
        # x,y,r
        boxes[:, 0] = boxes[:, 0] / self.input_size[1] * self.output_size[1]
        boxes[:, 1] = boxes[:, 1] / self.input_size[0] * self.output_size[0]
        boxes[:, 2] = boxes[:, 2] / self.input_size[1] * self.output_size[1]

        for i in range(2):
            bbox, cls_id = boxes[i].copy(), i
            ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 1, self.output_size[1] - 2)  # 0, -1
            bbox[[1, 2]] = np.clip(bbox[[1, 2]], 1, self.output_size[0] - 2)  # 0, -1
            r = bbox[2]
            if i == 0:
                radius = gaussian_radius((math.ceil(r), math.ceil(r)), min_overlap=0.5)
            else:
                radius = gaussian_radius((math.ceil(r), math.ceil(r)), min_overlap=0.8)
            radius = max(1, int(radius))

            if self.is_pad:
                batch_hm[:, :, cls_id] = -1
                batch_hm[:, :, cls_id] = draw_circle_on_mask(batch_hm[:, :, cls_id], bbox, 0)
            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)

        # # concentric circle
        alpha_inner, alpha_outer = self.alphas[0], self.alphas[1]
        concentric_hm = batch_hm[:, :, 0] * alpha_inner + batch_hm[:, :, 1] * alpha_outer
        # # # 
        batch_hm[:, :, 0] = batch_hm[:, :, 1] = concentric_hm
        batch_hm = np.clip(batch_hm, 0, 0.7)

        # gaussian: 1, 0.8
        for i in range(2):
            bbox, cls_id = boxes[i].copy(), i
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 1, self.output_size[1] - 2)  # 0, -1
            bbox[[1, 2]] = np.clip(bbox[[1, 2]], 1, self.output_size[0] - 2)  # 0, -1

            ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            # add 1, 0.8
            batch_hm[ct_int[1], ct_int[0], cls_id] = 1
            batch_hm[ct_int[1] + 1, ct_int[0], cls_id] = 0.8
            batch_hm[ct_int[1] - 1, ct_int[0], cls_id] = 0.8
            batch_hm[ct_int[1], ct_int[0] + 1, cls_id] = 0.8
            batch_hm[ct_int[1], ct_int[0] - 1, cls_id] = 0.8

            r = bbox[2]
            if self.is_pad:
                if cls_id == 0:
                    nonzero_mask = batch_hm[:, :, cls_id] >= 0
                else:
                    nonzero_mask = batch_hm[:, :, cls_id] > 0
                batch_wh[nonzero_mask, :, cls_id] = 1. * r
                batch_reg[ct_int[1], ct_int[0], :, cls_id] = ct - ct_int
                batch_reg_mask[nonzero_mask, cls_id] = 1
            else:
                batch_wh[ct_int[1], ct_int[0], :, cls_id] = 1. * r
                batch_reg[ct_int[1], ct_int[0], :, cls_id] = ct - ct_int
                batch_reg_mask[ct_int[1], ct_int[0], cls_id] = 1

        # (h w c) to (c h w) and contiguous memory
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


def collate_fn(batch_data):
    imgs, masks, batch_hms,\
    inner_batch_whs, inner_batch_regs, inner_batch_reg_masks, \
    outer_batch_whs, outer_batch_regs, outer_batch_reg_masks \
        = [], [], [], [], [], [], [], [], []

    for img, mask, batch_hm, \
        inner_batch_wh, inner_batch_reg, inner_batch_reg_mask, \
        outer_batch_wh, outer_batch_reg, outer_batch_reg_mask \
            in batch_data:
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

    train_data = load_images_labels_circles(cfg.root, cfg.dataName, "train")
    val_data = load_images_labels_circles(cfg.root, cfg.dataName, "test")

    datasets = {
        'train': CustomDataset(train_data, custom_transform, cfg.input_size, cfg.scale_size, cfg.num_class, cfg.alpha, cfg.is_pad),
        'val': CustomDataset(val_data, custom_transform, cfg.input_size, cfg.scale_size, cfg.num_class, cfg.alpha, cfg.is_pad),
    }

    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, num_workers=0,
                            shuffle=True, pin_memory=False, drop_last=True, collate_fn=collate_fn),
        "val": DataLoader(datasets["val"], batch_size=cfg.batch_size, num_workers=0,
                            shuffle=False, pin_memory=False, drop_last=True, collate_fn=collate_fn)
    }

    return dataloaders
