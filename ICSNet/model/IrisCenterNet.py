

import torch.nn as nn
import torch
import numpy as np


from model.backbone.simple_dla import SimpleDLA_2
from .head import Local_Head, Mask_Head


# from .backbone.att_module import SpatialAttentionMaskHead
# class CenterNet_SEG(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.cfg = config
#
#         # self.backbone = UNet_D4_L3(3, 32)
#         self.backbone = SimpleDLA_2()
#         self.local_head = Local_Head(channel=32, num_classes=2)
#         if self.cfg.state:
#             self.backbone.load_state_dict(torch.load(self.cfg.backbone_weight_path))
#             self.local_head.load_state_dict(torch.load(self.cfg.local_weight_path))
#
#         self.mask_head = SpatialAttentionMaskHead(channel=32, num_classes=2)
#
#     def forward(self, x):
#         feature_map = self.backbone(x)
#         output = self.local_head(feature_map)
#
#         heatmap = output[0]
#         roi_coord = get_roi(heatmap)
#         croped_rois, hook_coord = crop_roi(feature_map, roi_coord, self.cfg.mask_size)
#
#         # focus mask
#         outer_wh = output[2]["wh"].detach().clone()
#         roi_masks = get_wh_mask(roi_coord, outer_wh, self.cfg.input_size, self.cfg.r_expand)
#         roi_masks = roi_masks.to(feature_map.device)
#         roi_masks = roi_masks.unsqueeze(1).repeat(1, 32, 1, 1)
#         croped_masks, _ = crop_roi(roi_masks, roi_coord, self.cfg.mask_size)
#
#         assert croped_rois.shape == croped_masks.shape
#         focus_featmap = croped_rois * croped_masks
#         mask_logit = self.mask_head(focus_featmap)
#
#         # mask_logit = self.mask_head(croped_rois)
#
#         return [output, mask_logit, hook_coord]


class CenterNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        # self.backbone = UNet_D4_L3(3, 32)
        self.backbone = SimpleDLA_2()

        self.local_head = Local_Head(channel=32, num_classes=2)

        self.mask_head = Mask_Head(32, num_classes=2)

    def forward(self, x):
        feature_map = self.backbone(x)
        output = self.local_head(feature_map)

        heatmap = output[0]
        roi_coord = get_roi(heatmap)
        croped_rois, hook_coord = crop_roi(feature_map, roi_coord, self.cfg.mask_size)

        # focus mask
        outer_wh = output[2]["wh"].detach().clone()
        roi_masks = get_wh_mask(roi_coord, outer_wh, self.cfg.input_size, self.cfg.r_expand)
        roi_masks = roi_masks.to(feature_map.device)
        roi_masks = roi_masks.unsqueeze(1).repeat(1, 32, 1, 1)
        croped_masks, _ = crop_roi(roi_masks, roi_coord, self.cfg.mask_size)

        assert croped_rois.shape == croped_masks.shape
        focus_featmap = croped_rois * croped_masks
        mask_logit = self.mask_head(focus_featmap)

        # mask_logit = self.mask_head(croped_rois)

        return [output, mask_logit, hook_coord]


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_roi(heatmapss, topk=1):

    heatmaps = heatmapss.detach().clone()

    b, c, output_h, output_w = heatmaps.shape
    assert c == 2

    heatmap = heatmaps[:, 1, :, :].unsqueeze(1)
    heatmap = pool_nms(heatmap)

    # hm_for_view = heatmap[0].cpu().numpy()

    roi_list = []
    for batch in range(b):

        heat_map = heatmap[batch].reshape([-1, 1])

        xv, yv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        xv, yv = xv.flatten().float(), yv.flatten().float()

        xv = xv.to(heat_map.device)
        yv = yv.to(heat_map.device)

        all_coordinate = torch.cat([xv.unsqueeze(1), yv.unsqueeze(1), heat_map], dim=1)

        arg_sort = torch.argsort(all_coordinate[:, -1], descending=True)
        roi = all_coordinate[arg_sort]

        roi_list.append(roi.cpu().numpy()[:1][0])
        # roi_list.append(roi.cpu().numpy()[:topk])

    roi_list = np.array(roi_list)

    return roi_list


def crop_roi(feature_map, rois, size=(128, 128)):

    b, c, h, w = feature_map.shape

    radius = size[0] // 2

    crop_list = []
    hook_coord = []
    for batch in range(b):
        feature = feature_map[batch]
        roi = rois[batch]
        center_x, center_y = roi[0], roi[1]
        top = center_x - radius
        left = center_y - radius
        bottom = center_x + radius
        right = center_y + radius

        if top < 0:
            top = 0
            bottom += (radius - center_x)
        if bottom > h - 1:
            bottom = h - 1
            top -= (center_x + radius - h) + 1
        if left < 0:
            left = 0
            right += (radius - center_y)
        if right > w - 1:
            right = w - 1
            left -= (center_y + radius - w) + 1

        top, left = int(max(top, 0)), int(max(left, 0))
        bottom, right = int(min(bottom, h-1)), int(min(right, w-1))

        crop_feat = feature[:, top:bottom, left:right]
        crop_list.append(crop_feat)
        hook_coord.append([top, bottom, left, right])

    crop_list = torch.stack(crop_list)
    hook_coord = np.array(hook_coord)

    return crop_list, hook_coord


import cv2


def get_wh_mask(rois, outer_whs, input_size, r_expand):

    r_max = int(input_size[0] / 2 - 1)

    assert len(rois) == len(outer_whs)
    b = outer_whs.shape[0]

    reg_list = []
    for batch in range(b):
        roi = rois[batch]
        outer_wh = outer_whs[batch]

        center_x, center_y = int(roi[0]), int(roi[1])
        radius = outer_wh[0, center_x, center_y]
        radius += r_expand
        radius = min(max(0, int(radius)), r_max - 1)

        assert radius < r_max

        mask = np.zeros((input_size[0], input_size[1]), dtype=np.int32)
        mask = cv2.circle(mask, (center_y, center_x), radius, color=1, thickness=-1)
        reg_list.append(mask)

    reg_list = np.array(reg_list)
    batch_mask = torch.from_numpy(reg_list).type(torch.LongTensor)

    return batch_mask

















