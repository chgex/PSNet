

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


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


def compute_mask_iou(logits, targets):

    log_prob = F.softmax(logits, dim=1).data.cpu().numpy()  # b c h w to b h w
    preds = np.argmax(log_prob, axis=1)  # b h w, class index:0 is background, so the vale is 0 or 1.
    # b h w
    gts = targets.data.cpu().numpy()
    # if np.sum(np.logical_or(preds, gts)) == 0:
    #     IoU = 0
    # else:
    #     IoU = np.sum(np.logical_and(preds, gts)) / np.sum(np.logical_or(preds, gts))
    # return IoU
    b = logits.shape[0]
    batch_iou = 0
    batch_e1, batch_e2, batch_f1 = 0, 0, 0
    for index in range(b):
        pred = preds[index]
        gt = gts[index]
        if np.sum(np.logical_or(pred, gt)) == 0:
            IoU = 0
        else:
            IoU = np.sum(np.logical_and(pred, gt)) / np.sum(np.logical_or(pred, gt))
        batch_iou += IoU

        e1 = compute_e1(gt, pred)
        e2, f1, _, _ = compute_F1_socre(gt, pred)
        batch_e1 += e1
        batch_e2 += e2
        batch_f1 += f1

    return batch_iou, batch_e1, batch_e2, batch_f1
