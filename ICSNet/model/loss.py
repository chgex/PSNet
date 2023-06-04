

import torch
import torch.nn as nn
import numpy as np

def reg_l1_loss(pred, target, mask):
    """
    compute l1_loss
    :param pred:    b c h w
    :param target:  b c h w
    :param mask:    b h w
    :return:
    """
    # # pred = pred.permute(0,2,3,1)
    # # (b h w) to (b h w 1) to (b h w 2)
    # expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)
    # # (b, h, w, 2) to (b 2 h w)
    # expand_mask = expand_mask.permute(0,3,1,2)

    expand_mask = torch.unsqueeze(mask, 1)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)

    return loss


def offset_l1_loss(pred, target, mask):

    expand_mask = torch.unsqueeze(mask,1).repeat(1,2,1,1)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)

    return loss


def focal_loss_niceii(pred, target):
    """
    :param pred:    b 2 128 128
    :param target:  b 2 128 128
    :return:  loss
    """
    # pred_for_view = pred.cpu().detach().numpy()
    # # target = torch.unsqueeze(target, 1)
    # target_for_iew = target.cpu().detach().numpy()

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


# if pad with -1
def focal_loss_miche(pred, target):
    """
    :param pred:    b 2 128 128
    :param target:  b 2 128 128
    :return:  loss
    """
    assert target.shape == pred.shape and target.shape[1] == 2

    pos_inds = target.ge(0).float()
    neg_inds = target.lt(0).float()

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def iou_loss(preds, targets, masks):
    # (b 1 h w), (b 1 h w), (b h w)
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(1)

    assert preds.shape == targets.shape == masks.shape
    masks = masks.long()

    total_loss = 0
    batch = preds.shape[0]
    for index in range(batch):
        pred = torch.reshape(preds[index], [-1, 1])  # (h*w, 1)
        target = torch.reshape(targets[index], [-1, 1])  # (h*w, 1)
        mask = torch.reshape(masks[index], [-1, 1])

        assert pred.shape == target.shape == mask.shape

        mask_pos = mask > 0
        tmp = pred[mask_pos]

        radius = torch.cat([pred[mask_pos].unsqueeze(1), target[mask_pos].unsqueeze(1)], dim=1)
        min_radius, _p = torch.min(radius, dim=1)
        max_radius, _p = torch.max(radius, dim=1)
        loss = -(min_radius / max_radius).log_()

        total_loss += loss.sum()

    return total_loss / batch


def bce_loss(logits, targets, masks):

    assert logits.shape == targets.shape
    masks = masks.long()

    loss = 0
    batch = logits.shape[0]
    for index in range(batch):
        logit = torch.reshape(logits[index], [-1, 1])
        target = torch.reshape(targets[index], [-1, 1])
        mask = torch.reshape(masks[index], [-1, 1])

        loss += F.binary_cross_entropy_with_logits(input=logit[mask], target=target[mask]).view(1)

    return loss / batch


import torch.nn.functional as F

loss_fn = nn.CrossEntropyLoss()

def compute_mask_loss(logits, targets):

    loss = loss_fn(logits, targets)

    # compute IoU
    log_prob = F.softmax(logits, dim=1).data.cpu().numpy()  # b c h w to b h w
    pred = np.argmax(log_prob, axis=1)  # b h w, class index:0 is background, so the vale is 0 or 1.
    # b h w
    gt = targets.data.cpu().numpy()
    if np.sum(np.logical_or(pred, gt)) == 0:
        IoU = 0
    else:
        IoU = np.sum(np.logical_and(pred, gt)) / np.sum(np.logical_or(pred, gt))

    return loss, IoU


def crop_mask_targets(rois, batch_mask):
    b = batch_mask.shape[0]
    assert len(rois) == b

    croped_list = []
    for batch in range(b):
        mask = batch_mask[batch]
        coord = rois[batch]
        top, bottom, left, right = coord
        croped_mask = mask[top:bottom, left:right]
        croped_list.append(croped_mask)

    croped_list = torch.stack(croped_list)

    return croped_list


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
    pred = np.argmax(log_prob, axis=1)  # b h w, class index:0 is background, so the vale is 0 or 1.
    # b h w
    gt = targets.data.cpu().numpy()
    if np.sum(np.logical_or(pred, gt)) == 0:
        IoU = 0
    else:
        IoU = np.sum(np.logical_and(pred, gt)) / np.sum(np.logical_or(pred, gt))

    return IoU


