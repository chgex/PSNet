

import os
import time
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F

from utils import get_circle_and_score, get_iou, draw_circle, convert_image, vision_image
from utils_a import compute_e1, compute_F1_socre


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


def get_dist(circleA, circleB):
    import math
    (x1, y1, r1), (x2, y2, r2) = circleA, circleB
    distance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return distance


def process(circle):
    assert len(circle) == 6
    dist = get_dist(circle)

    if circle[0] == 0 or circle[1] == 0 or dist >= circle[5]:
        circle[0], circle[1] = circle[3], circle[4]

    if circle[2] == 0:
        circle[2] = circle[5] / 3

    return circle



from train import load_data, load_model_and_weights, cfg


if __name__ == "__main__":

    model = load_model_and_weights("./checkpoints/" + cfg.checkpoints + "/" + "model.pth")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    model.to(device)
    model.eval()

    # dataloader
    data_loaders = load_data(cfg)
    # train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]

    print("===> eval model: compute mIoU")
    model.eval()

    total_iou, total_mask_iou = 0, 0
    total_mask_e1, total_mask_e2, total_mask_f1 = 0, 0, 0

    total = 0
    total_inner_iou, total_outer_iou = 0, 0

    det0_cnt, det1_cnt = 0, 0
    det0_list, det1_list = [], []
    low_score_list = []

    with torch.no_grad():
        for batch_idx, batch_all_data in enumerate(val_loader):
            batch_data = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch_all_data[:-1]]
            batch_masks = batch_all_data[-1].to(device)

            batch_images, batch_hms, \
            inner_batch_whs, inner_batch_regs, inner_batch_reg_masks, \
            outer_batch_whs, outer_batch_regs, outer_batch_reg_masks = batch_data

            print("---")
            outputs = model(batch_images)
            output = outputs[0]

            mask_logits, roi_coords = outputs[1], outputs[2]
            mask_preds = paste_cropped_mask(mask_logits, roi_coords, cfg.input_size)

            mask_iou, mask_e1, mask_e2, mask_f1 = compute_mask_iou(mask_preds, batch_masks)
            total_mask_iou += mask_iou
            total_mask_e1 += mask_e1
            total_mask_e2 += mask_e2
            total_mask_f1 += mask_f1

            heatmaps = output[0]
            inner_det_wh, inner_det_offset = output[1]["wh"], output[1]["offset"]
            outer_det_wh, outer_det_offset = output[2]["wh"], output[2]["offset"]

            # post process
            gt_inner_batch_info = get_circle_and_score(batch_hms, 0, inner_batch_whs, inner_batch_regs, 0.5,
                                                       device)
            gt_outer_batch_info = get_circle_and_score(batch_hms, 1, outer_batch_whs, outer_batch_regs, 0.5,
                                                       device)

            det_inner_batch_info = get_circle_and_score(heatmaps, 0, inner_det_wh, inner_det_offset, 0.01,
                                                        device)
            det_outer_batch_info = get_circle_and_score(heatmaps, 1, outer_det_wh, outer_det_offset, 0.01,
                                                        device)

            for gt_inner_info, det_inner_info, \
                gt_outer_info, det_outer_info in zip(gt_inner_batch_info, det_inner_batch_info,
                                                     gt_outer_batch_info, det_outer_batch_info):

                gt_inner_circle, scores1 = gt_inner_info
                gt_outer_circle, scores2 = gt_outer_info

                det_inner_circle, scores3 = det_inner_info
                det_outer_circle, scores4 = det_outer_info

                # process 0
                if det_inner_circle[0] == 0 or det_inner_circle[1] == 0 or get_dist(det_inner_circle, det_outer_circle) >= det_outer_circle[2]:
                    det_inner_circle[0], det_inner_circle[1] = det_outer_circle[0], det_outer_circle[1]
                    det_inner_circle[2] = det_outer_circle[2] / 3
                if det_outer_circle[0] == 0 or det_outer_circle[1] == 0:
                    det_outer_circle[0], det_outer_circle[1] = det_inner_circle[0], det_inner_circle[1]
                    det_outer_circle[2] = det_inner_circle[2] * 3

                """ compute inner/outer iou """
                inner_iou = get_iou(gt_inner_circle, det_inner_circle)
                outer_iou = get_iou(gt_outer_circle, det_outer_circle)

                if inner_iou == 0:
                    det0_cnt += 1
                    det0_list.append(total + 1)
                if outer_iou == 0:
                    det1_cnt += 1
                    det1_list.append(total + 1)
                if 0 < outer_iou <= 0.5:
                    low_score_list.append(total + 1)

                total_inner_iou += inner_iou
                total_outer_iou += outer_iou
                total += 1
                print("index:%d, inner_iou:%f, outer_iou:%f, mask_IoU: %.5f" % (total, inner_iou, outer_iou, mask_iou / cfg.batch_size))
                # vision_image(image_with_circle, total, iou)

    # compute mean iou
    mask_miou = total_mask_iou / total

    inner_iou = total_inner_iou / total
    outer_iou = total_outer_iou / total
    local_miou = (inner_iou + outer_iou) / 2

    mE1 = total_mask_e1 / total
    mE2 = total_mask_e2 / total
    mF1 = total_mask_f1 / total

    print("---")
    print("local mIou: ", local_miou, ", mask mIoU: ", mask_miou)
    print("inner mIoU: ", inner_iou, ", outer mIoU: ", outer_iou)
    print("mE1:", mE1, ", mE2:", mE2, ", mF1: ", mF1)

    print("det 0 cnt: ", det0_cnt, ", det 1 cnt:", det1_cnt)
    print("det0_list: ", det0_list)
    print("det1_list: ", det1_list)
    print("outer_low_list: ", low_score_list)


# 2021.10.30 (after fine turn)
""" v5.0 copy (add r_expand) (train with loader load_for_train and it's focal loss)
NICEII, 54.pth
    local mIou:  0.8429135636783531 , mask mIoU:  0.8964407456649105
    inner mIoU:  0.7397294213678488 , outer mIoU:  0.9460977059888563
    det 0 cnt:  0 , det 1 cnt: 0
    mE1: 0.006671449497767856 , mE2: 0.003335724748883928 , mF1:  0.9419639101033008

GS, 54.pth
use det process
    local mIou:  0.7514646013948557 , mask mIoU:  0.7001510021974908
    inner mIoU:  0.6403079684135319 , outer mIoU:  0.8620431173337337
    det 0 cnt:  10 , det 1 cnt: 1
    mE1: 0.004731469348848452 , mE2: 0.002365734674424226 , mF1:  0.80529679029688

IP, 73.pth, use det process
    local mIou:  0.7535048517306356 , mask mIoU:  0.7531673263918883
    inner mIoU:  0.6304324682195676 , outer mIoU:  0.8732973446574943
    det 0 cnt:  10 , det 1 cnt: 7
    mE1: 0.0034106682827700976 , mE2: 0.0017053341413850488 , mF1:  0.835653930093476 (when compute e1, e2, batch_size=20)

GT2, 65.pth, use det process
    local mIou:  0.7124689649480359 , mask mIoU:  0.6905465348041592
    inner mIoU:  0.5706539556229788 , outer mIoU:  0.8488281993066346
    det 0 cnt:  13 , det 1 cnt: 2
    mE1: 0.005073846726190476 , mE2: 0.002536923363095238 , mF1:  0.7951445843100557
"""


# 2021.12.1
""" model.pth
GS,
    local mIou:  0.7513558818861894 , mask mIoU:  0.694646066534001
    inner mIoU:  0.6406171085321581 , outer mIoU:  0.8620946552402208
    mE1: 0.004720687770705018 , mE2: 0.002360343885352509 , mF1:  0.8057773211165784
    det 0 cnt:  11 , det 1 cnt: 10
    det0_list:  [55, 200, 213, 226, 258, 268, 449, 545, 557, 579, 596]
    det1_list:  [55, 200, 213, 226, 258, 268, 545, 557, 579, 596]
    outer_low_list:  [48, 449, 568, 590, 597]
IP
    local mIou:  0.752116813887695 , mask mIoU:  0.7431433390259012
    inner mIoU:  0.6307888962942232 , outer mIoU:  0.8734447314811669
    mE1: 0.0033759367414094533 , mE2: 0.0016879683707047267 , mF1:  0.8375456483713415
    det 0 cnt:  17 , det 1 cnt: 10
    det0_list:  [10, 15, 49, 153, 158, 192, 193, 194, 210, 396, 434, 446, 468, 501, 512, 534, 620]
    det1_list:  [49, 153, 158, 210, 396, 434, 446, 468, 534, 620]
    outer_low_list:  [15, 48, 136, 154, 191, 192, 193, 194, 217, 218, 219, 490, 501, 512]
GT2
    local mIou:  0.7094227841493848 , mask mIoU:  0.6843874164272523
    inner mIoU:  0.5698561487769972 , outer mIoU:  0.8489894195217724
    mE1: 0.005116792748047773 , mE2: 0.0025583963740238866 , mF1:  0.7947771774659307
    det 0 cnt:  15 , det 1 cnt: 13
    det0_list:  [33, 34, 47, 103, 112, 168, 171, 172, 173, 177, 179, 183, 190, 201, 271]
    det1_list:  [33, 34, 112, 168, 171, 172, 173, 177, 179, 183, 190, 201, 271]
    outer_low_list:  []
"""
