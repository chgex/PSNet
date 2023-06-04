

__all__ = ['decode_bbox', 'get_correct_boxes', 'get_bbox', 'get_circle_and_score']


import torch
import torch.nn as nn

import numpy as np


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_bbox(pred_hms, cls, pred_whs, pred_offsets, device, threshold=0.001, topk=1, scale_size=1):
    """
    :param pred_hms: Tensor of (b,num_class, h, w)  b 2 128 128
    :param pred_whs: Tensor of (b,num_class, h, w)  b 1 128 128
    :param pred_offsets:  Tensor of (b,num_class, h, w)  b 2 128 128
    :param device:  cuda or cpu
    :param threshold: confidence of bbox
    :param topk: remain topK bbox
    :return:
    """
    # -------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128,
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    # -------------------------------------------------------------------------#
    pred_hms = pool_nms(pred_hms)
    pred_offsets = torch.clamp(pred_offsets, 0, 1)

    # hm_for_view = pred_hms[0].cpu().numpy()

    # cls = 0 or 1
    pred_hm = pred_hms[:, cls, :, :]
    # b 128 128

    b, c, output_h, output_w = pred_hms.shape
    detects = []
    for batch in range(b):
        # -------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        # -------------------------------------------------------------------------#
        heat_map = pred_hm[batch].view([-1, 1])
        pred_wh = pred_whs[batch].permute(1, 2, 0).view([-1, 1])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if device:
            xv = xv.to(device)
            yv = yv.to(device)

        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > threshold

        #   取出得分筛选后对应的结果
        pred_wh_mask = pred_wh[mask]
        pred_offset_mask = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue

        #   计算调整后预测框的中心
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)

        #   计算预测框的宽高
        r = pred_wh_mask[..., 0:1]
        h = w = r

        # box: x,y,w,h
        bboxes = torch.cat([xv_mask, yv_mask, h, w], dim=1)
        bboxes[:, [0, 1, 2, 3]] *= scale_size


        detect = torch.cat(
            [bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)

        arg_sort = torch.argsort(detect[:, -2], descending=True)
        detect = detect[arg_sort]

        detects.append(detect.cpu().numpy()[:topk])

    return detects


def get_correct_boxes(x, y, w, h):
    # image_shape = np.array(cfg.input_size)
    # image_shape = np.array([320, 448])
    image_shape = np.array([1, 1])

    box_xy = np.concatenate((x, y), axis=-1)
    box_wh = np.concatenate((w, h), axis=-1)

    boxes = np.concatenate([
        box_xy[:, 0:1],
        box_xy[:, 1:2],
        box_wh[:, 0:1],
        box_wh[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def get_bbox(det_boxes, det_conf, det_label, confidence = 0.5):
    """
    given det_boxes, sort by confidence and class
    :param det_boxes:  ndarray: (num_box, 4)
    :param det_conf:   ndarray: (num_box,)
    :param det_label:  ndarray: (num_box,)
    :return:    det_box with specify cls and conf > confidence
    """
    x, y, w, h = det_boxes[:, 0], det_boxes[:, 1], det_boxes[:, 2], det_boxes[:, 3]

    # top indices, sorted by conf
    # top_indices = [i for i, dt_c in enumerate(det_conf) if dt_c >= confidence]
    top_indices = [i for i, dt_c in enumerate(det_conf)]

    top_x, top_y = np.expand_dims(x[top_indices], -1), np.expand_dims(y[top_indices], -1)
    top_w, top_h = np.expand_dims(w[top_indices], -1), np.expand_dims(h[top_indices], -1)

    # convert location (0., 1.)  to fit image size (512,512)
    # boxes = get_correct_boxes(top_x, top_y, top_h, top_w)

    box_xy = np.concatenate((top_x, top_y), axis=-1)
    box_wh = np.concatenate((top_w, top_h), axis=-1)
    boxes = np.concatenate([
        box_xy[:, 0:1],
        box_xy[:, 1:2],
        box_wh[:, 0:1],
        box_wh[:, 1:2]
    ], axis=-1)

    # boxes = get_correct_boxes(top_x, top_y, top_h, top_w)
    # 0.8211532199553594


    return boxes, det_conf[top_indices]


def get_circle_and_score(hms, cls, whs, regs, confidence, device):
    batch_info = decode_bbox(hms, cls, whs, regs, device)

    circle_and_score_list = []
    for batch, info in enumerate(batch_info):
        # info = info[0]
        if len(info) == 0:
            circle, score = np.array([0, 0, 0]), 0
        else:
            t_bbox, conf, label = info[:, :4], info[:, 4], info[:, 5]
            boxes, scores = get_bbox(t_bbox, conf, label, confidence)
            # remain only one circle
            x, y, w, h = boxes[0]
            circle, score = np.array([x, y, w]), scores[0]
        circle_and_score_list.append([circle, score])

    return circle_and_score_list
