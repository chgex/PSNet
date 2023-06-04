

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import numpy as np
import cv2
import os

# evaluating indicator
from utils_a import get_circle_and_score, get_iou
from utils_a import compute_e1, compute_F1_socre, norm_iris_mask, norm_iris_image
from utils_a.hdist import hausdorff, get_edg_mask


# local and seg utils
from utils_a.decode_mask import paste_cropped_mask

def transform_image(image, size):
    """
    resize image, mask to given size, and modify center point (x,y)
    :param image: PIL, 'RGB'
    :return: PIL, PIL,
    """
    transform = T.Compose([
        T.CenterCrop(size=size)
    ])
    trans_image = transform(image)

    return trans_image

def transform_circle(im_size, circle, to_size):
    """
    transform circle location to fit new size
    :param im_size: old image size, which formard is PIL.size, note: w, h = pil.size
    :param to_size: new image size, which formard is also PIL.size
    :param circle:  old circle, array: x1,y1,r1,x2,y2,r2
    :return:        new circle
    """
    iw, ih = im_size
    ia_w, ia_h = to_size
    circle_trans = circle.copy()
    # new center_x center_y
    sc_w, sc_h = (ia_w - iw)//2, (ia_h - ih)//2
    # radius is not change
    circle_trans[0] += sc_w
    circle_trans[1] += sc_h
    circle_trans[3] += sc_w
    circle_trans[4] += sc_h

    circle_trans[[0, 3]] = np.clip(circle_trans[[0, 3]], 0, ia_w - 1)
    circle_trans[[1, 4]] = np.clip(circle_trans[[1, 4]], 0, ia_h - 1)

    return circle_trans

def arrange(circle):
    """
    return [inner_x, inner_y, inner_r, outer_x, outer_y, outer_r]
    :param circle:
    :return:
    """
    assert len(circle) == 6
    if circle[2] < circle[5]:
        new_circle = [circle[i] for i in [0, 1, 2, 3, 4, 5]]
    else:
        new_circle = [circle[i] for i in [3, 4, 5, 0, 1, 2]]

    return np.array(new_circle)

def draw_circle(image_pil, circle, color=(255, 0, 0), thickness=1):
    """
    draw circle on image or mask image
    :param image: ndarray, shape is H W C
    :param circle: ndarray, shanpe is (6,0) or (3,0)
    :return: image
    """
    import cv2

    assert len(circle) == 3 or len(circle) == 6

    image = np.array(image_pil)
    image = image.astype(np.int32)
    circle = circle.astype(np.int32)

    if len(circle) == 3:
        inner_center, inner_radius = (circle[0], circle[1]), circle[2]
        img = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
    else:
        if circle[2] < circle[5]:
            inner_center, inner_radius = (circle[0], circle[1]), circle[2]
            outer_center, outer_radius = (circle[3], circle[4]), circle[5]
        else:
            inner_center, inner_radius = (circle[3], circle[4]), circle[5]
            outer_center, outer_radius = (circle[0], circle[1]), circle[2]
        img = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
        img = cv2.circle(img, outer_center, outer_radius, color=color, thickness=thickness)

    img_pil = Image.fromarray(img.astype("uint8"))
    return img_pil

def get_dist(circle):
    import math
    x1, y1, r1, x2, y2, r2 = circle
    distance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return distance

def process(circle):
    assert len(circle) == 6
    dist = get_dist(circle)

    if circle[0] == 0 or circle[1] == 0 or circle[2] == 0 or dist >= circle[5]:
        circle[0], circle[1], circle[2] = 0, 0, 0
    if circle[3] == 0 or circle[4] == 0 or circle[5] == 0:
        circle[3], circle[4], circle[5] = 0, 0, 0

    return circle


# load data
import glob
import pandas as pd
def load_images_and_labels(root_dataset_folder, dataName, signal):
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


# load weight
from train import load_model_and_weights, cfg


if __name__ == "__main__":

    model = load_model_and_weights("./checkpoints/" + cfg.checkpoints + "/" + "model.pth")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device.type))

    model.to(device)

    watch_list = []

    df = load_images_and_labels(cfg.root, cfg.dataName, "train")
    total_iou = 0
    total_inner_hdist, total_outer_hdist = 0, 0
    total_E1, total_E2, total_F1 = 0, 0, 0
    total_norm_e1 = 0
    total = 0
    det0_cnt = 0
    total_inner_iou, total_outer_iou = 0, 0
    for item in range(len(df)):
        # if item < 209:
        #     continue
        """ ground truth """
        image_path = df.iloc[item]['image_path']
        mask_path = df.iloc[item]['mask_path']
        circle_pth = df.iloc[item]['circle_path']
        # open
        image_pil = Image.open(image_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        gt_circle = np.load(circle_pth)

        gt_circle = arrange(gt_circle)
        gt_circle = np.around(gt_circle, decimals=5)
        image_w, image_h = image_pil.size

        """ detect circle """
        image = transform_image(image_pil, cfg.input_size)  # local_cfg.size
        # image.show()
        img = np.array(image, np.float32)
        img = np.transpose(img, (2, 0, 1))
        input = torch.from_numpy(img).type(torch.FloatTensor)   # .to(device)
        input = input.unsqueeze(0)
        if device:
            input = input.to(device)
        with torch.no_grad():
            output = model(input)

        out_local, mask_logits, roi_coords = output[0], output[1], output[2]

        # decode predicts
        heatmaps = out_local[0]
        inner_det_wh, inner_det_offset = out_local[1]["wh"], out_local[1]["offset"]
        outer_det_wh, outer_det_offset = out_local[2]["wh"], out_local[2]["offset"]

        det_inner_batch_info = get_circle_and_score(heatmaps, 0, inner_det_wh, inner_det_offset, 0.01, device)
        det_outer_batch_info = get_circle_and_score(heatmaps, 1, outer_det_wh, outer_det_offset, 0.01, device)
        # because batch_size = 1
        det_inner_circle, scores1 = det_inner_batch_info[0]
        det_outer_circle, scores2 = det_outer_batch_info[0]

        det_circle = np.append(det_inner_circle, det_outer_circle).astype("float")
        out_circle = transform_circle(image.size, det_circle, image_pil.size)
        out_circle = process(out_circle)
        print("---")

        """ compute inner/outer iou """
        inner_iou = get_iou(gt_circle[:3], out_circle[:3])
        outer_iou = get_iou(gt_circle[3:], out_circle[3:])
        total_inner_iou += inner_iou
        total_outer_iou += outer_iou
        # print(out_circle)
        # print(gt_circle)
        # # vision detected circle
        # image_for_vision = draw_circle(image_pil, out_circle)
        # image_for_vision.show()

        """ compute inner/outer circle's norm hausdorff dist"""
        out_circle = out_circle.astype("int")
        gt_circle = gt_circle.astype("int")
        if out_circle[5] == 0:
            outer_h_dist = 1
        else:
            det_outer_edg_mask, gt_outer_edg_mask = get_edg_mask(out_circle, gt_circle, size=(image_h, image_w), sign="outer")
            outer_h_dist = hausdorff(det_outer_edg_mask, gt_outer_edg_mask, size=(image_h, image_w))
        total_outer_hdist += outer_h_dist

        if out_circle[2] == 0:
            inner_h_dist = 1
        else:
            det_inner_edg_mask, gt_inner_edg_mask = get_edg_mask(out_circle, gt_circle, size=(image_h, image_w), sign="inner")
            inner_h_dist = hausdorff(det_inner_edg_mask, gt_inner_edg_mask, size=(image_h, image_w))
        total_inner_hdist += inner_h_dist

        assert 0 <= inner_h_dist <= 1 and 0 <= outer_h_dist <= 1
        # image1 = Image.fromarray(det_outer_edg_mask)
        # image1.show()
        # image2 = Image.fromarray(gt_outer_edg_mask)
        # image2.show()

        """ decode SEG Mask """
        seg_output = paste_cropped_mask(mask_logits, roi_coords, cfg.input_size)
        log_prob = F.softmax(seg_output, dim=1).data.cpu().numpy()  # b c h w
        pred = np.argmax(log_prob, axis=1)  # b h w, class index:0 is background, so the vale is 0 or 1.
        vision_mask = Image.fromarray((pred[0] * 255).astype("uint8"))
        seg_out = transform_image(vision_mask, (mask_pil.size[1], mask_pil.size[0]))
        # vision predicted mask and gt mask
        # seg_out.show()
        # mask_pil.show()

        """ compute mask IoU """
        gt_mask = np.array(mask_pil, dtype="int")
        gt_mask[gt_mask == 255] = 1

        pred_mask = np.array(seg_out, dtype="int")
        pred_mask[pred_mask == 255] = 1
        if np.sum(np.logical_or(pred_mask, gt_mask)) == 0:
            iou = 0
        else:
            iou = np.sum(np.logical_and(pred_mask, gt_mask)) / np.sum(np.logical_or(pred_mask, gt_mask))
        total_iou += iou

        """ compute mask e1, e2, f1_score """
        e1 = compute_e1(gt_mask, pred_mask)
        e2, f1_score, _, _ = compute_F1_socre(gt_mask, pred_mask)
        total_E1 += e1
        total_E2 += e2
        total_F1 += f1_score
        assert 0 <= e1 <= 1 and 0 <= e2 <= 1 and 0 <= f1_score <= 1

        """ compute norm iris e1 """
        if out_circle[2] == 0 or out_circle[5] == 0:
            norm_e1 = 1
            det0_cnt += 1
            watch_list.append(total + 1)
        else:
            gt_norm_iris = norm_iris_mask(gt_mask, gt_circle)
            # import matplotlib.pyplot as plt
            # plt.imshow(gt_norm_iris, cmap="gray")
            # plt.show()
            pred_norm_iris = norm_iris_mask(pred_mask, out_circle)
            # plt.imshow(pred_norm_iris, cmap="gray")
            # plt.show()
            norm_e1 = compute_e1(gt_norm_iris, pred_norm_iris)
            total_norm_e1 += norm_e1
        total += 1

        """ vision, blue, red """
        # img = np.array(image)
        # norm_image = norm_iris_image(img, out_circle)
        # norm_image = norm_image.astype(int)

        print("index:%d, inner_iou: %.5f, outer_iou: %.5f, mask_ioU: %.5f, inner_hdist: %.5f, outer_hdist: %.5f, e1: %.5f, norn_e1: %.5f"
              % (total, inner_iou, outer_iou, iou, inner_h_dist, outer_h_dist, e1, norm_e1))

    inner_miou = total_inner_iou / total
    outer_miou = total_outer_iou / total
    avg_miou = (inner_miou + outer_miou) / 2
    outer_mHdist = total_outer_hdist / total
    inner_mHdist = total_inner_hdist / total
    avg_mHdist = (inner_mHdist + outer_mHdist) / 2
    mIoU = total_iou / total
    mE1 = total_E1 / total
    mE2 = total_E2 / total
    mF1 = total_F1 / total
    norm_mE1 = total_norm_e1 / total
    print("\n total:", total, "det0_cnt:", det0_cnt)
    print(" inner_mIoU: %f, outer_mIoU: %f, loc_avg_mIoU:%.5f" % (inner_miou, outer_miou, avg_miou))
    print(" inner_mHdist: %f, outer_Hdist: %f, avg_mHdist:%.5f" % (inner_mHdist, outer_mHdist, avg_mHdist))
    print(" mE1:%.5f , mE2:%.5f , mF1_score:%.5f, mask_iou:%.5f" % (mE1, mE2, mF1, mIoU))
    print(" norm_mE1: %.5f" % (norm_mE1))


""" 
GS model-best.pth
     total: 634 det0_cnt: 0
     inner_mIoU: 0.595846, outer_mIoU: 0.838010, loc_avg_mIoU:0.71693
     inner_mHdist: 0.014980, outer_Hdist: 0.019428, avg_mHdist:0.01720
     mE1:0.00487 , mE2:0.00243 , mF1_score:0.81571, mask_iou:0.70830
     norm_mE1: 0.25838

# after fine turn (model.pth)
GS 
    total: 634 det0_cnt: 1
    inner_mIoU: 0.645266, outer_mIoU: 0.863545, loc_avg_mIoU:0.75441
    inner_mHdist: 0.015446, outer_Hdist: 0.016939, avg_mHdist:0.01619
    mE1:0.00505 , mE2:0.00252 , mF1_score:0.80829, mask_iou:0.69803
    norm_mE1: 0.25198
IP  **
    total: 631 det0_cnt: 9
    inner_mIoU: 0.638943, outer_mIoU: 0.878774, loc_avg_mIoU:0.75886
    inner_mHdist: 0.024999, outer_Hdist: 0.014794, avg_mHdist:0.01990
    mE1:0.00356 , mE2:0.00178 , mF1_score:0.84320, mask_iou:0.74906
    norm_mE1: 0.20702
GT2 **
    total: 316 det0_cnt: 5
    inner_mIoU: 0.574708, outer_mIoU: 0.860641, loc_avg_mIoU:0.71767
    inner_mHdist: 0.038183, outer_Hdist: 0.022779, avg_mHdist:0.03048
    mE1:0.00606 , mE2:0.00303 , mF1_score:0.79912, mask_iou:0.68572
    norm_mE1: 0.23199
NICEII **
    total: 1000 det0_cnt: 1
    inner_mIoU: 0.740537, outer_mIoU: 0.945687, loc_avg_mIoU:0.84311
    inner_mHdist: 0.014396, outer_Hdist: 0.010122, avg_mHdist:0.01226
    mE1:0.00791 , mE2:0.00396 , mF1_score:0.94233, mask_iou:0.89221
    norm_mE1: 0.09252
"""


