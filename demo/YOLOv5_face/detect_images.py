

import os
import copy
import numpy as np
import torch

from utils import letterbox, check_img_size
from utils import non_max_suppression_face, scale_coords


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def show_results(img, xyxy, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    # img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    lx, ly = int(landmarks[0]), int(landmarks[1])
    rx, ry = int(landmarks[2]), int(landmarks[3])
    dist = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
    dist_w, dist_h = int(np.ceil(dist / 4)), int(np.ceil(dist / 5))
    cv2.rectangle(img, (lx - dist_w, ly - dist_h), (lx + dist_w, ly + dist_h), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (rx - dist_w, ry - dist_h), (rx + dist_w, ry + dist_h), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)
        cv2.putText(img, str(i), (point_x-5, point_y-5),
                    0, 0.2, [225, 255, 255],
                    thickness=1, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


import cv2


def detect(model, source, device):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (640, 640)

    save_dir = "./outputs"
    print("save_dir: ", save_dir)

    print('loading images', source)
    image_list = os.listdir(source)
    total = len(image_list)

    for index, image_name in enumerate(image_list):
        image_path = os.path.join(source, image_name)
        org_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
        orgimg = cv2.cvtColor(org_cv2, cv2.COLOR_BGR2RGB)

        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from [w,h,c] to [c,w,h]
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        im0 = org_cv2.copy()
        # Process detections
        det = pred[0]

        # Rescale boxes from img_size to im0 size
        # for bounding box
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
        # for landmarks
        det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

        for j in range(det.size()[0]):
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            landmarks = det[j, 5:15].view(-1).tolist()
            class_num = det[j, 15].cpu().numpy()

            _ = show_results(im0, xyxy, conf, landmarks, class_num)

        cv2.imwrite(os.path.join("outputs/", image_name), im0)
        print("save to outputs/", image_name)
        # Save results (image with detections)
        print("-" * 10)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.create_model import load_model_and_weight
    model = load_model_and_weight('weights/yolov5n-face.pt', device)

    # model = load_model('weights/yolov5n-face.pt', device)
    detect(model, "./images", device)
