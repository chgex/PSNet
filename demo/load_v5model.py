

from YOLOv5_face import yolov5_face

import sys
sys.path.append('./YOLOv5_face/')


def load_yolov5face(weight, device):
    model = yolov5_face(weight, device)
    return model


# import torch
# yolo_face = torch.load("YOLOv5_face/model.pth")


# if __name__ == "__main__":
#
#     print("-" * 20)
#     # t = torch.load("./YOLOv5_face/model.pth")
#     model = load_yolov5face("./YOLOv5_face/yolov5n-face.pt", 'cpu')
#     print("*" * 20)


