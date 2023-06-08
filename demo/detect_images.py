

import os

import torch
import cv2
import copy

from YOLOv5_face import detect_image_face
from load_v5model import load_yolov5face

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device ...".format(device.type))

yolo_face = load_yolov5face("YOLOv5_face/yolov5n-face.pt", device)
# from ICSNet import create_model, detect_image_iris
# icsnet = create_model("ICSNet/model.pth", device)
from PSNet import load_model_and_weight, detect_image_iris
psnet = load_model_and_weight("PSNet/model.pth", device)


def detect_one_image(image_path, save_to):
    os.makedirs(save_to, exist_ok=True)
    org_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    org_cv2 = cv2.cvtColor(org_cv2, cv2.COLOR_BGR2RGB)

    image_show = copy.deepcopy(org_cv2)
    lr_list = detect_image_face(yolo_face, image_show, device, vis_face=False, vis_mark=False, vis_box=False)

    for lr_box in lr_list:
        for xyxy in lr_box:
            # generally, len(lr_box) == 2, indicate left eye and right eye.
            x1, y1, x2, y2 = xyxy
            if min(x1, y1, x2, y2) <= 10:
                continue
            else:
                cropped_region = org_cv2[y1:y2, x1:x2, :]
                print("  cropped_region size: ", cropped_region.shape)
                circle, mask, vision = detect_image_iris(cropped_region, psnet, device, vis=True)
                image_show[y1:y2, x1:x2, :] = vision
                # break
    # show for debug
    # from PIL import Image
    # pil_show = Image.fromarray(image_show.astype('uint8'))
    # pil_show.show()

    image_show = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
    to_path = os.path.join(save_to, os.path.basename(image_path))
    cv2.imwrite(to_path, image_show)
    print("-" * 30)


if __name__ == "__main__":

    image_root = "images-002"
    out_root = "outputs-002"

    image_List = os.listdir(image_root)
    total = len(image_List)
    for idx, image_name in enumerate(image_List):
        image_path = os.path.join(image_root, image_name)

        detect_one_image(image_path, out_root)

        print("current:", idx, "total:", total)

    print("*" * 20)

    # image_path = os.path.join("images", "0031.jpg")
    # detect_one_image(image_path)
    # print('*' * 20)

