

import os

import torch
import cv2
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device ...".format(device.type))

# from ICSNet import create_model, detect_image_iris
# icsnet = create_model("ICSNet/model.pth", device)
from PSNet import load_model_and_weight, detect_image_iris
psnet = load_model_and_weight("PSNet/model.pth", device)


def detect_one_image(image_path, save_to):
    os.makedirs(save_to, exist_ok=True)
    org_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    org_cv2 = cv2.cvtColor(org_cv2, cv2.COLOR_BGR2RGB)

    # image_show = copy.deepcopy(org_cv2)
    circle, mask, vision = detect_image_iris(org_cv2, psnet, device, vis=True)
    # show for debug
    # from PIL import Image
    # pil_show = Image.fromarray(image_show.astype('uint8'))
    # pil_show.show()

    image_show = cv2.cvtColor(vision, cv2.COLOR_RGB2BGR)
    to_path = os.path.join(save_to, os.path.basename(image_path))
    cv2.imwrite(to_path, image_show)
    print("-" * 30)


if __name__ == "__main__":
    image_root = "images-001"
    save_root = "outputs-001"

    image_List = os.listdir(image_root)
    total = len(image_List)

    for idx, image_name in enumerate(image_List):
        image_path = os.path.join(image_root, image_name)
        detect_one_image(image_path, save_to=save_root)

        print("current:", idx + 1, "total:", total)

    print("*" * 20)

    # image_path = os.path.join("images", "0031.jpg")
    # detect_one_image(image_path)
    # print('*' * 20)

