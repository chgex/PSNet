

import torch

# print("*" * 20)
# t = torch.load("model.pth")
# print("*" * 20)


if __name__ == "__main__":

    # weights = "yolov5n-face.pt"
    # pt = torch.load(weights)
    # print("*" * 20)

    from models import create_and_load
    config = "./models/yamls/yolov5n.yaml"
    model = create_and_load(config, "yolov5n-face.pt")

    print("-" * 20)


