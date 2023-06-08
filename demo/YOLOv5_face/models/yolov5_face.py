

"""File for accessing YOLOv5 via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, channels=3, classes=80)
"""


import os
import torch

from .yolo import Model
from utils.google_utils import attempt_download


def create(name, channels, classes, autoshape):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    # model.yaml path
    config = os.path.join('yamls', f'{name}.yaml')

    model = Model(config, channels, classes)

    # checkpoint filename
    fname = os.path.join('../weights', f'{name}-face.pt')
    # download if not found locally
    attempt_download(fname)

    # state dict
    ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter

    # load
    model.load_state_dict(state_dict, strict=False)

    if len(ckpt['model'].names) == classes:
        # set class names attribute
        model.names = ckpt['model'].names
    if autoshape:
        # for file/URI/PIL/cv2/np inputs and NMS
        model = model.autoshape()

    return model


def yolov5s(pretrained=False, channels=3, classes=80, autoshape=True):
    """YOLOv5-small model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5s', pretrained, channels, classes, autoshape)


def yolov5m(pretrained=False, channels=3, classes=80, autoshape=True):
    """YOLOv5-medium model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5m', pretrained, channels, classes, autoshape)


def yolov5l(pretrained=False, channels=3, classes=80, autoshape=True):
    """YOLOv5-large model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5l', pretrained, channels, classes, autoshape)


def yolov5x(pretrained=False, channels=3, classes=80, autoshape=True):
    """YOLOv5-xlarge model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5x', pretrained, channels, classes, autoshape)


if __name__ == '__main__':

    # pretrained example
    model = create(name='yolov5n', channels=3, classes=80, autoshape=True)
    print("-" * 10)

    # Verify inference
    from PIL import Image
    from pathlib import Path

    imgs = [Image.open(x) for x in Path('../images').glob('*.jpg')]
    results = model(imgs)
    results.show()
    results.print()
