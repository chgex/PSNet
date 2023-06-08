

import os
import torch

from .model import CenterNet
from .default_config import Config as cfg


def load_model_and_weight(weight_path, device=None):
    print("load model weight from: {}".format(weight_path))

    assert os.path.exists(weight_path)

    model = CenterNet(cfg)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    if device:
        model = model.to(device)
    model.eval()
    return model

