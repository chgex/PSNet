

"""
NICEII, image wh size: 300, 400, input size: (320, 448)
IPhone5, image wh size: 270, 444, input_size: (320, 448)
GS, image wh size: 270, 444, input_size: (320, 448)
SamsungGalaxyTab2, image wh size: 400, 300, input_size: (448, 320)  <--- note this
"""

input_size_dic = {"IP": (448, 288), "GS": (448, 288), "GT2": (320, 448), "NICEII": (320, 448)}
mask_size_dict = {"IP": (128, 128), "GS": (128, 128), "GT2": (128, 128), "NICEII": (224, 224)}

alpha_dict = {"MICHE": [0.8, 0.2], "NICEII": [0, 0]}
pad_dict = {"MICHE": True, "NICEII": False}


class Config():

    # dataset
    root = "../IPGSGT"
    dataName = "NICEII"
    # dataName = "IP"
    # dataName = "GS"
    # dataName = "GT2"

    # Custom Data Loader
    input_size = input_size_dic[dataName]
    scale_size = 1
    num_class = 2  # inner/outer
    alpha = alpha_dict["NICEII"] if dataName == "NICEII" else alpha_dict["MICHE"]
    is_pad = False if dataName == "NICEII" else True

    # Model
    mask_size = mask_size_dict[dataName]
    feature_map_channels = 32
    r_expand = 10

    # Training
    batch_size = 3
    epoch_val = 5
    epoch_eval = 5

    # Log info
    checkpoints = dataName
    log_info = dataName


""" train strategy

transfer training:  NICEII --> IP --> GS --> GT2 --> NICEII --> IP --> GS --> GT2.

"""