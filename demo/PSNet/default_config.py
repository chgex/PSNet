

"""
    Default Config
"""

# input_size = image_h, image_w
input_size_dic = {"IP": (448, 288), "GS": (448, 288), "GT2": (320, 448),
                  "NICEII": (320, 448), "ISDB": (320, 352),
                  "CASIA-D4": (480, 640), "CASIA-M1": (416, 416),
                  "VisAll": (320, 448)
                  }


class Config():
    # Iris DataSet
    root = "../IrisDataSet"
    # dataName = "NICEII"
    # dataName = "MICHE-IP"
    # dataName = "MICHE-GS"
    # dataName = "MICHE-GT2"
    # dataName = "ISDB"
    # dataName = "CASIA-D4"
    # dataName = "CASIA-M1"
    dataName = "VisAll"

    input_size = input_size_dic[dataName]
    scale_size = 1
    num_class = 2
    feature_map_channels = 32

    # Training, dataset and batch size: CASDA-D4 is 2; CASIA-M is 4; NICEII is 4; ISDB is 6; because total GPU is 6G.
    batch_size = 4

    epoch_lr_step = 5
    epoch_eval_iou = 5

    # Log info
    checkpoints = dataName

