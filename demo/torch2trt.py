

# import torch
# import tensorrt
#
# from torch2trt import torch2trt
# from torchvision.models.alexnet import alexnet
#
# # create some regular pytorch model...
# model = alexnet(pretrained=True).eval().cuda()
#
# # create example data
# x = torch.ones((1, 3, 224, 224)).cuda()
#
# # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])
#
# print("-" * 20)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from PSNet import load_model_and_weight
psnet = load_model_and_weight("PSNet/model.pth", device)
psnet.eval()


dummy_input = torch.ones(1, 3, 320, 448, device=device)

out = psnet(dummy_input)
print("---")

torch.onnx.export(psnet, dummy_input, 'psnet.onnx',
                  input_names=["inputs"],
                  output_names=['cls', 'radii1', 'radii2', 'off1', 'off2', 'logics'],
                  opset_version=11,
                  export_params=True,
                  verbose=True)

print("*" * 20)


""" 
Then, use ./trtexec.exe to convert 'ONNX model to TensorRT model: 
    use win terminal
    ./trtexec --onnx=psnet.onnx --saveEngine=psnet.trt
    
"""