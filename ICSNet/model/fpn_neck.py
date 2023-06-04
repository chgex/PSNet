

import torch.nn as nn
import torch.nn.functional as F
import math


class FPN(nn.Module):
    def __init__(self, in_chan, features=256, use_p5=True):
        super(FPN, self).__init__()
        self.prj_5 = nn.Conv2d(in_chan[0], features, kernel_size=1)
        self.prj_4 = nn.Conv2d(in_chan[1], features, kernel_size=1)
        self.prj_3 = nn.Conv2d(in_chan[2], features, kernel_size=1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P3 = self.conv_3(P3)
        # P4 = self.conv_4(P4)
        # P5 = self.conv_5(P5)

        # P5 = P5 if self.use_p5 else C5
        # P6 = self.conv_out6(P5)
        # P7 = self.conv_out7(F.relu(P6))
        # return [P3, P4, P5, P6, P7]
        return P3


if __name__ == "__main__":
    import torch

    x1 = torch.randn((1, 512, 64, 64))
    x2 = torch.randn((1, 1024, 32, 32))
    x3 = torch.randn((1, 2048, 16, 16))

    x = [x1, x2, x3]

    net = FPN([2048, 1024, 512], 64)

    o = net(x)

    for i in o:
        print(i.shape)

    # torch.Size([1, 256, 64, 64])
    # torch.Size([1, 256, 32, 32])
    # torch.Size([1, 256, 16, 16])
    # torch.Size([1, 256, 8, 8])
    # torch.Size([1, 256, 4, 4])