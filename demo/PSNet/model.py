

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class CustomNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=2,  **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvX(input_channels, nb_filter[0])
        self.conv1_0 = ConvX(nb_filter[0], nb_filter[1])
        self.conv2_0 = CatBottleneck(nb_filter[1], nb_filter[2], 2)
        self.conv3_0 = CatBottleneck(nb_filter[2], nb_filter[3], 2)

        self.stdc = CatBottleneck(nb_filter[3], nb_filter[4], 4)

        self.conv3_1 = ConvX(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = ConvX(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = ConvX(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = ConvX(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x4_0 = self.stdc(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output



class CenterNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.backbone = CustomNet(3, 32)

        self.local_head = Local_Head(channel=32, num_classes=2)
        self.mask_head = Mask_Head(32, num_classes=2)

    def forward(self, x):
        feature_map = self.backbone(x)

        output = self.local_head(feature_map)
        mask_logit = self.mask_head(feature_map)

        return [output, mask_logit]


class Local_Head(nn.Module):
    def __init__(self, channel=64, num_classes=2, bn_momentum=0.1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0)
        )

        # inner radius
        self.b1_wh_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0))

        # offset
        self.b1_reg_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

        # outer radius
        self.b2_wh_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0))

        # offset
        self.b2_reg_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        heatmap = self.cls_head(x).sigmoid_()

        inner_radii = torch.exp(self.b1_wh_head(x))
        inner_offset = self.b1_reg_head(x).sigmoid_()

        outer_radii = self.b2_wh_head(x)
        outer_offset = self.b2_reg_head(x).sigmoid_()

        return heatmap, inner_radii, inner_offset, outer_radii, outer_offset


class Mask_Head(nn.Module):
    def __init__(self, channel, num_classes=2,  **kwargs):
        super().__init__()

        self.mask_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        output = self.mask_head(x).sigmoid_()
        return output

