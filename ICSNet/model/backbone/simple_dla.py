import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2*out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level-1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level-1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class SimpleDLA_2(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer0 = Tree(block,  16, 32, level=1, stride=1)
        self.layer1 = Tree(block,  32, 64, level=2, stride=2)
        self.layer2 = Tree(block, 64, 128, level=2, stride=2)
        self.layer3 = Tree(block, 128, 256, level=2, stride=2)

        self.layer4 = Tree(block, 256, 512, level=1, stride=2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        nb_filter = [32, 64, 128, 256, 512]

        self.conv3 = ConvX(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2 = ConvX(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1 = ConvX(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0 = ConvX(nb_filter[1] + nb_filter[0], nb_filter[0])

    def forward(self, x):
        out = self.base(x)
        out0 = self.layer0(out)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        out4 = self.layer4(out3)

        x3_1 = self.conv3(torch.cat([out3, self.up(out4)], 1))
        x2_2 = self.conv2(torch.cat([out2, self.up(x3_1)], 1))
        x1_3 = self.conv1(torch.cat([out1, self.up(x2_2)], 1))
        x0_4 = self.conv0(torch.cat([out0, self.up(x1_3)], 1))

        out = x0_4

        return out


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out



if __name__ == '__main__':

    net = SimpleDLA_2()
    # print(net)
    x = torch.randn(1, 3, 128, 128)
    y = net(x)
    print(y.size())


