import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    # input size = output size
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1,
                     stride=stride, bias=False)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        # 带有groups的形式。
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class BottleneckWithSeparableConv2d(nn.Module):
    def __init__(self):
        super(BottleneckWithSeparableConv2d, self).__init__()
        pass

    def forward(self, x):
        pass


if __name__ == '__main__':
    # data = torch.rand(10, 64, 521, 512)
    # se = SeparableConv2d(64, 32, kernel_size=5, dilation=3)
    # out = se(data)
    # print(out.shape)
    for i in range(16):
        exec(f'print(block{i + 4})')
