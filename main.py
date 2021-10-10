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


if __name__ == '__main__':
    data = torch.rand(10, 64, 521, 512)
    se = CBAM_Attention(64)
    out = se(data)
    print(out.shape)
