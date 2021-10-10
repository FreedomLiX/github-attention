import torch.nn as nn
import torch


class SE_Attention(nn.Module):
    """
    ====通道上分配不同的权值====
    Squeeze-and-Excitation Networks
    paper: https://arxiv.org/abs/1709.01507
    ① Input，Output： B C H W ==》 B C H W
    ② Squeeze: avg pooling
    ③ Excitation: linear
    """

    def __init__(self, channel=512, reduction=16):
        super(SE_Attention, self).__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.sq(x).view(b, c)
        y = self.ex(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out


class SE_AttentionCBR(nn.Module):
    """
    SE_Attention used Linear to realize the channel attention,
    SE_AttentionBNR use Conv BatchNormal Relu.
    "目的：light weight",轻量级。
    用途：同 SE_Attention
    """

    def __init__(self, channel=512, reduction=16):
        super(SE_AttentionCBR, self).__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.sq(x)
        y = self.ex(y)
        out = x * y.expand_as(x)
        return out
