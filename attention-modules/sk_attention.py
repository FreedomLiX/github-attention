import torch.nn as nn
import torch


class SK_Attention(nn.Module):
    """
    Selective Kernel Networks
    https://arxiv.org/pdf/1903.06586.pdf
    """

    def __init__(self, channels, reduction=16):
        super(SK_Attention, self).__init__()
        self.split_conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # split 5*5 used 3*3 dilation=2 实现。
        self.split_conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      dilation=2,
                      padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.fuse_avg = nn.AdaptiveAvgPool2d(1)
        d = max(channels // reduction, 32)
        self.fuse_squeeze = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU()
        )
        self.fuse_excitation_a = nn.Sequential(
            nn.Linear(d, channels, bias=False),
            nn.Softmax()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # split
        split_3 = self.split_conv3x3(x)
        split_5 = self.split_conv5x5(x)

        # fuse
        fuse = self.fuse_avg(split_3 + split_5).view(b, c)
        fuse = self.fuse_squeeze(fuse)
        fuse_a = self.fuse_excitation_a(fuse).view(b, c, 1, 1)
        fuse_b = 1 - fuse_a
        # select
        select_a = split_3 * fuse_a.expand_as(x)
        select_b = split_5 * fuse_b.expand_as(x)
        out = select_a + select_b

        return out

