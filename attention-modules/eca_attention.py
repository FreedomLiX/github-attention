import torch.nn as nn
import math
import torch


class ECA_Attention(nn.Module):
    """
    ECA-Net: Efficient Channel Attention for
    Deep Convolutional Neural Networks---CVPR2020
    论文地址：https://arxiv.org/pdf/1910.03151.pdf
    对比 SE-Block,实现 without dims reduction
    """

    def __init__(self, gamma=2, b=1):
        super(ECA_Attention, self).__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # function k = f(C)
        t = int(abs(math.log(c, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = nn.Conv1d(1, 1, k, padding=int(k / 2), bias=False)(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sig(y)
        return x * y.expand_as(x)


if __name__ == '__main__':
    data = torch.rand(10, 64, 521, 512)
    se = ECA_Attention()
    out = se(data)
    print(out.shape)
