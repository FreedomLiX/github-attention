import torch.nn as nn
import torch


class BAM_Attention(nn.Module):
    """
    BAM: Bottleneck Attention Module---BMCV2018
    论文地址：https://arxiv.org/pdf/1807.06514.pdf
    channel attention : input= B C H W , out= B C 1 1
    spatial attention : input= B C H W , out= B 1 H W
    Conv2D with dilation when kernel size = 3,
    padding should equal dilation value, in older to keep same shape(H,W)
    """

    def __init__(self, channels, reduction=16, dia_val=4):
        super(BAM_Attention, self).__init__()
        # channel attention branch
        self.channel_atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # spatial attention branch
        self.spatial_atten = nn.Sequential(
            # 1*1 ==>3*3 ==>3*3 ==> 1*1 ==>BN
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            # 第一组
            nn.Conv2d(kernel_size=3, in_channels=channels // reduction,
                      out_channels=channels // reduction, padding=4,
                      dilation=dia_val),
            # 第二组
            nn.Conv2d(kernel_size=3, in_channels=channels // reduction,
                      out_channels=channels // reduction, padding=4,
                      dilation=dia_val),
            nn.Conv2d(channels // reduction, 1, kernel_size=1),
            nn.BatchNorm2d(1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # channel
        ch_x = self.channel_atten(x)
        ch_x = ch_x.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        # spatial
        sp_x = self.spatial_atten(x)
        sp_x = sp_x.expand_as(x)
        # channel + spatial BAM直接相加,
        # 对比CBAM方法，CBAM是先channel后spatial
        out = self.sig(ch_x + sp_x)
        x = x + out * x
        return x


if __name__ == '__main__':
    input = torch.randn(50, 512, 17, 17)
    bam = BAM_Attention(channels=512, reduction=16)
    output = bam(input)
    print(output.shape)
