"""
（1） self-attention and multi-head attention
    self-attention 实现 及 multi-head 实现。
（2） self attention ViT Net paper, from yolov5.
    Transformer block.
    B C H W ==> to_qkv()
"""
import torch
from torch import nn
# Muti-head Attention 机制的实现
from math import sqrt


# self-attention
class Self_Attention(nn.Module):
    """
    # input shape : （batch_size * seq_len , input_dim）
    # q : （batch_size * input_dim , dim_k）
    # k : （batch_size * input_dim , dim_k）
    # v : （batch_size * input_dim , dim_v）
    """
    def __init__(self, input_dim, dim_k, dim_v):
        super(Self_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v

        atten = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output


# multi-heads attention
class Self_Attention_Muti_Head(nn.Module):
    """
    # input shape : （batch_size * seq_len , input_dim）
    # q : （batch_size * input_dim , dim_k）
    # k : （batch_size * input_dim , dim_k）
    # v : （batch_size * input_dim , dim_v）
    """
    def __init__(self, input_dim, dim_k, dim_v, nums_head):
        super(Self_Attention_Muti_Head, self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)

        self.nums_head = nums_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # multi-head 的实现方法，即是 （self.dim_k // self.nums_head）
        # 输出的q,k,v 切分成了多个 “批次”
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        V = self.v(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.nums_head)
        print(x.shape)
        print(Q.size())

        atten = nn.Softmax(dim=-1)(
            torch.matmul(Q, K.permute(0, 1, 3, 2)))  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.matmul(atten, V).reshape(x.shape[0], x.shape[1],
                                                -1)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output


""" 
以下代码来源于yolov5 用于处理卷积块 B C H W 为输入数据，输出为B C H W 的注意力机制实现。
"""


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    # ① input shape: B C1 H W
    # ② input shape data operate by flatten, transpose
    #  B C1 H W ==>B C1 H*W ==> H*W B C1
    # H*W B C1 :
    # nn.Linear() 输入的[batch_size, size]
    #             输出的[batch_size, out-size]
    # 最后一维进行全连接操作，“ C1 ” 代表了二维特征(B C1 H W)
    # 不同 channels 之间的 “多头注意力”方式。
    # ③ output shape: B C2 H W
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        # multi heads , and multi layers
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


if __name__ == '__main__':
    x = torch.randn(10, 32, 64, 64)
    b, _, w, h = x.shape
    p = x.flatten(2)
    print(p.shape)
    p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
    print(p.shape)
    # sa = TransformerBlock(32, 16, 4, 2)
    # output = sa(x)
    # print(output.shape)
