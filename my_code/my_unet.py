import torch
import torch.nn as nn
from embedding import EmbedFC, SinusoidalPosEmb


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchnorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchnorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchnorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchnorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class my_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, n_steps=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.n_steps = n_steps
        self.inc = ConvBatchnorm(self.in_channels, self.dim)
        self.down1 = DownBlock(self.dim, self.dim*2, nb_Conv=2)
        self.down2 = DownBlock(self.dim*2, self.dim*4, nb_Conv=2)
        self.down3 = DownBlock(self.dim*4, self.dim*4, nb_Conv=2)

        self.timeembed1 = SinusoidalPosEmb(dim=4 * self.dim)
        self.timeembed2 = SinusoidalPosEmb(dim=2 * self.dim)
        self.timeembed3 = SinusoidalPosEmb(dim=1 * self.dim)
        self.contextembed1 = EmbedFC(1000, 4 * self.dim)
        self.contextembed2 = EmbedFC(1000, 2 * self.dim)
        self.contextembed3 = EmbedFC(1000, 1 * self.dim)
        self.posembed1 = SinusoidalPosEmb(dim=(4 * self.dim) / 4)
        self.posembed2 = SinusoidalPosEmb(dim=(2 * self.dim) / 4)
        self.posembed3 = SinusoidalPosEmb(dim=(1 * self.dim) / 4)

        self.up3 = UpBlock(self.dim*8, self.dim*2, nb_Conv=2)
        self.up2 = UpBlock(self.dim*4, self.dim, nb_Conv=2)
        self.up1 = UpBlock(self.dim*2, self.dim, nb_Conv=2)
        self.outc = nn.Conv2d(self.dim, self.out_channels, kernel_size=(1, 1))

    def forward(self, x, t, cemb, pos):

        c = cemb
        t = t / self.n_steps
        x = x.float()  # 3 224 64
        x1 = self.inc(x)  # 64 224 64
        x2 = self.down1(x1)  # 128 112 32
        x3 = self.down2(x2)  # 256 56 16
        x4 = self.down3(x3)  # 256 28 8

        cemb1 = self.contextembed1(c).view(-1, self.dim * 4, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.dim * 4, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.dim * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.dim * 2, 1, 1)
        cemb3 = self.contextembed3(c).view(-1, self.dim, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.dim, 1, 1)

        pemb1 = torch.zeros(pos.shape[0], self.dim * 4).to(x.device)
        pemb2 = torch.zeros(pos.shape[0], self.dim * 2).to(x.device)
        pemb3 = torch.zeros(pos.shape[0], self.dim * 1).to(x.device)
        for i, pos_i in enumerate(pos):
            pemb1[i] = self.posembed1(pos_i).flatten()
            pemb2[i] = self.posembed2(pos_i).flatten()
            pemb3[i] = self.posembed3(pos_i).flatten()
        pemb1 = pemb1.view(-1, self.dim * 4, 1, 1)
        pemb2 = pemb2.view(-1, self.dim * 2, 1, 1)
        pemb3 = pemb3.view(-1, self.dim * 1, 1, 1)

        x = self.up3(x4 * cemb1 + temb1 + pemb1, x3)   # 128 56 16
        x = self.up2(x * cemb2 + temb2 + pemb2, x2)  # 64 112 32
        x = self.up1(x * cemb3 + temb3 + pemb3, x1)  # 64 224 64
        out = self.outc(x)  # 3 224 64

        return out