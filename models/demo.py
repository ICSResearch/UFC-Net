import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from torch.nn import init
import numpy as np
from convnext.convnextv2 import *
from utils.util import init_weights, count_param
from einops.layers.torch import Rearrange
import numbers
from einops import rearrange


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_bool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_bool(x)
        y = self.conv_du(y)
        return x * y


# Channel Atttion Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernal_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernal_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernal_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = res + x  # b, 32, w, h
        return res


class MHSA(nn.Module):
    def __init__(self, num_head, dim, attn_drop=0., proj_drop=0.):
        super(MHSA, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        b, c, h, w = q.shape
        head_dim = self.dim // self.num_head
        q = Rearrange('b (nh hd) h w -> b nh hd (h w)',
                      hd=head_dim)(q)  # b, nh, hd, hw
        k = Rearrange('b (nh hd) h w -> b nh (h w) hd',
                      hd=head_dim)(k)  # b, nh, hw, hd
        v = Rearrange('b (nh hd) h w -> b nh hd (h w)',
                      hd=head_dim)(v)  # b, nh, hd, hw

        attention = torch.matmul(q, k)  # b, nh, hd, hd
        scale = head_dim ** -0.5
        attention = attention * scale
        attention = self.softmax(attention)
        attention = self.attn_drop(attention)

        x = torch.matmul(attention, v)  # b, nh, hd, hw
        x = Rearrange('b nh hd (h w) -> b (nh hd) h w',
                      h=h, w=w)(x)  # b, c, h, w

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attention


class Tunit(nn.Module):
    def __init__(self, num_head, dim, drop_out=0.1):
        super(Tunit, self).__init__()
        self.layernorm1 = LayerNorm(dim, data_format="channels_first")
        self.attention = MHSA(num_head, dim, attn_drop=drop_out, proj_drop=drop_out)
        self.layernorm2 = LayerNorm(dim, data_format="channels_first")
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout(drop_out),
        )

    def forward(self, x):
        x_t, atten = self.attention(self.layernorm1(x))
        x = x_t + x
        x_t = self.ff(self.layernorm2(x))
        x = x_t + x
        return x

class shallow_pre(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, reduction, act):
        super(shallow_pre, self).__init__()
        self.c1 = conv(1, n_feat, kernel_size, bias)
        self.relu = nn.LeakyReLU(inplace=False)
        self.c = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)

    def forward(self, inputs):
        inputs = self.c(self.relu(self.c1(inputs)))
        return inputs

class shallow_end(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, reduction, act):
        super(shallow_end, self).__init__()
        self.c1 = conv(n_feat, 1, kernel_size, bias)
        self.relu = nn.LeakyReLU(inplace=False)
        self.c = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)

    def forward(self, inputs):
        inputs = self.c1(self.relu(self.c(inputs)))
        return inputs

class ITB(nn.Module):
    def __init__(self, dim, is_last=False):
        super(ITB, self).__init__()
        self.is_last = is_last
        self.dc = nn.Sequential(
            # nn.Conv2d(dim, 2*dim, kernel_size=1, stride=1, padding=0, bias=True),
            Block(dim)
        )
        self.tunit = Tunit(num_head=4, dim = 2*dim, drop_out=0.1)
        if not self.is_last:
            self.dc2 = nn.Sequential(
                nn.Conv2d(2*dim, dim, kernel_size=1,stride=1,padding=0,bias=True)
            )
        self.dc3 = nn.Sequential(
            nn.Conv2d(2*dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, z):
        x = self.dc(x)
        x = torch.cat([x, z], dim=1)
        x = self.tunit(x)
        if not self.is_last:
            z = self.dc2(x)
        x = self.dc3(x)
        return x, z


class FPC(nn.Module):
    def __init__(self, LayerNo, in_c=1, out_c=1, n_feat=32, kernel_size=3, reduction=4, bias=False, cs_ratio=0.10):
        super(FPC, self).__init__()
        self.LayerNo = LayerNo
        self.Phi = nn.Parameter(
            init.xavier_normal_(torch.Tensor(np.ceil(cs_ratio * 1024).astype(int), 1024)))  # 1024*cs_ratio, 1024

        act = nn.PReLU()
        self.shallow_feat1 = nn.ModuleList()
        self.shallow_feat2 = nn.ModuleList()
        self.block1 = nn.ModuleList()
        self.block2 = nn.ModuleList()
        self.Tunit1 = nn.ModuleList()
        self.Tunit2 = nn.ModuleList()
        self.ITB1 = nn.ModuleList()
        self.ITB2 = nn.ModuleList()
        for i in range(self.LayerNo):
            self.shallow_feat1.append(shallow_pre(n_feat, kernel_size, bias, reduction, act))
            self.block1.append(Block(n_feat))
            self.block2.append(Block(n_feat))
            self.shallow_feat2.append(shallow_end(n_feat, kernel_size, bias, reduction, act))
            self.Tunit1.append(Tunit(4, n_feat))
            self.Tunit2.append(Tunit(4, n_feat))
            if i is not self.LayerNo - 1:
                self.ITB1.append(ITB(n_feat, is_last=False))
                self.ITB2.append(ITB(n_feat, is_last=False))
            else:
                self.ITB1.append(ITB(n_feat, is_last=True))
                self.ITB2.append(ITB(n_feat, is_last=True))

        self.pre = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=1, stride=1, padding=0, bias=True),
            Block(n_feat),
        )
        self.end = nn.Sequential(
            nn.Conv2d(n_feat*2, n_feat, kernel_size=1, stride=1, padding=0, bias=True),
        )

        t = min(1 + 1.665 * (1 - cs_ratio), 1.999)

        self.gama = 0.9
        self.miu_end = 5000
        self.taus = []
        self.mius = []
        self.betas = []
        self.lambdas = []
        # self.lambda_step = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        for i in range(self.LayerNo):
            self.register_parameter("tau_" + str(i + 1), nn.Parameter(torch.tensor(t), requires_grad=True))  # todo
            self.taus.append(eval("self.tau_" + str(i + 1)))
            self.register_parameter("miu_" + str(i + 1), nn.Parameter(torch.tensor(1.), requires_grad=True))  # todo
            self.mius.append(eval("self.miu_" + str(i + 1)))
            self.register_parameter("beta_" + str(i + 1), nn.Parameter(torch.tensor(2.), requires_grad=True))  # todo
            self.betas.append(eval("self.beta_" + str(i + 1)))
            self.register_parameter("lambda_" + str(i + 1), nn.Parameter(torch.tensor(0.5), requires_grad=True))
            self.lambdas.append(eval("self.lambda_" + str(i + 1)))

    def forward(self, x):
        b, c, w, h = x.shape
        PhiTPhi = torch.mm(torch.transpose(self.Phi, 0, 1), self.Phi)  # 1024, 1024
        # x = x.to("cuda:0")
        Phix = torch.mm(x.view(b, -1), torch.transpose(self.Phi, 0, 1))  # b, 1024*cs_ratio
        PhiTb = torch.mm(Phix, self.Phi)  # b, 1024 初始重建值

        recon = PhiTb

        t2 = torch.norm(recon, float('inf'))
        self.mius[0] = self.taus[0] / t2 / self.gama

        for i in range(self.LayerNo):
            recon = recon.view(b, -1)
            recon = recon - self.lambdas[i] * torch.mm(recon, PhiTPhi)
            recon = recon + self.lambdas[i] * PhiTb
            recon_g0 = recon.view(b, c, w, h)
            if i==0:
                z = self.pre(recon_g0)
            # shallow_feat1 or ConvBlockV2
            recon = self.shallow_feat1[i](recon_g0)
            recon = self.block1[i](recon)
            recon1 = self.Tunit1[i](recon)
            recon, z1 = self.ITB1[i](recon1, z)
            recon = recon + recon1

            # threshold
            if i != 0:
                self.mius[i] = min(self.mius[i - 1] * self.betas[i], self.miu_end)
            nu = self.taus[i] / self.mius[i]
            recon = torch.mul(torch.sign(recon), F.relu(torch.abs(recon) - nu))

            recon2, z2 = self.ITB2[i](recon, z)
            recon = self.Tunit2[i](recon2)
            recon = recon + recon2
            recon = self.block2[i](recon)

            recon = self.shallow_feat2[i](recon)
            z = torch.cat([z1, z2], dim=1)
            z = self.end(z)
            recon = recon + recon_g0

        return recon