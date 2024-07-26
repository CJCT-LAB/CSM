import torch.nn.functional as F
import torch.fft
import numpy as np
from mamba_ssm import Mamba
from einops import rearrange, repeat, einsum
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import torch.utils.checkpoint as cp
import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, trunc_normal_
import math
import warnings


class CS_SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            interval=8,
            lsm=0,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.1,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.interval = interval
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.lsmode = lsm
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4


        if self.lsmode == 0:

            x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                                 dim=1).view(B, 2, -1, L)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

            xs = xs.float().view(B, -1, L)  # (b, k * d, l)
            dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
            Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
            Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
            Ds = self.Ds.float().view(-1)  # (k * d)
            As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
            dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)
            assert out_y.dtype == torch.float

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        elif self.lsmode == 1:


            _, _, Hp, Wp  = x.shape

            I, Gh, Gw = self.interval, Hp // self.interval, Wp // self.interval
            x_l = x.reshape(B, C, Gh, I, Gw, I).permute(0, 3, 5, 1,2, 4).contiguous()
            x_l = x_l.reshape(B * I * I, C, Gh ,Gw)
            # nG = I ** 2

            x_hwwh = torch.stack([x_l.view(B * I * I, -1, Gh * Gw), torch.transpose(x_l, dim0=2, dim1=3).contiguous().view(B * I * I, -1, Gh * Gw)],
                                 dim=1).view(B * I * I, 2, -1, Gh * Gw)
            xsl = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

            xl_dbl = torch.einsum("b k d l, k c d -> b k c l", xsl.view(B * I * I, K, -1, Gh * Gw), self.x_proj_weight)
            dts, Bs, Cs = torch.split(xl_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B * I * I, K, -1, Gh * Gw), self.dt_projs_weight)

            xsl = xsl.float().view(B * I * I, -1, Gh * Gw)  # (b, k * d, l)
            dts = dts.contiguous().float().view(B * I * I, -1, Gh * Gw)  # (b, k * d, l)
            Bs = Bs.float().view(B * I * I, K, -1, Gh * Gw)  # (b, k, d_state, l)
            Cs = Cs.float().view(B * I * I, K, -1, Gh * Gw)  # (b, k, d_state, l)
            Ds = self.Ds.float().view(-1)  # (k * d)
            As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
            dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

            out_y = self.selective_scan(
                xsl, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B * I * I, K, -1, Gh * Gw)
            assert out_y.dtype == torch.float

            out_y = out_y.reshape(B, I, I,K, C,Gh, Gw).permute(0, 3, 4, 5, 1, 6, 2).contiguous()  # B, Gh, I, Gw, I, C
            out_y = out_y.reshape(B, K, C, -1)

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        elif self.lsmode == 2:

            size_div = G = Gh = Gw = 7
            x = x.permute(0,2,3,1).contiguous()
            pad_l = pad_t = 0
            pad_r = (size_div - W % size_div) % size_div
            pad_b = (size_div - H % size_div) % size_div
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            x = x.permute(0,3,1,2).contiguous()
            _, _, Hp, Wp = x.shape
            x = x.reshape(B, C, Hp // G, G, Wp // G, G).permute(0, 2, 4, 1,3, 5 ).contiguous()
            x = x.reshape(B * Hp * Wp // G ** 2, C, G,G)
            x_hwwh = torch.stack([x.view(B * Hp * Wp // G ** 2, -1, G ** 2), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B * Hp * Wp // G ** 2, -1, G * G)],
                                 dim=1).view(B * Hp * Wp // G ** 2, 2, -1, Gh * Gw)
            xsl = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

            xl_dbl = torch.einsum("b k d l, k c d -> b k c l", xsl.view(B * Hp * Wp // G ** 2, K, -1, Gh * Gw), self.x_proj_weight)
            dts, Bs, Cs = torch.split(xl_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B * Hp * Wp // G ** 2, K, -1, Gh * Gw), self.dt_projs_weight)

            xsl = xsl.float().view(B * Hp * Wp // G ** 2, -1, Gh * Gw)  # (b, k * d, l)
            dts = dts.contiguous().float().view(B * Hp * Wp // G ** 2, -1, Gh * Gw)  # (b, k * d, l)
            Bs = Bs.float().view(B * Hp * Wp // G ** 2, K, -1, Gh * Gw)  # (b, k, d_state, l)
            Cs = Cs.float().view(B * Hp * Wp // G ** 2, K, -1, Gh * Gw)  # (b, k, d_state, l)
            Ds = self.Ds.float().view(-1)  # (k * d)
            As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
            dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

            out_y = self.selective_scan(
                xsl, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B * Hp * Wp // G ** 2, K, -1, Gh * Gw)
            assert out_y.dtype == torch.float

            out_y = out_y.reshape(B, Hp // G, Wp // G,K,C, G, G).permute(0, 3, 4, 1, 5,2,6).contiguous()

            out_y = out_y.reshape(B, K, C, Hp, Wp)

            # remove padding
            if pad_r > 0 or pad_b > 0:
                out_y = out_y[:, :, :, :H,:W].contiguous()
            out_y = out_y.view(B, K, C,-1)

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, H,W):
        B, L, C = x.shape
        x = x.reshape(B, H, W, -1)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y).reshape(B,L,-1)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

def rand_bbox(size, lam, scale=1):
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class CSM_Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 interval = 8,
                 lsm = 0,
                 d_conv=3,
                 local_conv=True,
                 ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CS_SS2D(dim,interval=interval,lsm=lsm,d_conv=d_conv)
        self.local_conv=local_conv
        if self.local_conv:
            self.conv1b3 = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(dim),
                nn.SiLU(),
            )
            self.conv1a3 = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(dim),
                nn.SiLU(),
            )
            self.conv33 = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                          bias=False,
                          groups=dim),
                nn.BatchNorm2d(dim),
                nn.SiLU(),
            )
            # self.finalconv11 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)
            # self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.mlp = FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        def _inner_forward(x):
            if self.local_conv:
                B,_,C = x.shape
                x = x + self.drop_path(self.attn(x,H,W).reshape(B,H,W,-1).permute(0, 3, 1, 2).contiguous()+self.conv1a3(self.conv33(self.conv1b3(x.reshape(B,H,W,-1).permute(0, 3, 1, 2).contiguous())))).reshape(B,C,-1).permute(0,2,1).contiguous()
            else:
                x = x + self.drop_path(self.attn(x,H,W))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            return x
        if x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

@ROTATED_BACKBONES.register_module()
class CSM(BaseModule):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 stem_hidden_dim=32,
                 embed_dims=[64, 128, 320, 448],
                 mlp_ratios=[8, 8, 4, 4],
                 interval=[8,4,2,1],
                 lsm = 0,
                 d_conv=3,
                 local_conv=False,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[4, 2, 1, 1],
                 num_stages=4,
                 token_label=False,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList([CSM_Block(
                dim=embed_dims[i],
                mlp_ratio=mlp_ratios[i],
                drop_path=dpr[cur + j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[i],
                interval=interval[i],
                lsm=lsm[i][j],
                d_conv=d_conv[i][j],
                local_conv=local_conv[i]
               )
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)


        self.return_dense = token_label
        self.mix_token = token_label
        self.beta = 1.0
        self.pooling_scale = 8

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CSM, self).init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_cls(self, x, H, W):
        B, N, C = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)


            norm = getattr(self, f"norm{i + 1}")
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        if not self.return_dense:
            x = self.forward_features(x)
            return x
        else:
            x, H, W = self.forward_embeddings(x)
            if self.mix_token and self.training:
                lam = np.random.beta(self.beta, self.beta)
                patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                    2] // self.pooling_scale
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
                temp_x = x.clone()
                sbbx1, sbby1, sbbx2, sbby2 = self.pooling_scale * bbx1, self.pooling_scale * bby1, \
                                             self.pooling_scale * bbx2, self.pooling_scale * bby2
                temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
                x = temp_x
            else:
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
            x,outs = self.forward_tokens(x, H, W)
        return tuple(outs)

    def forward_tokens(self, x, H, W):
        outs = []
        B = x.shape[0]
        x = x.view(B, -1, x.size(-1))

        for i in range(self.num_stages):
            if i != 0:
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                x, H, W = patch_embed(x)
            block = getattr(self, f"block{i + 1}")
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                # print("x.shape:", x.shape)
                outs.append(x)
            else:
                norm = getattr(self, f"norm{i + 1}")
                x_temp = norm(x)
                x_temp = x_temp.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x_temp)

        x = self.forward_cls(x, H, W)
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x,outs

    def forward_embeddings(self, x):
        patch_embed = getattr(self, f"patch_embed{0 + 1}")
        x, H, W = patch_embed(x)
        x = x.view(x.size(0), H, W, -1)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x