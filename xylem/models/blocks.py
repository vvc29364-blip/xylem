"""Shared building blocks for the re-implemented baselines (pure PyTorch)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import fps, gather, knn_idx


def mlp(dims, last_act=False, norm=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            if norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class PointNetEncoder(nn.Module):
    """PCN encoder: two stacked PointNets with feature concatenation."""

    def __init__(self, out=1024):
        super().__init__()
        self.m1 = mlp([3, 128, 256])
        self.m2 = mlp([512, 512, out])

    def forward(self, x):
        f = self.m1(x)
        g = f.amax(1, keepdim=True).expand_as(f)
        f = self.m2(torch.cat([f, g], -1))
        return f.amax(1)


class SetAbstraction(nn.Module):
    """PointNet++ SA: FPS + kNN grouping + shared MLP + max."""

    def __init__(self, n_center, k, in_dim, dims):
        super().__init__()
        self.n, self.k = n_center, k
        self.m = mlp([in_dim + 3] + dims, last_act=True)

    def forward(self, xyz, feat=None):
        idx = fps(xyz, self.n)
        c = gather(xyz, idx)
        nb = knn_idx(c, xyz, self.k)
        g = gather(xyz, nb) - c.unsqueeze(2)
        if feat is not None:
            g = torch.cat([g, gather(feat, nb)], -1)
        return c, self.m(g).amax(2)


class PointTransformerLayer(nn.Module):
    """Vector self-attention on kNN neighbourhoods (Zhao et al. 2021)."""

    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.q, self.kk, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.pos = mlp([3, 64, dim])
        self.attn = mlp([dim, dim * 2, dim])
        self.out = nn.Linear(dim, dim)

    def forward(self, xyz, f, key_xyz=None, key_f=None):
        key_xyz = xyz if key_xyz is None else key_xyz
        key_f = f if key_f is None else key_f
        nb = knn_idx(xyz, key_xyz, self.k)
        q = self.q(f).unsqueeze(2)
        kf = gather(self.kk(key_f), nb)
        vf = gather(self.v(key_f), nb)
        pe = self.pos(xyz.unsqueeze(2) - gather(key_xyz, nb))
        a = torch.softmax(self.attn(q - kf + pe), dim=2)
        return f + self.out((a * (vf + pe)).sum(2))


class SkipTransformer(nn.Module):
    """SnowflakeNet skip-transformer: query = current features, key = previous-step features."""

    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.q, self.kk, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim * 2, dim)
        self.pos = mlp([3, 64, dim])
        self.attn = mlp([dim, dim * 2, dim])
        self.out = nn.Linear(dim, dim)

    def forward(self, xyz, f, prev):
        nb = knn_idx(xyz, xyz, self.k)
        v = self.v(torch.cat([f, prev], -1))
        q = self.q(prev).unsqueeze(2)
        kf = gather(self.kk(f), nb)
        vf = gather(v, nb)
        pe = self.pos(xyz.unsqueeze(2) - gather(xyz, nb))
        a = torch.softmax(self.attn(q - kf + pe), dim=2)
        return v + self.out((a * (vf + pe)).sum(2))


class Folding(nn.Module):
    """FoldingNet-style patch decoder: code + 2-D grid -> 3-D offsets."""

    def __init__(self, code_dim, n_pts, hidden=256):
        super().__init__()
        s = int(math.ceil(math.sqrt(n_pts)))
        g = torch.linspace(-0.05, 0.05, s)
        grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)[:n_pts]
        self.register_buffer("grid", grid)
        self.f1 = mlp([code_dim + 2, hidden, hidden // 2, 3])
        self.f2 = mlp([code_dim + 3, hidden, hidden // 2, 3])

    def forward(self, code):  # code (B,Q,C) -> (B,Q,n,3)
        B, Q, C = code.shape
        n = self.grid.shape[0]
        c = code.unsqueeze(2).expand(-1, -1, n, -1)
        g = self.grid.view(1, 1, n, 2).expand(B, Q, -1, -1)
        x = self.f1(torch.cat([c, g], -1))
        return self.f2(torch.cat([c, x], -1))


class MHA(nn.Module):
    def __init__(self, dim, heads=6):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, ctx=None, mask=None):
        ctx = x if ctx is None else ctx
        B, N, C = x.shape
        M = ctx.shape[1]
        h = self.h
        q = self.q(x).view(B, N, h, C // h).transpose(1, 2)
        k = self.k(ctx).view(B, M, h, C // h).transpose(1, 2)
        v = self.v(ctx).view(B, M, h, C // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(o.transpose(1, 2).reshape(B, N, C))


class GeoAwareBlock(nn.Module):
    """PoinTr geometry-aware transformer block: self-attn || kNN edge features, merged."""

    def __init__(self, dim, heads=6, k=8, cross=False):
        super().__init__()
        self.k = k
        self.n1 = nn.LayerNorm(dim)
        self.sa = MHA(dim, heads)
        self.knn_mlp = mlp([dim * 2, dim], last_act=True)
        self.merge = nn.Linear(dim * 2, dim)
        self.cross = cross
        if cross:
            self.nc = nn.LayerNorm(dim)
            self.ca = MHA(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.ff = mlp([dim, dim * 4, dim])

    def forward(self, x, pos, ctx=None, mask=None):
        h = self.n1(x)
        a = self.sa(h, mask=mask)
        nb = knn_idx(pos, pos, self.k)
        e = gather(h, nb)
        e = self.knn_mlp(torch.cat([e - h.unsqueeze(2), h.unsqueeze(2).expand_as(e)], -1)).amax(2)
        x = x + self.merge(torch.cat([a, e], -1))
        if self.cross and ctx is not None:
            x = x + self.ca(self.nc(x), ctx)
        return x + self.ff(self.n2(x))
