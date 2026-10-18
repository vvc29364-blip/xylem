"""Weight-sharing *elastic* layers (slimmable width, per-call).

Every layer is allocated at the maximal size.  A smaller sub-network uses the leading slice of
each weight matrix (first `d` channels / heads), so one set of weights contains every subnet and
a subnet can later be *physically cut out* (see xylem/engine/export.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ELinear(nn.Linear):
    def forward(self, x, out_dim=None):
        o = out_dim or self.out_features
        i = x.shape[-1]
        b = self.bias[:o] if self.bias is not None else None
        return F.linear(x, self.weight[:o, :i], b)


class ELayerNorm(nn.LayerNorm):
    def forward(self, x):
        c = x.shape[-1]
        return F.layer_norm(x, (c,), self.weight[:c], self.bias[:c], self.eps)


class EDWConv(nn.Module):
    """Depth-wise 1-D conv along a serialized sequence (conditional positional encoding)."""

    def __init__(self, dim, k=5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, k))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.k = k

    def forward(self, x):  # (B,N,C)
        c = x.shape[-1]
        y = F.conv1d(x.transpose(1, 2), self.weight[:c], self.bias[:c], padding=self.k // 2, groups=c)
        return y.transpose(1, 2)


class EMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = ELinear(dim, hidden)
        self.fc2 = ELinear(hidden, dim)

    def forward(self, x, ratio):
        d = x.shape[-1]
        h = F.gelu(self.fc1(x, int(d * ratio)))
        return self.fc2(h, d)


class EAttention(nn.Module):
    """Multi-head (self or cross) attention with elastic width; head_dim is fixed."""

    def __init__(self, dim, ctx_dim=None, head_dim=32):
        super().__init__()
        ctx_dim = ctx_dim or dim
        self.q = ELinear(dim, dim)
        self.k = ELinear(ctx_dim, dim)
        self.v = ELinear(ctx_dim, dim)
        self.o = ELinear(dim, dim)
        self.hd = head_dim

    def forward(self, x, ctx=None, attn_mask=None):
        ctx = x if ctx is None else ctx
        B, N, d = x.shape
        M = ctx.shape[1]
        h = max(1, d // self.hd)
        q = self.q(x, d).view(B, N, h, d // h).transpose(1, 2)
        k = self.k(ctx, d).view(B, M, h, d // h).transpose(1, 2)
        v = self.v(ctx, d).view(B, M, h, d // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(o.transpose(1, 2).reshape(B, N, d), d)


def window_apply(x, window, fn):
    """Apply fn to non-overlapping windows of a sequence (B,N,C).  window<=0 -> global."""
    B, N, C = x.shape
    if window <= 0 or window >= N:
        return fn(x)
    pad = (-N) % window
    if pad:
        x = torch.cat([x, x[:, -pad:].flip(1)], 1)
    n = x.shape[1] // window
    y = fn(x.reshape(B * n, window, C)).reshape(B, n * window, C)
    return y[:, :N]


class SerialBlock(nn.Module):
    """Pre-norm transformer block on a serialized point sequence with windowed attention."""

    def __init__(self, dim, head_dim=32, mlp_ratio=4, ctx_dim=None):
        super().__init__()
        self.cpe = EDWConv(dim)
        self.n1 = ELayerNorm(dim)
        self.attn = EAttention(dim, head_dim=head_dim)
        self.cross = None
        if ctx_dim:
            self.nc = ELayerNorm(dim)
            self.cross = EAttention(dim, ctx_dim=ctx_dim, head_dim=head_dim)
        self.n2 = ELayerNorm(dim)
        self.mlp = EMLP(dim, int(dim * mlp_ratio))

    def forward(self, x, window, ratio, ctx=None):
        x = x + self.cpe(x)
        x = x + window_apply(self.n1(x), window, self.attn)
        if self.cross is not None and ctx is not None:
            x = x + self.cross(self.nc(x), ctx)
        return x + self.mlp(self.n2(x), ratio)


class DecoderLayer(nn.Module):
    """Global self-attention + cross-attention to point tokens + MLP."""

    def __init__(self, dim, ctx_dim, head_dim=32, mlp_ratio=4):
        super().__init__()
        self.n1 = ELayerNorm(dim)
        self.sa = EAttention(dim, head_dim=head_dim)
        self.n2 = ELayerNorm(dim)
        self.ca = EAttention(dim, ctx_dim=ctx_dim, head_dim=head_dim)
        self.n3 = ELayerNorm(dim)
        self.mlp = EMLP(dim, int(dim * mlp_ratio))

    def forward(self, x, ctx, ratio):
        x = x + self.sa(self.n1(x))
        x = x + self.ca(self.n2(x), ctx)
        return x + self.mlp(self.n3(x), ratio)
