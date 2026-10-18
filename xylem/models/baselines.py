"""Re-implementations of point-cloud completion baselines (identical data, losses, schedule).

All models map a partial cloud (B,N,3) to a list of point sets `stages` (coarse -> fine);
`stages[-1]` has 16384 points.  Training loss = sum of CD-L1 over stages (as in the papers,
with EMD replaced by CD where the original used EMD for the coarse output).

Faithfulness notes (documented in paper/appendix):
  PCN        Yuan et al. 3DV'18        PointNet x2 encoder, MLP coarse (1024) + folding (x16)
  FoldingNet Yang et al. CVPR'18       graph encoder, two folding ops on a 128x128 grid
  TopNet     Tchapmi et al. CVPR'19    rooted tree decoder, arity 4, 7 levels
  GRNet      Xie et al. ECCV'20        gridding (64^3), 3D CNN U-Net, reverse gridding, cubic feat. sampling
  PoinTr     Yu et al. ICCV'21         point proxies, geometry-aware transformer, 256 queries x 64 folding
  AdaPoinTr  Yu et al. TPAMI'23        + adaptive query generation (ranking) + denoising queries
  SnowflakeNet Xiang et al. ICCV'21    PN++/PT encoder, seed generator, 3x SPD with skip-transformer
  SeedFormer Zhou et al. ECCV'22       patch seeds, seed-feature interpolation, upsample transformer
  PointAttN  Wang et al. AAAI'24       GDP/SFA attention encoder, seed generator, attention refiners
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import chamfer, fps, gather, knn_idx, nn_idx
from .blocks import (Folding, GeoAwareBlock, MHA, PointNetEncoder, PointTransformerLayer, SetAbstraction,
                     SkipTransformer, mlp)

N_OUT = 16384


def _fps_pts(x, n):
    return gather(x, fps(x, n))


class CompletionBase(nn.Module):
    kind = "completion"

    def loss(self, out, batch):
        tot = 0
        logs = {}
        w = getattr(self, "stage_w", None) or [1.0] * len(out["stages"])
        for i, (s, wi) in enumerate(zip(out["stages"], w)):
            l, _, _ = chamfer(s, batch["gt"])
            tot = tot + wi * l
            logs[f"cd{i}"] = l.detach()
        if "extra_loss" in out:
            tot = tot + out["extra_loss"]
            logs["extra"] = out["extra_loss"].detach()
        logs["total"] = tot.detach()
        return tot, logs


# ------------------------------------------------------------------------------------------
class PCN(CompletionBase):
    def __init__(self, n_coarse=1024, grid=4):
        super().__init__()
        self.enc = PointNetEncoder(1024)
        self.nc = n_coarse
        self.coarse = mlp([1024, 1024, 1024, 3 * n_coarse])
        g = torch.linspace(-0.02, 0.02, grid)
        self.register_buffer("grid", torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2))
        self.fold = mlp([1024 + 2 + 3, 512, 512, 3])

    def forward(self, x):
        B = x.shape[0]
        g = self.enc(x)
        c = self.coarse(g).view(B, self.nc, 3)
        n = self.grid.shape[0]
        cc = c.unsqueeze(2).expand(-1, -1, n, -1)
        z = torch.cat([g[:, None, None].expand(-1, self.nc, n, -1), self.grid.expand(B, self.nc, -1, -1), cc], -1)
        fine = (self.fold(z) + cc).reshape(B, -1, 3)
        return dict(stages=[c, fine], dense=fine)


# ------------------------------------------------------------------------------------------
class FoldingNet(CompletionBase):
    def __init__(self, code=512, k=16):
        super().__init__()
        self.k = k
        self.local = mlp([12, 64, 64], last_act=True)
        self.g1 = mlp([64, 128], last_act=True)
        self.g2 = mlp([128, 1024], last_act=True)
        self.code = mlp([1024, code, code])
        self.fold = Folding(code, N_OUT, hidden=512)
        g = torch.linspace(-0.5, 0.5, 128)
        self.fold.grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    def forward(self, x):
        nb = knn_idx(x, x, self.k)
        loc = gather(x, nb) - x.unsqueeze(2)
        cov = torch.einsum("bnki,bnkj->bnij", loc, loc).reshape(*x.shape[:2], 9) / self.k   # local covariance
        f = self.local(torch.cat([x, cov], -1))
        f = self.g1(gather(f, nb).amax(2))
        f = self.g2(gather(f, nb).amax(2))
        code = self.code(f.amax(1))
        self.fold.grid = self.fold.grid.to(x.device)
        out = self.fold(code.unsqueeze(1))[:, 0] + torch.tensor([0, 0, 0.5], device=x.device)
        return dict(stages=[out], dense=out)


# ------------------------------------------------------------------------------------------
class TopNet(CompletionBase):
    def __init__(self, arity=(4, 4, 4, 4, 4, 4, 4), nfeat=8, code=1024):
        super().__init__()
        self.enc = PointNetEncoder(code)
        self.arity = arity
        self.nf = nfeat
        self.root = mlp([code, 256, 64, nfeat * arity[0]])
        self.levels = nn.ModuleList()
        for i, a in enumerate(arity[1:]):
            last = i == len(arity) - 2
            self.levels.append(mlp([code + nfeat, 256, 64, (3 if last else nfeat) * a]))

    def forward(self, x):
        B = x.shape[0]
        g = self.enc(x)
        f = torch.tanh(self.root(g).view(B, self.arity[0], self.nf))
        for i, lv in enumerate(self.levels):
            a = self.arity[i + 1]
            last = i == len(self.levels) - 1
            z = torch.cat([g.unsqueeze(1).expand(-1, f.shape[1], -1), f], -1)
            f = lv(z).view(B, -1, 3 if last else self.nf)
            if not last:
                f = torch.tanh(f)
        out = f + torch.tensor([0, 0, 0.5], device=x.device)
        return dict(stages=[out], dense=out)


# ------------------------------------------------------------------------------------------
def _to_grid(p, R):  # normalised tree coords -> [0, R-1]
    q = (p - torch.tensor([0, 0, 0.5], device=p.device)) * 1.4 + 0.5
    return q.clamp(0, 1 - 1e-6) * (R - 1)


def _from_grid(q, R):
    return (q / (R - 1) - 0.5) / 1.4 + torch.tensor([0, 0, 0.5], device=q.device)


class GRNet(CompletionBase):
    def __init__(self, R=64, n_sparse=2048):
        super().__init__()
        self.R = R
        self.ns = n_sparse
        c = [32, 64, 128, 256]
        self.e = nn.ModuleList()
        ci = 1
        for co in c:
            self.e.append(nn.Sequential(nn.Conv3d(ci, co, 4, padding=2), nn.BatchNorm3d(co), nn.LeakyReLU(0.2),
                                        nn.MaxPool3d(2)))
            ci = co
        s = R // 16
        self.fc = nn.Sequential(nn.Linear(256 * s ** 3, 2048), nn.ReLU(), nn.Linear(2048, 256 * s ** 3), nn.ReLU())
        self.d = nn.ModuleList()
        for ci, co in ((256, 128), (128, 64), (64, 32), (32, 1)):
            self.d.append(nn.Sequential(nn.ConvTranspose3d(ci, co, 4, stride=2, padding=1), nn.BatchNorm3d(co) if co > 1 else nn.Identity(),
                                        nn.ReLU() if co > 1 else nn.Identity()))
        self.feat_mlp = mlp([(32 + 64 + 128) * 8, 1792, 448, 112, 24])

    def gridding(self, p):
        B, N, _ = p.shape
        R = self.R
        q = _to_grid(p, R)
        base = q.floor().long().clamp(0, R - 2)
        fr = q - base
        vol = torch.zeros(B, R * R * R, device=p.device)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((fr[..., 0] if dx else 1 - fr[..., 0]) * (fr[..., 1] if dy else 1 - fr[..., 1])
                         * (fr[..., 2] if dz else 1 - fr[..., 2]))
                    idx = ((base[..., 0] + dx) * R + base[..., 1] + dy) * R + base[..., 2] + dz
                    vol.scatter_add_(1, idx, w)
        return vol.view(B, 1, R, R, R)

    def reverse_gridding(self, v):
        """one point per cell = weighted mean of its 8 vertices (weights = grid values)."""
        B, _, R, _, _ = v.shape
        v = v[:, 0]
        ar = torch.arange(R - 1, device=v.device, dtype=v.dtype)
        gx, gy, gz = torch.meshgrid(ar, ar, ar, indexing="ij")
        num = torch.zeros(B, R - 1, R - 1, R - 1, 3, device=v.device, dtype=v.dtype)
        den = torch.zeros(B, R - 1, R - 1, R - 1, device=v.device, dtype=v.dtype)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = v[:, dx:R - 1 + dx, dy:R - 1 + dy, dz:R - 1 + dz]
                    c = torch.stack([gx + dx, gy + dy, gz + dz], -1).to(v.dtype)
                    num = num + w.unsqueeze(-1) * c
                    den = den + w
        pts = num / den.unsqueeze(-1).clamp_min(1e-6)
        return pts.view(B, -1, 3), den.view(B, -1)

    def cubic_features(self, pts_grid, feats):
        """gather features of the 8 vertices of each point's cell from 3 feature volumes."""
        out = []
        for f in feats:  # f (B,C,r,r,r)
            B, C, r = f.shape[:3]
            q = pts_grid * (r - 1) / (self.R - 1)
            base = q.floor().long().clamp(0, r - 2)
            ff = f.view(B, C, -1)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        idx = ((base[..., 0] + dx) * r + base[..., 1] + dy) * r + base[..., 2] + dz
                        out.append(torch.gather(ff, 2, idx.unsqueeze(1).expand(-1, C, -1)).transpose(1, 2))
        return torch.cat(out, -1)

    def forward(self, x):
        B = x.shape[0]
        v = self.gridding(x)
        skips = []
        h = v
        for e in self.e:
            skips.append(h)
            h = e(h)
        s = h.shape
        h = self.fc(h.flatten(1)).view(s) + h
        dec_feats = []
        for i, d in enumerate(self.d):
            h = d(h)
            h = h[..., : skips[-1 - i].shape[-1], : skips[-1 - i].shape[-1], : skips[-1 - i].shape[-1]]
            if i < 3:
                dec_feats.append(h)
            h = h + skips[-1 - i]
        pts_g, den = self.reverse_gridding(h.relu())
        # keep the n_sparse cells with the highest density (random tie-break)
        idx = torch.topk(den + 1e-4 * torch.rand_like(den), self.ns, dim=1).indices
        sp = gather(pts_g, idx)
        feat = self.cubic_features(sp, dec_feats[::-1][:3][::-1])
        off = self.feat_mlp(feat).view(B, self.ns, 8, 3)
        sparse = _from_grid(sp, self.R)
        dense = (sparse.unsqueeze(2) + off * 0.02).reshape(B, -1, 3)
        return dict(stages=[sparse, dense], dense=dense)


# ------------------------------------------------------------------------------------------
class ProxyEncoder(nn.Module):
    """PoinTr point proxies: FPS centres + local mini-PointNet + positional embedding."""

    def __init__(self, n_center=128, k=32, dim=384):
        super().__init__()
        self.n, self.k = n_center, k
        self.local = mlp([3, 128, 256], last_act=True)
        self.local2 = mlp([512, 512, dim])
        self.pos = mlp([3, 128, dim])

    def forward(self, x):
        c = _fps_pts(x, self.n)
        nb = knn_idx(c, x, self.k)
        g = gather(x, nb) - c.unsqueeze(2)
        f = self.local(g)
        f = self.local2(torch.cat([f, f.amax(2, keepdim=True).expand_as(f)], -1)).amax(2)
        return c, f, self.pos(c)


class PoinTr(CompletionBase):
    def __init__(self, dim=384, n_query=256, enc_depth=6, dec_depth=8, fold_pts=64, adaptive=False, denoise=0):
        super().__init__()
        self.proxy = ProxyEncoder(128, 32, dim)
        self.enc = nn.ModuleList([GeoAwareBlock(dim) for _ in range(enc_depth)])
        self.dec = nn.ModuleList([GeoAwareBlock(dim, cross=True) for _ in range(dec_depth)])
        self.nq = n_query
        self.adaptive = adaptive
        self.denoise = denoise
        self.inc = mlp([dim, 1024], last_act=True)
        self.coarse = mlp([1024, 1024, 3 * n_query])
        if adaptive:
            self.coarse_in = mlp([dim, 256, 3])
            self.score = mlp([dim + 3, 256, 1])
            self.q_from_in = mlp([dim + 3, dim])
        self.q_mlp = mlp([1024 + 3, 1024, dim])
        self.q_pos = mlp([3, 128, dim])
        self.fold = Folding(dim + 1024, fold_pts, hidden=256)
        self.stage_w = [1.0, 1.0]

    def forward(self, x, gt=None):
        B = x.shape[0]
        c, f, pe = self.proxy(x)
        h = f + pe
        for blk in self.enc:
            h = blk(h, c)
        glob = self.inc(h).amax(1)
        coarse = self.coarse(glob).view(B, self.nq, 3)
        qf = self.q_mlp(torch.cat([glob.unsqueeze(1).expand(-1, self.nq, -1), coarse], -1))
        qpos = coarse
        extra = {}
        if self.adaptive:
            # adaptive query generation: candidates from input proxies + global prediction, ranked
            cin = c + self.coarse_in(h) * 0.1
            q_in = self.q_from_in(torch.cat([h, cin], -1))
            cand_q = torch.cat([q_in, qf], 1)
            cand_p = torch.cat([cin, coarse], 1)
            sc = self.score(torch.cat([cand_q, cand_p], -1))[..., 0]
            top = sc.topk(self.nq, dim=1).indices
            qf = gather(cand_q, top) * torch.sigmoid(gather(sc.unsqueeze(-1), top))
            qpos = gather(cand_p, top)
            coarse = torch.cat([coarse, cin], 1)
        n_dn = 0
        mask = None
        if self.denoise and self.training and gt is not None:
            n_dn = self.denoise
            dn_c = _fps_pts(gt, n_dn)
            dn_noisy = dn_c + torch.randn_like(dn_c) * 0.01
            dn_q = self.q_mlp(torch.cat([glob.unsqueeze(1).expand(-1, n_dn, -1), dn_noisy], -1))
            qf = torch.cat([qf, dn_q], 1)
            qpos = torch.cat([qpos, dn_noisy], 1)
            L = qf.shape[1]
            mask = torch.zeros(L, L, dtype=torch.bool, device=x.device)
            mask[: self.nq, : self.nq] = True
            mask[self.nq:, :] = True          # denoising queries may see everything, normal ones not them
            extra["dn_c"] = dn_c
        q = qf + self.q_pos(qpos)
        for blk in self.dec:
            q = blk(q, qpos, ctx=h, mask=mask)
        code = torch.cat([q, glob.unsqueeze(1).expand(-1, q.shape[1], -1)], -1)
        patch = self.fold(code) + qpos.unsqueeze(2)          # (B,Q,64,3)
        dense = patch[:, : self.nq].reshape(B, -1, 3)
        if not self.adaptive:     # official PoinTr concatenates the (sparse) input to both outputs
            coarse = torch.cat([coarse, _fps_pts(x, self.nq)], 1)
            dense = torch.cat([dense, x], 1)
        out = dict(stages=[coarse, dense], dense=dense)
        if n_dn:
            # denoising target: GT points near each clean centre
            dn_patch = patch[:, self.nq:]
            nb = knn_idx(extra["dn_c"], gt, dn_patch.shape[2])
            tgt = gather(gt, nb)
            l, _, _ = chamfer(dn_patch.reshape(-1, dn_patch.shape[2], 3), tgt.reshape(-1, tgt.shape[2], 3))
            out["extra_loss"] = 0.5 * l
        return out


class AdaPoinTr(PoinTr):
    def __init__(self):
        super().__init__(adaptive=True, denoise=64)


# ------------------------------------------------------------------------------------------
class SnowflakeEncoder(nn.Module):
    def __init__(self, out=512):
        super().__init__()
        self.sa1 = SetAbstraction(512, 16, 0, [64, 128])
        self.pt1 = PointTransformerLayer(128)
        self.sa2 = SetAbstraction(128, 16, 128, [128, 256])
        self.pt2 = PointTransformerLayer(256)
        self.sa3 = mlp([256 + 3, 512, out])

    def forward(self, x):
        c1, f1 = self.sa1(x)
        f1 = self.pt1(c1, f1)
        c2, f2 = self.sa2(c1, f1)
        f2 = self.pt2(c2, f2)
        return self.sa3(torch.cat([c2, f2], -1)).amax(1)


class SeedGenerator(nn.Module):
    def __init__(self, dim=512, n_seed=256, feat=128):
        super().__init__()
        self.n = n_seed
        self.ps = nn.ConvTranspose1d(dim, feat, n_seed, n_seed)
        self.m1 = mlp([feat + dim, 256, feat], last_act=True)
        self.m2 = mlp([feat, 64, 3])

    def forward(self, g):
        f = self.ps(g.unsqueeze(-1)).transpose(1, 2)            # (B,n,feat)
        f = self.m1(torch.cat([f, g.unsqueeze(1).expand(-1, self.n, -1)], -1))
        return self.m2(f) + torch.tensor([0, 0, 0.5], device=g.device), f


class SPD(nn.Module):
    """Snowflake point deconvolution."""

    def __init__(self, up, radius, dim=128, gdim=512, skip=True):
        super().__init__()
        self.up, self.radius = up, radius
        self.m1 = mlp([3, 64, 128], last_act=True)
        self.m2 = mlp([128 + 128 + gdim, 256, dim], last_act=True)
        self.skip = SkipTransformer(dim) if skip else None
        self.ps = nn.ConvTranspose1d(dim, dim, up, up, bias=False)
        self.up_mlp = mlp([dim * 2, 64, dim], last_act=True)
        self.delta = mlp([dim, 64, 3])

    def forward(self, p, prev, g):
        f = self.m1(p)
        f = self.m2(torch.cat([f, f.amax(1, keepdim=True).expand_as(f), g.unsqueeze(1).expand(-1, p.shape[1], -1)], -1))
        h = self.skip(p, f, prev if prev is not None else f) if self.skip is not None else f
        hc = self.ps(h.transpose(1, 2)).transpose(1, 2)
        hr = h.repeat_interleave(self.up, 1)
        k = self.up_mlp(torch.cat([hc, hr], -1)) + hr
        d = torch.tanh(self.delta(F.leaky_relu(k))) * self.radius
        return p.repeat_interleave(self.up, 1) + d, k


class SnowflakeNet(CompletionBase):
    def __init__(self, ups=(1, 4, 8)):
        super().__init__()
        self.enc = SnowflakeEncoder(512)
        self.seed = SeedGenerator(512, 256)
        self.spd = nn.ModuleList([SPD(u, 0.05 / (2 ** i)) for i, u in enumerate(ups)])

    def forward(self, x):
        g = self.enc(x)
        seed, _ = self.seed(g)
        p = _fps_pts(torch.cat([seed, x], 1), 512)
        stages = [seed, p]
        prev = None
        for s in self.spd:
            p, prev = s(p, prev, g)
            stages.append(p)
        return dict(stages=stages, dense=p)


# ------------------------------------------------------------------------------------------
class UpTransformer(nn.Module):
    """SeedFormer upsample transformer (local attention with seed features)."""

    def __init__(self, dim=128, k=16, up=1, radius=0.05):
        super().__init__()
        self.k, self.up, self.radius = k, up, radius
        self.q, self.kk, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.pos = mlp([3, 64, dim])
        self.attn = mlp([dim, dim * 2, dim * up])
        self.upf = mlp([dim, dim])
        self.delta = mlp([dim, 64, 3])
        self.dim = dim

    def forward(self, p, f, seedf):
        B, N, C = f.shape
        nb = knn_idx(p, p, self.k)
        q = self.q(f + seedf).unsqueeze(2)
        kf = gather(self.kk(f), nb)
        vf = gather(self.v(f), nb)
        pe = self.pos(p.unsqueeze(2) - gather(p, nb))
        a = self.attn(q - kf + pe).view(B, N, self.k, self.up, C)
        a = torch.softmax(a, 2)
        h = ((vf + pe).unsqueeze(3) * a).sum(2)                   # (B,N,up,C)
        h = self.upf(h).reshape(B, N * self.up, C) + f.repeat_interleave(self.up, 1)
        d = torch.tanh(self.delta(h)) * self.radius
        return p.repeat_interleave(self.up, 1) + d, h


class SeedFormer(CompletionBase):
    def __init__(self, ups=(1, 4, 8), dim=128):
        super().__init__()
        self.sa1 = SetAbstraction(512, 16, 0, [64, 128])
        self.pt1 = PointTransformerLayer(128)
        self.sa2 = SetAbstraction(128, 16, 128, [128, 256])
        self.pt2 = PointTransformerLayer(256)
        self.glob = mlp([256 + 3, 512, 512])
        self.seed = SeedGenerator(512, 256, feat=dim)
        self.seed_up = PointTransformerLayer(dim)
        self.inp = mlp([3, 64, dim])
        self.ups = nn.ModuleList([UpTransformer(dim, up=u, radius=0.05 / (2 ** i)) for i, u in enumerate(ups)])

    def interp(self, p, sp, sf):
        nb = knn_idx(p, sp, 3)
        d = (p.unsqueeze(2) - gather(sp, nb)).norm(dim=-1).clamp_min(1e-6)
        w = (1 / d) / (1 / d).sum(-1, keepdim=True)
        return (gather(sf, nb) * w.unsqueeze(-1)).sum(2)

    def forward(self, x):
        c1, f1 = self.sa1(x)
        f1 = self.pt1(c1, f1)
        c2, f2 = self.sa2(c1, f1)
        f2 = self.pt2(c2, f2)
        g = self.glob(torch.cat([c2, f2], -1)).amax(1)
        seed, sf = self.seed(g)
        sf = self.seed_up(seed, sf)
        p = _fps_pts(torch.cat([seed, x], 1), 512)
        f = self.inp(p)
        stages = [seed, p]
        for u in self.ups:
            p, f = u(p, f, self.interp(p, seed, sf))
            stages.append(p)
        return dict(stages=stages, dense=p)


# ------------------------------------------------------------------------------------------
class SFA(nn.Module):
    """Self-feature augment (global self-attention)."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.a = MHA(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.ff = mlp([dim, dim * 2, dim])

    def forward(self, x, ctx=None):
        x = x + self.a(self.n1(x), None if ctx is None else self.n1(ctx))
        return x + self.ff(self.n2(x))


class PointAttN(CompletionBase):
    def __init__(self, dim=256):
        super().__init__()
        self.emb = mlp([3, 64, 128])
        self.gdp1 = SFA(128)       # geometric detail perception = cross-attn FPS pts -> all pts
        self.up1 = nn.Linear(128, dim)
        self.sfa1 = SFA(dim)
        self.gdp2 = SFA(dim)
        self.sfa2 = SFA(dim)
        self.glob = mlp([dim, 512, 512])
        self.seed_ps = nn.ConvTranspose1d(512, 128, 128, 128)
        self.seed_sfa = SFA(128)
        self.seed_out = mlp([128, 64, 3])
        self.ref = nn.ModuleList()
        for up, r in ((4, 0.03), (8, 0.012)):
            self.ref.append(nn.ModuleDict(dict(e=mlp([3, 64, 128]), g=nn.Linear(512, 128), s1=SFA(128), s2=SFA(128),
                                               ps=nn.ConvTranspose1d(128, 128, up, up), d=mlp([128, 64, 3]))))
        self.ups = (4, 8)
        self.rad = (0.03, 0.012)

    def forward(self, x):
        B = x.shape[0]
        f = self.emb(x)
        i1 = fps(x, 512)
        c1 = gather(x, i1)
        f1 = self.gdp1(gather(f, i1), f)
        f1 = self.sfa1(self.up1(f1))
        i2 = fps(c1, 128)
        c2 = gather(c1, i2)
        f2 = self.sfa2(self.gdp2(gather(f1, i2), f1))
        g = self.glob(f2).amax(1)
        sf = self.seed_ps(g.unsqueeze(-1)).transpose(1, 2)
        sf = self.seed_sfa(sf)
        seed = self.seed_out(sf) + torch.tensor([0, 0, 0.5], device=x.device)
        p = _fps_pts(torch.cat([seed, x], 1), 512)
        stages = [seed, p]
        for m, up, r in zip(self.ref, self.ups, self.rad):
            h = m["e"](p) + m["g"](g).unsqueeze(1)
            h = m["s2"](m["s1"](h)) if p.shape[1] <= 2048 else m["s1"](h, gather(h, fps(p, 1024)))
            hu = m["ps"](h.transpose(1, 2)).transpose(1, 2)
            p = p.repeat_interleave(up, 1) + torch.tanh(m["d"](hu)) * r
            stages.append(p)
        return dict(stages=stages, dense=p)


BASELINES = dict(pcn=PCN, foldingnet=FoldingNet, topnet=TopNet, grnet=GRNet, pointr=PoinTr, adapointr=AdaPoinTr,
                 snowflakenet=SnowflakeNet, seedformer=SeedFormer, pointattn=PointAttN)
