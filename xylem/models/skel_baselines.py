"""Skeleton-learning baselines, re-implemented on the same serialized-transformer backbone as
XylemNet for a controlled comparison.

Smart-Tree* (Dobbs et al., IbPRIA'23): per-point medial-axis vector + radius + wood/leaf
    regression; skeleton by projecting wood points onto the medial axis followed by greedy
    geodesic path extraction (here: our geodesic-bin skeletonizer).  No completion of
    occluded parts.
DiffTS* (Marks et al., ICCV'25): conditional denoising diffusion over a set of skeleton nodes
    (position + branch direction) with a transformer denoiser cross-attending to point tokens;
    topology by a minimum spanning tree.  We add a radius head (DiffTS does not estimate radii)
    so that the dense/forestry metrics can be computed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import field_losses
from ..ops import chamfer, gather, nn_idx, normalize, sinusoidal
from .blocks import MHA, mlp
from .xylemnet import XylemNet


def _strip_decoder(net):
    """Remove the XylemNet decoder so only the shared encoder (+ point head) remains."""
    for n in ("dec", "grow_block", "grow_head", "grow_mlp", "grow_n", "off_head", "child_code", "mem3", "mem2",
              "mem_pe", "seed_proj", "q_pe", "head0", "tube_feat", "res1", "res2", "fold1", "fold2"):
        if hasattr(net, n):
            setattr(net, n, None)
    for n in ("learned_q", "learned_anchor", "step"):
        setattr(net, n, None)


class SmartTree(nn.Module):
    kind = "smarttree"

    def __init__(self):
        super().__init__()
        self.net = XylemNet(use_learned_q=False)
        # only encoder + point head are used
        _strip_decoder(self.net)

    def forward(self, x):
        cfg = self.net.cfg
        n = min(cfg["n_in"], x.shape[1])
        feats, poss, inv0 = self.net.encode(x[:, :n], cfg)
        return dict(point=self.net.point_head(feats, inv0, n), n_in=n)

    def loss(self, out, batch):
        l, logs = field_losses(out["point"], batch, out["n_in"], dict(seg=1.0, p_flow=0.3, p_geo=0.3, p_rad=0.5, p_med=1.0, field_on_leaves=0.0))
        logs["total"] = l.detach()
        return l, logs


class DiffTS(nn.Module):
    kind = "diffts"

    def __init__(self, n_nodes=1024, dim=256, depth=6, T=1000):
        super().__init__()
        self.enc = XylemNet(use_learned_q=False, use_point_head=False)
        _strip_decoder(self.enc)
        self.K, self.T = n_nodes, T
        self.inp = mlp([6 + 6 * 2 * 6, dim, dim])
        self.t_emb = mlp([128, dim, dim])
        self.mem = nn.Linear(512, dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict(dict(n1=nn.LayerNorm(dim), sa=MHA(dim, 8), n2=nn.LayerNorm(dim),
                                                  ca=MHA(dim, 8), n3=nn.LayerNorm(dim), ff=mlp([dim, dim * 4, dim]))))
        self.out = mlp([dim, dim, 6 + 1])
        s = 0.008
        t = torch.linspace(0, T, T + 1) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        self.register_buffer("abar", ab[1:].clamp(1e-5, 1.0))
        # data normalisation of the 6-D node state
        self.register_buffer("mu", torch.tensor([0, 0, 0.5, 0, 0, 0.0]))
        self.register_buffer("sd", torch.tensor([0.3, 0.3, 0.3, 1, 1, 1.0]))

    def cond(self, x):
        cfg = self.enc.cfg
        feats, poss, _ = self.enc.encode(x, cfg)
        return self.mem(feats[2]) + 0, poss[2]

    def denoise(self, z, t, mem):
        te = self.t_emb(sinusoidal((t.float() / self.T).view(-1, 1, 1), 64).view(z.shape[0], 1, -1))
        h = self.inp(torch.cat([z, sinusoidal(z, 6)], -1)) + te
        for l in self.layers:
            h = h + l["sa"](l["n1"](h))
            h = h + l["ca"](l["n2"](h), mem)
            h = h + l["ff"](l["n3"](h))
        o = self.out(h)
        return o[..., :6], o[..., 6]      # x0 prediction (normalised), log-radius

    def forward(self, x, batch=None):
        mem, _ = self.cond(x)
        return dict(mem=mem)

    def loss(self, out, batch):
        mem = out["mem"]
        B = mem.shape[0]
        idx = torch.randint(0, batch["sk"].shape[1], (B, self.K), device=mem.device)
        x0 = torch.cat([gather(batch["sk"], idx), gather(batch["sk_flow"], idx)], -1)
        x0n = (x0 - self.mu) / self.sd
        t = torch.randint(0, self.T, (B,), device=mem.device)
        ab = self.abar[t].view(B, 1, 1)
        noise = torch.randn_like(x0n)
        zt = ab.sqrt() * x0n + (1 - ab).sqrt() * noise
        pred, logr = self.denoise(zt, t, mem)
        l_pos = F.mse_loss(pred[..., :3], x0n[..., :3])
        l_dir = (1 - (normalize(pred[..., 3:]) * x0[..., 3:]).sum(-1)).mean()
        rt = gather(batch["sk_r"].unsqueeze(-1), idx)[..., 0]
        mr = torch.isfinite(rt).float()
        l_r = ((logr - torch.log(torch.nan_to_num(rt, nan=1.0).clamp_min(1e-5))).abs() * mr).sum() / mr.sum().clamp_min(1.0)
        tot = l_pos + 0.3 * l_dir + 0.3 * l_r
        return tot, dict(total=tot.detach(), pos=l_pos.detach(), dir=l_dir.detach(), rad=l_r.detach())

    @torch.no_grad()
    def sample(self, x, steps=20):
        mem, _ = self.cond(x)
        B = x.shape[0]
        z = torch.randn(B, self.K, 6, device=x.device)
        ts = torch.linspace(self.T - 1, 0, steps, device=x.device).long()
        logr = None
        for i, t in enumerate(ts):
            tt = t.expand(B)
            x0, logr = self.denoise(z, tt, mem)
            ab = self.abar[t]
            eps = (z - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-5)
            if i + 1 < len(ts):
                ab_n = self.abar[ts[i + 1]]
                z = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps        # DDIM (eta = 0)
            else:
                z = x0
        x0 = z * self.sd + self.mu
        return x0[..., :3], normalize(x0[..., 3:]), logr.exp()
