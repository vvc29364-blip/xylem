"""XylemNet: botanically-constrained, elastic skeleton-first tree reconstruction.

Pipeline (see paper/main.tex, Sec. 3)
  1. Serialized elastic point transformer encoder (multi-order space-filling-curve windows,
     PTv3-style, pure tensor ops -> ONNX/TensorRT friendly on Jetson).
  2. Xylem field head: per input point wood/leaf, flow-to-root, geodesic-to-root, radius and
     medial-axis offset.
  3. Skeleton query decoder: seeds = pooled tokens + learned "occlusion" queries.
  4. Hierarchical xylem growth: every node spawns children (x4, x2) that are refined by
     serialized window attention + cross attention to finer tokens.  Every node carries
     (position, radius, flow, geodesic, half-length, existence).
  5. Tubular splatting: dense surface = generalized cylinders around the nodes (radius-exact, so
     thin twigs are not blurred) + a bounded learned residual.  Resolution-free at test time.
Topology is recovered by flow-guided arborescence decoding (models/topology.py): parents are
chosen among nodes with *smaller* predicted geodesic, which guarantees an acyclic single tree.
"""
import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import PERMS, gather, normalize, orthonormal_basis, serialize, sinusoidal
from .elastic import DecoderLayer, ELayerNorm, ELinear, EMLP, SerialBlock

# ------------------------------------------------------------------------------------------
# architecture / search space
# ------------------------------------------------------------------------------------------
ARCH = dict(dims=(128, 256, 512), depths=(2, 2, 6), dec_dim=384, dec_layers=4, n_learned=64,
            grow=(4, 2), head_dim=32, mlp_ratio=4, windows=(64, 64, 0), node_window=32, tube_m=8,
            n_perm=4, n_anchor=256, use_tube=True, use_growth=True, use_point_head=True, use_learned_q=True,
            use_anchor=True)

SPACE = dict(n_in=[2048, 3072, 4096], width=[0.5, 0.75, 1.0], d1=[1, 2], d2=[1, 2], d3=[2, 4, 6],
             mlp=[2, 3, 4], dec_layers=[1, 2, 3, 4], dec_width=[0.5, 0.75, 1.0], n_learned=[32, 64],
             grow2=[1, 2], n_anchor=[128, 192, 256])
MAX_CFG = {k: v[-1] for k, v in SPACE.items()}
MIN_CFG = {k: v[0] for k, v in SPACE.items()}


def sample_cfg(rng=random):
    return {k: rng.choice(v) for k, v in SPACE.items()}


def cfg_key(cfg):
    return "-".join(f"{k}{cfg[k]}" for k in SPACE)


def _rd(x, hd):
    return max(hd, int(round(x / hd)) * hd)


# ------------------------------------------------------------------------------------------
class NodeHead(nn.Module):
    """(feat) -> dpos(3) logr(1) flow(3) geo(1) half_len(1) exist(1)"""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = ELinear(dim, dim)
        self.fc2 = ELinear(dim, 10)
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias.zero_()
            self.fc2.bias[3] = math.log(0.004)
            self.fc2.bias[6] = -1.0      # flow prior: pointing down (towards the root)
            self.fc2.bias[9] = 2.0

    def forward(self, x):
        o = self.fc2(F.gelu(self.fc1(x, x.shape[-1])))
        return dict(dpos=o[..., 0:3], logr=o[..., 3].clamp(-12, 0), flow=normalize(o[..., 4:7] + 1e-4),
                    geo=F.softplus(o[..., 7]), h=F.softplus(o[..., 8]) * 0.01 + 1e-3, exist=o[..., 9])


class XylemNet(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        a = copy.deepcopy(ARCH)
        a.update(kw)
        self.a = a
        dims, hd = a["dims"], a["head_dim"]
        D = a["dec_dim"]
        self.pe_dim = 3 + 3 * 2 * 6
        # ---- encoder ----
        self.embed1 = ELinear(self.pe_dim, dims[0])
        self.embed_n = ELayerNorm(dims[0])
        self.embed2 = ELinear(dims[0], dims[0])
        self.stage_pe = nn.ModuleList([ELinear(self.pe_dim, d) for d in dims])
        self.stages = nn.ModuleList([
            nn.ModuleList([SerialBlock(dims[s], hd, a["mlp_ratio"]) for _ in range(a["depths"][s])])
            for s in range(3)])
        self.down = nn.ModuleList([ELinear(dims[s], dims[s + 1]) for s in range(2)])
        self.down_n = nn.ModuleList([ELayerNorm(dims[s + 1]) for s in range(2)])
        # ---- xylem field head (FPN) ----
        if a["use_point_head"]:
            self.up = nn.ModuleList([ELinear(dims[1], dims[0]), ELinear(dims[2], dims[1])])
            self.ph_n = ELayerNorm(dims[0])
            self.ph1 = ELinear(dims[0], dims[0])
            self.ph2 = ELinear(dims[0], 10)
        # ---- decoder ----
        self.mem3 = ELinear(dims[2], D)
        self.mem2 = ELinear(dims[1], D)
        self.mem_pe = ELinear(self.pe_dim, D)
        self.seed_proj = ELinear(dims[2], D)
        self.learned_q = nn.Parameter(torch.randn(a["n_learned"], D) * 0.02)
        self.learned_anchor = nn.Parameter(torch.rand(a["n_learned"], 3) * torch.tensor([0.8, 0.8, 1.0])
                                           - torch.tensor([0.4, 0.4, 0.0]))
        self.q_pe = ELinear(self.pe_dim, D)
        if a["use_anchor"] and a["use_point_head"]:
            self.anchor_proj = ELinear(dims[0], D)
        self.dec = nn.ModuleList([DecoderLayer(D, D, hd, a["mlp_ratio"]) for _ in range(a["dec_layers"])])
        self.head0 = NodeHead(D)
        n_grow = len(a["grow"]) if a["use_growth"] else 1
        grow = a["grow"] if a["use_growth"] else (int(math.prod(a["grow"])),)
        self.grow_ratio = grow
        self.child_code = nn.ParameterList([nn.Parameter(torch.randn(r, D) * 0.02) for r in grow])
        self.grow_mlp = nn.ModuleList([EMLP(D, D * 2) for _ in range(n_grow)])
        self.grow_n = nn.ModuleList([ELayerNorm(D) for _ in range(n_grow)])
        self.off_head = nn.ModuleList([ELinear(D, 3) for _ in range(n_grow)])
        self.step = nn.Parameter(torch.tensor([0.03, 0.015, 0.01][:n_grow]))
        self.grow_block = nn.ModuleList([SerialBlock(D, hd, a["mlp_ratio"], ctx_dim=D) for _ in range(n_grow)])
        self.grow_head = nn.ModuleList([NodeHead(D) for _ in range(n_grow)])
        # ---- dense decoder ----
        self.tube_feat = ELinear(D, 64)
        tin = 64 + 3 * 2 * 4 + 3
        if a["use_tube"]:
            self.res1 = nn.Linear(tin, 64)
            self.res2 = nn.Linear(64, 3)
            nn.init.zeros_(self.res2.weight)
            nn.init.zeros_(self.res2.bias)
        else:  # ablation: folding patches (no generalized-cylinder prior)
            self.fold1 = nn.Linear(64 + 2 * 2 * 4 + 2, 128)
            self.fold2 = nn.Linear(128, 3)
        self.cfg = dict(MAX_CFG)

    # --------------------------------------------------------------------------------------
    def set_cfg(self, cfg):
        self.cfg = dict(cfg)

    def active_dims(self, cfg):
        hd = self.a["head_dim"]
        dims = [min(d, _rd(d * cfg["width"], hd)) for d in self.a["dims"]]
        D = min(self.a["dec_dim"], _rd(self.a["dec_dim"] * cfg["dec_width"], hd))
        return dims, D

    def pe(self, x):
        return torch.cat([x, sinusoidal(x, 6)], -1)

    # --------------------------------------------------------------------------------------
    def encode(self, xyz, cfg):
        a = self.a
        dims, _ = self.active_dims(cfg)
        depths = [min(cfg["d1"], a["depths"][0]), min(cfg["d2"], a["depths"][1]), min(cfg["d3"], a["depths"][2])]
        order0, inv0 = serialize(xyz, PERMS[0])
        pos = gather(xyz, order0)
        x = self.embed2(F.gelu(self.embed_n(self.embed1(self.pe(pos), dims[0]))), dims[0])
        feats, poss = [], []
        for s in range(3):
            x = x + self.stage_pe[s](self.pe(pos), dims[s])
            n_perm = max(1, min(a["n_perm"], len(PERMS)))
            rel = [None] + [serialize(pos, PERMS[p]) for p in range(1, n_perm)]
            for b in range(depths[s]):
                p = b % n_perm
                if p == 0:
                    x = self.stages[s][b](x, a["windows"][s], cfg["mlp"])
                else:
                    o, iv = rel[p]
                    x = gather(self.stages[s][b](gather(x, o), a["windows"][s], cfg["mlp"]), iv)
            feats.append(x)
            poss.append(pos)
            if s < 2:
                B, N, _ = x.shape
                pad = (-N) % 4
                if pad:
                    x = torch.cat([x, x[:, -pad:]], 1)
                    pos = torch.cat([pos, pos[:, -pad:]], 1)
                y = self.down[s](x, dims[s + 1])
                x = self.down_n[s](y.view(B, -1, 4, dims[s + 1]).amax(2))
                pos = pos.view(B, -1, 4, 3).mean(2)
        return feats, poss, inv0

    def point_head(self, feats, inv0, n):
        f1, f2, f3 = feats
        up2 = f2 + self.up[1](f3.repeat_interleave(4, 1)[:, : f2.shape[1]], f2.shape[-1])
        up1 = f1 + self.up[0](up2.repeat_interleave(4, 1)[:, : f1.shape[1]], f1.shape[-1])
        o = self.ph2(F.gelu(self.ph1(self.ph_n(up1), up1.shape[-1])), 10)
        o = gather(o, inv0)[:, :n]
        self._up1 = gather(up1, inv0)[:, :n]
        return dict(seg=o[..., 0], flow=normalize(o[..., 1:4] + 1e-4), geo=F.softplus(o[..., 4]),
                    logr=(o[..., 5] + math.log(0.004)).clamp(-12, 0), med=o[..., 6:9] * 0.05, tree=o[..., 9] + 3.0)

    # --------------------------------------------------------------------------------------
    def forward(self, inp, cfg=None, tube_phase=None):
        cfg = dict(self.cfg if cfg is None else cfg)
        a = self.a
        n = min(cfg["n_in"], inp.shape[1])
        xyz = inp[:, :n]
        dims, D = self.active_dims(cfg)
        feats, poss, inv0 = self.encode(xyz, cfg)
        out = dict(n_in=n)
        if a["use_point_head"]:
            out["point"] = self.point_head(feats, inv0, n)
        B = xyz.shape[0]
        mem3 = self.mem3(feats[2], D) + self.mem_pe(self.pe(poss[2]), D)
        mem2 = self.mem2(feats[1], D) + self.mem_pe(self.pe(poss[1]), D)
        # ---- seeds ----
        q = self.seed_proj(feats[2], D)
        anchor = poss[2]
        if a["use_learned_q"]:
            nl = min(cfg["n_learned"], a["n_learned"])
            q = torch.cat([q, self.learned_q[:nl, :D].unsqueeze(0).expand(B, -1, -1)], 1)
            anchor = torch.cat([anchor, self.learned_anchor[:nl].unsqueeze(0).expand(B, -1, -1)], 1)
        if a["use_anchor"] and a["use_point_head"]:
            # observation-anchored queries: most confident wood points projected on their medial axis
            na = min(cfg.get("n_anchor", a["n_anchor"]), a["n_anchor"], n)
            top = out["point"]["seg"].detach().topk(na, dim=1).indices
            apos = gather(xyz + out["point"]["med"], top)
            q = torch.cat([q, self.anchor_proj(gather(self._up1, top), D)], 1)
            anchor = torch.cat([anchor, apos], 1)
        q = q + self.q_pe(self.pe(anchor), D)
        for layer in self.dec[: min(cfg["dec_layers"], a["dec_layers"])]:
            q = layer(q, mem3, cfg["mlp"])
        h = self.head0(q)
        pos = anchor + h["dpos"]
        stages = [dict(pos=pos, **{k: v for k, v in h.items() if k != "dpos"})]
        # ---- hierarchical growth ----
        for g, R in enumerate(self.grow_ratio):
            if a["use_growth"] and g == 1:
                R = min(R, cfg["grow2"])
            K = q.shape[1]
            qc = (q.unsqueeze(2) + self.child_code[g][:R, :D]).reshape(B, K * R, D)
            qc = qc + self.grow_mlp[g](self.grow_n[g](qc), 2)
            off = self.off_head[g](qc, 3) * self.step[g]
            pc = pos.repeat_interleave(R, 1) + off
            o, _ = serialize(pc, PERMS[g % len(PERMS)])
            qc, pc = gather(qc, o), gather(pc, o)
            qc = qc + self.q_pe(self.pe(pc), D)
            qc = self.grow_block[g](qc, a["node_window"], cfg["mlp"], ctx=mem2)
            h = self.grow_head[g](qc)
            pos = pc + h["dpos"]
            q = qc
            stages.append(dict(pos=pos, **{k: v for k, v in h.items() if k != "dpos"}))
        out["nodes"] = stages
        out["node_feat"] = self.tube_feat(q, 64)
        out["dense"] = self.tube_points(stages[-1], out["node_feat"], phase=tube_phase)
        return out

    # --------------------------------------------------------------------------------------
    def _tube(self, X, t, r, hl, feat, s, th):
        """X,t (B,K,3) r,hl (B,K)  feat (B,K,64)  s,th (B,K,M) -> (B,K,M,3)"""
        u, v = orthonormal_basis(t)
        ring = th.cos().unsqueeze(-1) * u.unsqueeze(2) + th.sin().unsqueeze(-1) * v.unsqueeze(2)
        p = X.unsqueeze(2) + (s * hl.unsqueeze(-1)).unsqueeze(-1) * t.unsqueeze(2) + r[..., None, None] * ring
        code = torch.stack([s, th.cos(), th.sin()], -1)
        z = torch.cat([feat.unsqueeze(2).expand(-1, -1, s.shape[-1], -1), code, sinusoidal(code, 4)], -1)
        if self.a["use_tube"]:
            res = torch.tanh(self.res2(F.gelu(self.res1(z)))) * (0.5 * r + 0.002)[..., None, None]
            return p + res
        # folding-patch ablation: ignore the cylinder, learn the patch from a 2-D code
        c2 = torch.stack([s, th / math.pi - 1], -1)
        z = torch.cat([feat.unsqueeze(2).expand(-1, -1, s.shape[-1], -1), c2, sinusoidal(c2, 4)], -1)
        return X.unsqueeze(2) + self.fold2(F.gelu(self.fold1(z))) * 0.05

    def tube_points(self, node, feat, phase=None):
        X, t, hl = node["pos"], node["flow"], node["h"]
        r = node["logr"].exp()
        B, K = X.shape[:2]
        M = self.a["tube_m"]
        m = torch.arange(M, device=X.device, dtype=X.dtype)
        s = (2 * (m + 0.5) / M - 1).expand(B, K, M)
        if phase is None:
            phase = torch.rand(B, K, 1, device=X.device, dtype=X.dtype) * 2 * math.pi if self.training \
                else torch.zeros(B, K, 1, device=X.device, dtype=X.dtype)
        th = (m * 2.399963 + phase) % (2 * math.pi)
        return self._tube(X, t, r, hl, feat, s, th).reshape(B, K * M, 3)

    @torch.no_grad()
    def sample_dense(self, out, n, exist_thr=0.0, generator=None, area_weighted=False):
        """Resolution-free resampling of the tubular surface (test time).  Nodes are sampled
        uniformly (the distribution the tubes were trained with); area weighting is optional."""
        node = out["nodes"][-1]
        X, t, hl = node["pos"], node["flow"], node["h"]
        r = node["logr"].exp()
        keep = (node["exist"].sigmoid() > exist_thr).float()
        w = ((r + 1e-4) * hl if area_weighted else torch.ones_like(r)) * keep
        w = torch.where(w.sum(1, keepdim=True) > 0, w, torch.ones_like(r))
        idx = torch.multinomial(w, n, replacement=True, generator=generator)       # (B,n)
        g = lambda z: gather(z, idx)
        B = X.shape[0]
        s = torch.rand(B, n, 1, device=X.device, generator=generator) * 2 - 1
        th = torch.rand(B, n, 1, device=X.device, generator=generator) * 2 * math.pi
        p = self._tube(g(X), g(t), g(r[..., None])[..., 0], g(hl[..., None])[..., 0], g(out["node_feat"]), s, th)
        return p.reshape(B, n, 3)


def build_xylemnet(variant=None, **kw):
    """variant: None (supernet) or an explicit cfg dict -> standalone net with max = cfg."""
    if variant is None:
        return XylemNet(**kw)
    c = variant
    tmp = XylemNet(**kw)
    dims, D = tmp.active_dims(c)
    a = dict(kw)
    a.update(dims=tuple(dims), dec_dim=D, depths=(c["d1"], c["d2"], c["d3"]), dec_layers=c["dec_layers"],
             n_learned=c["n_learned"], mlp_ratio=c["mlp"], n_anchor=c.get("n_anchor", tmp.a["n_anchor"]),
             grow=(tmp.a["grow"][0], c["grow2"]) if tmp.a["use_growth"] else tmp.a["grow"])
    m = XylemNet(**a)
    full = {k: v[-1] for k, v in SPACE.items()}
    full.update(n_in=c["n_in"], mlp=c["mlp"], grow2=c["grow2"], n_anchor=c.get("n_anchor", tmp.a["n_anchor"]))
    m.set_cfg(full)
    m.fixed_cfg = dict(c)
    return m
