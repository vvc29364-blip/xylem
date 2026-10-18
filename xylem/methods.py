"""Unified method interface: build / loss / predict for every compared approach."""
import numpy as np
import torch

from .data.skeleton import sample_surface
from .losses import kd_loss, xylem_loss
from .models.baselines import BASELINES
from .models.skel_baselines import DiffTS, SmartTree
from .models.topology import extract_tree, mst_tree, refine_radii
from .models.xylemnet import MAX_CFG, XylemNet, build_xylemnet
from .skeletonize import geobin_skeleton

N_EVAL = 16384

# display names used in tables
DISPLAY = dict(geobin="GeoBin-QSM (classical)", pcn="PCN", foldingnet="FoldingNet", topnet="TopNet", grnet="GRNet",
               pointr="PoinTr", adapointr="AdaPoinTr", snowflakenet="SnowflakeNet", seedformer="SeedFormer",
               pointattn="PointAttN", smarttree="Smart-Tree*", diffts="DiffTS*", xylem="XylemNet (ours)")
VENUE = dict(geobin="TOG'07/RS'19", pcn="3DV'18", foldingnet="CVPR'18", topnet="CVPR'19", grnet="ECCV'20",
             pointr="ICCV'21", adapointr="TPAMI'23", snowflakenet="ICCV'21", seedformer="ECCV'22",
             pointattn="AAAI'24", smarttree="IbPRIA'23", diffts="ICCV'25", xylem="")


class XylemMethod(torch.nn.Module):
    kind = "xylem"

    def __init__(self, model_kw=None, loss_w=None, fixed_cfg=None, topology="flow", merge_input=True,
                 skel_source="union", dense_source="graph"):
        super().__init__()
        self.dense_source = dense_source    # 'graph' = surface rendered on the decoded skeleton, 'nodes' = node tubes
        self.node_thr = 0.5                 # confidence threshold of generated nodes (tuned on validation)
        self.skel_source = skel_source      # 'union' = generated nodes + medially projected observations
        self.merge_input = merge_input      # leaf-aware observation preservation at test time
        model_kw = model_kw or {}
        self.net = build_xylemnet(fixed_cfg, **model_kw) if fixed_cfg else XylemNet(**model_kw)
        self.loss_w = loss_w or {}
        self.topology = topology
        self.bot_on = True

    def forward(self, x, cfg=None):
        return self.net(x, cfg)

    def loss(self, out, batch):
        return xylem_loss(out, batch, self.loss_w, bot=self.bot_on)

    def kd(self, s, t):
        return kd_loss(s, t, self.loss_w)


class Classical(torch.nn.Module):
    kind = "classical"

    def forward(self, x):
        return {}


class CircleFit(torch.nn.Module):
    """Classical forestry baseline for real trees: DBH by RANSAC circle fit at 1.3 m, height = max z."""
    kind = "circlefit"

    def forward(self, x):
        return {}


def build_method(name, model_kw=None, loss_w=None, fixed_cfg=None, topology="flow"):
    if name == "xylem":
        return XylemMethod(model_kw, loss_w, fixed_cfg, topology)
    if name in BASELINES:
        return BASELINES[name]()
    if name == "smarttree":
        return SmartTree()
    if name == "diffts":
        return DiffTS()
    if name == "geobin":
        return Classical()
    if name == "circlefit":
        return CircleFit()
    raise KeyError(name)


def _fit_n(p, n, rng):
    if len(p) >= n:
        return p[rng.choice(len(p), n, replace=False)]
    return np.concatenate([p, p[rng.choice(len(p), n - len(p))]])


def _tubes(graph, rng, n=N_EVAL):
    pos, rad, par = graph
    if len(pos) < 2:
        return np.repeat(pos[:1], n, 0)
    return sample_surface(pos.astype(np.float64), np.maximum(rad, 1e-4).astype(np.float64), par.astype(np.int64),
                          n, rng, return_attr=False)


@torch.no_grad()
def predict(method, batch, cfg=None, with_graph=True, bin_size=0.02, seed=0):
    """Returns a list (one per sample) of dict(dense, graph, seg, extra)."""
    inp = batch["inp"]
    B = inp.shape[0]
    rng = np.random.default_rng(seed)
    res = []
    kind = method.kind
    if kind == "xylem":
        out = method(inp, cfg)
        gen = torch.Generator(device=inp.device)
        gen.manual_seed(seed)
        dense = method.net.sample_dense(out, N_EVAL, generator=gen).float().cpu().numpy()
        node = {k: v.float().cpu().numpy() for k, v in out["nodes"][-1].items()}
        seg = out["point"]["seg"].float().cpu().numpy() if "point" in out else None
        pt = {k: v.float().cpu().numpy() for k, v in out["point"].items()} if "point" in out else None
        xin = inp[:, : out["n_in"]].float().cpu().numpy()
        for b in range(B):
            g = None
            if with_graph:
                P, Fl, G, R = node["pos"][b], node["flow"][b], node["geo"][b], np.exp(node["logr"][b])
                E = 1 / (1 + np.exp(-node["exist"][b]))
                E = np.where(E > getattr(method, "node_thr", 0.5), E, 0.0)
                if getattr(method, "skel_source", "nodes") == "union" and pt is not None:
                    # observed points (wood AND leaves) projected on the axis of their supporting branch
                    ok = (pt["tree"][b] > 0) if "tree" in pt else np.ones(len(xin[b]), bool)
                    P = np.concatenate([P, (xin[b] + pt["med"][b])[ok]])
                    Fl = np.concatenate([Fl, pt["flow"][b][ok]])
                    G = np.concatenate([G, pt["geo"][b][ok]])
                    R = np.concatenate([R, np.exp(pt["logr"][b][ok])])
                    E = np.concatenate([E, 1 / (1 + np.exp(-pt["tree"][b][ok])) if "tree" in pt else np.ones(ok.sum())])
                if method.topology == "flow":
                    g = extract_tree(P, Fl, G, R, E, nms_dist=getattr(method, "nms_dist", 0.003))
                else:   # ablation: MST topology from positions only
                    keep = E > 0.5
                    g = mst_tree(P[keep], R[keep])
            if g is not None and getattr(method, "refine_radius", True) and seg is not None and len(g[0]) > 1:
                obs = [xin[b][seg[b] > 0]]
                if "bh_pts" in batch and batch["bh_pts"][b] is not None:    # full-resolution band if provided
                    bh = batch["bh_pts"][b]
                    obs.append(bh.cpu().numpy() if torch.is_tensor(bh) else np.asarray(bh))
                g = (g[0], refine_radii(g[0], g[1], g[2], np.concatenate(obs), float(batch["scale"][b])), g[2])
            d = dense[b]
            if g is not None and getattr(method, "dense_source", "graph") == "graph" and len(g[0]) > 1:
                d = _tubes(g, rng)                   # surface rendered on the decoded tree (QSM-consistent)
            if getattr(method, "merge_input", False) and seg is not None:
                w = xin[b][seg[b] > 0]               # leaf-aware observation preservation
                if len(w):
                    d = np.concatenate([w, d[rng.choice(len(d), N_EVAL - len(w), replace=False)]])
            res.append(dict(dense=d, graph=g, seg=None if seg is None else seg[b]))
        return res
    if kind == "completion":
        out = method(inp)
        dense = out["dense"].float().cpu().numpy()
        for b in range(B):
            d = _fit_n(dense[b], N_EVAL, rng)
            g = geobin_skeleton(d, bin_size=bin_size) if with_graph else None
            res.append(dict(dense=d, graph=g, seg=None))
        return res
    if kind == "smarttree":
        out = method(inp)
        pt = {k: v.float().cpu().numpy() for k, v in out["point"].items()}
        x = inp[:, : out["n_in"]].float().cpu().numpy()
        from scipy.spatial import cKDTree
        for b in range(B):
            wood = pt["seg"][b] > 0
            if wood.sum() < 16:
                wood = np.argsort(-pt["seg"][b])[:256]
            med = (x[b] + pt["med"][b])[wood]
            rr = np.exp(pt["logr"][b])[wood]
            pos, rad, par = geobin_skeleton(med, bin_size=bin_size * 0.75)
            _, j = cKDTree(pos).query(med)
            r = np.array([np.median(rr[j == i]) if (j == i).any() else np.nan for i in range(len(pos))])
            r = np.where(np.isfinite(r), r, np.nanmedian(r) if np.isfinite(r).any() else 0.003)
            g = (pos, r, par)
            res.append(dict(dense=_tubes(g, rng), graph=g, seg=pt["seg"][b]))
        return res
    if kind == "diffts":
        pos, fl, r = method.sample(inp)
        pos, r = pos.float().cpu().numpy(), r.float().cpu().numpy()
        for b in range(B):
            g = mst_tree(pos[b], r[b])
            res.append(dict(dense=_tubes(g, rng), graph=g, seg=None))
        return res
    if kind == "circlefit":
        from .skeletonize import circle_fit_dbh
        x = inp.float().cpu().numpy()
        for b in range(B):
            sc = float(batch["scale"][b])
            gz = float(batch["ground_n"][b]) * sc if "ground_n" in batch else 0.0
            pm = x[b].astype(np.float64) * sc
            bh = batch["bh_pts"][b] if "bh_pts" in batch else None      # full-resolution breast-height band
            if bh is not None:
                bh = bh.cpu().numpy() if torch.is_tensor(bh) else np.asarray(bh)
            src = bh.astype(np.float64) * sc if bh is not None and len(bh) >= 8 else pm
            if "stem_n" in batch:           # known stem position: ignore neighbouring vegetation
                st = batch["stem_n"][b]
                st = (st.cpu().numpy() if torch.is_tensor(st) else np.asarray(st)).astype(np.float64) * sc
                src = src[np.hypot(src[:, 0] - st[0], src[:, 1] - st[1]) < 1.0]
            res.append(dict(dense=x[b], graph=None, seg=None,
                            extra=dict(dbh_m=circle_fit_dbh(src, gz), height_m=float(np.percentile(pm[:, 2], 99.9) - gz))))
        return res
    if kind == "classical":
        x = inp.float().cpu().numpy()
        for b in range(B):
            g = geobin_skeleton(x[b], bin_size=bin_size)
            res.append(dict(dense=_tubes(g, rng), graph=g, seg=None))
        return res
    raise ValueError(kind)
