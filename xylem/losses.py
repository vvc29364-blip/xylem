"""Training objectives.

  L = L_dense (tube surface vs GT surface, CD-L1)
    + sum_stages w_s * L_skel (nodes vs GT skeleton samples, CD-L1 + attributes)
    + L_field (per-input-point xylem field: wood/leaf, flow, geodesic, radius, medial offset)
    + lambda_bot * L_botanical  (label-free priors; also used for test-time adaptation)
    [+ L_kd for elastic sub-networks (in-place distillation from the largest net)]
"""
import math

import torch
import torch.nn.functional as F

from .models.topology import hard_parents
from .ops import chamfer, gather, knn_idx, nn_idx, normalize, one_sided

DEFAULT_W = dict(dense=20.0, skel=20.0, skel_stage=(0.5, 0.75, 1.0), radius=0.3, flow=0.3, geo=0.5, exist=0.1,
                 seg=0.3, p_flow=0.2, p_geo=0.3, p_rad=0.2, p_med=1.0, bot=0.1, kd=1.0,
                 bot_flow=1.0, bot_geo=1.0, bot_mono=1.0, bot_pipe=1.0, exist_tau=0.015, field_on_leaves=1.0, pipe_lo=0.9, pipe_hi=1.35)


def node_losses(node, batch, w):
    X = node["pos"]
    S = batch["sk"]
    cd, d_ab, _ = chamfer(X, S)
    j, _ = nn_idx(X, S)
    near = (d_ab < w["exist_tau"]).float()
    rt = gather(batch["sk_r"].unsqueeze(-1), j)[..., 0]
    ft = gather(batch["sk_flow"], j)
    gt = gather(batch["sk_geo"].unsqueeze(-1), j)[..., 0]
    wsum = near.sum().clamp_min(1.0)
    rv = near * torch.isfinite(rt).float()
    l_r = ((node["logr"] - torch.log(torch.nan_to_num(rt, nan=1.0).clamp_min(1e-5))).abs() * rv).sum() / rv.sum().clamp_min(1.0)
    l_f = ((1 - (node["flow"] * ft).sum(-1)) * near).sum() / wsum
    l_g = ((node["geo"] - gt).abs() * near).sum() / wsum
    l_e = F.binary_cross_entropy_with_logits(node["exist"], near)
    tot = w["skel"] * cd + w["radius"] * l_r + w["flow"] * l_f + w["geo"] * l_g + w["exist"] * l_e
    return tot, dict(skel_cd=cd.detach(), rad=l_r.detach(), flow=l_f.detach(), geo=l_g.detach(), exist=l_e.detach())


def field_losses(pt, batch, n, w):
    """Per-point xylem field.  Wood points regress their own branch axis; *leaves regress the axis,
    flow, geodesic and radius of the twig that carries them* (foliage-aware skeleton recovery);
    outliers only get the tree/outlier classification."""
    seg = batch["seg"][:, :n].float()
    l_seg = F.binary_cross_entropy_with_logits(pt["seg"], seg)
    valid = batch["p_valid"][:, :n].float() if "p_valid" in batch else torch.ones_like(seg)
    if "tree" in pt:
        l_seg = l_seg + 0.5 * F.binary_cross_entropy_with_logits(pt["tree"], valid)
    m = valid if w.get("field_on_leaves", 1.0) > 0 else seg
    ms = m.sum().clamp_min(1.0)
    l_f = ((1 - (pt["flow"] * batch["p_flow"][:, :n]).sum(-1)) * m).sum() / ms
    l_g = ((pt["geo"] - batch["p_geo"][:, :n]).abs() * m).sum() / ms
    pr = batch["p_rad"][:, :n]
    mr = m * torch.isfinite(pr).float()
    l_r = ((pt["logr"] - torch.log(torch.nan_to_num(pr, nan=1.0).clamp_min(1e-5))).abs() * mr).sum() / mr.sum().clamp_min(1.0)
    l_m = (F.smooth_l1_loss(pt["med"], batch["p_med"][:, :n], reduction="none", beta=0.002).sum(-1) * m).sum() / ms
    tot = w["seg"] * l_seg + w["p_flow"] * l_f + w["p_geo"] * l_g + w["p_rad"] * l_r + w["p_med"] * l_m * 100
    return tot, dict(seg=l_seg.detach(), p_flow=l_f.detach(), p_geo=l_g.detach(), p_rad=l_r.detach(), p_med=l_m.detach())


def botanical_losses(node, w, k=16):
    """Label-free botanical priors on the predicted nodes (hard topology, differentiable values).

    flow consistency : predicted flow points to the chosen parent
    geodesic consist.: g_i = g_parent + |X_i - X_parent|
    radius monotone  : r_i <= r_parent
    pipe model band  : at junctions  sum_children r_c^2 in [0.9, 1.35] r_parent^2
                       (Leonardo da Vinci's rule for pipe exponents 2..3); near-duplicate
                       siblings (same direction) are soft-merged so redundant nodes do not count twice.
    """
    X, Fl, g = node["pos"], node["flow"], node["geo"]
    r = node["logr"].exp()
    ex = (node["exist"].detach().sigmoid() > 0.5).float()
    with torch.no_grad():
        par, idx = hard_parents(X.detach(), Fl.detach(), g.detach(), k=k)
    has = (par >= 0).float() * ex
    p = par.clamp_min(0)
    Xp = gather(X, p)
    gp = gather(g.unsqueeze(-1), p)[..., 0]
    rp = gather(r.unsqueeze(-1), p)[..., 0]
    d = Xp - X
    dist = d.norm(dim=-1)
    hs = has.sum().clamp_min(1.0)
    l_flow = ((1 - (Fl * normalize(d)).sum(-1)) * has).sum() / hs
    l_geo = ((g - gp - dist).abs() * has).sum() / hs
    l_mono = (F.relu(r - rp) / (rp.detach() + 1e-4) * has).sum() / hs
    # ---- pipe model at junctions ----
    B, K = X.shape[:2]
    cdir = normalize(X - Xp)                                   # child direction from parent
    nb_par = gather(par.unsqueeze(-1).float(), idx)[..., 0].long()
    nb_dir = gather(cdir, idx)
    same = (nb_par == par.unsqueeze(-1)) & ((nb_dir * cdir.unsqueeze(2)).sum(-1) > 0.9)
    mult = 1.0 + same.float().sum(-1)                          # near-duplicate siblings share weight
    contrib = has * r ** 2 / mult
    S = torch.zeros(B, K, device=X.device, dtype=contrib.dtype).scatter_add(1, p, contrib)
    nch = torch.zeros(B, K, device=X.device, dtype=contrib.dtype).scatter_add(1, p, (has / mult).to(contrib.dtype))
    junc = (nch > 1.6).float()
    ratio = S / (r ** 2 + 1e-8)
    lo, hi = w.get("pipe_lo", 0.9), w.get("pipe_hi", 1.35)
    l_pipe = ((F.relu(lo - ratio) + F.relu(ratio - hi)) * junc).sum() / junc.sum().clamp_min(1.0)
    tot = w["bot_flow"] * l_flow + w["bot_geo"] * l_geo + w["bot_mono"] * l_mono + w["bot_pipe"] * l_pipe
    return tot, dict(b_flow=l_flow.detach(), b_geo=l_geo.detach(), b_mono=l_mono.detach(), b_pipe=l_pipe.detach(),
                     b_junc=junc.sum().detach() / B)


def xylem_loss(out, batch, w=None, bot=True):
    w = dict(DEFAULT_W, **(w or {}))
    logs = {}
    l_dense, _, _ = chamfer(out["dense"], batch["gt"])
    total = w["dense"] * l_dense
    logs["dense_cd"] = l_dense.detach()
    ws = w["skel_stage"]
    for s, node in enumerate(out["nodes"]):
        ls, lg = node_losses(node, batch, w)
        wt = ws[-len(out["nodes"]):][s] if len(out["nodes"]) <= len(ws) else 1.0
        total = total + wt * ls
        if s == len(out["nodes"]) - 1:
            logs.update(lg)
    if "point" in out:
        lf, lg = field_losses(out["point"], batch, out["n_in"], w)
        total = total + lf
        logs.update(lg)
    if bot and w["bot"] > 0:
        lb, lg = botanical_losses(out["nodes"][-1], w)
        total = total + w["bot"] * lb
        logs.update(lg)
    logs["total"] = total.detach()
    return total, logs


def kd_loss(student, teacher, w=None):
    """In-place distillation: sub-network mimics the (detached) largest network."""
    w = dict(DEFAULT_W, **(w or {}))
    l, _, _ = chamfer(student["dense"], teacher["dense"].detach())
    ns, nt = student["nodes"][-1], teacher["nodes"][-1]
    ln, _, _ = chamfer(ns["pos"], nt["pos"].detach())
    j, _ = nn_idx(ns["pos"], nt["pos"])
    g = lambda z: gather(z.detach().unsqueeze(-1) if z.dim() == 2 else z.detach(), j)
    la = (ns["logr"] - g(nt["logr"])[..., 0]).abs().mean() * 0.3 + (1 - (ns["flow"] * g(nt["flow"])).sum(-1)).mean() * 0.3 \
        + (ns["geo"] - g(nt["geo"])[..., 0]).abs().mean() * 0.5
    tot = l + ln + la
    if "point" in student and "point" in teacher:
        n = student["n_in"]
        ps, pt = student["point"], teacher["point"]
        tot = tot + 0.3 * F.binary_cross_entropy_with_logits(ps["seg"], pt["seg"][:, :n].detach().sigmoid()) \
            + 0.2 * (1 - (ps["flow"] * pt["flow"][:, :n].detach()).sum(-1)).mean() \
            + 0.3 * (ps["geo"] - pt["geo"][:, :n].detach()).abs().mean() \
            + 100 * (ps["med"] - pt["med"][:, :n].detach()).abs().mean()
    return w["kd"] * tot
