"""Evaluation metrics (all computed per sample, in normalised units unless stated otherwise).

Dense geometry  : CD-L1 (x1e3), CD-L2 (x1e4), F-Score@1% / @0.5%, DCD
Thin structures : recall of GT surface points by *metric* branch-radius bin
                  (twig <1 cm, 1-3 cm, 3-10 cm, >10 cm) -> "TBR" = recall on twigs <1 cm
Skeleton        : Skel-CD, Skel-P/R/F@1% (and F@2%), radius MARE (relative), radius MAE [mm]
Topology        : geodesic (root-path) error of matched samples, tip-F, junction-F
Forestry        : height error [m], DBH error [cm], wood volume error [%], branch length error [%]
"""
import numpy as np
from scipy.spatial import cKDTree

from .data.skeleton import (dbh, geodesic, junctions_and_tips, resample_skeleton, topo_order, total_length,
                            wood_volume)

TAU = 0.01          # F-score threshold (1% of the normalised tree size)
TAU_HALF = 0.005
TAU_SK = 0.01       # skeleton matching threshold (1% of tree size); F@2% also reported
RADIUS_BINS_M = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 10.0)]


def _nn(a, b):
    d, j = cKDTree(b).query(a, workers=1)
    return d, j


ABS_TAUS_CM = (0.5, 1, 2, 3, 5, 10)      # absolute (metric) thresholds for the threshold curves


def dense_metrics(pred, gt, gt_radius_m=None, scale=None):
    d_pg, _ = _nn(pred, gt)
    d_gp, _ = _nn(gt, pred)
    out = {}
    out["cd_l1"] = 1e3 * 0.5 * (d_pg.mean() + d_gp.mean())
    out["cd_l2"] = 1e4 * 0.5 * ((d_pg ** 2).mean() + (d_gp ** 2).mean())
    for tau, name in ((TAU, "f1"), (TAU_HALF, "f05")):
        p = (d_pg < tau).mean()
        r = (d_gp < tau).mean()
        out[name] = 2 * p * r / (p + r + 1e-12)
        if name == "f1":
            out["prec"], out["rec"] = p, r
    # density-aware CD (alpha=1000 on squared distances, Wu et al. 2021)
    _, j_pg = _nn(pred, gt)
    _, j_gp = _nn(gt, pred)
    cg = np.bincount(j_pg, minlength=len(gt))
    cp = np.bincount(j_gp, minlength=len(pred))
    out["dcd"] = 0.5 * ((1 - np.exp(-1000 * d_pg ** 2) / np.maximum(cg[j_pg], 1)).mean()
                        + (1 - np.exp(-1000 * d_gp ** 2) / np.maximum(cp[j_gp], 1)).mean())
    if gt_radius_m is not None:
        for (lo, hi) in RADIUS_BINS_M:
            m = (gt_radius_m >= lo) & (gt_radius_m < hi)
            out[f"rec_r{lo:g}-{hi:g}"] = (d_gp[m] < TAU).mean() if m.any() else np.nan
        out["tbr"] = out["rec_r0-0.01"]
    if scale is not None:
        # metric-scale thresholds (independent of the tree size): F@t cm and twig recall @2 cm
        for t in ABS_TAUS_CM:
            tau = t / 100.0 / scale
            p, r = (d_pg < tau).mean(), (d_gp < tau).mean()
            out[f"f_{t:g}cm"] = 2 * p * r / (p + r + 1e-12)
        if gt_radius_m is not None:
            m = gt_radius_m < 0.01
            out["tbr_2cm"] = (d_gp[m] < 0.02 / scale).mean() if m.any() else np.nan
    return out


def _depths(par):
    order = topo_order(par)
    dep = np.zeros(len(par), np.int64)
    for n in order:
        if par[n] >= 0:
            dep[n] = dep[par[n]] + 1
    return dep


def _path_len(par, geo, dep, a, b):
    ga, gb = geo[a], geo[b]
    while dep[a] > dep[b]:
        a = par[a]
    while dep[b] > dep[a]:
        b = par[b]
    while a != b:
        a, b = par[a], par[b]
    return ga + gb - 2 * geo[a]


def path_score(pp, ppar, gp, gpar, n_pairs=150, tau=0.02, seed=0):
    """APLS-style topology score (Van Etten et al.) adapted to trees: for random node pairs of the
    reference tree, compare the along-tree path length with the path length between the matched
    predicted nodes (missing match -> 0).  Symmetrised over GT->pred and pred->GT."""
    rng = np.random.default_rng(seed)

    def one(A, Apar, Bpos, Bpar):
        if len(A) < 3 or len(Bpos) < 3:
            return 0.0
        gA, gB = geodesic(A, Apar), geodesic(Bpos, Bpar)
        dA, dB = _depths(Apar), _depths(Bpar)
        d, j = _nn(A, Bpos)
        i1 = rng.integers(0, len(A), n_pairs)
        i2 = rng.integers(0, len(A), n_pairs)
        sc = []
        for a, b in zip(i1, i2):
            if a == b:
                continue
            La = _path_len(Apar, gA, dA, a, b)
            if La < 5 * tau:
                continue
            if d[a] > tau or d[b] > tau:
                sc.append(0.0)
                continue
            Lb = _path_len(Bpar, gB, dB, j[a], j[b])
            sc.append(1 - min(1.0, abs(Lb - La) / La))
        return float(np.mean(sc)) if sc else 0.0
    return 0.5 * (one(gp, gpar, pp, ppar) + one(pp, ppar, gp, gpar))


def skeleton_metrics(pp, pr, ppar, gp, gr, gpar, scale):
    """pp/pr/ppar: predicted graph; gp/gr/gpar: GT graph (normalised units)."""
    out = {}
    P = resample_skeleton(pp, pr, ppar, step=0.005)
    G = resample_skeleton(gp, gr, gpar, step=0.005)
    d_pg, j_pg = _nn(P["pts"], G["pts"])
    d_gp, j_gp = _nn(G["pts"], P["pts"])
    out["skel_cd"] = 1e3 * 0.5 * (d_pg.mean() + d_gp.mean())
    p = (d_pg < TAU_SK).mean()
    r = (d_gp < TAU_SK).mean()
    out["skel_p"], out["skel_r"] = p, r
    out["skel_f"] = 2 * p * r / (p + r + 1e-12)
    p2, r2 = (d_pg < 2 * TAU_SK).mean(), (d_gp < 2 * TAU_SK).mean()
    out["skel_f2"] = 2 * p2 * r2 / (p2 + r2 + 1e-12)
    m = d_gp < TAU_SK
    if m.any():
        rg = G["radius"][m]
        rp = P["radius"][j_gp[m]]
        out["rad_mare"] = float(np.mean(np.abs(rp - rg) / np.maximum(rg, 1e-5)))
        out["rad_mae_mm"] = float(np.mean(np.abs(rp - rg)) * scale * 1e3)
        gg = G["geo"][m]
        gpd = P["geo"][j_gp[m]]
        out["geo_err"] = float(np.mean(np.abs(gpd - gg) / np.maximum(gg, 0.05)))
    else:
        out["rad_mare"] = out["rad_mae_mm"] = out["geo_err"] = np.nan
    # tips & junctions
    for kind, ki in (("tip", 1), ("junc", 0)):
        a = junctions_and_tips(np.asarray(ppar))[ki]
        b = junctions_and_tips(np.asarray(gpar))[ki]
        if len(a) == 0 or len(b) == 0:
            out[f"{kind}_f"] = 0.0
            continue
        tau = 2 * TAU_SK
        da, _ = _nn(pp[a], gp[b])
        db, _ = _nn(gp[b], pp[a])
        pa, rb = (da < tau).mean(), (db < tau).mean()
        out[f"{kind}_f"] = 2 * pa * rb / (pa + rb + 1e-12)
    for t in (5, 10):   # metric-scale skeleton F-scores
        tau = t / 100.0 / scale
        p_, r_ = (d_pg < tau).mean(), (d_gp < tau).mean()
        out[f"skel_f_{t}cm"] = 2 * p_ * r_ / (p_ + r_ + 1e-12)
    out["path_score"] = path_score(pp, np.asarray(ppar), gp, np.asarray(gpar))
    return out


def forestry_metrics(pp, pr, ppar, gp, gr, gpar, scale):
    """Metric-scale structural attributes. z is measured from the GT root (ground)."""
    s = float(scale)
    Pp, Pr = pp * s, pr * s
    Gp, Gr = gp * s, gr * s
    z0 = Gp[0, 2]
    out = {}
    out["height_err_m"] = abs((Pp[:, 2].max() - z0) - (Gp[:, 2].max() - z0))
    Pp2 = Pp.copy()
    # DBH: shift the predicted root to ground level only in z-reference (use GT ground)
    out["dbh_err_cm"] = 100 * abs(_dbh_ground(Pp2, Pr, ppar, z0) - _dbh_ground(Gp, Gr, gpar, z0))
    vg = wood_volume(Gp, Gr, gpar)
    out["vol_err_pct"] = 100 * abs(wood_volume(Pp, Pr, ppar) - vg) / max(vg, 1e-9)
    lg = total_length(Gp, gpar)
    out["len_err_pct"] = 100 * abs(total_length(Pp, ppar) - lg) / max(lg, 1e-9)
    return out


def _dbh_ground(pos, rad, par, z0, h=1.3):
    """Radius of the lowest skeleton edge crossing z0+h (fallback: nearest node in height)."""
    a = pos[par[par >= 0]]
    b = pos[par >= 0]
    ra, rb = rad[par[par >= 0]], rad[par >= 0]
    zt = z0 + h
    lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
    m = (lo <= zt) & (hi >= zt) & (hi > lo)
    if m.any():
        t = (zt - a[m, 2]) / (b[m, 2] - a[m, 2])
        r = ra[m] + t * (rb[m] - ra[m])
        return 2 * float(r.max())
    i = int(np.argmin(np.abs(pos[:, 2] - zt)))
    return 2 * float(rad[i])


def forest_metrics(pred_dense, graph, sample, extra=None):
    """Real forest trees: DBH and height against field / TLS references (metric units)."""
    scale = float(sample["scale"])
    ground = float(sample["ground_n"]) * scale
    out = {}
    extra = extra or {}
    dbh = extra.get("dbh_m", np.nan)
    dbh_model = np.nan
    if not np.isfinite(dbh) and graph is not None and len(graph[0]) > 1:
        pos, rad, par = [np.asarray(a) for a in graph]
        Pm, Rm, par = pos.astype(np.float64) * scale, rad.astype(np.float64) * scale, par.astype(np.int64)
        dbh_model = _dbh_ground(Pm, Rm, par, ground)
        # stem-localised circle fit on the observed points (same post-processing for every method);
        # falls back to the reconstructed radius when the stem is not observed at breast height
        axis = _stem_point(Pm, Rm, par, ground + 1.3)
        obs = [np.asarray(sample["inp"], np.float64) * scale]
        if "bh_pts" in sample and sample["bh_pts"] is not None and len(sample["bh_pts"]):
            obs.append(np.asarray(sample["bh_pts"], np.float64) * scale)
        obs = np.concatenate(obs)
        obs = obs[np.hypot(obs[:, 0] - axis[0], obs[:, 1] - axis[1]) < 1.0]
        from .skeletonize import circle_fit_dbh
        fit = circle_fit_dbh(obs, ground) if len(obs) >= 8 else np.nan
        dbh = fit if np.isfinite(fit) else dbh_model
        out["dbh_fit_ok"] = float(np.isfinite(fit))
    out["dbh_model_cm"] = 100 * dbh_model if np.isfinite(dbh_model) else np.nan
    z = np.asarray(pred_dense, np.float64)[:, 2] * scale
    h = extra.get("height_m", np.percentile(z, 99.9) - ground if len(z) else np.nan)
    ref_d, ref_h = float(sample["dbh_ref_cm"]), float(sample["height_ref_m"])
    out["dbh_pred_cm"], out["dbh_ref_cm"] = 100 * dbh, ref_d
    out["dbh_err_cm"] = 100 * dbh - ref_d
    out["dbh_abs_cm"] = abs(out["dbh_err_cm"])
    out["height_pred_m"], out["height_ref_m"] = h, ref_h
    out["height_err_m"] = h - ref_h
    out["height_abs_m"] = abs(out["height_err_m"])
    return out


def _stem_point(P, R, par, z):
    """Point of the stem axis at height z: from the root follow the thickest child."""
    from .data.skeleton import children_lists
    ch = children_lists(par)
    root = int(np.nonzero(par < 0)[0][0])
    n = root
    while ch[n]:
        c = max(ch[n], key=lambda k: R[k])
        if (P[n, 2] - z) * (P[c, 2] - z) <= 0 and P[c, 2] != P[n, 2]:
            t = (z - P[n, 2]) / (P[c, 2] - P[n, 2])
            return P[n] + t * (P[c] - P[n])
        n = c
    return P[int(np.argmin(np.abs(P[:, 2] - z)))]


def forest_summary(df):
    s = {}
    if "dbh_model_cm" in df:        # radius-only DBH (no stem fit) of reconstruction methods
        df = df.copy()
        df["dbhmodel_pred_cm"] = df["dbh_model_cm"]
        df["dbhmodel_ref_cm"] = df["dbh_ref_cm"]
        df["dbhmodel_err_cm"] = df["dbh_model_cm"] - df["dbh_ref_cm"]
    for key, unit in (("dbh", "cm"), ("height", "m"), ("dbhmodel", "cm")):
        ok = np.isfinite(df.get(f"{key}_pred_{unit}", pd_nan(df))) & np.isfinite(df.get(f"{key}_ref_{unit}", pd_nan(df)))
        n_ref = int(np.isfinite(df.get(f"{key}_ref_{unit}", pd_nan(df))).sum())
        if ok.sum() < 2:
            continue
        e = df.loc[ok, f"{key}_err_{unit}"].to_numpy()
        ref = df.loc[ok, f"{key}_ref_{unit}"].to_numpy()
        pred = df.loc[ok, f"{key}_pred_{unit}"].to_numpy()
        s[f"{key}_rmse_{unit}"] = float(np.sqrt(np.mean(e ** 2)))
        s[f"{key}_bias_{unit}"] = float(np.mean(e))
        s[f"{key}_mae_{unit}"] = float(np.mean(np.abs(e)))
        s[f"{key}_rrmse_pct"] = float(100 * np.sqrt(np.mean(e ** 2)) / np.mean(ref))
        s[f"{key}_r2"] = float(1 - np.sum((pred - ref) ** 2) / max(np.sum((ref - ref.mean()) ** 2), 1e-9))
        s[f"{key}_coverage"] = float(ok.sum() / max(n_ref, 1))
        s[f"{key}_n"] = int(ok.sum())
    return s


def pd_nan(df):
    return np.full(len(df), np.nan)


def gt_point_radius(gt, gp, gr, gpar):
    s = resample_skeleton(gp, gr, gpar, step=0.004)
    _, j = _nn(gt, s["pts"])
    return s["radius"][j]


def seg_metrics(pred_logit, label):
    p = pred_logit > 0
    t = label > 0
    iou_w = (p & t).sum() / max((p | t).sum(), 1)
    iou_l = (~p & ~t).sum() / max((~p | ~t).sum(), 1)
    return dict(seg_miou=0.5 * (iou_w + iou_l), seg_acc=(p == t).mean())


def evaluate_sample(pred_dense, pred_graph, sample, seg_logit=None):
    gp, gr, gpar = sample["g_pos"].astype(np.float64), sample["g_rad"].astype(np.float64), sample["g_par"].astype(np.int64)
    scale = float(sample["scale"])
    gt = sample["gt"].astype(np.float64)
    has_r = bool((gr > 0).any())          # BRANCH3D has no reference radii
    r_m = gt_point_radius(gt, gp, gr, gpar) * scale if has_r else None
    out = dense_metrics(np.asarray(pred_dense, np.float64), gt, r_m, scale)
    if "inp" in sample:   # visibility: fraction of the GT surface covered by the partial input
        d_vis, _ = _nn(gt, np.asarray(sample["inp"], np.float64))
        out["vis"] = float((d_vis < TAU).mean())
    if pred_graph is not None:
        pp, pr, ppar = [np.asarray(x) for x in pred_graph]
        pp = pp.astype(np.float64)
        pr = pr.astype(np.float64)
        ppar = ppar.astype(np.int64)
        if len(pp) >= 2:
            sm = skeleton_metrics(pp, pr, ppar, gp, gr, gpar, scale)
            if not has_r:
                sm["rad_mare"] = sm["rad_mae_mm"] = np.nan
            out.update(sm)
            if has_r:
                out.update(forestry_metrics(pp, pr, ppar, gp, gr, gpar, scale))
    if seg_logit is not None:
        out.update(seg_metrics(np.asarray(seg_logit)[: len(sample["seg"])], np.asarray(sample["seg"])[: len(seg_logit)]))
    return out


# metrics where lower is better (for tables / arrows)
LOWER_BETTER = {"dbh_mae_cm", "height_mae_m", "dbh_rrmse_pct", "height_rrmse_pct", "dbh_rmse_cm", "dbh_bias_cm", "height_rmse_m", "height_bias_m", "cd_l1", "cd_l2", "dcd", "skel_cd", "rad_mare", "rad_mae_mm", "geo_err", "height_err_m",
                "dbh_err_cm", "vol_err_pct", "len_err_pct"}
