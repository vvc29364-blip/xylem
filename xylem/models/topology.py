"""Flow-guided arborescence decoding.

Given nodes with position X, flow F (unit, towards the root) and geodesic g, the parent of
node i is the neighbour j that (a) is strictly closer to the root (g_j < g_i), (b) lies in the
flow direction and (c) is geodesically consistent (g_i ~ g_j + |X_i - X_j|).  Because parents
always have a smaller geodesic value the result is acyclic, and with the global fallback every
node except the root gets a parent -> a single tree.  Fully vectorised (torch) so it can run
inside the training loop for the botanical losses.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

from ..ops import gather, knn_idx, normalize


def hard_parents(X, F, g, k=16, eps_g=1e-3, w_dir=2.0, w_geo=1.0):
    """X (B,K,3) F (B,K,3) g (B,K) -> parent (B,K) long (-1 if none among kNN)."""
    idx = knn_idx(X, X, k, exclude_self=True)            # (B,K,k)
    Xn = gather(X, idx)
    gn = gather(g.unsqueeze(-1), idx)[..., 0]
    d = Xn - X.unsqueeze(2)
    dist = d.norm(dim=-1)
    cos = (normalize(d) * F.unsqueeze(2)).sum(-1)
    valid = gn < (g.unsqueeze(-1) - eps_g)
    cost = dist * (1 + w_dir * (1 - cos)) + w_geo * (g.unsqueeze(-1) - gn - dist).abs()
    cost = cost.masked_fill(~valid, float("inf"))
    j = cost.argmin(-1)
    par = torch.gather(idx, 2, j.unsqueeze(-1))[..., 0]
    return torch.where(valid.any(-1), par, torch.full_like(par, -1)), idx


def nms_nodes(X, r, score, min_dist=0.003, rel=0.6):
    """Greedy radius-aware NMS (numpy).  Suppress nodes within max(min_dist, rel*r_i) of a kept node."""
    order = np.argsort(-score)
    kd = cKDTree(X)
    suppressed = np.zeros(len(X), bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        nb = kd.query_ball_point(X[i], max(min_dist, rel * r[i]))
        suppressed[nb] = True
    return np.asarray(sorted(keep))


def prune_spurs(pos, rad, par, min_len):
    """Iteratively remove terminal branches (tip -> first junction) shorter than min_len."""
    if min_len <= 0 or len(pos) < 3:
        return pos, rad, par
    par = par.copy()
    alive = np.ones(len(pos), bool)
    for _ in range(3):
        cnt = np.bincount(par[(par >= 0) & alive], minlength=len(pos))
        tips = np.nonzero(alive & (cnt == 0) & (par >= 0))[0]
        removed = False
        for t in tips:
            chain, L, n = [t], 0.0, t
            while par[n] >= 0 and cnt[par[n]] == 1 and par[par[n]] >= 0:
                L += np.linalg.norm(pos[n] - pos[par[n]])
                n = par[n]
                chain.append(n)
                if L >= min_len:
                    break
            if par[n] >= 0 and cnt[par[n]] >= 2:
                L += np.linalg.norm(pos[n] - pos[par[n]])
                if L < min_len:
                    alive[chain] = False
                    cnt[par[n]] -= 1
                    removed = True
        if not removed:
            break
    new = -np.ones(len(pos), np.int64)
    new[alive] = np.arange(alive.sum())
    p2 = par[alive]
    p2 = np.where(p2 >= 0, new[np.maximum(p2, 0)], -1)
    return pos[alive], rad[alive], p2


def smooth_chains(pos, par, iters=2, lam=0.5):
    """Laplacian smoothing of chain nodes (exactly one child); junctions, tips and root are fixed."""
    if iters <= 0 or len(pos) < 3:
        return pos
    cnt = np.bincount(par[par >= 0], minlength=len(pos))
    child = -np.ones(len(pos), np.int64)
    c = np.nonzero(par >= 0)[0]
    child[par[c]] = c
    m = (cnt == 1) & (par >= 0)
    idx = np.nonzero(m)[0]
    pos = pos.copy()
    for _ in range(iters):
        mid = 0.5 * (pos[par[idx]] + pos[child[idx]])
        pos[idx] = (1 - lam) * pos[idx] + lam * mid
    return pos


def extract_tree(X, F, g, r, exist=None, exist_thr=0.5, nms=True, k=16, nms_dist=0.003, spur=0.0, smooth=0):
    """numpy arrays for ONE sample -> (pos, radius, parent) arborescence rooted at the node with min geodesic."""
    X, F, g, r = [np.asarray(a, dtype=np.float64) for a in (X, F, g, r)]
    keep = np.arange(len(X))
    if exist is not None:
        keep = np.nonzero(np.asarray(exist) > exist_thr)[0]
        if len(keep) < 8:
            keep = np.argsort(-np.asarray(exist))[: max(8, min(64, len(X)))]
    X, F, g, r = X[keep], F[keep], g[keep], r[keep]
    if nms and len(X) > 8:
        sc = -g * 0 + (np.asarray(exist)[keep] if exist is not None else np.ones(len(X)))
        kk = nms_nodes(X, r, sc, min_dist=nms_dist)
        X, F, g, r = X[kk], F[kk], g[kk], r[kk]
    K = len(X)
    # root: smallest geodesic; tie-break by height
    root = int(np.argmin(g + 1e-3 * X[:, 2]))
    g = g.copy()
    g[root] = -1.0
    kq = min(k + 1, K)
    dd, ii = cKDTree(X).query(X, kq)
    ii = ii[:, 1:]
    d = X[ii] - X[:, None]
    dist = np.linalg.norm(d, axis=-1)
    cos = np.sum(d / (dist[..., None] + 1e-12) * F[:, None], -1)
    gn = g[ii]
    valid = gn < g[:, None] - 1e-6
    cost = dist * (1 + 2.0 * (1 - cos)) + np.abs(g[:, None] - gn - dist)
    cost[~valid] = np.inf
    j = np.argmin(cost, 1)
    parent = ii[np.arange(K), j]
    bad = ~valid.any(1)
    bad[root] = False
    for i in np.nonzero(bad)[0]:          # global fallback: nearest node with smaller geodesic
        cand = np.nonzero(g < g[i])[0]
        parent[i] = cand[np.argmin(np.linalg.norm(X[cand] - X[i], axis=1))]
    parent[root] = -1
    # re-order so that the root is node 0
    perm = np.concatenate([[root], np.delete(np.arange(K), root)])
    inv = np.empty(K, dtype=np.int64)
    inv[perm] = np.arange(K)
    par = parent[perm]
    par = np.where(par >= 0, inv[np.maximum(par, 0)], -1)
    P = smooth_chains(X[perm], par, smooth)
    return prune_spurs(P, r[perm], par, spur)


def mst_tree(X, r, k=10):
    """Topology baseline (used by DiffTS-style and ablation w/o flow/geodesic): Euclidean MST
    over a kNN graph rooted at the lowest node."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, connected_components
    X = np.asarray(X, np.float64)
    K = len(X)
    kq = min(k + 1, K)
    dd, ii = cKDTree(X).query(X, kq)
    rows = np.repeat(np.arange(K), kq - 1)
    A = coo_matrix((dd[:, 1:].ravel() + 1e-9, (rows, ii[:, 1:].ravel())), shape=(K, K)).tocsr()
    A = A.maximum(A.T)
    # connect components by their closest pairs
    nc, lab = connected_components(A, directed=False)
    if nc > 1:
        extra_r, extra_c, extra_w = [], [], []
        main = np.bincount(lab).argmax()
        mi = np.nonzero(lab == main)[0]
        kd = cKDTree(X[mi])
        for c in range(nc):
            if c == main:
                continue
            ci = np.nonzero(lab == c)[0]
            d, j = kd.query(X[ci])
            a = np.argmin(d)
            extra_r.append(ci[a]); extra_c.append(mi[j[a]]); extra_w.append(d[a] + 1e-9)
        E = coo_matrix((extra_w, (extra_r, extra_c)), shape=(K, K)).tocsr()
        A = A.maximum(E).maximum(E.T)
    T = minimum_spanning_tree(A)
    T = T.maximum(T.T)
    root = int(np.argmin(X[:, 2]))
    order, pred = breadth_first_order(T, root, directed=False, return_predecessors=True)
    parent = np.where(pred >= 0, pred, -1)
    parent[root] = -1
    reach = np.zeros(K, bool)
    reach[order] = True
    Xs, rs, ps = X[reach], np.asarray(r)[reach], parent[reach]
    new = -np.ones(K, dtype=np.int64)
    new[np.nonzero(reach)[0]] = np.arange(reach.sum())
    ps = np.where(ps >= 0, new[np.maximum(ps, 0)], -1)
    ro = int(new[root])
    perm = np.concatenate([[ro], np.delete(np.arange(len(Xs)), ro)])
    inv = np.empty(len(Xs), dtype=np.int64)
    inv[perm] = np.arange(len(Xs))
    ps = ps[perm]
    ps = np.where(ps >= 0, inv[np.maximum(ps, 0)], -1)
    return Xs[perm], rs[perm], ps


def _circle_ransac(p2, r_lo, r_hi, thr, iters=60, seed=0):
    """RANSAC + algebraic circle fit in 2-D.  Returns (center, r, inlier mask) or None."""
    rng = np.random.default_rng(seed)

    def kasa(q):
        A = np.c_[2 * q, np.ones(len(q))]
        c, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
        return c[:2], np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0.0))
    best = None
    for _ in range(iters):
        idx = rng.choice(len(p2), 3, replace=False)
        try:
            c, r = kasa(p2[idx])
        except np.linalg.LinAlgError:
            continue
        if not (r_lo < r < r_hi):
            continue
        inl = np.abs(np.linalg.norm(p2 - c, axis=1) - r) < max(thr, 0.08 * r)
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 8:
        return None
    c, r = kasa(p2[best])
    if not (r_lo < r < r_hi):
        return None
    inl = np.abs(np.linalg.norm(p2 - c, axis=1) - r) < max(thr, 0.08 * r)
    return c, r, inl


def refine_radii(pos, rad, par, obs, scale=None, min_r_m=0.025, n0=15.0):
    """Observation-refined radii (QSM-style): for every *thick* decoded edge, observed points in a
    cylinder around the predicted axis (window independent of the predicted radius) are projected
    on the cross-section plane and a RANSAC circle is fitted.  Accepted only if reliable (>=8 inliers,
    inlier ratio >= 0.5, >= 90 deg arc coverage); then r = w r_fit + (1-w) r_pred, w = n/(n+n0).
    Occluded / thin parts keep the learned radius."""
    obs = np.asarray(obs, np.float64)
    if len(obs) < 8 or len(pos) < 2:
        return rad
    rad = np.asarray(rad, np.float64).copy()
    s = float(scale) if scale else 15.0
    kd = cKDTree(obs)
    c = np.nonzero((par >= 0) & (rad * s >= min_r_m))[0]
    for i in c:
        a, b = pos[par[i]], pos[i]
        ax = b - a
        L = float(np.linalg.norm(ax))
        if L < 1e-9:
            continue
        u = ax / L
        Rs = max(3.0 * rad[i], 0.35 / s)
        idx = kd.query_ball_point(0.5 * (a + b), 0.5 * L + Rs + 0.1 / s)
        if len(idx) < 8:
            continue
        v = obs[idx] - a
        t = v @ u
        keep = (t > -0.5 * L) & (t < 1.5 * L)
        w_ = v[keep] - t[keep, None] * u
        w_ = w_[np.linalg.norm(w_, axis=1) < Rs]
        if len(w_) < 8:
            continue
        helper = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(u, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        p2 = np.stack([w_ @ e1, w_ @ e2], 1)
        fit = _circle_ransac(p2, 0.3 * rad[i], min(4.0 * rad[i] + 0.05 / s, 0.8 / s), 0.01 / s, seed=int(i))
        if fit is None:
            continue
        cc, r_fit, inl = fit
        if inl.mean() < 0.5:
            continue
        ang = np.sort(np.arctan2(p2[inl, 1] - cc[1], p2[inl, 0] - cc[0]))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        if 2 * np.pi - gaps.max() < np.pi / 2:
            continue
        n = int(inl.sum())
        w = n / (n + n0)
        rad[i] = w * r_fit + (1 - w) * rad[i]
    return rad
