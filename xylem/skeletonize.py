"""Classical point-cloud -> skeleton graph + radii (geodesic level-set binning).

This is the standard QSM-style pipeline (Xu et al. TOG'07; Verroust & Lazarus'00; AdTree'19;
TreeQSM-like segment growing):
  kNN graph -> Dijkstra geodesic from the root (lowest points) -> bin points by geodesic
  distance -> connected components within each bin become skeleton nodes (centroids) ->
  parent = adjacent component in the previous bin -> radius = median distance of the
  component's points to the local axis.
Used (i) as the classical baseline on raw scans and (ii) to turn the dense outputs of
completion networks into skeletons so that every method is evaluated with the same metrics.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


def geobin_skeleton(pts, bin_size=0.02, k=10, max_edge=None, radius_mode="median"):
    pts = np.asarray(pts, np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    N = len(pts)
    if N < 10:
        return pts[:1], np.full(1, 0.005), np.array([-1])
    kd = cKDTree(pts)
    k = min(k + 1, N)
    dd, ii = kd.query(pts, k)
    if max_edge is None:
        max_edge = max(4 * np.median(dd[:, 1]), 1e-4)
    rows = np.repeat(np.arange(N), k - 1)
    cols = ii[:, 1:].ravel()
    w = dd[:, 1:].ravel()
    m = w < max_edge
    A = coo_matrix((w[m] + 1e-9, (rows[m], cols[m])), shape=(N, N)).tocsr()
    A = A.maximum(A.T)
    # bridge disconnected components: MST over the component graph (one pass, O(N k log N))
    nc, lab = connected_components(A, directed=False)
    root = int(np.argmin(pts[:, 2]))
    if nc > 1:
        A = A.maximum(_bridge_components(pts, lab, nc, kd))
    # root = lowest few centimetres
    low = np.nonzero(pts[:, 2] < pts[:, 2].min() + bin_size)[0]
    g = dijkstra(A, directed=False, indices=low, min_only=True)
    g[~np.isfinite(g)] = g[np.isfinite(g)].max() if np.isfinite(g).any() else 0
    b = np.floor(g / bin_size).astype(np.int64)
    # components per bin (edges only inside the same bin)
    Ac = A.tocoo()
    same = b[Ac.row] == b[Ac.col]
    Ab = coo_matrix((np.ones(same.sum()), (Ac.row[same], Ac.col[same])), shape=(N, N)).tocsr()
    nc2, comp = connected_components(Ab, directed=False)
    cnt = np.bincount(comp, minlength=nc2)
    cen = np.zeros((nc2, 3))
    np.add.at(cen, comp, pts)
    cen /= np.maximum(cnt, 1)[:, None]
    cbin = np.zeros(nc2, np.int64)
    cbin[comp] = b
    # parent: component in a lower bin connected by an edge (choose the most connected)
    diff = cbin[comp[Ac.row]] < cbin[comp[Ac.col]]
    ch_c, pa_c = comp[Ac.col[diff]], comp[Ac.row[diff]]
    parent = -np.ones(nc2, np.int64)
    if len(ch_c):
        key = ch_c * nc2 + pa_c
        u, c = np.unique(key, return_counts=True)
        o = np.lexsort((-c, u // nc2))
        u = u[o]
        first = np.ones(len(u), bool)
        first[1:] = (u[1:] // nc2) != (u[:-1] // nc2)
        parent[u[first] // nc2] = u[first] % nc2
    # radius: median distance to the component centroid, projected orthogonally to the parent edge
    rad = np.zeros(nc2)
    dirv = np.zeros((nc2, 3))
    hp = parent >= 0
    dirv[hp] = cen[hp] - cen[parent[hp]]
    dirv /= np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-12
    off = pts - cen[comp]
    along = np.sum(off * dirv[comp], 1, keepdims=True) * dirv[comp]
    dist = np.linalg.norm(off - along, axis=1)
    order = np.argsort(comp)
    splits = np.cumsum(cnt)[:-1]
    for c, ds in enumerate(np.split(dist[order], splits)):
        rad[c] = np.median(ds) if len(ds) else 0.0
    # cross-sections seen from one side under-estimate the radius -> use a mild correction
    rad = np.maximum(rad * (1.0 if radius_mode == "median" else 1.2), 1e-4)
    # roots: components without parent; keep the tree reachable from the main root
    root_c = comp[root]
    # attach orphan components to the nearest lower-bin component
    orphans = np.nonzero((parent < 0) & (np.arange(nc2) != root_c))[0]
    for o in orphans:
        cand = np.nonzero(cbin < cbin[o])[0]
        if len(cand) == 0:
            cand = np.array([root_c])
        parent[o] = cand[np.argmin(np.linalg.norm(cen[cand] - cen[o], axis=1))]
    parent[root_c] = -1
    # guard against cycles (should not happen as parents are in lower bins)
    perm = np.concatenate([[root_c], np.delete(np.arange(nc2), root_c)])
    inv = np.empty(nc2, np.int64)
    inv[perm] = np.arange(nc2)
    par = parent[perm]
    par = np.where(par >= 0, inv[np.maximum(par, 0)], -1)
    return cen[perm], rad[perm], par


def _bridge_components(pts, lab, nc, kd, k=32):
    """Edges that connect all components with (approximately) minimal total length."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    N = len(pts)
    kk = min(k, N)
    dd, ii = kd.query(pts, kk)
    src = np.repeat(np.arange(N), kk)
    dst = ii.ravel()
    w = dd.ravel()
    cross = lab[src] != lab[dst]
    src, dst, w = src[cross], dst[cross], w[cross]
    # components without any cross neighbour among the k-NN: nearest point of another component
    has = np.zeros(nc, bool)
    has[lab[src]] = True
    if (~has).any():
        centroids = np.stack([np.bincount(lab, weights=pts[:, d], minlength=nc) for d in range(3)], 1)
        centroids /= np.maximum(np.bincount(lab, minlength=nc), 1)[:, None]
        for c in np.nonzero(~has)[0]:
            members = np.nonzero(lab == c)[0]
            _, cand = kd.query(centroids[c], min(N, len(members) + 64))
            cand = np.atleast_1d(cand)
            cand = cand[lab[cand] != c]
            if len(cand) == 0:
                cand = np.nonzero(lab != c)[0]
            m = members[np.argmin(np.linalg.norm(pts[members][:, None] - pts[cand[:1]][None], axis=-1)[:, 0])]
            j = cand[np.argmin(np.linalg.norm(pts[cand] - pts[m], axis=1))]
            src = np.append(src, m); dst = np.append(dst, j); w = np.append(w, np.linalg.norm(pts[m] - pts[j]))
    # cheapest edge per component pair, then MST on the component graph
    a, b = lab[src], lab[dst]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * nc + hi
    o = np.lexsort((w, key))
    key, src, dst, w = key[o], src[o], dst[o], w[o]
    first = np.ones(len(key), bool)
    first[1:] = key[1:] != key[:-1]
    key, src, dst, w = key[first], src[first], dst[first], w[first]
    C = coo_matrix((w + 1e-9, (key // nc, key % nc)), shape=(nc, nc)).tocsr()
    T = minimum_spanning_tree(C).tocoo()
    pick = {(min(r, c), max(r, c)) for r, c in zip(T.row, T.col)}
    sel = np.array([(int(k // nc), int(k % nc)) in pick for k in key], bool)
    return coo_matrix((w[sel] + 1e-9, (src[sel], dst[sel])), shape=(len(pts), len(pts))).tocsr().maximum(
        coo_matrix((w[sel] + 1e-9, (dst[sel], src[sel])), shape=(len(pts), len(pts))).tocsr())


def radius_from_points(skel_pos, skel_par, pts):
    """Assign each point to its nearest skeleton edge and take the median point-axis distance."""
    from .data.skeleton import resample_skeleton
    s = resample_skeleton(skel_pos, np.zeros(len(skel_pos)), skel_par, step=0.004)
    kd = cKDTree(s["pts"])
    d, j = kd.query(pts)
    e = s["edge"][j]
    r = np.zeros(len(skel_pos))
    for node in np.unique(e[e >= 0]):
        r[node] = np.median(d[e == node])
    has = r > 0
    if has.any():
        r[~has] = np.median(r[has])
    return np.maximum(r, 1e-4)


def circle_fit_dbh(pts_m, ground_z, h=1.3, half=0.1, iters=200, thr=0.02, seed=0):
    """Standard forestry baseline: RANSAC + algebraic (Kasa) circle fit on the breast-height slice.
    Returns DBH [m] or NaN when the slice has too few points (typical for UAV / airborne scans)."""
    z = pts_m[:, 2] - ground_z
    for w in (half, 2 * half, 3 * half):
        sl = pts_m[np.abs(z - h) < w][:, :2]
        if len(sl) >= 8:
            break
    if len(sl) < 8:
        return np.nan

    def kasa(p):
        A = np.c_[2 * p, np.ones(len(p))]
        b = (p ** 2).sum(1)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        r2 = c[2] + c[0] ** 2 + c[1] ** 2
        return c[:2], np.sqrt(max(r2, 0))
    rng = np.random.default_rng(seed)
    best, best_in = None, -1
    for _ in range(iters):
        idx = rng.choice(len(sl), 3, replace=False)
        try:
            c, r = kasa(sl[idx])
        except np.linalg.LinAlgError:
            continue
        if not (0.02 < r < 0.8):
            continue
        inl = np.abs(np.linalg.norm(sl - c, axis=1) - r) < thr
        if inl.sum() > best_in:
            best, best_in = inl, inl.sum()
    if best is None or best_in < 8 or best_in < 0.4 * len(sl):
        return np.nan
    c, r = kasa(sl[best])
    if not (0.02 < r < 0.8):
        return np.nan
    # reliability: the inliers must cover at least 90 degrees of the stem circumference
    ang = np.sort(np.arctan2(sl[best, 1] - c[1], sl[best, 0] - c[0]))
    if 2 * np.pi - np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max() < np.pi / 2:
        return np.nan
    return 2 * r
