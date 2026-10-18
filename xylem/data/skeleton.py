"""Skeleton-graph utilities shared by the generator, datasets, metrics and models.

A tree skeleton is stored as
    pos    (K,3) float   node positions, node 0 is the root (trunk base)
    radius (K,)  float   branch radius at the node
    parent (K,)  int     parent index, -1 for the root
Edges go child -> parent.  The "flow" of a node is the unit vector pointing to its
parent (i.e. towards the root, the direction sap flows in the xylem in reverse).
"""
import numpy as np


def children_lists(parent):
    ch = [[] for _ in range(len(parent))]
    for i, p in enumerate(parent):
        if p >= 0:
            ch[p].append(i)
    return ch


def topo_order(parent):
    """Root-first ordering (BFS).  Works for any valid arborescence."""
    ch = children_lists(parent)
    roots = [i for i, p in enumerate(parent) if p < 0]
    order, q = [], list(roots)
    while q:
        n = q.pop()
        order.append(n)
        q.extend(ch[n])
    return np.asarray(order, dtype=np.int64)


def geodesic(pos, parent, order=None):
    """Path length from each node to its root (C-speed Dijkstra on the tree graph)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    K = len(pos)
    c = np.nonzero(parent >= 0)[0]
    if len(c) == 0:
        return np.zeros(K)
    w = np.linalg.norm(pos[c] - pos[parent[c]], axis=1) + 1e-12
    A = coo_matrix((w, (parent[c], c)), shape=(K, K)).tocsr()
    roots = np.nonzero(parent < 0)[0]
    d = dijkstra(A, directed=True, indices=roots)
    return np.nan_to_num(np.min(np.atleast_2d(d), axis=0), posinf=0.0)


def flow_dirs(pos, parent):
    f = np.zeros_like(pos)
    has = parent >= 0
    d = pos[parent[has]] - pos[has]
    f[has] = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    f[~has] = np.array([0, 0, -1.0])
    return f


def branch_order(parent, pos=None):
    """Gravelius-like order: trunk = 0; a child that is not the 'main' continuation gets +1.
    Main continuation = child with the largest subtree size."""
    ch = children_lists(parent)
    order = topo_order(parent)
    size = np.ones(len(parent), dtype=np.int64)
    for n in order[::-1]:
        p = parent[n]
        if p >= 0:
            size[p] += size[n]
    bo = np.zeros(len(parent), dtype=np.int64)
    for n in order:
        if not ch[n]:
            continue
        main = max(ch[n], key=lambda c: size[c])
        for c in ch[n]:
            bo[c] = bo[n] + (0 if c == main else 1)
    return bo


def pipe_radii(parent, r_tip, exponent, order=None):
    """Leonardo / pipe-model radii: r_p^e = sum_c r_c^e, tips get r_tip."""
    if order is None:
        order = topo_order(parent)
    ch = children_lists(parent)
    r = np.zeros(len(parent), dtype=np.float64)
    for n in order[::-1]:
        if not ch[n]:
            r[n] = r_tip[n] if np.ndim(r_tip) else r_tip
        else:
            r[n] = np.sum(r[ch[n]] ** exponent) ** (1.0 / exponent)
    return r


def edge_arrays(pos, radius, parent):
    """Return (a, b, ra, rb, child_idx) for every edge parent(a) -> child(b)."""
    idx = np.nonzero(parent >= 0)[0]
    return pos[parent[idx]], pos[idx], radius[parent[idx]], radius[idx], idx


def resample_skeleton(pos, radius, parent, step):
    """Uniformly resample every edge with spacing <= step.  Returns samples with attributes.

    Output dict: pts (S,3), radius (S,), flow (S,3), geo (S,), edge (S,) child index of the edge,
    is_node (S,) bool.
    """
    g = geodesic(pos, parent)
    a, b, ra, rb, ci = edge_arrays(pos, radius, parent)
    L = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(L / step).astype(np.int64))
    tot = int(n.sum())
    e = np.repeat(np.arange(len(L)), n)
    # t in (0,1]; t=1 -> child node, so every node except root is emitted exactly once
    start = np.cumsum(n) - n
    k = np.arange(tot) - np.repeat(start, n) + 1
    t = k / np.repeat(n, n)
    pts = a[e] + (b[e] - a[e]) * t[:, None]
    rad = ra[e] + (rb[e] - ra[e]) * t
    fl = (a[e] - b[e]) / (L[e][:, None] + 1e-12)
    geo = g[parent[ci[e]]] + L[e] * t
    is_node = k == np.repeat(n, n)
    root = np.nonzero(parent < 0)[0]
    pts = np.concatenate([pos[root], pts])
    rad = np.concatenate([radius[root], rad])
    fl = np.concatenate([np.tile([[0, 0, -1.0]], (len(root), 1)), fl])
    geo = np.concatenate([np.zeros(len(root)), geo])
    edge = np.concatenate([-np.ones(len(root), dtype=np.int64), ci[e]])
    is_node = np.concatenate([np.ones(len(root), bool), is_node])
    return dict(pts=pts, radius=rad, flow=fl, geo=geo, edge=edge, is_node=is_node)


def _orthonormal(axis):
    """Two unit vectors orthogonal to each axis row (vectorised, degeneracy-free)."""
    helper = np.where(np.abs(axis[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    v = np.cross(axis, u)
    return u, v


def sample_surface(pos, radius, parent, n, rng, return_attr=True, min_radius=0.0):
    """Area-weighted sampling of the generalized-cylinder (frustum) surface of the skeleton.

    Returns pts (n,3) and, optionally, attribute dict with per-point radius, flow, geo,
    axis (closest medial-axis point) and edge (child index).
    """
    a, b, ra, rb, ci = edge_arrays(pos, radius, parent)
    ra = np.maximum(ra, min_radius)
    rb = np.maximum(rb, min_radius)
    axis = b - a
    L = np.linalg.norm(axis, axis=1)
    keep = L > 1e-9
    a, b, ra, rb, ci, axis, L = a[keep], b[keep], ra[keep], rb[keep], ci[keep], axis[keep], L[keep]
    axis = axis / L[:, None]
    area = np.pi * (ra + rb) * np.sqrt(L ** 2 + (ra - rb) ** 2)
    prob = area / area.sum()
    e = rng.choice(len(L), size=n, p=prob)
    t = rng.random(n)
    # area-correct t for cones: density along t ~ r(t); use inverse CDF of linear density
    r0, r1 = ra[e], rb[e]
    lin = np.abs(r1 - r0) > 1e-9
    u = t.copy()
    tt = np.where(lin, (np.sqrt(np.maximum(r0 ** 2 + (r1 ** 2 - r0 ** 2) * u, 0)) - r0) / np.where(lin, r1 - r0, 1), u)
    t = np.clip(tt, 0, 1)
    r = r0 + (r1 - r0) * t
    uu, vv = _orthonormal(axis[e])
    th = rng.random(n) * 2 * np.pi
    c = a[e] + axis[e] * (L[e] * t)[:, None]
    normal = uu * np.cos(th)[:, None] + vv * np.sin(th)[:, None]
    pts = c + normal * r[:, None]
    if not return_attr:
        return pts
    g = geodesic(pos, parent)
    attr = dict(
        radius=r,
        flow=-axis[e],
        geo=g[parent[ci[e]]] + L[e] * t,
        axis=c,
        normal=normal,
        edge=ci[e],
    )
    return pts, attr


def tree_height(pos):
    return float(pos[:, 2].max() - pos[:, 2].min())


def wood_volume(pos, radius, parent):
    a, b, ra, rb, _ = edge_arrays(pos, radius, parent)
    L = np.linalg.norm(b - a, axis=1)
    return float(np.sum(np.pi * L * (ra ** 2 + ra * rb + rb ** 2) / 3.0))


def total_length(pos, parent):
    a, b, _, _, _ = edge_arrays(pos, np.zeros(len(pos)), parent)
    return float(np.linalg.norm(b - a, axis=1).sum())


def dbh(pos, radius, parent, height=1.3):
    """Diameter at breast height: radius of the trunk path interpolated at z=height
    (z measured from the root node).  Trunk path = from root, always follow the child with
    the largest radius."""
    ch = children_lists(parent)
    root = int(np.nonzero(parent < 0)[0][0])
    z0 = pos[root, 2]
    n = root
    while ch[n]:
        c = max(ch[n], key=lambda k: radius[k])
        za, zb = pos[n, 2] - z0, pos[c, 2] - z0
        if za <= height <= zb and zb > za:
            t = (height - za) / (zb - za)
            return 2.0 * float(radius[n] + t * (radius[c] - radius[n]))
        n = c
    return 2.0 * float(radius[root])


def count_branches(parent):
    """Number of branch segments = number of tips + number of junction-to-junction segments.
    We report the number of tips (terminal nodes) which is robust to over-segmentation."""
    has_child = np.zeros(len(parent), bool)
    has_child[parent[parent >= 0]] = True
    return int((~has_child).sum())


def junctions_and_tips(parent):
    cnt = np.bincount(parent[parent >= 0], minlength=len(parent))
    return np.nonzero(cnt >= 2)[0], np.nonzero(cnt == 0)[0]


def simplify(pos, radius, parent, min_seg):
    """Merge chains of nodes closer than min_seg (keeps junctions/tips).  Returns new arrays."""
    ch = children_lists(parent)
    order = topo_order(parent)
    keep = np.zeros(len(parent), bool)
    last_kept = -np.ones(len(parent), dtype=np.int64)
    for n in order:
        p = parent[n]
        if p < 0 or len(ch[n]) != 1:
            keep[n] = True
        else:
            anc = last_kept[p] if not keep[p] else p
            keep[n] = np.linalg.norm(pos[n] - pos[anc]) >= min_seg
        last_kept[n] = n if keep[n] else (last_kept[p] if p >= 0 and not keep[p] else p)
    new_id = -np.ones(len(parent), dtype=np.int64)
    new_id[keep] = np.arange(keep.sum())
    new_parent = -np.ones(keep.sum(), dtype=np.int64)
    for n in np.nonzero(keep)[0]:
        p = parent[n]
        while p >= 0 and not keep[p]:
            p = parent[p]
        new_parent[new_id[n]] = new_id[p] if p >= 0 else -1
    return pos[keep], radius[keep], new_parent
