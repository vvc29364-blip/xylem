"""Virtual laser scanning of a tree scene (wood surfaces + leaves) with physically motivated
occlusion.

Each scene point carries attributes (wood/leaf label, radius, flow, geodesic, medial axis point).
A scanner is a spherical range image: for every angular pixel only the closest surface sample
survives (z-buffer), which reproduces self-occlusion by the trunk, branches and foliage and the
loss of thin, distant twigs whose cross-section is below the beam footprint.

Sensors
  tls1  single terrestrial scan (heavy occlusion, back side missing)
  tls3  three terrestrial scans around the tree (moderate occlusion)
  uls   UAV/airborne scan from above (canopy dominant, trunk sparse)
"""
import numpy as np

from .skeleton import sample_surface, children_lists

SENSORS = ("tls1", "tls3", "uls")


# --------------------------------------------------------------------------------------
# leaves
# --------------------------------------------------------------------------------------
def make_leaves(pos, radius, parent, leaf_size, leaf_density, kind, rng, max_leaves=40000):
    """Place leaves around thin terminal twigs.  Returns (centers, normals, sizes, node_idx)."""
    ch = children_lists(parent)
    thin = radius < max(np.quantile(radius, 0.35), 0.006)
    nodes = np.nonzero(thin & (parent >= 0))[0]
    if len(nodes) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int64)
    depth_bonus = np.array([1.5 if len(ch[n]) == 0 else 1.0 for n in nodes])
    cnt = rng.poisson(leaf_density * depth_bonus)
    idx = np.repeat(nodes, cnt)
    if len(idx) > max_leaves:
        idx = rng.choice(idx, max_leaves, replace=False)
    seg = pos[idx] - pos[parent[idx]]
    t = rng.random(len(idx))[:, None]
    base = pos[parent[idx]] + seg * t
    spread = leaf_size * (1.2 if kind == "broad" else 0.6)
    centers = base + rng.normal(0, spread, (len(idx), 3))
    normals = rng.normal(0, 1, (len(idx), 3))
    normals[:, 2] += 1.5 if kind == "broad" else 0.0      # broad leaves tend to face upwards
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    sizes = leaf_size * rng.uniform(0.7, 1.3, len(idx))
    return centers, normals, sizes, idx


def sample_leaves(centers, normals, sizes, n_per_leaf, rng, kind="broad"):
    L = len(centers)
    if L == 0:
        return np.zeros((0, 3))
    k = n_per_leaf
    helper = np.where(np.abs(normals[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    u = np.cross(normals, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    v = np.cross(normals, u)
    rr = np.sqrt(rng.random((L, k))) * sizes[:, None] * 0.5
    th = rng.random((L, k)) * 2 * np.pi
    elong = 1.0 if kind == "broad" else 0.25       # needles: thin ellipses
    p = (centers[:, None, :] + (rr * np.cos(th))[..., None] * u[:, None, :]
         + (rr * np.sin(th) * elong)[..., None] * v[:, None, :])
    return p.reshape(-1, 3)


# --------------------------------------------------------------------------------------
# scene & scanning
# --------------------------------------------------------------------------------------
def build_scene(tree, rng, leaf_on, spacing):
    """Dense surface samples of the tree.  spacing ~ desired mean distance between samples."""
    pos, rad, par = tree["pos"].astype(np.float64), tree["radius"].astype(np.float64), tree["parent"].astype(np.int64)
    a = pos[par[par >= 0]]
    b = pos[par >= 0]
    L = np.linalg.norm(b - a, axis=1)
    area = float(np.sum(np.pi * (rad[par >= 0] + rad[par[par >= 0]]) * L))
    n_wood = int(np.clip(area / spacing ** 2, 30000, 350000))
    wpts, wattr = sample_surface(pos, rad, par, n_wood, rng)
    scene = dict(pts=wpts, label=np.ones(n_wood, np.int8), radius=wattr["radius"], flow=wattr["flow"],
                 geo=wattr["geo"], axis=wattr["axis"])
    if leaf_on:
        c, nrm, sz, node = make_leaves(pos, rad, par, float(tree["leaf_size"]), float(tree["leaf_density"]),
                                       str(tree["leaf_kind"]), rng)
        n_per = int(np.clip(np.ceil((float(tree["leaf_size"]) / spacing) ** 2), 2, 10))
        lp = sample_leaves(c, nrm, sz, n_per, rng, str(tree["leaf_kind"]))
        if len(lp):
            node = np.repeat(node, n_per)
            g = np.zeros(len(pos))
            from .skeleton import geodesic
            g = geodesic(pos, par)
            fl = pos[par[node]] - pos[node]
            fl /= np.linalg.norm(fl, axis=1, keepdims=True) + 1e-12
            scene["pts"] = np.concatenate([scene["pts"], lp])
            scene["label"] = np.concatenate([scene["label"], np.zeros(len(lp), np.int8)])
            scene["radius"] = np.concatenate([scene["radius"], rad[node]])
            scene["flow"] = np.concatenate([scene["flow"], fl])
            scene["geo"] = np.concatenate([scene["geo"], g[node]])
            scene["axis"] = np.concatenate([scene["axis"], pos[node]])
    return scene


def zbuffer_scan(pts, origin, ang_res, rng, max_el=np.pi / 2, min_el=-np.pi / 2):
    """Return indices of the points hit by a spherical scanner at `origin`."""
    v = pts - origin
    rng_ = np.linalg.norm(v, axis=1) + 1e-9
    az = np.arctan2(v[:, 1], v[:, 0])
    el = np.arcsin(np.clip(v[:, 2] / rng_, -1, 1))
    ok = (el <= max_el) & (el >= min_el)
    # jitter the pixel grid so that every scan samples differently
    off = rng.random(2) * ang_res
    iu = np.floor((az + np.pi + off[0]) / ang_res).astype(np.int64)
    iv = np.floor((el + np.pi / 2 + off[1]) / ang_res).astype(np.int64)
    key = iu * 1000003 + iv
    idx = np.nonzero(ok)[0]
    # single float64 sort: integer pixel key + normalised range in [0,1)
    o = np.argsort(key[idx].astype(np.float64) + rng_[idx] / (rng_[idx].max() * 1.001))
    ks = key[idx][o]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    return idx[o][first]


def scanner_positions(sensor, H, rng):
    if sensor == "tls1":
        d = H * rng.uniform(0.7, 1.4)
        a = rng.uniform(0, 2 * np.pi)
        return [np.array([d * np.cos(a), d * np.sin(a), rng.uniform(1.2, 1.8)])]
    if sensor == "tls3":
        d = H * rng.uniform(0.7, 1.4)
        a0 = rng.uniform(0, 2 * np.pi)
        return [np.array([d * np.cos(a0 + k * 2 * np.pi / 3 + rng.normal(0, 0.2)),
                          d * np.sin(a0 + k * 2 * np.pi / 3 + rng.normal(0, 0.2)),
                          rng.uniform(1.2, 1.8)]) for k in range(3)]
    if sensor == "uls":
        alt = H + rng.uniform(25, 50)
        a0 = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(5, 25)
        return [np.array([off * np.cos(a0 + k * np.pi / 2), off * np.sin(a0 + k * np.pi / 2), alt]) for k in range(4)]
    raise ValueError(sensor)


def scene_from_points(tree, leaf_on):
    """Scene from an external dense point cloud (e.g. SpeedTree / real TLS) with labels.
    Attributes are transferred from the nearest skeleton sample."""
    from scipy.spatial import cKDTree
    from .skeleton import resample_skeleton
    pts = tree["scene_pts"].astype(np.float64)
    lab = tree["scene_label"].astype(np.int8)
    if not leaf_on:
        pts, lab = pts[lab == 1], lab[lab == 1]
    sk = resample_skeleton(tree["pos"].astype(np.float64), tree["radius"].astype(np.float64),
                           tree["parent"].astype(np.int64), step=float(tree["height"]) / 300)
    _, nn = cKDTree(sk["pts"]).query(pts)
    return dict(pts=pts, label=lab, radius=sk["radius"][nn], flow=sk["flow"][nn], geo=sk["geo"][nn],
                axis=sk["pts"][nn])


def scan_tree(tree, sensor, rng, leaf_on, n_out=4096, noise=None, outlier_frac=None, density=1.0):
    """Simulate a scan.  Returns dict with pts (n_out,3) in metres and per-point attributes.
    density < 1 thins the raw returns before resampling (sparser sensors)."""
    H = float(tree["height"])
    spacing = H / 500.0
    if "scene_pts" in tree:
        scene = scene_from_points(tree, leaf_on)
    else:
        scene = build_scene(tree, rng, leaf_on, spacing)
    origins = scanner_positions(sensor, H, rng)
    hits = []
    for o in origins:
        dist = np.linalg.norm(np.array([0, 0, H * 0.5]) - o)
        ang = (H / 260.0) / dist * rng.uniform(0.8, 1.25)   # footprint ~ H/260 at the tree
        if sensor == "uls":
            ang *= 1.5
        h = zbuffer_scan(scene["pts"], o, ang, rng,
                         min_el=-np.pi / 2 if sensor == "uls" else np.deg2rad(-45))
        hits.append((h, o))
    idx = np.concatenate([h for h, _ in hits])
    org = np.concatenate([np.repeat(o[None], len(h), 0) for h, o in hits])
    # beam dropout (low reflectance, wind-blown leaves) and optional density reduction
    keep = rng.random(len(idx)) > rng.uniform(0.02, 0.12)
    if density < 1.0:
        keep &= rng.random(len(idx)) < density
    idx, org = idx[keep], org[keep]
    pts = scene["pts"][idx].copy()
    ray = pts - org
    ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-12
    sigma = noise if noise is not None else rng.uniform(0.002, 0.012)       # metres
    pts += ray * rng.normal(0, sigma, (len(pts), 1)) + rng.normal(0, sigma * 0.3, pts.shape)
    label = scene["label"][idx].astype(np.int64)
    # mixed-pixel / edge outliers: pushed along the beam behind the hit
    of = outlier_frac if outlier_frac is not None else rng.uniform(0.0, 0.02)
    n_out_l = int(of * len(pts))
    if n_out_l:
        oi = rng.choice(len(pts), n_out_l, replace=False)
        pts[oi] += ray[oi] * rng.uniform(0.05, 0.6, (n_out_l, 1)) * H / 15
        label[oi] = 2
    # equalise density (voxel) then random subsample to n_out
    vs = H / 400.0
    key = np.floor(pts / vs).astype(np.int64)
    _, first = np.unique(key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791, return_index=True)
    rng.shuffle(first)
    if density < 1.0:   # sparse sensor: do not re-densify by voxel equalisation
        first = first[: max(64, int(len(first) * density))]
    sel = first
    if len(sel) >= n_out:
        sel = sel[:n_out]
    else:
        sel = np.concatenate([sel, rng.choice(first, n_out - len(sel))])
    out = dict(pts=pts[sel], label=label[sel])
    for k in ("radius", "flow", "geo", "axis"):
        out[k] = scene[k][idx][sel]
    out["n_raw"] = len(idx)
    return out
