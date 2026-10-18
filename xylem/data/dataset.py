"""PyTorch datasets.

A *sample* is a partial scan of one tree plus supervision:
    inp      (N,3)   normalised input points
    seg      (N,)    1 = wood, 0 = leaf / outlier
    p_flow   (N,3)   flow (unit vector towards the root along the branch) of the source surface
    p_geo    (N,)    geodesic distance to the root
    p_rad    (N,)    branch radius
    p_med    (N,3)   offset from the point to the medial axis
    gt       (M,3)   dense complete wood surface (area-uniform)
    sk, sk_r, sk_flow, sk_geo   uniform skeleton samples (S points) with attributes
    center, scale    normalisation (x_norm = (x - center) / scale); metres = scale * units
Fixed (val/test) samples additionally store the full GT graph g_pos, g_rad, g_par for metrics.
"""
import glob
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .scanner import scan_tree
from .skeleton import resample_skeleton, sample_surface


def stable_seed(*parts):
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:15], 16)


def load_tree(path):
    d = np.load(path, allow_pickle=True)
    t = {k: d[k] for k in d.files}
    for k in ("species", "leaf_kind"):
        if k in t:
            t[k] = str(t[k])
    if "height" not in t:
        t["height"] = np.float32(t["pos"][:, 2].max() - t["pos"][:, 2].min())
    if "leaf_size" not in t:
        t.update(leaf_size=np.float32(0.05), leaf_density=np.float32(5), leaf_kind="broad")
    if "points" in t:   # external dense clouds (Synthetic-Trees / BRANCH3D)
        t["scene_pts"] = t.pop("points")
        cls = t.pop("cls") if "cls" in t else np.zeros(len(t["scene_pts"]), np.int8)
        t["scene_label"] = (cls == 0).astype(np.int8) if "cls_is_leaf" not in t else 1 - cls
    return t


def rotate_tree(tree, theta, flip):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]], dtype=np.float64)
    if flip:
        R = R @ np.diag([1.0, -1.0, 1.0])
    t = dict(tree)
    t["pos"] = tree["pos"].astype(np.float64) @ R.T
    if "scene_pts" in tree:
        t["scene_pts"] = tree["scene_pts"].astype(np.float64) @ R.T
    return t


def normalisation(pts):
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    scale = float(max((hi - lo).max(), 1e-3))
    return center, scale


def make_sample(tree, sensor, leaf_on, rng, n_in=4096, n_gt=16384, n_skel=2048,
                noise=None, outlier_frac=None, density=1.0, with_graph=False):
    scan = scan_tree(tree, sensor, rng, leaf_on, n_out=n_in, noise=noise, outlier_frac=outlier_frac,
                     density=density)
    pts = scan["pts"]
    center, scale = normalisation(pts)
    nz = lambda x: ((x - center) / scale).astype(np.float32)
    pos = tree["pos"].astype(np.float64)
    rad = tree["radius"].astype(np.float64)
    par = tree["parent"].astype(np.int64)
    if "scene_pts" in tree:   # external: GT surface = the dense wood points of the source
        wp = tree["scene_pts"][tree["scene_label"] == 1]
        gt = wp[rng.choice(len(wp), n_gt, replace=len(wp) < n_gt)]
    else:
        gt = sample_surface(pos, rad, par, n_gt, rng, return_attr=False)
    sk = resample_skeleton(pos, rad, par, step=scale * 0.006)
    si = rng.choice(len(sk["pts"]), n_skel, replace=len(sk["pts"]) < n_skel)
    s = dict(
        inp=nz(pts),
        seg=(scan["label"] == 1).astype(np.int64),
        p_valid=(scan["label"] != 2).astype(np.float32),
        p_flow=scan["flow"].astype(np.float32),
        p_geo=(scan["geo"] / scale).astype(np.float32),
        p_rad=(scan["radius"] / scale).astype(np.float32),
        p_med=((scan["axis"] - pts) / scale).astype(np.float32),
        gt=nz(gt),
        sk=nz(sk["pts"][si]),
        sk_r=(sk["radius"][si] / scale).astype(np.float32),
        sk_flow=sk["flow"][si].astype(np.float32),
        sk_geo=(sk["geo"][si] / scale).astype(np.float32),
        center=center.astype(np.float32),
        scale=np.float32(scale),
    )
    if with_graph:
        s.update(g_pos=nz(pos), g_rad=(rad / scale).astype(np.float32), g_par=par.astype(np.int32))
    return s


class OnlineTreeDataset(Dataset):
    """Training set: every access simulates a *new* scan (random sensor, foliage, noise, pose)."""

    def __init__(self, tree_files, sensors=("tls1", "tls3", "uls"), leaf_prob=0.5, n_in=4096, n_gt=16384,
                 n_skel=2048, repeat=1, seed=0):
        self.files = list(tree_files)
        self.sensors = sensors
        self.leaf_prob = leaf_prob
        self.kw = dict(n_in=n_in, n_gt=n_gt, n_skel=n_skel)
        self.repeat = repeat
        self.seed = seed
        self._cache = {}

    def __len__(self):
        return len(self.files) * self.repeat

    def _tree(self, i):
        if i not in self._cache:
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[i] = load_tree(self.files[i])
        return self._cache[i]

    def __getitem__(self, idx):
        i = idx % len(self.files)
        rng = np.random.default_rng([self.seed, idx, torch.initial_seed() % (2 ** 31)])
        tree = rotate_tree(self._tree(i), rng.uniform(0, 2 * np.pi), rng.random() < 0.5)
        sensor = self.sensors[rng.integers(len(self.sensors))]
        for _ in range(3):
            try:
                return make_sample(tree, sensor, rng.random() < self.leaf_prob, rng, **self.kw)
            except Exception:
                continue
        return make_sample(tree, sensor, False, rng, **self.kw)


def build_fixed_sample(args):
    """Worker: deterministic sample for (tree file, condition) -> cached npz."""
    path, out, cond = args
    if os.path.exists(out):
        return out
    tree = load_tree(path)
    rng = np.random.default_rng(stable_seed(os.path.basename(path), sorted(cond.items())))
    tree = rotate_tree(tree, rng.uniform(0, 2 * np.pi), False)
    c = dict(cond)
    sensor = c.pop("sensor")
    leaf_on = c.pop("leaf_on")
    s = make_sample(tree, sensor, leaf_on, rng, with_graph=True, **c)
    s["species"] = np.array(tree.get("species", "unknown"))
    s["sensor"] = np.array(sensor)
    s["leaf_on"] = np.array(leaf_on)
    s["tree"] = np.array(os.path.basename(path))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **s)
    os.replace(tmp, out)
    return out


class FixedDataset(Dataset):
    """Pre-simulated (cached) evaluation samples."""

    def __init__(self, folder, limit=None):
        folders = folder if isinstance(folder, (list, tuple)) else [folder]
        self.files = []
        for fo in folders:
            fs = sorted(glob.glob(os.path.join(fo, "*.npz")))
            if not fs:
                fs = sorted(glob.glob(os.path.join(fo, "**", "*.npz"), recursive=True))
            self.files += [f for f in fs if not f.endswith(".tmp.npz")]
        if limit and len(self.files) > limit:
            # deterministic, spread over sub-folders
            idx = np.linspace(0, len(self.files) - 1, limit).round().astype(int)
            self.files = [self.files[i] for i in idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        s = {k: d[k] for k in d.files}
        s["name"] = os.path.basename(self.files[i])
        return s


def collate(batch):
    out = {}
    keys = list(dict.fromkeys(k for b in batch for k in b))
    for k in keys:
        vals = [b.get(k) for b in batch]
        if any(v is None for v in vals):      # optional per-sample field (e.g. ref_pts) -> keep as list
            out[k] = vals
            continue
        if isinstance(vals[0], np.ndarray) and vals[0].dtype.kind in "fiub" and all(v.shape == vals[0].shape for v in vals):
            out[k] = torch.from_numpy(np.stack(vals))
        elif isinstance(vals[0], (np.floating, float)):
            out[k] = torch.tensor(np.array(vals, dtype=np.float32))
        else:
            out[k] = vals
    return out
