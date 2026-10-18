"""Real central-European forest trees (Weiser et al., ESSD 2022) -> evaluation / adaptation data.

Inputs per tree (whatever exists):  uls_off (UAV, leaf-off), uls_on (UAV, leaf-on), als (airborne),
tls1 (a *single-station* view simulated from the real multi-station TLS cloud with the z-buffer
scanner).  References: DBH (field or TLS) and tree height (field or ALS) from the dataset metrics.

Tree-level split by hash: odd -> evaluation (fixed sets data/weiser/fixed/<input>/),
even -> unlabelled adaptation pool for botanical self-supervised adaptation (no labels used).
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
from torch.utils.data import Dataset

from .dataset import normalisation, stable_seed
from .scanner import zbuffer_scan

INPUTS = {"uls_off": "ULS_off", "uls_on": "ULS_on", "als": "ALS_on", "tls1": "TLS_on"}


def references(d):
    """DBH: field inventory (FI) > TLS;  height: field > ALS > ULS leaf-off > ULS leaf-on."""
    import json
    m = json.loads(str(d["metrics"])) if "metrics" in d.files else {}

    def first(srcs, key, canopy=None):
        for sname in srcs:
            for rec in m.get(sname, []):
                if key in rec and (canopy is None or rec.get("canopy") == canopy):
                    return float(rec[key])
        return np.nan
    dbh = first(("FI", "field", "Field", "TLS"), "dbh_cm")
    h = first(("FI", "field", "Field", "ALS"), "height_m")
    if not np.isfinite(h):
        h = first(("ULS",), "height_m", "leaf-off")
    if not np.isfinite(h):
        h = first(("ULS",), "height_m")
    return dbh, h


def is_eval_tree(name):
    return stable_seed("weiser-split", name) % 2 == 1


def _root(data):
    from .external import ext_root
    return ext_root(data, "weiser")


def tls_single_view(pts, rng, return_origin=False):
    H = max(float(pts[:, 2].max()), 2.0)
    d = H * rng.uniform(0.6, 1.0)
    a = rng.uniform(0, 2 * np.pi)
    o = np.array([d * np.cos(a), d * np.sin(a), 1.5])
    ang = (H / 300.0) / d
    vis = pts[zbuffer_scan(pts, o, ang, rng, min_el=np.deg2rad(-45))]
    return (vis, o) if return_origin else vis


def make_input(pts, rng, n_in=4096):
    if len(pts) < 128:
        return None
    sel = rng.choice(len(pts), n_in, replace=len(pts) < n_in)
    return pts[sel]


def _build(args):
    f, out_dir = args
    d = np.load(f, allow_pickle=True)
    name = os.path.basename(f)
    ref_d, ref_h = references(d)
    if not (np.isfinite(ref_d) or np.isfinite(ref_h)):
        return 0
    n = 0
    for key, src in INPUTS.items():
        if src not in d.files:
            continue
        out = os.path.join(out_dir, key, name)
        if os.path.exists(out):
            n += 1
            continue
        rng = np.random.default_rng(stable_seed(name, key))
        pts = d[src].astype(np.float64)
        full = pts
        if key == "tls1":
            pts, origin = tls_single_view(pts, rng, return_origin=True)
            # a real single station samples the facing half of the stem at full resolution
            full = full[(full[:, :2] @ origin[:2]) > 0]
        x = make_input(pts, rng)
        if x is None:
            continue
        center, scale = normalisation(x)
        nz = lambda a: ((a - center) / scale).astype(np.float32)
        s = dict(inp=nz(x), center=center.astype(np.float32), scale=np.float32(scale),
                 ground_n=np.float32((0.0 - center[2]) / scale), dbh_ref_cm=np.float32(ref_d),
                 height_ref_m=np.float32(ref_h), species=np.array(str(d["species"])), sensor=np.array(key),
                 leaf_on=np.array(key in ("uls_on", "als", "tls1")), tree=np.array(name))
        # full-resolution breast-height band near the stem (for the classical circle-fit baseline only)
        bh = full[(np.abs(full[:, 2] - 1.3) < 0.5) & (np.hypot(full[:, 0], full[:, 1]) < 1.0)]
        if len(bh) > 4000:
            bh = bh[rng.choice(len(bh), 4000, replace=False)]
        s["bh_pts"] = nz(bh) if len(bh) else np.zeros((0, 3), np.float32)
        s["stem_n"] = (-center[:2] / scale).astype(np.float32)
        if "TLS_on" in d.files:
            t = d["TLS_on"]
            s["ref_pts"] = nz(t[rng.choice(len(t), min(8000, len(t)), replace=False)])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, **s)
        n += 1
    return n


def build_weiser_fixed(data, workers=8):
    root = _root(data)
    files = [f for f in sorted(glob.glob(os.path.join(root, "trees", "*.npz"))) if is_eval_tree(os.path.basename(f))]
    with Pool(workers) as p:
        n = sum(p.map(_build, [(f, os.path.join(root, "fixed")) for f in files], chunksize=4))
    print(f"Weiser fixed samples: {n} from {len(files)} evaluation trees")


class WeiserTargetDataset(Dataset):
    """Unlabelled real forest scans (adaptation half) for botanical self-supervised adaptation."""

    def __init__(self, data, n_in=4096):
        root = _root(data)
        self.files = [f for f in sorted(glob.glob(os.path.join(root, "trees", "*.npz")))
                      if not is_eval_tree(os.path.basename(f))]
        self.n_in = n_in

    def __len__(self):
        return len(self.files) * 4

    def __getitem__(self, i):
        rng = np.random.default_rng()
        d = np.load(self.files[i % len(self.files)], allow_pickle=True)
        keys = [src for src in INPUTS.values() if src in d.files]
        src = keys[rng.integers(len(keys))]
        pts = d[src].astype(np.float64)
        if src == "TLS_on":
            pts = tls_single_view(pts, rng)
        if len(pts) < 128:
            pts = d[keys[0]].astype(np.float64)
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        x = pts[rng.choice(len(pts), self.n_in, replace=len(pts) < self.n_in)] @ R.T
        center, scale = normalisation(x)
        return dict(inp=((x - center) / scale).astype(np.float32), center=center.astype(np.float32),
                    scale=np.float32(scale))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    build_weiser_fixed(a.data, a.workers)
