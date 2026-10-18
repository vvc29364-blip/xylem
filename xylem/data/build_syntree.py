"""Build the SynTree benchmark.

    python -m xylem.data.build_syntree --root data --budget full --workers 32

Layout
    data/syntree/trees/{train,val,test_seen,test_unseen}/<species>_<i>.npz      skeleton graphs
    data/syntree/fixed/{val,test_seen,test_unseen}/<sensor>/...npz             cached eval scans
    data/syntree/robust/<factor>_<level>/...npz                               robustness sets
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from .dataset import build_fixed_sample, stable_seed
from .treegen import SEEN_SPECIES, UNSEEN_SPECIES, generate_tree
from .scanner import SENSORS

BUDGETS = {
    #            train/sp  val/sp  test_seen/sp  test_unseen/sp  robust trees
    "smoke": dict(train=12, val=2, test_seen=2, test_unseen=3, robust=6),
    "small": dict(train=250, val=10, test_seen=20, test_unseen=40, robust=60),
    "full": dict(train=1500, val=40, test_seen=120, test_unseen=200, robust=240),
}

ROBUST = {
    "noise": [0.0, 0.01, 0.02, 0.04, 0.08],        # metres (range noise sigma)
    "outlier": [0.0, 0.02, 0.05, 0.10],
    "density": [1.0, 0.5, 0.25, 0.1],
}


def _gen(args):
    species, idx, split, out = args
    if os.path.exists(out):
        return out
    t = generate_tree(species, stable_seed("syntree", split, species, idx) % (2 ** 31))
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **t)
    os.replace(tmp, out)
    return out


def alt_radius_tree(src, dst, seed=0):
    """Radius = r_tip + (r_base - r_tip) * (1 - g/g_max)^gamma  (+-15% noise): children inherit the
    parent radius at junctions, i.e. sum r_c^2 ~ n r_p^2 -- a strong violation of Leonardo's rule."""
    from .skeleton import geodesic
    d = dict(np.load(src, allow_pickle=True))
    rng = np.random.default_rng(seed)
    pos, par, r = d["pos"].astype(np.float64), d["parent"].astype(np.int64), d["radius"].astype(np.float64)
    g = geodesic(pos, par)
    gamma = rng.uniform(1.0, 2.0)
    r_tip = float(np.quantile(r, 0.05))
    rn = r_tip + (r[0] - r_tip) * np.clip(1 - g / max(g.max(), 1e-6), 0, 1) ** gamma
    rn *= rng.uniform(0.85, 1.15, len(rn))
    d["radius"] = np.maximum(rn, 0.0015).astype(np.float32)
    tmp = dst + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, dst)


def build(root, budget, workers):
    B = BUDGETS[budget]
    base = os.path.join(root, "syntree")
    jobs = []
    for split, species in (("train", SEEN_SPECIES), ("val", SEEN_SPECIES), ("test_seen", SEEN_SPECIES),
                           ("test_unseen", UNSEEN_SPECIES)):
        for sp in species:
            for i in range(B[split]):
                jobs.append((sp, i, split, os.path.join(base, "trees", split, f"{sp}_{i:05d}.npz")))
    for split in ("train", "val", "test_seen", "test_unseen"):
        os.makedirs(os.path.join(base, "trees", split), exist_ok=True)
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_gen, jobs, chunksize=8)):
            if k % 1000 == 0:
                print(f"[trees] {k}/{len(jobs)}", flush=True)
    # ---- fixed evaluation scans ----
    fx = []
    for split in ("val", "test_seen", "test_unseen"):
        files = sorted(glob.glob(os.path.join(base, "trees", split, "*.npz")))
        for j, f in enumerate(files):
            sensors = [SENSORS[j % 3]] if split == "val" else SENSORS
            for s in sensors:
                leaf = bool((j + SENSORS.index(s)) % 2)
                out = os.path.join(base, "fixed", split, s, os.path.basename(f))
                fx.append((f, out, dict(sensor=s, leaf_on=leaf)))
    # ---- robustness sets: tls3, leaf-on, one factor varied ----
    rfiles = sorted(glob.glob(os.path.join(base, "trees", "test_seen", "*.npz")))
    rng = np.random.default_rng(0)
    rfiles = [rfiles[i] for i in rng.permutation(len(rfiles))[: B["robust"]]]
    for factor, levels in ROBUST.items():
        for lv in levels:
            for f in rfiles:
                cond = dict(sensor="tls3", leaf_on=True, noise=0.005, outlier_frac=0.0, density=1.0)
                cond[{"noise": "noise", "outlier": "outlier_frac", "density": "density"}[factor]] = lv
                out = os.path.join(base, "robust", f"{factor}_{lv}", os.path.basename(f))
                fx.append((f, out, cond))
    # ---- SynTree-AltR: same trees, radii from a path-length taper that violates the pipe model ----
    adir = os.path.join(base, "trees", "test_altr")
    os.makedirs(adir, exist_ok=True)
    for j, f in enumerate(rfiles):
        out_t = os.path.join(adir, os.path.basename(f))
        if not os.path.exists(out_t):
            alt_radius_tree(f, out_t, seed=j)
        s = SENSORS[j % 3]
        fx.append((out_t, os.path.join(base, "fixed", "test_altr", s, os.path.basename(f)),
                   dict(sensor=s, leaf_on=bool(j % 2))))
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(build_fixed_sample, fx, chunksize=4)):
            if k % 500 == 0:
                print(f"[fixed] {k}/{len(fx)}", flush=True)
    with open(os.path.join(base, "meta.json"), "w") as f:
        json.dump(dict(budget=budget, counts={k: len(glob.glob(os.path.join(base, "trees", k, "*.npz")))
                                              for k in ("train", "val", "test_seen", "test_unseen")},
                       robust=ROBUST, seen=SEEN_SPECIES, unseen=UNSEEN_SPECIES), f, indent=1)
    print("SynTree ready at", base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--budget", default="full", choices=list(BUDGETS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    build(a.root, a.budget, a.workers)
