"""External datasets.

Synthetic-Trees (Dobbs et al. 2023, SpeedTree, 6 species, GT skeleton + radii)
    converted by tools/stream_synthetic_trees.py into data/synthetic_trees/{branches,branches_foliage}/<sp>/<name>.npz
    -> used for cross-generator generalisation (train on SynTree, test here) and as a second
       training source.  Inputs are produced with the same virtual scanner (on their dense points).
BRANCH3D (Marks et al., ICCV 2025, real orchard trees, reference skeletons without radii)
    DiffTS_dataset/{train,val,test}/*.pt with scan_points, scan_point_classes, skeleton_vertices,
    skeleton_edges -> data/branch3d/{train,val,test}/<name>.npz
    -> real-world evaluation (skeleton + topology metrics) and unlabelled target domain for the
       botanical self-supervised adaptation.
"""
import glob
import io
import json
import os
import zipfile
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import build_fixed_sample, normalisation, stable_seed
from .skeleton import resample_skeleton, simplify

SENSORS = ("tls1", "tls3", "uls")
_ROOT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


def ext_root(data, name):
    """External datasets live in <repo>/data even when experiments use another data root (smoke)."""
    p = os.path.join(data, name)
    return p if os.path.exists(p) else os.path.join(_ROOT_DATA, name)


# ------------------------------------------------------------------------------------------
# Synthetic-Trees
# ------------------------------------------------------------------------------------------
def st_split(data):
    base = ext_root(data, "synthetic_trees")
    sp = json.load(open(os.path.join(base, "split.json")))
    return base, sp


def external_train_files(data, source):
    if source == "synthetic_trees":
        base, sp = st_split(data)
        out = []
        for variant in ("branches", "branches_foliage"):
            for f in sp["train"]:
                p = os.path.join(base, "simplified", variant, f)
                if os.path.exists(p):
                    out.append(p)
        return out
    if source == "branch3d":
        return sorted(glob.glob(os.path.join(ext_root(data, "branch3d"), "train", "*.npz")))
    raise KeyError(source)


def _simplify_one(args):
    src, dst = args
    if os.path.exists(dst):
        return dst
    d = np.load(src)
    pos, rad, par = d["pos"].astype(np.float64), d["radius"].astype(np.float64), d["parent"].astype(np.int64)
    H = float(pos[:, 2].max() - pos[:, 2].min())
    pos2, rad2, par2 = simplify(pos, rad, par, min_seg=H / 250)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, pos=pos2.astype(np.float32), radius=rad2.astype(np.float32), parent=par2.astype(np.int32),
                        points=d["points"], cls=d["cls"], height=np.float32(H), species=os.path.basename(os.path.dirname(src)),
                        leaf_size=np.float32(0.05), leaf_density=np.float32(5), leaf_kind="broad")
    return dst


def prepare_synthetic_trees(data, workers=8, max_test=None):
    base, sp = st_split(data)
    jobs = []
    for variant in ("branches", "branches_foliage"):
        for split in ("train", "validation", "val", "test"):
            for f in sp.get(split, []):
                src = os.path.join(base, variant, f)
                if os.path.exists(src):
                    jobs.append((src, os.path.join(base, "simplified", variant, f)))
    with Pool(workers) as p:
        p.map(_simplify_one, jobs, chunksize=4)
    # fixed evaluation scans (virtual scanner on the SpeedTree dense clouds)
    fx = []
    for variant, folder, leaf in (("branches", "test", False), ("branches_foliage", "test_foliage", True)):
        files = [os.path.join(base, "simplified", variant, f) for f in sp.get("test", [])]
        files = [f for f in files if os.path.exists(f)]
        if max_test:
            files = files[:max_test]
        for j, f in enumerate(files):
            for s in SENSORS:
                fx.append((f, os.path.join(base, "fixed", folder, s, os.path.basename(f)), dict(sensor=s, leaf_on=leaf)))
        vfiles = [os.path.join(base, "simplified", variant, f) for f in sp.get("validation", sp.get("val", []))]
        for j, f in enumerate([v for v in vfiles if os.path.exists(v)][:20]):
            fx.append((f, os.path.join(base, "fixed", "val", variant + "_" + os.path.basename(f)),
                       dict(sensor=SENSORS[j % 3], leaf_on=leaf)))
    with Pool(workers) as p:
        p.map(build_fixed_sample, fx, chunksize=2)
    print("Synthetic-Trees prepared:", len(jobs), "trees,", len(fx), "fixed scans")


# ------------------------------------------------------------------------------------------
# BRANCH3D
# ------------------------------------------------------------------------------------------
def _orient(V, E):
    """undirected edges -> parent array rooted at the lowest vertex of the largest component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components
    K = len(V)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(K, K)).tocsr()
    A = A.maximum(A.T)
    nc, lab = connected_components(A, directed=False)
    main = np.bincount(lab).argmax()
    cand = np.nonzero(lab == main)[0]
    root = cand[np.argmin(V[cand, 2])]
    order, pred = breadth_first_order(A, root, directed=False, return_predecessors=True)
    keep = np.zeros(K, bool)
    keep[order] = True
    par = np.where(pred >= 0, pred, -1)
    par[root] = -1
    new = -np.ones(K, np.int64)
    idx = np.concatenate([[root], [i for i in order if i != root]])
    new[idx] = np.arange(len(idx))
    P = par[idx]
    P = np.where(P >= 0, new[np.maximum(P, 0)], -1)
    return V[idx], P


def convert_branch3d(zip_path, data):
    out_base = os.path.join(data, "branch3d")
    z = zipfile.ZipFile(zip_path)
    n = 0
    for name in z.namelist():
        if not name.endswith(".pt"):
            continue
        split = name.split("/")[-2]
        dst = os.path.join(out_base, split, os.path.basename(name).replace(".pt", ".npz"))
        if os.path.exists(dst):
            n += 1
            continue
        obj = torch.load(io.BytesIO(z.read(name)), map_location="cpu", weights_only=False)
        V = np.asarray(obj["skeleton_vertices"], np.float64)
        E = np.asarray(obj["skeleton_edges"], np.int64)
        pos, par = _orient(V, E)
        pts = np.asarray(obj["scan_points"], np.float64)
        cls = np.asarray(obj["scan_point_classes"]).astype(np.int8)
        base = pos[0].copy()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, pos=(pos - base).astype(np.float32), parent=par.astype(np.int32),
                            radius=np.zeros(len(pos), np.float32), points=(pts - base).astype(np.float32), cls=cls,
                            height=np.float32(pos[:, 2].max() - pos[:, 2].min()), species="apple_real")
        n += 1
    print("BRANCH3D converted:", n)


class RealScanDataset(Dataset):
    """Real scans used as-is (natural occlusion).  Radii unknown (NaN -> masked in the losses)."""

    def __init__(self, data, source="branch3d", split="test", augment=False, n_in=4096, n_gt=16384, n_skel=2048):
        self.files = sorted(glob.glob(os.path.join(ext_root(data, source), split, "*.npz")))
        self.augment = augment
        self.n_in, self.n_gt, self.n_skel = n_in, n_gt, n_skel
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return make_real_sample(self.files[i], self.augment, self.n_in, self.n_gt, self.n_skel,
                                seed=None if self.augment else stable_seed(self.files[i]))


def make_real_sample(path, augment, n_in=4096, n_gt=16384, n_skel=2048, seed=None, with_graph=False):
    d = np.load(path, allow_pickle=True)
    rng = np.random.default_rng(seed)
    pts = d["points"].astype(np.float64)
    cls = d["cls"].astype(np.int64)
    pos = d["pos"].astype(np.float64)
    par = d["parent"].astype(np.int64)
    if augment:
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        pts, pos = pts @ R.T, pos @ R.T
    sel = rng.choice(len(pts), n_in, replace=len(pts) < n_in)
    x = pts[sel]
    center, scale = normalisation(x)
    nz = lambda a: ((a - center) / scale).astype(np.float32)
    wood = cls == 1
    wp = pts[wood] if wood.sum() > 100 else pts
    gt = wp[rng.choice(len(wp), n_gt, replace=len(wp) < n_gt)]
    sk = resample_skeleton(pos, np.zeros(len(pos)), par, step=scale * 0.006)
    si = rng.choice(len(sk["pts"]), n_skel, replace=len(sk["pts"]) < n_skel)
    from scipy.spatial import cKDTree
    _, nn = cKDTree(sk["pts"]).query(x)
    s = dict(inp=nz(x), seg=(cls[sel] == 1).astype(np.int64), p_valid=np.ones(n_in, np.float32),
             p_flow=sk["flow"][nn].astype(np.float32), p_geo=(sk["geo"][nn] / scale).astype(np.float32),
             p_rad=np.full(n_in, np.nan, np.float32), p_med=((sk["pts"][nn] - x) / scale).astype(np.float32),
             gt=nz(gt), sk=nz(sk["pts"][si]), sk_r=np.full(n_skel, np.nan, np.float32),
             sk_flow=sk["flow"][si].astype(np.float32), sk_geo=(sk["geo"][si] / scale).astype(np.float32),
             center=center.astype(np.float32), scale=np.float32(scale))
    if with_graph:
        s.update(g_pos=nz(pos), g_rad=np.zeros(len(pos), np.float32), g_par=par.astype(np.int32),
                 species=np.array("apple_real"), sensor=np.array("real"), leaf_on=np.array(True),
                 tree=np.array(os.path.basename(path)))
    return s


def prepare_branch3d(data, zip_path):
    if zip_path and os.path.exists(zip_path):
        convert_branch3d(zip_path, data)
    for split in ("val", "test"):
        for f in sorted(glob.glob(os.path.join(data, "branch3d", split, "*.npz"))):
            out = os.path.join(data, "branch3d", "fixed", split, os.path.basename(f))
            if os.path.exists(out):
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            s = make_real_sample(f, False, seed=stable_seed(f), with_graph=True)
            np.savez_compressed(out, **s)
    # use part of train as val if no val split exists
    if not glob.glob(os.path.join(data, "branch3d", "fixed", "val", "*.npz")):
        for f in sorted(glob.glob(os.path.join(data, "branch3d", "train", "*.npz")))[:8]:
            out = os.path.join(data, "branch3d", "fixed", "val", os.path.basename(f))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(out, **make_real_sample(f, False, seed=stable_seed(f), with_graph=True))
