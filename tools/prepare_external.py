"""Download (if missing) and prepare the external datasets: BRANCH3D (real) and Synthetic-Trees.

    python tools/prepare_external.py --data data
Both steps are optional: failures (e.g. no internet on the compute server) are reported and the
corresponding experiments are skipped by run_all.py.
"""
import argparse
import glob
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B3D_URL = "https://www.ipb.uni-bonn.de/html/projects/DiffTS/BRANCH3D_dataset.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    data = a.data if os.path.isabs(a.data) else os.path.join(ROOT, a.data)
    raw = os.path.join(ROOT, "data", "raw")
    os.makedirs(raw, exist_ok=True)
    # ---------------- BRANCH3D ----------------
    from xylem.data.external import prepare_branch3d, prepare_synthetic_trees
    z = os.path.join(raw, "BRANCH3D_dataset.zip")
    have_b3d = len(glob.glob(os.path.join(ROOT, "data", "branch3d", "test", "*.npz"))) > 0
    try:
        if not have_b3d and not os.path.exists(z):
            print("downloading BRANCH3D (1.9 GB) ...", flush=True)
            urllib.request.urlretrieve(B3D_URL, z + ".part")
            os.replace(z + ".part", z)
        prepare_branch3d(os.path.join(ROOT, "data"), z if os.path.exists(z) else None)
    except Exception as ex:
        print("[WARN] BRANCH3D unavailable:", ex)
    # ---------------- Synthetic-Trees ----------------
    st = os.path.join(ROOT, "data", "synthetic_trees")
    try:
        if not os.path.exists(os.path.join(st, "split.json")) or len(glob.glob(os.path.join(st, "branches*", "*", "*.npz"))) < 1200:
            print("streaming Synthetic-Trees (converted on the fly, ~2 GB on disk) ...", flush=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "tools", "stream_synthetic_trees.py")], check=True)
        prepare_synthetic_trees(os.path.join(ROOT, "data"), workers=a.workers)
    except Exception as ex:
        print("[WARN] Synthetic-Trees unavailable:", ex)


def _mirror(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        import shutil
        shutil.copytree(src, dst)


if __name__ == "__main__":
    main()
