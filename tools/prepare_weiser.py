"""Real forest trees with field / TLS reference measurements (Weiser et al., ESSD 2022,
PANGAEA 942856, CC-BY-SA 4.0).

Only the single-tree files are fetched from the 5-18 GB per-plot archives (HTTP range requests):
ULS leaf-off / leaf-on, ALS and TLS single-tree clouds + metrics (DBH, height, crown base height).
TLS clouds are subsampled to <= 400k points.  Output: data/weiser/trees/<tree_id>.npz

    python tools/prepare_weiser.py --threads 6
"""
import argparse
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from remote_zip import RemoteZip  # noqa: E402

URL = "https://download.pangaea.de/dataset/942856/files/{}.zip"
PLOTS = ["BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "KA09", "KA10", "KA11", "SP02"]
CAP = {"TLS-on": 400000, "ULS-off": 150000, "ULS-on": 150000, "ALS-on": 60000}


def read_laz(blob):
    import laspy
    las = laspy.read(io.BytesIO(blob))
    return np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], 1)


def parse_metrics(txt):
    rows = [r.split("\t") for r in txt.strip().splitlines()]
    head = rows[0]
    out = {}
    for r in rows[1:]:
        d = dict(zip(head, r))
        src = d.get("Source", "?").strip()
        rec = {}
        for k, v in d.items():
            v = v.strip()
            if not v:
                continue
            if k.startswith("Diameter at breast height"):
                rec["dbh_cm"] = float(v)
            elif k.startswith("Height"):
                rec["height_m"] = float(v)
            elif k.startswith("Crown base height"):
                rec["cbh_m"] = float(v)
        rec["canopy"] = d.get("Canopy condition", "")
        out.setdefault(src, []).append(rec)
    return out


def process_tree(z, tid, files, out_dir):
    dst = os.path.join(out_dir, tid + ".npz")
    if os.path.exists(dst):
        return tid, "exists"
    gj = json.loads(z.read(files["geojson"]).decode())
    pos = None
    for m in gj["properties"]["measurements"]:
        if "position_xyz" in m:
            pos = np.array(m["position_xyz"], np.float64)
    metrics = parse_metrics(z.read(files["metrics"]).decode()) if "metrics" in files else {}
    clouds = {}
    rng = np.random.default_rng(0)
    for key in ("ULS-off", "ULS-on", "ALS-on", "TLS-on"):
        if key in files:
            try:
                p = read_laz(z.read(files[key]))
            except Exception as ex:
                print("laz fail", tid, key, ex, flush=True)
                continue
            if len(p) > CAP[key]:
                p = p[rng.choice(len(p), CAP[key], replace=False)]
            clouds[key] = p
    if not clouds or pos is None:
        return tid, "skip"
    # tree-local coordinates: stem position at the origin, z = height above the stem-base terrain
    save = {k.replace("-", "_"): (v - pos).astype(np.float32) for k, v in clouds.items()}
    ref = {}
    for src in ("FI", "field", "Field", "TLS"):
        for rec in metrics.get(src, []):
            if "dbh_cm" in rec and "dbh_cm" not in ref:
                ref["dbh_cm"], ref["dbh_src"] = rec["dbh_cm"], src
    for src in ("FI", "field", "Field", "ALS", "ULS"):
        for rec in metrics.get(src, []):
            if "height_m" in rec and "height_m" not in ref:
                ref["height_m"], ref["height_src"] = rec["height_m"], src
    np.savez_compressed(dst, species=gj["properties"].get("species", ""), plot=tid.split("_")[1] if "_" in tid else "",
                        dbh_cm=np.float32(ref.get("dbh_cm", np.nan)), dbh_src=ref.get("dbh_src", ""),
                        height_m=np.float32(ref.get("height_m", np.nan)), height_src=ref.get("height_src", ""),
                        metrics=json.dumps(metrics), **save)
    return tid, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "weiser", "trees"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--plots", default=",".join(PLOTS))
    ap.add_argument("--build_data", default="", help="after downloading, build the fixed evaluation sets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_ok = 0
    for plot in a.plots.split(","):
        try:
            z = RemoteZip(URL.format(plot))
        except Exception as ex:
            print("[weiser] cannot open", plot, ex, flush=True)
            continue
        trees = {}
        for n in z.entries:
            m = re.match(r"single_trees/([^/]+)/(.+)$", n)
            if not m or not m.group(2):
                continue
            tid, fn = m.group(1), m.group(2)
            d = trees.setdefault(tid, {})
            if fn.endswith(".geojson"):
                d["geojson"] = n
            elif fn.endswith("_metrics.txt"):
                d["metrics"] = n
            else:
                mm = re.search(r"_(TLS|ULS|ALS)-(on|off)\.laz$", fn)
                if mm:
                    d[f"{mm.group(1)}-{mm.group(2)}"] = n
        trees = {k: v for k, v in trees.items() if "geojson" in v}
        print(f"[weiser] {plot}: {len(trees)} trees", flush=True)
        with ThreadPoolExecutor(a.threads) as ex:
            for tid, st in ex.map(lambda kv: process_tree(z, kv[0], kv[1], a.out), trees.items()):
                n_ok += st in ("ok", "exists")
        print(f"[weiser] {plot} done, total trees {n_ok}", flush=True)
    print("WEISER_DONE", n_ok)
    if a.build_data:
        sys.path.insert(0, ROOT)
        from xylem.data.weiser import build_weiser_fixed
        build_weiser_fixed(a.build_data)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:          # no internet on the server -> forest experiments are skipped gracefully
        print("[WARN] Weiser data unavailable:", ex)
