"""Evaluate a trained method on one or more fixed test sets.

    python -m xylem.engine.evaluate --exp main/pointr --sets syn_seen,syn_unseen
    python -m xylem.engine.evaluate --exp main/xylem_supernet --cfg results/main/xylem_supernet/subnets.json:S --tag S

Writes results/<exp>/eval/<tag>__<set>.csv (per sample) and <tag>__<set>.json (means) and keeps
a few predictions for qualitative figures in results/<exp>/eval/pred__<tag>__<set>/.
"""
import argparse
import heapq
import json
import os
import time
from collections import deque
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import FixedDataset, collate
from ..methods import build_method, predict
from ..metrics import evaluate_sample, forest_metrics, forest_summary
from ..methods import _tubes
from ..skeletonize import geobin_skeleton

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_folders(data, name):
    """Named evaluation sets -> list of folders."""
    s = os.path.join(data, "syntree")
    if name == "syn_seen":
        return [os.path.join(s, "fixed", "test_seen")]
    if name == "syn_unseen":
        return [os.path.join(s, "fixed", "test_unseen")]
    if name == "syn_val":
        return [os.path.join(s, "fixed", "val")]
    if name.startswith("robust/"):
        return [os.path.join(s, name)]
    if name == "syn_altr":            # radii violating the pipe model (prior-validity test)
        return [os.path.join(s, "fixed", "test_altr")]
    if name.startswith("weiser_"):    # real forest trees: weiser_uls_off / weiser_uls_on / weiser_als / weiser_tls1
        from ..data.external import ext_root
        return [os.path.join(ext_root(data, "weiser"), "fixed", name[len("weiser_"):])]
    from ..data.external import ext_root
    if name in ("st_test", "st_foliage_test"):
        return [os.path.join(ext_root(data, "synthetic_trees"), "fixed", "test" if name == "st_test" else "test_foliage")]
    if name == "branch3d_test":
        return [os.path.join(ext_root(data, "branch3d"), "fixed", "test")]
    raise KeyError(name)


def _worker(args):
    pred, sample, kind, bin_size = args[:4]
    render = len(args) > 4 and args[4]
    t = time.time()
    g = pred["graph"]
    if g is None and kind in ("completion", "classical"):
        g = geobin_skeleton(pred["dense"], bin_size=bin_size)
        if render and len(g[0]) > 1:      # Baseline+QSM: surface rendered on the skeleton of the completion
            pred = dict(pred, dense=_tubes(g, np.random.default_rng(0)))
    t_graph = time.time() - t
    try:
        if "dbh_ref_cm" in sample:        # real forest tree: DBH / height against references
            m = forest_metrics(pred["dense"], g, sample, pred.get("extra"))
        else:
            m = evaluate_sample(pred["dense"], g, sample, pred.get("seg"))
    except Exception as ex:  # never let a single degenerate prediction kill the evaluation
        m = dict(error=str(ex)[:100])
    m["t_graph"] = t_graph
    for k in ("species", "sensor", "leaf_on", "name"):
        if k in sample:
            v = sample[k]
            m[k] = v.item() if hasattr(v, "item") else v
    return m, g


def load_method(exp_dir, ckpt="best", dev="cuda", fixed_cfg=None):
    a = json.load(open(os.path.join(exp_dir, "train_args.json")))
    fc = json.loads(a["fixed_cfg"]) if a.get("fixed_cfg") else fixed_cfg
    m = build_method(a["method"], json.loads(a["model_kw"]), json.loads(a["loss_w"]), fc, a.get("topology", "flow"))
    if a["method"] not in ("geobin", "circlefit"):
        p = os.path.join(exp_dir, f"ckpt_{ckpt}.pt")
        if not os.path.exists(p):
            p = os.path.join(exp_dir, "ckpt_final.pt")
        sd = torch.load(p, map_location="cpu")
        m.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
    return m.to(dev).eval(), a


def resolve_cfg(spec):
    """'path.json:S' -> cfg dict,  '{json}' -> dict, '' -> None"""
    if not spec:
        return None
    if spec.startswith("{"):
        return json.loads(spec)
    path, key = spec.rsplit(":", 1)
    return json.load(open(path))[key]["cfg"]


def select_node_thr(args, method, cfg, exp_dir, pool, dev, amp, cands=(0.5, 0.7, 0.9)):
    """XylemNet: tune the confidence threshold of generated nodes on the validation set."""
    f = os.path.join(exp_dir, "decode.json")
    if os.path.exists(f):
        return json.load(open(f))["node_thr"]
    ds = FixedDataset(set_folders(args.data, "syn_val"), limit=48)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=collate)
    scores = {}
    for thr in cands:
        method.node_thr = thr
        jobs = []
        for bi, b in enumerate(dl):
            bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                preds = predict(method, bt, cfg=cfg, with_graph=True, seed=bi)
            for i, p in enumerate(preds):
                jobs.append((p, {k: (v[i].numpy() if torch.is_tensor(v) else v[i]) for k, v in b.items()}, method.kind, 0.02))
        ms = [m for m, _ in pool.imap(_worker, jobs)]
        scores[thr] = float(np.nanmean([0.5 * (m.get("skel_f", 0.0) + m.get("f1", 0.0)) for m in ms]))
    best = max(scores, key=scores.get)
    json.dump(dict(node_thr=best, scores=scores), open(f, "w"), indent=1)
    print("[decode] node_thr", best, scores, flush=True)
    return best


def select_bin(args, method, cfg, exp_dir, pool, dev, amp, cands=(0.01, 0.015, 0.02, 0.03)):
    """Tune the skeletonisation bin of methods that rely on it (fair to every baseline)."""
    if method.kind == "xylem":
        method.node_thr = select_node_thr(args, method, cfg, exp_dir, pool, dev, amp)
        return 0.02
    if method.kind not in ("completion", "classical", "smarttree"):
        return 0.02
    f = os.path.join(exp_dir, "bin.json")
    if os.path.exists(f):
        return json.load(open(f))["bin"]
    ds = FixedDataset(set_folders(args.data, "syn_val"), limit=48)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=collate)
    scores = {}
    for bsz in cands:
        jobs = []
        for bi, b in enumerate(dl):
            bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                preds = predict(method, bt, cfg=cfg, with_graph=method.kind == "smarttree", bin_size=bsz, seed=bi)
            for i, p in enumerate(preds):
                jobs.append((p, {k: (v[i].numpy() if torch.is_tensor(v) else v[i]) for k, v in b.items()}, method.kind, bsz))
        ms = [m for m, _ in pool.imap(_worker, jobs)]
        scores[bsz] = float(np.nanmean([m.get("skel_f", 0.0) for m in ms]))
    best = max(scores, key=scores.get)
    json.dump(dict(bin=best, scores=scores), open(f, "w"), indent=1)
    print("[bin] selected", best, scores, flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--results", default=os.environ.get("XYLEM_RESULTS", os.path.join(ROOT, "results")))
    ap.add_argument("--sets", default="syn_seen,syn_unseen")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--keep_pred", type=int, default=16)
    ap.add_argument("--bin", default="auto", help="skeletonisation bin size, 'auto' = tuned on the validation set")
    ap.add_argument("--exp_dir", default="", help="explicit experiment dir (e.g. exported subnet)")
    ap.add_argument("--topology", default="", help="override XylemNet topology decoding: flow | mst")
    ap.add_argument("--no_merge", action="store_true", help="XylemNet: do not merge observed wood points")
    ap.add_argument("--skel_source", default="", help="XylemNet skeleton decoding source: union | nodes")
    ap.add_argument("--dense_source", default="", help="graph | nodes (XylemNet); graph = +QSM rendering for baselines")
    ap.add_argument("--node_thr", type=float, default=-1, help="XylemNet: fixed node threshold (skip validation tuning)")
    ap.add_argument("--nms_dist", type=float, default=-1, help="XylemNet: NMS radius of the topology decoder")
    ap.add_argument("--no_refine", action="store_true", help="XylemNet: disable observation-refined radii")
    args = ap.parse_args()
    exp_dir = args.exp_dir or os.path.join(args.results, args.exp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = resolve_cfg(args.cfg)
    method, targs = load_method(exp_dir, args.ckpt, dev)
    if args.topology and method.kind == "xylem":
        method.topology = args.topology
    if args.no_merge and method.kind == "xylem":
        method.merge_input = False
    if args.skel_source and method.kind == "xylem":
        method.skel_source = args.skel_source
    if args.dense_source and method.kind == "xylem":
        method.dense_source = args.dense_source
    render = args.dense_source == "graph" and method.kind in ("completion", "classical")
    if args.nms_dist > 0:
        method.nms_dist = args.nms_dist
    if args.no_refine:
        method.refine_radius = False
    amp = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else None
    ev_dir = os.path.join(exp_dir, "eval")
    os.makedirs(ev_dir, exist_ok=True)
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        if args.node_thr >= 0 and method.kind == "xylem":
            method.node_thr = args.node_thr
            args.bin = 0.02 if args.bin == "auto" else float(args.bin)
        else:
            args.bin = select_bin(args, method, cfg, exp_dir, pool, dev, amp) if args.bin == "auto" else float(args.bin)
        for set_name in args.sets.split(","):
            tag = f"{args.tag}__{set_name.replace('/', '_')}"
            out_csv = os.path.join(ev_dir, tag + ".csv")
            if os.path.exists(out_csv.replace(".csv", ".json")):
                print("exists", out_csv)
                continue
            try:
                ds = FixedDataset(set_folders(args.data, set_name), limit=args.limit or None)
            except KeyError:
                continue
            if len(ds) == 0:
                print("empty set", set_name)
                continue
            dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)
            rows = []
            t_inf, n_inf = 0.0, 0
            keep_dir = os.path.join(ev_dir, "pred__" + tag)
            # keep qualitative samples only for the sets used in figures
            kept = 0 if set_name in ("syn_seen", "syn_unseen", "branch3d_test", "st_test", "st_foliage_test", "weiser_uls_off", "weiser_tls1") else args.keep_pred
            # keep qualitative samples: the first `kept..keep_pred` ones and the 8 worst (failure cases).
            # Metrics are streamed through the pool so that only these few samples stay in RAM (holding every
            # sample + prediction of the 2880-sample syn_* sets got the process OOM-killed).
            n_first = max(0, args.keep_pred - kept)
            first, worst_heap, pending = {}, [], deque()
            max_pending = 4 * args.workers + 2 * args.bs

            def consume(limit):
                while len(pending) > limit:
                    j, p, sample, ar = pending.popleft()
                    m, g = ar.get()
                    rows.append(m)
                    if n_first == 0:
                        continue
                    forest = "dbh_ref_cm" in sample      # forest sets: worst = largest DBH error, else lowest F1
                    v = float(m.get("dbh_abs_cm" if forest else "f1", np.nan))
                    if j < n_first:
                        first[j] = (p, sample, g, v)
                    if np.isfinite(v):
                        item = (v if forest else -v, -j, j, (p, sample, g, v))
                        if len(worst_heap) < 8:
                            heapq.heappush(worst_heap, item)
                        elif item[:2] > worst_heap[0][:2]:
                            heapq.heapreplace(worst_heap, item)

            n_seen = 0
            for bi, b in enumerate(dl):
                bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                torch.cuda.synchronize() if dev == "cuda" else None
                t = time.time()
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    preds = predict(method, bt, cfg=cfg, with_graph=method.kind not in ("completion", "classical"),
                                    bin_size=args.bin, seed=bi)
                torch.cuda.synchronize() if dev == "cuda" else None
                t_inf += time.time() - t
                n_inf += len(preds)
                for i, p in enumerate(preds):
                    # copy out of the worker's shared-memory storage: keeping views alive holds one fd per tensor
                    # per batch -> "received 0 items of ancdata" (fd exhaustion) on the large syn_* sets
                    sample = {k: (v[i].numpy().copy() if torch.is_tensor(v) else v[i]) for k, v in b.items()}
                    ar = pool.apply_async(_worker, ((p, sample, method.kind, args.bin, render),))
                    pending.append((n_seen, p, sample, ar))
                    n_seen += 1
                del b, bt
                consume(max_pending)
            consume(0)
            worst = {it[2]: it[3] for it in worst_heap}
            for j in sorted(set(first) | set(worst)):
                p, sample, g, v = first[j] if j in first else worst[j]
                os.makedirs(keep_dir, exist_ok=True)
                fname = f"{sample.get('sensor', '')}__{sample.get('name', j)}".replace(".npz", "")
                np.savez_compressed(os.path.join(keep_dir, fname),
                                    dense=p["dense"].astype(np.float32), inp=sample["inp"],
                                    gt=sample.get("gt", sample.get("ref_pts", np.zeros((0, 3)))),
                                    g_pos=sample.get("g_pos", np.zeros((0, 3))), g_rad=sample.get("g_rad", np.zeros(0)),
                                    g_par=sample.get("g_par", np.zeros(0, int)),
                                    p_pos=g[0] if g is not None else np.zeros((0, 3)),
                                    p_rad=g[1] if g is not None else np.zeros(0),
                                    p_par=g[2] if g is not None else np.zeros(0, int),
                                    seg=p["seg"] if p.get("seg") is not None else np.zeros(0),
                                    scale=np.float32(sample.get("scale", 1.0)), metric=np.float32(v),
                                    worst=np.array(j in worst))
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            num = df.select_dtypes(include=[np.number])
            summ = {k: float(v) for k, v in num.mean(skipna=True).items()}
            summ["n"] = len(df)
            if "dbh_ref_cm" in df:
                summ.update(forest_summary(df))
            summ["sec_per_sample_gpu"] = t_inf / max(n_inf, 1)
            for key in ("sensor", "leaf_on", "species"):
                if key in df:
                    summ[f"by_{key}"] = {str(k): {c: float(v) for c, v in g.select_dtypes(include=[np.number]).mean().items()}
                                         for k, g in df.groupby(key)}
            json.dump(summ, open(out_csv.replace(".csv", ".json"), "w"), indent=1)
            print(f"[{args.exp}:{tag}] n={len(df)} cd_l1={summ.get('cd_l1', np.nan):.3f} f1={summ.get('f1', np.nan):.3f} "
                  f"tbr={summ.get('tbr', np.nan):.3f} skel_f={summ.get('skel_f', np.nan):.3f} "
                  f"rad_mare={summ.get('rad_mare', np.nan):.3f}", flush=True)


if __name__ == "__main__":
    main()
