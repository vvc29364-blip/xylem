"""Sub-network search on the trained elastic supernet and extraction of XylemNet-L / -M / -S.

    python -m xylem.engine.search --exp main/xylem_supernet --n_random 300

L = largest network.  M / S = best validation F-score under FLOP budgets (fractions of L) found by
random sampling + a small evolutionary refinement.  S additionally has a parameter cap so that
it runs in real time on a Jetson Orin (16 GB) in FP16 via TensorRT.  Writes:
    results/<exp>/search.csv      every evaluated candidate (for the Pareto figure)
    results/<exp>/subnets.json    the chosen L/M/S configs
    results/<exp>_{L,M,S}/        standalone (physically cut) networks
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import FixedDataset, collate
from ..models.xylemnet import MAX_CFG, MIN_CFG, SPACE, build_xylemnet, cfg_key, sample_cfg
from ..ops import nn_idx
from .evaluate import load_method
from .export import export_subnet, save_as_experiment, verify
from .profile import count_params, flops, latency

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@torch.no_grad()
def quality(method, batches, cfg, dev, amp):
    cds, fs = [], []
    for b in batches:
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = method(b["inp"], cfg)
            d = method.net.sample_dense(out, 16384).float()
        gt = b["gt"].float()
        _, d1 = nn_idx(d, gt)
        _, d2 = nn_idx(gt, d)
        cds.append(0.5 * (d1.mean(1) + d2.mean(1)))
        p = (d1 < 0.01).float().mean(1)
        r = (d2 < 0.01).float().mean(1)
        fs.append(2 * p * r / (p + r + 1e-9))
    return float(torch.cat(cds).mean()) * 1e3, float(torch.cat(fs).mean())


def full_quality(method, batches, cfg, dev, amp, workers=8):
    """F@1% and skeleton-F of the complete XylemNet pipeline (decoded tree + rendered surface)."""
    from multiprocessing import get_context
    from ..methods import predict
    from .evaluate import _worker
    jobs = []
    for bi, b in enumerate(batches):
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            preds = predict(method, b, cfg=cfg, with_graph=True, seed=bi)
        for i, p in enumerate(preds):
            jobs.append((p, {k: (v[i].cpu().numpy() if torch.is_tensor(v) else v[i]) for k, v in b.items()}, "xylem", 0.02))
    with get_context("spawn").Pool(workers) as pool:
        ms = [m for m, _ in pool.map(_worker, jobs)]
    f1 = float(np.nanmean([m.get("f1", 0.0) for m in ms]))
    sk = float(np.nanmean([m.get("skel_f", 0.0) for m in ms]))
    return dict(full_f1=f1, full_skel_f=sk, full_score=0.5 * (f1 + sk))


def mutate(cfg, rng, p=0.3):
    c = dict(cfg)
    for k, v in SPACE.items():
        if rng.random() < p:
            c[k] = rng.choice(v)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--results", default=os.environ.get("XYLEM_RESULTS", os.path.join(ROOT, "results")))
    ap.add_argument("--n_random", type=int, default=300)
    ap.add_argument("--evo_gens", type=int, default=3)
    ap.add_argument("--evo_pop", type=int, default=30)
    ap.add_argument("--val_limit", type=int, default=64)
    ap.add_argument("--budget_m", type=float, default=0.35, help="M: FLOPs <= budget_m * FLOPs(L)")
    ap.add_argument("--budget_s", type=float, default=0.10, help="S: FLOPs <= budget_s * FLOPs(L)")
    ap.add_argument("--s_param_cap", type=float, default=6.0, help="S: parameters [M]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerank", type=int, default=5, help="top-k per budget re-ranked with the full pipeline")
    ap.add_argument("--full_workers", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    exp_dir = os.path.join(args.results, args.exp)
    dev = "cuda"
    method, targs = load_method(exp_dir, "best", dev)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    vds = FixedDataset(os.path.join(args.data, "syntree", "fixed", "val"), limit=args.val_limit)
    batches = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
               for b in DataLoader(vds, batch_size=8, collate_fn=collate)]
    cache = {}

    def evaluate(cfg):
        key = cfg_key(cfg)
        if key in cache:
            return cache[key]
        torch.manual_seed(0)
        cd, f1 = quality(method, batches, cfg, dev, amp)
        x = torch.rand(1, cfg["n_in"], 3, device=dev)
        fl = flops(lambda z: method(z, cfg), x) / 1e9
        lat, mem = latency(lambda z: method(z, cfg), x, warmup=3, runs=15, dtype=torch.float16)
        n_par = count_params(build_xylemnet(cfg).float()) / 1e6
        r = dict(cfg, key=key, cd_l1=cd, f1=f1, gflops=fl, params_m=n_par, lat_ms_fp16=lat, mem_mb=mem)
        cache[key] = r
        return r

    cands = [MAX_CFG, MIN_CFG] + [sample_cfg(rng) for _ in range(args.n_random)]
    for i, c in enumerate(cands):
        evaluate(c)
        if i % 25 == 0:
            print(f"[search] {i}/{len(cands)}", flush=True)
    L = cache[cfg_key(MAX_CFG)]

    def best_under(frac, cap=None):
        ok = [r for r in cache.values() if r["gflops"] <= frac * L["gflops"] and (cap is None or r["params_m"] <= cap)]
        return max(ok, key=lambda r: r["f1"]) if ok else min(cache.values(), key=lambda r: r["gflops"])

    for frac, cap in ((args.budget_m, None), (args.budget_s, args.s_param_cap)):
        for g in range(args.evo_gens):
            ok = [r for r in cache.values() if r["gflops"] <= frac * L["gflops"] and (cap is None or r["params_m"] <= cap)]
            parents = sorted(ok, key=lambda r: -r["f1"])[:10] or [cache[cfg_key(MIN_CFG)]]
            for _ in range(args.evo_pop):
                p = rng.choice(parents)
                evaluate(mutate({k: p[k] for k in SPACE}, rng))
            print(f"[search] evo budget {frac} gen {g}: best f1 {best_under(frac, cap)['f1']:.4f}", flush=True)
    # re-rank the top candidates of each budget with the *full* pipeline (decoded skeleton + rendered
    # surface), because the node-tube proxy used above is not the final metric
    full_cache = {}

    def full_score(r):
        if r["key"] not in full_cache:
            cfg = {k: r[k] for k in SPACE}
            full_cache[r["key"]] = full_quality(method, batches, cfg, dev, amp, args.full_workers)
            r.update(full_cache[r["key"]])
        return full_cache[r["key"]]["full_score"]

    def best_full(frac, cap=None, top=args.rerank, lo=0.0):
        ok = [r for r in cache.values() if lo * L["gflops"] < r["gflops"] <= frac * L["gflops"]
              and (cap is None or r["params_m"] <= cap)]
        if not ok:     # no candidate in the band: smallest network above the lower bound (never equal to S)
            above = [r for r in cache.values() if r["gflops"] > lo * L["gflops"]]
            return min(above, key=lambda r: r["gflops"]) if above else L
        ok = sorted(ok, key=lambda r: -r["f1"])[:top]
        return max(ok, key=full_score)

    full_score(L)
    S = best_full(args.budget_s, args.s_param_cap)
    # M = balanced model: strictly larger than the S budget (distinct from S), within the M budget
    M = best_full(args.budget_m, lo=args.budget_s)
    if M["key"] == S["key"]:
        M = best_full(args.budget_m, lo=S["gflops"] / L["gflops"] + 1e-6)
    df = pd.DataFrame(list(cache.values()))
    df.to_csv(os.path.join(exp_dir, "search.csv"), index=False)
    chosen = {}
    for name, r in (("L", L), ("M", M), ("S", S)):
        cfg = {k: r[k] for k in SPACE}
        chosen[name] = dict(cfg=cfg, **{k: r[k] for k in ("cd_l1", "f1", "gflops", "params_m", "lat_ms_fp16", "mem_mb")},
                            **{k: r.get(k) for k in ("full_f1", "full_skel_f", "full_score")})
        sub, kw = export_subnet(method, cfg)
        err = verify(method, sub, cfg, dev)
        chosen[name]["export_max_abs_err"] = err
        save_as_experiment(sub.cpu(), kw, cfg, exp_dir + f"_{name}", targs, info=chosen[name])
        print(f"[search] {name}: {cfg}  f1={r['f1']:.4f} GFLOPs={r['gflops']:.2f} params={r['params_m']:.2f}M "
              f"lat={r['lat_ms_fp16']:.1f}ms export_err={err:.2e}", flush=True)
    json.dump(chosen, open(os.path.join(exp_dir, "subnets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
