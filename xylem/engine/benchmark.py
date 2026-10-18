"""Efficiency benchmark of trained methods (batch 1, 4096 input points).

    python -m xylem.engine.benchmark --exps main/pointr,main/xylem_supernet_L,... --out results/efficiency.csv

Reports parameters, GFLOPs, GPU latency (FP32 and FP16 autocast), peak memory and the CPU
post-processing time (skeleton/topology extraction, taken from the evaluation CSVs).
The Jetson numbers are produced separately on the device with deploy/jetson_benchmark.py.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from ..methods import predict
from .evaluate import load_method
from .profile import count_params, flops, latency

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", required=True)
    ap.add_argument("--results", default=os.environ.get("XYLEM_RESULTS", os.path.join(ROOT, "results")))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    rows = []
    for exp in args.exps.split(","):
        d = os.path.join(args.results, exp)
        if not os.path.exists(os.path.join(d, "done.flag")):
            print("skip (not trained)", exp)
            continue
        m, a = load_method(d, "best", dev)
        n_in = json.loads(a["fixed_cfg"])["n_in"] if a.get("fixed_cfg") else 4096
        x = (torch.rand(1, n_in, 3, device=dev) - torch.tensor([0.5, 0.5, 0.0], device=dev))
        if m.kind == "classical":
            fn = None
        elif m.kind == "diffts":
            fn = lambda z: m.sample(z)
        else:
            fn = lambda z: m(z)
        r = dict(exp=exp, method=a["method"], n_in=n_in)
        if fn is not None:
            r["params_m"] = count_params(m) / 1e6
            try:
                r["gflops"] = flops(fn, x) / 1e9
            except Exception as ex:
                r["gflops"] = np.nan
            r["lat_fp32_ms"], r["mem_fp32_mb"] = latency(fn, x, dtype=None)
            r["lat_fp16_ms"], r["mem_fp16_mb"] = latency(fn, x, dtype=torch.float16)
        # post-processing time from the evaluation logs
        cs = glob.glob(os.path.join(d, "eval", "*syn_seen.csv"))
        if cs:
            df = pd.read_csv(cs[0])
            if "t_graph" in df:
                r["post_ms"] = float(df["t_graph"].mean() * 1000)
        rows.append(r)
        print(r, flush=True)
        del m
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(args.out or os.path.join(args.results, "efficiency.csv"), index=False)


if __name__ == "__main__":
    main()
