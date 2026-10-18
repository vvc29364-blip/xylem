"""Export XylemNet-L/M/S to ONNX for TensorRT (Jetson Orin 16 GB).

    python deploy/export_onnx.py --all            # every available L/M/S checkpoint
    python deploy/export_onnx.py --exp main/xylem_S_ft

The exported graph contains encoder + xylem-field head + skeleton decoder + tube splatting
(fixed phase).  Topology decoding and area-weighted resampling are light numpy post-processing
(deploy/jetson_benchmark.py).  Outputs: node_pos, node_logr, node_flow, node_geo, node_h,
node_exist, dense, point_seg.  The ONNX model is verified against PyTorch with onnxruntime.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xylem.engine.evaluate import load_method  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        n = o["nodes"][-1]
        p = o["point"]
        return (n["pos"], n["logr"], n["flow"], n["geo"], n["h"], n["exist"], o["dense"],
                p["seg"], p["tree"], p["med"], p["flow"], p["geo"], p["logr"])


def export(exp_dir, out_path, opset=17):
    m, a = load_method(exp_dir, "best", "cpu")
    m.eval()
    cfg = json.loads(a["fixed_cfg"]) if a.get("fixed_cfg") else None
    n_in = cfg["n_in"] if cfg else 4096
    w = Wrapper(m).eval()
    x = (torch.rand(1, n_in, 3) - torch.tensor([0.5, 0.5, 0.0])) * torch.tensor([1, 1, 1.0])
    names = ["node_pos", "node_logr", "node_flow", "node_geo", "node_h", "node_exist", "dense",
             "point_seg", "point_tree", "point_med", "point_flow", "point_geo", "point_logr"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kw = dict(input_names=["points"], output_names=names, opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(w, (x,), out_path, dynamo=False, **kw)      # torch >= 2.5: force the TorchScript exporter
    except TypeError:
        torch.onnx.export(w, (x,), out_path, **kw)
    info = dict(exp=exp_dir, n_in=n_in, cfg=cfg, onnx=out_path, size_mb=os.path.getsize(out_path) / 2 ** 20)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        ref = [t.detach().numpy() for t in w(x)]
        got = sess.run(None, {"points": x.numpy()})
        info["max_abs_err_node_pos"] = float(np.abs(ref[0] - got[0]).max())
        info["max_abs_err_dense"] = float(np.abs(ref[6] - got[6]).max())
        # measured CPU runtime (edge proxy): network via onnxruntime + numpy post-processing
        import time
        from deploy.jetson_benchmark import postprocess
        ts = []
        for _ in range(12):
            t0 = time.perf_counter()
            got = sess.run(None, {"points": x.numpy()})
            ts.append(time.perf_counter() - t0)
        outs = dict(zip(names, got))
        tp = []
        for _ in range(5):
            t0 = time.perf_counter()
            postprocess(outs, x.numpy())
            tp.append(time.perf_counter() - t0)
        info["ort_cpu_net_ms"] = 1000 * float(np.median(ts[2:]))
        info["cpu_post_ms"] = 1000 * float(np.median(tp))
    except Exception as ex:
        info["verify"] = f"skipped: {ex}"
    print(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--results", default=os.environ.get("XYLEM_RESULTS", os.path.join(ROOT, "results")))
    ap.add_argument("--out", default=os.path.join(ROOT, "deploy", "onnx"))
    args = ap.parse_args()
    exps = [args.exp] if args.exp else []
    if args.all:
        for v in ("L", "M", "S"):
            for cand in (f"main/xylem_{v}_ft", f"main/xylem_supernet_{v}"):
                if os.path.exists(os.path.join(args.results, cand, "done.flag")):
                    exps.append(cand)
                    break
    infos = []
    os.makedirs(args.out, exist_ok=True)
    for e in exps:
        name = e.replace("/", "_")
        try:
            infos.append(export(os.path.join(args.results, e), os.path.join(args.out, name + ".onnx")))
        except Exception as ex:
            print("[onnx] failed for", e, ex)
            infos.append(dict(exp=e, error=str(ex)))
    json.dump(infos, open(os.path.join(args.out, "export_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
