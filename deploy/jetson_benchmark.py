"""Benchmark XylemNet on NVIDIA Jetson (Orin NX / AGX Orin 16 GB) with TensorRT FP16.

On the Jetson (JetPack >= 6, TensorRT >= 8.6):
    python3 deploy/jetson_benchmark.py --onnx deploy/onnx/main_xylem_S_ft.onnx --fp16
It (1) builds a TensorRT engine (cached next to the ONNX file), (2) measures network latency,
(3) measures the numpy post-processing (flow-guided arborescence decoding + tube resampling),
(4) writes deploy/jetson_<name>.json -> copy it to results/ and re-run tools/make_report.py
to fill the Jetson column of the efficiency table.
Falls back to onnxruntime (TensorRT / CUDA / CPU EP) if the TensorRT python API is missing.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def build_trt(onnx_path, fp16=True, workspace_gb=4):
    import tensorrt as trt
    engine_path = onnx_path.replace(".onnx", "_fp16.engine" if fp16 else "_fp32.engine")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    if os.path.exists(engine_path):
        return runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    ser = builder.build_serialized_network(net, cfg)
    open(engine_path, "wb").write(ser)
    return runtime.deserialize_cuda_engine(ser)


def run_trt(engine, x, runs):
    import torch  # torch for CUDA buffers (available on Jetson via the NVIDIA wheels)
    ctx = engine.create_execution_context()
    bufs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        bufs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(name, bufs[name].data_ptr())
    bufs["points"].copy_(torch.from_numpy(x))
    stream = torch.cuda.Stream()
    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    outs = {k: v.cpu().numpy() for k, v in bufs.items() if k != "points"}
    return ts, outs, torch.cuda.max_memory_allocated() / 2 ** 20


def run_ort(onnx_path, x, runs):
    import onnxruntime as ort
    prov = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    s = ort.InferenceSession(onnx_path, providers=prov)
    for _ in range(5):
        s.run(None, {"points": x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        o = s.run(None, {"points": x})
        ts.append((time.perf_counter() - t) * 1000)
    names = [o.name for o in s.get_outputs()]
    return ts, dict(zip(names, o)), float("nan"), prov[0]


def postprocess(outs, x, node_thr=0.9):
    """union skeleton decoding (generated nodes + medially projected observations) and
    leaf-aware dense output, exactly as xylem/methods.py:predict."""
    from xylem.models.topology import extract_tree
    sig = lambda z: 1 / (1 + np.exp(-z))
    ok = outs["point_tree"][0] > 0
    P = np.concatenate([outs["node_pos"][0], (x[0] + outs["point_med"][0])[ok]])
    Fl = np.concatenate([outs["node_flow"][0], outs["point_flow"][0][ok]])
    G = np.concatenate([outs["node_geo"][0], outs["point_geo"][0][ok]])
    R = np.exp(np.concatenate([outs["node_logr"][0], outs["point_logr"][0][ok]]))
    En = sig(outs["node_exist"][0])
    E = np.concatenate([np.where(En > node_thr, En, 0.0), sig(outs["point_tree"][0][ok])])
    graph = extract_tree(P, Fl, G, R, E)
    dense = np.concatenate([x[0][outs["point_seg"][0] > 0], outs["dense"][0]])
    return graph, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--sample", default="", help="optional fixed test sample (.npz) as input")
    ap.add_argument("--node_thr", type=float, default=0.9, help="see results/<exp>/decode.json")
    a = ap.parse_args()
    info = json.load(open(os.path.join(os.path.dirname(a.onnx), "export_info.json")))
    n_in = next((i["n_in"] for i in info if os.path.basename(i.get("onnx", "")) == os.path.basename(a.onnx)), 4096)
    if a.sample:
        x = np.load(a.sample)["inp"][None, :n_in].astype(np.float32)
    else:
        x = (np.random.rand(1, n_in, 3).astype(np.float32) - np.array([0.5, 0.5, 0], np.float32))
    try:
        ts, outs, mem = run_trt(build_trt(a.onnx, a.fp16), x, a.runs)
        backend = "tensorrt-fp16" if a.fp16 else "tensorrt-fp32"
    except Exception as ex:
        print("TensorRT python API unavailable/failed -> onnxruntime:", ex)
        ts, outs, mem, backend = run_ort(a.onnx, x, a.runs)
    tp = []
    for _ in range(20):
        t = time.perf_counter()
        postprocess(outs, x, a.node_thr)
        tp.append((time.perf_counter() - t) * 1000)
    res = dict(onnx=os.path.basename(a.onnx), backend=backend, n_in=n_in, net_ms_median=float(np.median(ts)),
               net_ms_p90=float(np.percentile(ts, 90)), post_ms_median=float(np.median(tp)),
               total_ms=float(np.median(ts) + np.median(tp)), fps=1000.0 / float(np.median(ts) + np.median(tp)),
               gpu_mem_mb=mem)
    try:
        res["device"] = open("/proc/device-tree/model").read().strip("\x00")
    except Exception:
        import platform
        res["device"] = platform.node()
    print(json.dumps(res, indent=1))
    out = os.path.join(ROOT, "deploy", "jetson_" + os.path.basename(a.onnx).replace(".onnx", ".json"))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
