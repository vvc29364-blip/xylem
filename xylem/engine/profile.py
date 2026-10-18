"""Efficiency profiling: parameters, FLOPs, latency (fp32 / fp16), peak memory."""
import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p is not None)


@torch.no_grad()
def flops(fn, *a):
    fc = FlopCounterMode(display=False)
    with fc:
        fn(*a)
    return fc.get_total_flops()


@torch.no_grad()
def latency(fn, *a, warmup=10, runs=50, dtype=None):
    dev = a[0].device
    if dev.type != "cuda":
        t = time.time()
        for _ in range(5):
            fn(*a)
        return (time.time() - t) / 5 * 1000, 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
        for _ in range(warmup):
            fn(*a)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(runs):
            st.record()
            fn(*a)
            en.record()
            torch.cuda.synchronize()
            times.append(st.elapsed_time(en))
    times.sort()
    return float(sum(times[len(times) // 10: -len(times) // 10 or None]) / max(1, len(times) - 2 * (len(times) // 10))), \
        torch.cuda.max_memory_allocated() / 2 ** 20
