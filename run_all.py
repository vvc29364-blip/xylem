"""Run the complete XylemNet study with one command.

    python run_all.py                       # full study on all visible GPUs
    python run_all.py --budget medium       # half-length schedules
    python run_all.py --budget smoke        # tiny end-to-end check of the whole pipeline
    python run_all.py --gpus 0,1            # restrict GPUs
    python run_all.py --only main,abl       # subset of experiment groups
    python run_all.py --list                # show the job graph and exit

Everything is resumable: finished jobs are skipped (results/_jobs/*.done), interrupted trainings
resume from their last checkpoint.  Logs: logs/jobs/<job>.log.  Final tables/figures: results/report/.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def check_env():
    import importlib
    missing = []
    for mod in ("torch", "numpy", "scipy", "pandas", "matplotlib", "sklearn"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("Missing python packages:", missing, "->  pip install -r requirements.txt")
        sys.exit(1)
    import torch
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name} {p.total_memory / 2 ** 30:.1f} GB")
    try:
        import triton  # noqa
        print("triton available -> fused nearest-neighbour kernels enabled")
    except Exception:
        print("triton not found -> falling back to chunked torch.cdist (slower)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="full", choices=["smoke", "medium", "full"])
    ap.add_argument("--gpus", default="")
    ap.add_argument("--only", default="", help="comma separated groups: data,main,lms,seeds,abl,data,ext,real,eff,report")
    ap.add_argument("--max_cpu_jobs", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--retry_failed", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    n_gpu = check_env()
    # jobs are pinned to *physical* GPU ids: honour an outer CUDA_VISIBLE_DEVICES (e.g. "2,3")
    visible = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip() != ""]
    gpus = [g for g in args.gpus.split(",") if g != ""] or visible or [str(i) for i in range(max(n_gpu, 1))]
    from xylem.experiments import Registry, BUDGETS, results_root
    res_root = os.path.join(ROOT, results_root(args.budget))
    os.environ["XYLEM_RESULTS"] = res_root          # inherited by every job
    have_st = True
    have_b3d = True
    reg = Registry(args.budget, have_st, have_b3d)
    jobs = reg.build()
    if args.only:
        keep = set(args.only.split(",")) | {"data"}
        jobs = {k: v for k, v in jobs.items() if v["group"] in keep}
        for j in jobs.values():
            j["deps"] = [d for d in j["deps"] if d in jobs]
    if args.list:
        for j in jobs.values():
            print(f"[{j['group']:6s}] {j['name']:55s} gpu={j['gpu']} ~{j['hours']:.2f}h deps={len(j['deps'])}")
        tot = sum(j["hours"] for j in jobs.values() if j["gpu"])
        print(f"{len(jobs)} jobs, ~{tot:.0f} GPU-hours -> ~{tot / len(gpus):.0f} h on {len(gpus)} GPU(s)")
        return
    jd = os.path.join(res_root, "_jobs")
    ld = os.path.join(ROOT, "logs", "jobs" if args.budget == "full" else f"jobs_{args.budget}")
    os.makedirs(jd, exist_ok=True)
    os.makedirs(ld, exist_ok=True)
    json.dump(jobs, open(os.path.join(jd, f"jobs_{args.budget}.json"), "w"), indent=1)
    done = lambda n: os.path.exists(os.path.join(jd, safe(n) + ".done"))
    failed = set()
    if not args.retry_failed:
        failed = {n for n in jobs if os.path.exists(os.path.join(jd, safe(n) + ".failed"))}
    running = {}          # name -> (proc, gpu or None, t0, logfile)
    free = list(gpus)
    t_start = time.time()
    print(f"{len(jobs)} jobs, {sum(done(n) for n in jobs)} already done, GPUs {gpus}", flush=True)
    while True:
        # --- collect finished ---
        for n, (p, g, t0, lf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[n]
            if g is not None:
                free.append(g)
            if rc == 0:
                open(os.path.join(jd, safe(n) + ".done"), "w").write(f"{time.time() - t0:.0f}s\n")
                print(f"[ok]   {n}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            else:
                failed.add(n)
                open(os.path.join(jd, safe(n) + ".failed"), "w").write(f"rc={rc}\n")
                print(f"[FAIL] {n}  rc={rc}  see logs/jobs/{safe(n)}.log", flush=True)
        # --- blocked jobs (dependency failed) ---
        blocked = set()
        changed = True
        while changed:
            changed = False
            for n, j in jobs.items():
                if n in blocked or n == "report" or done(n):
                    continue
                if any(d in failed or d in blocked for d in j["deps"]):
                    blocked.add(n)
                    changed = True
        pending = [n for n in jobs if not done(n) and n not in failed and n not in blocked and n not in running]
        if not pending and not running:
            break
        ready = []
        for n in pending:
            j = jobs[n]
            if n == "report":
                others = [m for m in jobs if m != "report"]
                if all(done(m) or m in failed or m in blocked for m in others):
                    ready.append(n)
                continue
            if all(done(d) for d in j["deps"]):
                ready.append(n)
        ready.sort(key=lambda n: -jobs[n]["hours"])       # longest first
        n_cpu = sum(1 for (_, g, _, _) in running.values() if g is None)
        for n in ready:
            j = jobs[n]
            if j["gpu"]:
                if not free:
                    continue
                g = free.pop(0)
            else:
                if n_cpu >= args.max_cpu_jobs:
                    continue
                g = None
                n_cpu += 1
            env = dict(os.environ)
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = g
            env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
            env.setdefault("OMP_NUM_THREADS", "4")
            lf = open(os.path.join(ld, safe(n) + ".log"), "a")
            lf.write(f"\n===== {time.ctime()}  {' '.join(j['cmd'])}\n")
            lf.flush()
            p = subprocess.Popen(j["cmd"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running[n] = (p, g, time.time(), lf)
            print(f"[run]  {n}  on {'GPU' + g if g is not None else 'CPU'}", flush=True)
        time.sleep(5)
    el = (time.time() - t_start) / 3600
    n_done = sum(done(n) for n in jobs)
    print(f"\nfinished in {el:.1f} h: {n_done}/{len(jobs)} jobs done, {len(failed)} failed, "
          f"{len(jobs) - n_done - len(failed)} skipped")
    if failed:
        print("failed jobs:", sorted(failed))
        print("re-run with  python run_all.py --retry_failed  after fixing the cause (logs/jobs/)")
    print(f"tables & figures: {res_root}/report/")


if __name__ == "__main__":
    main()
