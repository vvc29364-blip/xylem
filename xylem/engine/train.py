"""Training entry point (single GPU, resumable).

    python -m xylem.engine.train --exp main/xylem_supernet --method xylem --supernet --iters 60000
    python -m xylem.engine.train --exp main/pointr --method pointr --iters 40000

Supernet training uses the sandwich rule (largest + smallest + n random sub-networks per
step) with in-place distillation from the largest network (BigNAS / OFA style).
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import FixedDataset, OnlineTreeDataset, collate
from ..data.external import external_train_files, RealScanDataset
from ..methods import build_method
from ..models.xylemnet import MAX_CFG, MIN_CFG, sample_cfg
from ..ops import chamfer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", default="xylem")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--results", default=os.environ.get("XYLEM_RESULTS", os.path.join(ROOT, "results")))
    ap.add_argument("--source", default="syntree", help="syntree | synthetic_trees | branch3d")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--supernet", action="store_true")
    ap.add_argument("--n_random", type=int, default=1)
    ap.add_argument("--no_kd", action="store_true")
    ap.add_argument("--model_kw", default="{}")
    ap.add_argument("--loss_w", default="{}")
    ap.add_argument("--fixed_cfg", default="")
    ap.add_argument("--topology", default="flow")
    ap.add_argument("--bot_start", type=float, default=0.2, help="fraction of training before botanical losses")
    ap.add_argument("--sensors", default="tls1,tls3,uls")
    ap.add_argument("--leaf_prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--val_limit", type=int, default=160)
    ap.add_argument("--init", default="", help="initialise weights from a checkpoint (fine-tuning)")
    ap.add_argument("--train_frac", type=float, default=1.0, help="use a fraction of the training trees")
    ap.add_argument("--target", default="", help="unlabelled target source for botanical adaptation (branch3d | weiser)")
    ap.add_argument("--teacher", default="", help="experiment dir of a frozen teacher for knowledge distillation")
    ap.add_argument("--target_margin", type=float, default=0.0,
                    help="BSDA: hinge margin (tree-size units) for nodes/surface far from the observed scan; 0 = off")
    return ap.parse_args()


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def detach(o):
    if torch.is_tensor(o):
        return o.detach()
    if isinstance(o, dict):
        return {k: detach(v) for k, v in o.items()}
    if isinstance(o, list):
        return [detach(v) for v in o]
    return o


def train_files(args):
    files = []
    for src in args.source.split("+"):          # e.g. "syntree+synthetic_trees" (mixed-source training)
        if src == "syntree":
            files += sorted(glob.glob(os.path.join(args.data, "syntree", "trees", "train", "*.npz")))
        else:
            files += external_train_files(args.data, src)
    if args.train_frac < 1.0:
        rng = random.Random(0)
        files = rng.sample(files, max(1, int(len(files) * args.train_frac)))
    return files


def val_dir(args):
    if args.source.split("+")[0] == "syntree":
        return os.path.join(args.data, "syntree", "fixed", "val")
    from ..data.external import ext_root
    return os.path.join(ext_root(args.data, args.source), "fixed", "val")


def forward_loss(method, batch, it, args, cfg=None):
    kind = method.kind
    if kind == "xylem":
        out = method(batch["inp"], cfg)
        return out, method.loss(out, batch)
    if kind == "completion":
        name = type(method).__name__
        out = method(batch["inp"], batch["gt"]) if name in ("PoinTr", "AdaPoinTr") else method(batch["inp"])
        return out, method.loss(out, batch)
    out = method(batch["inp"])
    return out, method.loss(out, batch)


@torch.no_grad()
def validate(method, loader, dev, cfg=None, amp_dtype=None):
    method.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            if method.kind == "xylem":
                out = method(b["inp"], cfg)
                d = method.net.sample_dense(out, 16384)
            elif method.kind == "completion":
                d = method(b["inp"])["dense"]
            else:
                out, (l, _) = forward_loss(method, b, 0, None)
                tot += float(l) * b["inp"].shape[0]
                n += b["inp"].shape[0]
                continue
        l, _, _ = chamfer(d.float(), b["gt"].float())
        tot += float(l) * b["inp"].shape[0]
        n += b["inp"].shape[0]
    method.train()
    return tot / max(n, 1)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out_dir = os.path.join(args.results, args.exp)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "done.flag")):
        print("already trained:", out_dir)
        return
    with open(os.path.join(out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    if args.method in ("geobin", "circlefit"):        # classical baselines: nothing to train
        open(os.path.join(out_dir, "done.flag"), "w").write("classical")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fixed_cfg.startswith("@"):          # "@results/.../subnets.json:S" -> cfg chosen by the search
        path, key = args.fixed_cfg[1:].rsplit(":", 1)
        args.fixed_cfg = json.dumps(json.load(open(path))[key]["cfg"])
        with open(os.path.join(out_dir, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
    fixed_cfg = json.loads(args.fixed_cfg) if args.fixed_cfg else None
    method = build_method(args.method, json.loads(args.model_kw), json.loads(args.loss_w), fixed_cfg, args.topology).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        sd = sd.get("model", sd)
        missing = method.load_state_dict(sd, strict=False)
        print("init from", args.init, missing)
    n_par = sum(p.numel() for p in method.parameters())
    print(f"[{args.exp}] {args.method} params {n_par / 1e6:.2f}M", flush=True)
    params = [p for p in method.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=(0.9, 0.999))
    use_amp = args.amp != "none" and dev.type == "cuda"
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.bfloat16 if (args.amp == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    start = 0
    best = float("inf")
    ck = os.path.join(out_dir, "ckpt_last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu")
        method.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        start, best = s["iter"], s.get("best", best)
        print("resumed at", start)
    if args.source == "branch3d":             # real scans are used as they are (no virtual scanner)
        ds = RealScanDataset(args.data, "branch3d", "train", augment=True)
        ds.files = ds.files * max(1, math.ceil(args.iters * args.bs / max(len(ds.files), 1)) + 1)
    else:
        files = train_files(args)
        ds = OnlineTreeDataset(files, sensors=tuple(args.sensors.split(",")), leaf_prob=args.leaf_prob,
                               repeat=max(1, math.ceil(args.iters * args.bs / max(len(files), 1)) + 1), seed=args.seed + start)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=args.workers > 0, pin_memory=True,
                    prefetch_factor=4 if args.workers > 0 else None)
    tgt_iter = None
    if args.target:
        if args.target == "weiser":
            from ..data.weiser import WeiserTargetDataset
            tds = WeiserTargetDataset(args.data)
        else:
            tds = RealScanDataset(args.data, args.target, "train", augment=True)
        tdl = DataLoader(tds, batch_size=max(2, args.bs // 2), shuffle=True, num_workers=2, collate_fn=collate,
                         drop_last=True, persistent_workers=True)

        def _cycle(loader):
            while True:
                for b in loader:
                    yield b
        tgt_iter = _cycle(tdl)
    teacher = None
    if args.teacher:
        from .evaluate import load_method
        teacher, _ = load_method(args.teacher, "best", dev)
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print("distilling from", args.teacher, flush=True)
    vds = FixedDataset(val_dir(args), limit=args.val_limit)
    vdl = DataLoader(vds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate)
    log_f = open(os.path.join(out_dir, "log.csv"), "a", newline="")
    writer = None
    method.train()
    it = start
    t0 = time.time()
    data_iter = iter(dl)
    eval_cfg = None if fixed_cfg else (MAX_CFG if args.method == "xylem" else None)
    while it < args.iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        batch = to_dev(batch, dev)
        lr = args.lr * min(1.0, (it + 1) / args.warmup) * (0.5 * (1 + math.cos(math.pi * it / args.iters)) * 0.99 + 0.01)
        for g in opt.param_groups:
            g["lr"] = lr
        if method.kind == "xylem":
            method.bot_on = it >= args.bot_start * args.iters
        opt.zero_grad(set_to_none=True)
        logs = {}
        if method.kind == "xylem" and args.supernet:
            cfgs = [MAX_CFG, MIN_CFG] + [sample_cfg() for _ in range(args.n_random)]
            teacher = None
            for ci, cfg in enumerate(cfgs):
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out, (loss, lg) = forward_loss(method, batch, it, args, cfg)
                    if ci > 0 and not args.no_kd:
                        loss = loss + method.kd(out, teacher)
                scaler.scale(loss / len(cfgs)).backward()
                if ci == 0:
                    teacher = detach(out)
                    logs = lg
                elif ci == 1:
                    logs["min_total"] = lg["total"]
            del teacher
        else:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out, (loss, logs) = forward_loss(method, batch, it, args)
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(batch["inp"])
                    l_kd = method.kd(out, t_out)
                    loss = loss + l_kd
                    logs["kd"] = l_kd.detach()
                if tgt_iter is not None:
                    from .adapt import target_loss
                    tb = to_dev(next(tgt_iter), dev)
                    lt, tl = target_loss(method, tb, margin=args.target_margin)
                    loss = loss + lt
                    logs.update(tl)
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(method.parameters(), args.clip)
        if torch.isfinite(gn):
            scaler.step(opt)
        scaler.update()
        it += 1
        if it % 50 == 0 or it == 1:
            row = dict(iter=it, lr=lr, sec=time.time() - t0, gnorm=float(gn), **{k: float(v) for k, v in logs.items()})
            if writer is None:
                writer = csv.DictWriter(log_f, fieldnames=list(row.keys()), extrasaction="ignore")
                if start == 0:
                    writer.writeheader()
            writer.writerow(row)
            log_f.flush()
            if it % 500 == 0 or it == 1:
                print(f"[{args.exp}] it {it}/{args.iters} lr {lr:.2e} " + " ".join(f"{k}={float(v):.4f}" for k, v in list(logs.items())[:8])
                      + f" {(time.time() - t0) / max(it - start, 1):.2f}s/it", flush=True)
        if it % args.val_every == 0 or it == args.iters:
            v = validate(method, vdl, dev, eval_cfg, amp_dtype)
            print(f"[{args.exp}] it {it} val {v:.5f} (best {best:.5f})", flush=True)
            state = dict(model=method.state_dict(), opt=opt.state_dict(), iter=it, best=min(best, v))
            if v < best:
                best = v
                torch.save(dict(model=method.state_dict(), iter=it, val=v), os.path.join(out_dir, "ckpt_best.pt"))
            torch.save(state, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    if not os.path.exists(os.path.join(out_dir, "ckpt_best.pt")):
        torch.save(dict(model=method.state_dict(), iter=it), os.path.join(out_dir, "ckpt_final.pt"))
    with open(os.path.join(out_dir, "done.flag"), "w") as f:
        f.write("best_val=%s" % best)
    if os.path.exists(ck):              # optimizer state is only needed for resuming
        os.remove(ck)
    print("done", out_dir)


if __name__ == "__main__":
    main()
