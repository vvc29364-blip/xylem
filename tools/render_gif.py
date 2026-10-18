"""Animated renders of a XylemNet reconstruction (used for the GIFs in the README).

    python tools/render_gif.py --exp results/main/xylem_L_ft \
        --scan data/syntree/fixed/test_seen/tls1/birch_00059.npz --out assets/birch.gif
    python tools/render_gif.py ... --mode grow      # skeleton revealed from the root outwards

The network is run on the given cached test scan; nothing is post-edited.  Three panels:
input scan (wood / leaf colours when the scan carries labels), decoded skeleton coloured by the
along-tree distance to the root, and the surface rendered on that skeleton.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xylem.data.dataset import collate  # noqa: E402
from xylem.engine.evaluate import load_method  # noqa: E402
from xylem.methods import predict  # noqa: E402

BG = "#f6f6f4"
WOOD, LEAF, BARK = "#7a5230", "#5aa04a", "#6b5544"


def root_dist(pos, par):
    n = len(pos)
    d = np.zeros(n)
    ch = [[] for _ in range(n)]
    for i in range(n):
        if par[i] >= 0:
            ch[par[i]].append(i)
    st = [i for i in range(n) if par[i] < 0]
    while st:
        i = st.pop()
        for c in ch[i]:
            d[c] = d[i] + np.linalg.norm(pos[c] - pos[i])
            st.append(c)
    return d


def rot(p, az):
    a = np.deg2rad(az)
    x = p[:, 0] * np.cos(a) - p[:, 1] * np.sin(a)
    y = p[:, 0] * np.sin(a) + p[:, 1] * np.cos(a)
    return x, p[:, 2], y


def scatter(ax, p, colors, s, az):
    u, v, dep = rot(p, az)
    o = np.argsort(-dep)
    t = ((dep - dep.min()) / (np.ptp(dep) + 1e-9))[o]
    c = np.asarray([matplotlib.colors.to_rgb(x) for x in colors]) if isinstance(colors, (list, np.ndarray)) and \
        np.ndim(colors) == 1 else np.asarray(colors)[:, :3]
    c = c[o] * (0.6 + 0.4 * (1 - t))[:, None] + 0.4 * t[:, None] * np.array(matplotlib.colors.to_rgb(BG))
    ax.scatter(u[o], v[o], s=s, c=np.clip(c, 0, 1), lw=0)


def frame(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.03, 2 * lim - 0.03)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment directory with train_args.json and ckpt_best.pt")
    ap.add_argument("--scan", required=True, help="cached test scan (.npz) as written by build_syntree / prepare_external")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="rotate", choices=["rotate", "grow"])
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--title", default="")
    ap.add_argument("--px", type=int, default=880, help="width of the GIF in pixels")
    args = ap.parse_args()

    m, a = load_method(args.exp, "best", "cuda" if torch.cuda.is_available() else "cpu")
    dj = os.path.join(args.exp, "decode.json")
    if os.path.exists(dj):
        m.node_thr = json.load(open(dj))["node_thr"]
    d = np.load(args.scan, allow_pickle=True)
    s = {k: d[k] for k in d.files}
    s["name"] = os.path.basename(args.scan)
    batch = collate([s])
    dev = next(m.parameters()).device
    bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        p = predict(m, bt, with_graph=True, seed=0)[0]
    pos, rad, par = p["graph"]
    par = par.astype(int)
    rd = root_dist(pos, par)
    norm = matplotlib.colors.Normalize(0, np.percentile(rd, 99))
    cmap = matplotlib.colormaps["viridis"]
    inp = s["inp"]
    if "seg" in s and s["seg"].max() > 0 and s["seg"].min() == 0:
        cin = np.where(s["seg"] == 1, WOOD, LEAF)
    else:
        cin = np.array([WOOD] * len(inp))
    dense = p["dense"]
    rng = np.random.default_rng(0)
    dense = dense[rng.choice(len(dense), min(12000, len(dense)), replace=False)]
    allp = np.concatenate([inp, pos])
    c = (allp.max(0) + allp.min(0)) / 2
    inp, pos, dense = inp - [c[0], c[1], 0], pos - [c[0], c[1], 0], dense - [c[0], c[1], 0]
    lim = max(np.abs(allp[:, :2] - c[:2]).max(), (allp[:, 2].max()) / 2) * 1.05
    i = np.nonzero(par >= 0)[0]
    lw = np.clip(rad[i] / (np.percentile(rad, 99) + 1e-9) * 2.2, 0.25, 2.2)

    frames = []
    dpi = 100
    W = args.px / dpi
    for f in range(args.frames):
        if args.mode == "rotate":
            az, reveal = 360.0 * f / args.frames, np.inf
        else:
            az, reveal = 30.0, (f + 1) / (args.frames * 0.8) * rd.max()
        fig, axs = plt.subplots(1, 3, figsize=(W, W * 0.42), dpi=dpi, facecolor=BG)
        fig.subplots_adjust(0.01, 0.02, 0.99, 0.88, wspace=0.02)
        scatter(axs[0], inp, cin, 1.2, az)
        u, v, _ = rot(pos, az)
        keep = rd[i] <= reveal
        seg = np.stack([np.c_[u[i], v[i]], np.c_[u[par[i]], v[par[i]]]], 1)[keep]
        lc = LineCollection(seg, linewidths=lw[keep], colors=cmap(norm(rd[i][keep])), capstyle="round")
        axs[1].add_collection(lc)
        if args.mode == "rotate" or reveal >= rd.max():
            scatter(axs[2], dense, np.tile(matplotlib.colors.to_rgba(BARK), (len(dense), 1)), 1.0, az)
        for ax, t in zip(axs, ("input scan", "decoded skeleton", "reconstructed surface")):
            frame(ax, lim)
            ax.set_title(t, fontsize=11, color="#3a3a37", family="serif")
        if args.title:
            fig.text(0.5, 0.965, args.title, ha="center", va="top", fontsize=12, color="#1f1f1d", family="serif")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(img).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        plt.close(fig)
    if args.mode == "grow":
        frames += [frames[-1]] * (args.frames // 3)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    print("wrote", args.out, f"{os.path.getsize(args.out) / 2 ** 20:.1f} MB", len(frames), "frames")


if __name__ == "__main__":
    main()
