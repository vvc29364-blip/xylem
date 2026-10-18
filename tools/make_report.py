"""Collect every evaluation into tables (LaTeX + Markdown) and figures.

    python tools/make_report.py
Outputs: results/report/{*.md,*.tex,figs/*.pdf|png} and results/report/SUMMARY.md.
Missing experiments are shown as '--' so the report can be generated at any time.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xylem.methods import DISPLAY, VENUE  # noqa: E402
from xylem.metrics import LOWER_BETTER  # noqa: E402

RES = os.environ.get("XYLEM_RESULTS", os.path.join(ROOT, "results"))
REP = os.path.join(RES, "report")
FIG = os.path.join(REP, "figs")
PT = REP
PF = FIG
for d in (REP, FIG, PT, PF):
    os.makedirs(d, exist_ok=True)

BASE = ["geobin", "pcn", "foldingnet", "topnet", "grnet", "pointr", "adapointr", "snowflakenet", "seedformer",
        "pointattn", "smarttree", "diffts"]
OURS = [("main/xylem_S_ft", "XylemNet-S (ours)"), ("main/xylem_M_ft", "XylemNet-M (ours)"),
        ("main/xylem_L_ft", "XylemNet-L (ours)")]
COLS = dict(cd_l1=r"CD-$\ell_1$", f1="F@1\\%", f05="F@0.5\\%", tbr="TBR", skel_f="Sk-F@1\\%", skel_f2="Sk-F@2\\%",
            skel_cd="Sk-CD", rad_mare="R-MARE", geo_err="Geo-Err", tip_f="Tip-F", junc_f="Jn-F",
            height_err_m="H-err[m]", dbh_err_cm="DBH-err[cm]", vol_err_pct="Vol-err[\\%]", len_err_pct="Len-err[\\%]",
            seg_miou="Seg-mIoU", dcd="DCD")
MD_COLS = {k: v.replace("\\%", "%").replace("$\\ell_1$", "L1").replace("$", "") for k, v in COLS.items()}
SUMMARY = []


def load(exp, set_name, tag="default"):
    if "@" in exp:                     # "exp@tag" selects an evaluation variant
        exp, tag = exp.split("@")
    f = os.path.join(RES, exp, "eval", f"{tag}__{set_name.replace('/', '_')}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def load_csv(exp, set_name, tag="default"):
    if "@" in exp:
        exp, tag = exp.split("@")
    f = os.path.join(RES, exp, "eval", f"{tag}__{set_name.replace('/', '_')}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def main_rows():
    rows = [(f"main/{m}", DISPLAY[m] + (f" [{VENUE[m]}]" if VENUE.get(m) else "")) for m in BASE]
    return rows + OURS


def tex_escape(t):
    """Escape plain-text row names for LaTeX (math segments $...$ are left untouched)."""
    parts = t.split("$")
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("\\", "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    return "$".join(parts)


def fmt(v, k):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if k in ("cd_l1", "skel_cd", "dcd"):
        return f"{v:.2f}" if k != "dcd" else f"{v:.3f}"
    if k.endswith("_m") or k.endswith("_cm") or k.endswith("_pct"):
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
    return f"{100 * v:.1f}" if k in ("f1", "f05", "tbr", "skel_f", "skel_f2", "tip_f", "junc_f", "seg_miou", "prec", "rec") \
        else f"{v:.3f}"


def table(name, rows, cols, caption, set_name=None, tag="default", getter=None, label=None):
    """rows: [(exp or key, display)], getter(exp)->dict of metrics."""
    data = []
    for exp, disp in rows:
        m = getter(exp) if getter else load(exp, set_name, tag)
        data.append((disp, m))
    # best / second best per column
    best = {}
    for k in cols:
        vals = [(m or {}).get(k) for _, m in data]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        s = sorted(set(vals), reverse=k not in LOWER_BETTER)
        best[k] = (s[0], s[1] if len(s) > 1 else None)
    # LaTeX
    arrow = lambda k: r"$\downarrow$" if k in LOWER_BETTER else r"$\uparrow$"
    tex = [r"\begin{table*}[t]\centering\small", r"\caption{" + caption + "}", r"\label{tab:" + (label or name) + "}",
           r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           "Method & " + " & ".join(COLS.get(k, k.replace("_", r"\_")) + arrow(k) for k in cols) + r" \\", r"\midrule"]
    md = [f"### {caption}", "", "| Method | " + " | ".join(MD_COLS.get(k, k) + ("↓" if k in LOWER_BETTER else "↑") for k in cols) + " |",
          "|---" * (len(cols) + 1) + "|"]
    for disp, m in data:
        cells, mcells = [], []
        for k in cols:
            v = (m or {}).get(k)
            s = fmt(v, k)
            ms = s
            if k in best and v is not None and np.isfinite(v):
                if v == best[k][0]:
                    s, ms = r"\textbf{" + s + "}", f"**{s}**"
                elif best[k][1] is not None and v == best[k][1]:
                    s, ms = r"\underline{" + s + "}", f"_{s}_"
            cells.append(s)
            mcells.append(ms)
        if "ours" in disp and disp.startswith("XylemNet-S"):
            tex.append(r"\midrule")
        tex.append(tex_escape(disp) + " & " + " & ".join(cells) + r" \\")
        md.append(f"| {disp} | " + " | ".join(mcells) + " |")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(PT, name + ".tex"), "w").write("\n".join(tex))
    open(os.path.join(REP, name + ".tex"), "w").write("\n".join(tex))
    open(os.path.join(REP, name + ".md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return data


# ------------------------------------------------------------------------------------------
def t_main():
    cols = ["cd_l1", "f1", "f05", "tbr", "skel_f", "skel_cd", "rad_mare", "geo_err", "path_score", "junc_f"]
    COLS["path_score"] = "Path"
    MD_COLS["path_score"] = "Path"
    rows = main_rows()
    rows = rows[:-3] + [(f"main/{m}@qsm", DISPLAY[m] + " +QSM") for m in ("pointr", "adapointr", "snowflakenet",
                                                                         "seedformer")] + rows[-3:]
    d = table("tab_main", rows, cols, "Tree reconstruction on SynTree (seen species, all sensors, leaf-on/off). "
              "CD-$\\ell_1\\times10^3$; F-scores, TBR (recall of twigs $<$1\\,cm) and skeleton scores in \\%.", "syn_seen")
    best_b = None
    for disp, m in d:
        if m and "ours" not in disp and (best_b is None or m.get("f1", 0) > best_b[1].get("f1", 0)):
            best_b = (disp, m)
    ours = [(disp, m) for disp, m in d if m and "XylemNet-L" in disp]
    if best_b and ours:
        o = ours[0][1]
        SUMMARY.append(f"* SynTree (seen): XylemNet-L F@1% = {100 * o.get('f1', np.nan):.1f} vs best baseline "
                       f"{best_b[0]} {100 * best_b[1].get('f1', np.nan):.1f}; twig recall TBR {100 * o.get('tbr', np.nan):.1f} vs "
                       f"{100 * best_b[1].get('tbr', np.nan):.1f}; skeleton F {100 * o.get('skel_f', np.nan):.1f} vs "
                       f"{100 * best_b[1].get('skel_f', np.nan):.1f}.")


def t_forestry():
    cols = ["height_err_m", "dbh_err_cm", "vol_err_pct", "len_err_pct", "rad_mare", "seg_miou"]
    table("tab_forestry", main_rows(), cols, "Forestry attributes derived from the reconstructions (SynTree, seen species).",
          "syn_seen")


def t_general():
    cols_sets = [("syn_unseen", "Unseen species"), ("st_test", "Synthetic-Trees"), ("st_foliage_test", "Synthetic-Trees + foliage")]

    def getter(exp):
        out = {}
        for s, _ in cols_sets:
            m = load(exp, s) or {}
            for k in ("cd_l1", "f1", "skel_f"):
                if k in m:
                    out[f"{s}:{k}"] = m[k]
        return out
    cols = [f"{s}:{k}" for s, _ in cols_sets for k in ("cd_l1", "f1", "skel_f")]
    for s, n in cols_sets:
        for k in ("cd_l1", "f1", "skel_f"):
            COLS[f"{s}:{k}"] = f"{n[:10]} {COLS[k]}"
            MD_COLS[f"{s}:{k}"] = f"{n} {MD_COLS[k]}"
            if k in LOWER_BETTER:
                LOWER_BETTER.add(f"{s}:{k}")
    table("tab_generalization", main_rows(), cols, "Zero-shot generalisation: unseen species (willow, acacia) and a "
          "different generator (Synthetic-Trees / SpeedTree) with and without foliage.", getter=getter)


def t_breakdown():
    keys = [("sensor", "tls1"), ("sensor", "tls3"), ("sensor", "uls"), ("leaf_on", "False"), ("leaf_on", "True")]

    def getter(exp):
        m = load(exp, "syn_seen") or {}
        out = {}
        for grp, val in keys:
            g = m.get(f"by_{grp}", {}).get(val, {})
            for k in ("f1", "skel_f"):
                if k in g:
                    out[f"{grp}={val}:{k}"] = g[k]
        return out
    cols = []
    for grp, val in keys:
        for k in ("f1", "skel_f"):
            c = f"{grp}={val}:{k}"
            cols.append(c)
            nm = {"tls1": "TLS-1", "tls3": "TLS-3", "uls": "ULS", "False": "leaf-off", "True": "leaf-on"}[val]
            COLS[c] = f"{nm} {COLS[k]}"
            MD_COLS[c] = f"{nm} {MD_COLS[k]}"
    table("tab_breakdown", main_rows(), cols, "Breakdown by sensor and foliage (SynTree, seen species).", getter=getter)


def t_real():
    rows = [(f"main/{m}", DISPLAY[m] + " (zero-shot)") for m in BASE] + \
           [(f"real/{m}_sft", DISPLAY[m] + " (fine-tuned)") for m in ("pointr", "snowflakenet", "smarttree", "diffts")] + \
           [("real/diffts_scratch", "DiffTS* (trained on BRANCH3D)"),
            ("main/xylem_S_ft", "XylemNet-S (zero-shot)"), ("real/xylem_S_bsda", "XylemNet-S + BSDA (label-free)"),
            ("main/xylem_L_ft", "XylemNet-L (zero-shot)"), ("real/xylem_bsda", "XylemNet-L + BSDA (label-free)"),
            ("real/xylem_sft", "XylemNet-L (fine-tuned)")]
    cols = ["skel_cd", "skel_f", "skel_f2", "tip_f", "junc_f", "geo_err", "f1"]
    d = table("tab_real", rows, cols, "Real orchard trees (BRANCH3D, ICCV'25). Reference skeletons only (no radii); "
              "F@1\\% here measures agreement with the observed wood points. BSDA = botanical self-supervised adaptation "
              "(no labels).", "branch3d_test")
    dd = {disp: m for disp, m in d if m}
    a, b = dd.get("XylemNet-L (zero-shot)"), dd.get("XylemNet-L + BSDA (label-free)")
    if a and b:
        SUMMARY.append(f"* BRANCH3D (real): skeleton F@1% zero-shot {100 * a['skel_f']:.1f} -> BSDA {100 * b['skel_f']:.1f} "
                       f"(label-free).")


def t_ablation():
    rows = [("abl/full", "Full model"), ("abl/no_tube", "w/o tubular splatting (folding patches)"),
            ("abl/no_growth", "w/o hierarchical growth"), ("abl/no_anchor", "w/o observation anchors"),
            ("abl/no_learned_q", "w/o occlusion queries"), ("abl/no_field", "w/o xylem field head"),
            ("abl/no_rootflow", "w/o root-flow/geodesic (MST topology)"), ("abl/no_bot", "w/o botanical losses"),
            ("abl/no_pipe", "w/o pipe-model loss"), ("abl/single_order", "single serialization order"),
            ("abl/cd_only", "CD only (no structural supervision)"),
            ("abl/full@nomerge", "w/o leaf-aware observation merging"),
            ("abl/full@nodesonly", "skeleton from generated nodes only"),
            ("abl/full@nodetubes", "surface from node tubes instead of the decoded tree"),
            ("abl/full@norefine", "w/o observation-refined radii"),
            ("abl/no_leaf_field", "w/o foliage-aware field (wood-only medial)")]

    def getter(exp):
        exp, tag = exp.split("@") if "@" in exp else (exp, "default")
        m = dict(load(exp, "syn_seen", tag) or {})
        u = load(exp, "syn_unseen", tag) or {}
        n = load(exp, "robust/noise_0.04", tag) or {}
        if u:
            m["unseen_f1"] = u.get("f1")
        if n:
            m["noise_f1"] = n.get("f1")
        return m
    COLS.update(unseen_f1="Unseen F@1\\%", noise_f1="Noise(4cm) F@1\\%")
    MD_COLS.update(unseen_f1="Unseen F@1%", noise_f1="Noise(4cm) F@1%")
    cols = ["cd_l1", "f1", "tbr", "skel_f", "rad_mare", "geo_err", "junc_f", "unseen_f1", "noise_f1"]
    table("tab_ablation", rows, cols, "Ablation study (standalone L architecture, shorter schedule).", getter=getter)
    # topology decoding ablation (same checkpoint)
    rows = [("flow", "Flow-guided arborescence (ours)"), ("mst", "Euclidean MST (DiffTS-style)")]
    table("tab_topology", rows, ["skel_f", "geo_err", "tip_f", "junc_f"], "Topology decoding on identical node predictions.",
          getter=lambda t: load("abl/full", "syn_seen", "default" if t == "flow" else "mst"))


def t_lms():
    sub = {}
    f = os.path.join(RES, "main", "xylem_supernet", "subnets.json")
    if os.path.exists(f):
        sub = json.load(open(f))
    eff = pd.read_csv(os.path.join(RES, "efficiency.csv")) if os.path.exists(os.path.join(RES, "efficiency.csv")) else None
    jet = {}
    for jf in glob.glob(os.path.join(ROOT, "deploy", "jetson_*.json")):
        j = json.load(open(jf))
        jet[j["onnx"]] = j
    rows = []
    for v in ("L", "M", "S"):
        rows += [(f"main/xylem_supernet_{v}", f"XylemNet-{v}: cut from supernet"),
                 (f"main/xylem_{v}_ft", f"XylemNet-{v}: cut + short fine-tune"),
                 (f"lms/xylem_{v}_scratch", f"XylemNet-{v}: trained from scratch")]
    rows.append(("lms/xylem_S_kd", "XylemNet-S: from scratch + KD from L"))
    ort = {}
    fo = os.path.join(ROOT, "deploy", "onnx", "export_info.json")
    if os.path.exists(fo):
        for i in json.load(open(fo)):
            if "ort_cpu_net_ms" in i:
                ort[os.path.basename(str(i.get("exp", ""))).replace("\\", "/")] = i["ort_cpu_net_ms"] + i.get("cpu_post_ms", 0)

    def getter(exp):
        m = dict(load(exp, "syn_seen") or {})
        v = exp.split("xylem_")[-1][0] if "supernet" not in exp else exp[-1]
        for k_, ms_ in ort.items():
            if exp.endswith(k_) or k_.endswith(exp.split("/")[-1]):
                m["cpu_ms"] = ms_
        s = sub.get(v, {})
        m["params_m"], m["gflops"] = s.get("params_m"), s.get("gflops")
        if eff is not None and (eff["exp"] == exp).any():
            r = eff[eff["exp"] == exp].iloc[0]
            m["lat_fp16_ms"] = r.get("lat_fp16_ms")
        j = jet.get(exp.replace("/", "_") + ".onnx")
        if j:
            m["jetson_ms"] = j["total_ms"]
        return m
    COLS.update(params_m="Params[M]", gflops="GFLOPs", lat_fp16_ms="GPU[ms]", jetson_ms="Jetson[ms]", cpu_ms="CPU(ORT)[ms]")
    MD_COLS["cpu_ms"] = "CPU(ORT)[ms]"
    LOWER_BETTER.add("cpu_ms")
    MD_COLS.update(params_m="Params[M]", gflops="GFLOPs", lat_fp16_ms="GPU[ms]", jetson_ms="Jetson[ms]")
    for k in ("params_m", "gflops", "lat_fp16_ms", "jetson_ms"):
        LOWER_BETTER.add(k)
    table("tab_lms", rows, ["params_m", "gflops", "lat_fp16_ms", "cpu_ms", "jetson_ms", "cd_l1", "f1", "tbr", "skel_f"],
          "One supernet cut into three deployable models vs. three separately trained models. "
          "GPU: batch 1, FP16; Jetson: Orin 16\\,GB, TensorRT FP16 incl. post-processing.", getter=getter)
    # KD ablation
    rows = [(("abl/supernet_kd", t), f"with in-place KD, {t}") for t in ("max", "mid", "min")] + \
           [(("abl/supernet_nokd", t), f"without KD, {t}") for t in ("max", "mid", "min")]
    table("tab_supernet_kd", rows, ["cd_l1", "f1", "tbr", "skel_f"], "Supernet training with and without in-place distillation.",
          getter=lambda e: load(e[0], "syn_seen", e[1]))


def t_efficiency():
    f = os.path.join(RES, "efficiency.csv")
    if not os.path.exists(f):
        return
    eff = pd.read_csv(f)
    rows = [(r["exp"], r["exp"]) for _, r in eff.iterrows()]

    def getter(exp):
        r = eff[eff["exp"] == exp].iloc[0].to_dict()
        m = load(exp, "syn_seen") or {}
        r["f1"] = m.get("f1")
        return r
    COLS.update(lat_fp32_ms="GPU fp32[ms]", mem_fp16_mb="Mem[MB]", post_ms="Post[ms]")
    MD_COLS.update(lat_fp32_ms="GPU fp32[ms]", mem_fp16_mb="Mem[MB]", post_ms="Post[ms]")
    for k in ("lat_fp32_ms", "mem_fp16_mb", "post_ms"):
        LOWER_BETTER.add(k)
    names = {f"main/{m}": DISPLAY[m] for m in BASE}
    names.update({e: d for e, d in OURS})
    names.update({f"main/xylem_supernet_{v}": f"XylemNet-{v} (cut)" for v in "LMS"})
    rows = [(e, names.get(e, e)) for e, _ in rows]
    table("tab_efficiency", rows, ["params_m", "gflops", "lat_fp32_ms", "lat_fp16_ms", "mem_fp16_mb", "post_ms", "f1"],
          "Efficiency (batch 1, 4096 points, server GPU).", getter=getter)


def t_seeds():
    groups = {"PoinTr": ["main/pointr"] + [f"seeds/pointr_seed{s}" for s in (1, 2)],
              "SnowflakeNet": ["main/snowflakenet"] + [f"seeds/snowflakenet_seed{s}" for s in (1, 2)],
              "Smart-Tree*": ["main/smarttree"] + [f"seeds/smarttree_seed{s}" for s in (1, 2)],
              "XylemNet-L (scratch)": ["lms/xylem_L_scratch"] + [f"seeds/xylem_L_seed{s}" for s in (1, 2)]}
    md = ["### Seed variance (mean ± std over seeds, SynTree seen / unseen)", "",
          "| Method | #seeds | F@1% seen | Sk-F seen | F@1% unseen | Sk-F unseen |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]\centering\small\caption{Mean $\pm$ std over training seeds.}\label{tab:seeds}",
           r"\begin{tabular}{lccccc}\toprule Method & \#seeds & F@1\% seen & Sk-F seen & F@1\% unseen & Sk-F unseen\\\midrule"]
    for name, exps in groups.items():
        cells = []
        n = 0
        for s in ("syn_seen", "syn_unseen"):
            for k in ("f1", "skel_f"):
                v = [load(e, s)[k] for e in exps if load(e, s) and k in load(e, s)]
                n = max(n, len(v))
                cells.append(f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}" if v else "--")
        md.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
        tex.append(f"{name} & {n} & " + " & ".join(c.replace("±", r"$\pm$") for c in cells) + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(REP, "tab_seeds.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(PT, "tab_seeds.tex"), "w").write("\n".join(tex))


def t_data_ext():
    rows = []
    for m in ("xylem", "pointr", "snowflakenet"):
        for f in ("0.1", "0.25"):
            rows.append((f"data/{m}_frac{f}", f"{DISPLAY[m]} {int(float(f) * 100)}% trees"))
        rows.append((f"main/{m}" if m != "xylem" else "abl/full", f"{DISPLAY[m]} 100% trees"))
    table("tab_data_eff", rows, ["cd_l1", "f1", "tbr", "skel_f"], "Data efficiency (fraction of SynTree training trees).",
          "syn_seen")
    rows = [(f"ext/st_{m}", DISPLAY[m]) for m in ("geobin", "pointr", "seedformer", "snowflakenet", "smarttree", "diffts", "xylem")]

    def getter(exp):
        a = load(exp, "st_test") or {}
        b = load(exp, "st_foliage_test") or {}
        return {"a_f1": a.get("f1"), "a_skel_f": a.get("skel_f"), "a_rad": a.get("rad_mare"),
                "b_f1": b.get("f1"), "b_skel_f": b.get("skel_f"), "b_rad": b.get("rad_mare")}
    COLS.update(a_f1="ST F@1\\%", a_skel_f="ST Sk-F", a_rad="ST R-MARE", b_f1="ST+fol. F@1\\%", b_skel_f="ST+fol. Sk-F",
                b_rad="ST+fol. R-MARE")
    MD_COLS.update(a_f1="ST F@1%", a_skel_f="ST Sk-F", a_rad="ST R-MARE", b_f1="ST+fol. F@1%", b_skel_f="ST+fol. Sk-F",
                   b_rad="ST+fol. R-MARE")
    LOWER_BETTER.update({"a_rad", "b_rad"})
    table("tab_synthetic_trees", rows, ["a_f1", "a_skel_f", "a_rad", "b_f1", "b_skel_f", "b_rad"],
          "In-domain training and testing on Synthetic-Trees (Dobbs et al.).", getter=getter)


# ------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def save(fig, name):
    for d in (FIG, PF):
        fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def f_radius_bins():
    bins = ["rec_r0-0.01", "rec_r0.01-0.03", "rec_r0.03-0.1", "rec_r0.1-10"]
    labels = ["<1 cm", "1-3 cm", "3-10 cm", ">10 cm"]
    meths = [(f"main/{m}", DISPLAY[m]) for m in ("geobin", "pointr", "adapointr", "snowflakenet", "seedformer", "smarttree", "diffts")] + OURS[2:]
    data = [(d, load(e, "syn_seen")) for e, d in meths]
    data = [(d, m) for d, m in data if m]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.8 / len(data)
    for i, (d, m) in enumerate(data):
        ax.bar(np.arange(4) + i * w, [100 * m.get(b, np.nan) for b in bins], w, label=d)
    ax.set_xticks(np.arange(4) + 0.4 - w / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("surface recall @1% [%]")
    ax.set_xlabel("branch radius")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_radius_recall")


def f_robust():
    facs = [("noise", [0.0, 0.01, 0.02, 0.04, 0.08], "range noise σ [m]"),
            ("outlier", [0.0, 0.02, 0.05, 0.1], "outlier ratio"),
            ("density", [1.0, 0.5, 0.25, 0.1], "point density")]
    meths = [(f"main/{m}", DISPLAY[m]) for m in ("geobin", "pointr", "snowflakenet", "seedformer", "smarttree", "diffts")] + OURS
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.5))
    any_ = False
    for j, (f, lv, xl) in enumerate(facs):
        for r, key in enumerate(("f1", "skel_f")):
            ax = axs[r, j]
            for i, (e, d) in enumerate(meths):
                ys = [(load(e, f"robust/{f}_{l}") or {}).get(key, np.nan) for l in lv]
                if np.all(np.isnan(ys)):
                    continue
                any_ = True
                ax.plot(lv, 100 * np.array(ys), "o-", ms=3, label=d, lw=2.2 if "ours" in d else 1.2)
            ax.set_xlabel(xl)
            ax.set_ylabel({"f1": "F@1% [%]", "skel_f": "Skeleton F@1% [%]"}[key])
            ax.grid(alpha=0.3)
            if f == "density":
                ax.invert_xaxis()
    if any_:
        axs[0, 0].legend(fontsize=6)
        save(fig, "fig_robustness")
    else:
        plt.close(fig)


def f_pareto():
    f = os.path.join(RES, "main", "xylem_supernet", "search.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(df["gflops"], 100 * df["f1"], s=8, c="lightgray", label="sub-networks of the supernet")
    sub = json.load(open(os.path.join(RES, "main", "xylem_supernet", "subnets.json")))
    for v, c in zip("LMS", COLORS):
        ax.scatter(sub[v]["gflops"], 100 * sub[v]["f1"], s=60, c=c, marker="*", label=f"XylemNet-{v}", zorder=3)
    ef = os.path.join(RES, "efficiency.csv")
    if os.path.exists(ef):
        eff = pd.read_csv(ef)
        for _, r in eff.iterrows():
            e = r["exp"]
            if e.startswith("main/") and "xylem" not in e:
                m = load(e, "syn_val") or load(e, "syn_seen")
                if m and np.isfinite(r.get("gflops", np.nan)):
                    ax.scatter(r["gflops"], 100 * m["f1"], s=20, marker="x", c="k")
                    ax.annotate(DISPLAY.get(e.split("/")[1], e), (r["gflops"], 100 * m["f1"]), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("GFLOPs (batch 1)")
    ax.set_ylabel("F@1% [%] (validation)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "fig_pareto")


def _draw(ax, pts, graph=None, title="", color="k"):
    ax.scatter(pts[:, 0], pts[:, 2], s=0.05, c=color, rasterized=True)
    if graph is not None and len(graph[0]) > 1:
        p, r, par = graph
        c = np.nonzero(par >= 0)[0]
        from matplotlib.collections import LineCollection
        segs = np.stack([p[c][:, [0, 2]], p[par[c]][:, [0, 2]]], 1)
        lw = np.clip(r[c] * 120, 0.15, 1.5)
        ax.add_collection(LineCollection(segs, linewidths=lw, colors="tab:red", alpha=0.8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7)


def f_qualitative(set_name="syn_seen", n=4, name="fig_qualitative"):
    meths = [(f"main/{m}", DISPLAY[m]) for m in ("geobin", "pointr", "snowflakenet", "seedformer", "smarttree", "diffts")] + \
            [(e, d.replace(" (ours)", "")) for e, d in OURS[2:]]
    ours_dir = os.path.join(RES, OURS[2][0], "eval", f"pred__default__{set_name}")
    files = sorted(f for f in glob.glob(os.path.join(ours_dir, "*.npz")) if not ("worst" in np.load(f).files and bool(np.load(f)["worst"])))
    if not files:
        return
    files = files[:: max(1, len(files) // n)][:n]
    ncol = len(meths) + 2
    fig, axs = plt.subplots(len(files), ncol, figsize=(1.6 * ncol, 2.4 * len(files)))
    axs = np.atleast_2d(axs)
    for i, f in enumerate(files):
        base = os.path.basename(f)
        d = np.load(f)
        _draw(axs[i, 0], d["inp"], title="Input (partial scan)" if i == 0 else "", color="tab:green")
        _draw(axs[i, 1], d["gt"], (d["g_pos"], d["g_rad"], d["g_par"]), "Ground truth" if i == 0 else "")
        for j, (e, disp) in enumerate(meths):
            pf = os.path.join(RES, e, "eval", f"pred__default__{set_name}", base)
            ax = axs[i, j + 2]
            if os.path.exists(pf):
                p = np.load(pf)
                g = (p["p_pos"], p["p_rad"], p["p_par"]) if len(p["p_pos"]) else None
                _draw(ax, p["dense"], g, disp if i == 0 else "")
            else:
                ax.axis("off")
    save(fig, name)


def f_species():
    meths = [(f"main/{m}", DISPLAY[m]) for m in ("pointr", "snowflakenet", "smarttree", "diffts")] + OURS[2:]
    fig, ax = plt.subplots(figsize=(9, 3))
    sp_all = None
    for i, (e, d) in enumerate(meths):
        m = load(e, "syn_seen")
        u = load(e, "syn_unseen")
        if not m:
            continue
        by = dict(m.get("by_species", {}))
        if u:
            by.update(u.get("by_species", {}))
        sp_all = sp_all or list(by.keys())
        ax.bar(np.arange(len(sp_all)) + i * 0.14, [100 * by.get(s, {}).get("f1", np.nan) for s in sp_all], 0.14, label=d)
    if sp_all:
        ax.set_xticks(np.arange(len(sp_all)) + 0.35)
        ax.set_xticklabels(sp_all, rotation=30, fontsize=7)
        ax.set_ylabel("F@1% [%]")
        ax.legend(fontsize=6, ncol=3)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "fig_species")
    else:
        plt.close(fig)


def f_curves():
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ok = False
    for e, d in [("main/xylem_supernet", "XylemNet supernet"), ("main/pointr", "PoinTr"), ("main/snowflakenet", "SnowflakeNet")]:
        f = os.path.join(RES, e, "log.csv")
        if os.path.exists(f):
            df = pd.read_csv(f)
            col = "dense_cd" if "dense_cd" in df else ("cd1" if "cd1" in df else "total")
            ax.plot(df["iter"], df[col].rolling(20, min_periods=1).mean(), label=f"{d} ({col})")
            ok = True
    if ok:
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("training CD-$\\ell_1$")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        save(fig, "fig_training")
    else:
        plt.close(fig)


def main():
    for fn in (t_main, t_forestry, t_general, t_breakdown, t_real, t_ablation, t_lms, t_efficiency, t_seeds, t_data_ext,
               f_radius_bins, f_robust, f_pareto, f_qualitative, f_species, f_curves,
               lambda: f_qualitative("branch3d_test", 4, "fig_qualitative_real"),
               lambda: f_qualitative("syn_unseen", 3, "fig_qualitative_unseen")):
        try:
            fn()
        except Exception as ex:
            print("[report] skipped", getattr(fn, "__name__", "fig"), ":", repr(ex)[:200])
    try:
        import report_extra
        report_extra.run_all()
    except Exception as ex:
        print("[report] extra section failed:", repr(ex)[:300])
    mds = sorted(glob.glob(os.path.join(REP, "tab_*.md")))
    with open(os.path.join(REP, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("# XylemNet - results summary (auto-generated)\n\n## Key findings\n\n" + "\n".join(SUMMARY) + "\n\n")
        for m in mds:
            f.write(open(m, encoding="utf-8").read() + "\n")
        f.write("\nFigures: results/report/figs/\n")
    print("report written to", REP)


if __name__ == "__main__":
    main()
