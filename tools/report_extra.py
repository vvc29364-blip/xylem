"""Reviewer-driven tables and figures (see docs/REVIEW.md, items R1-R16).  Called by make_report.py."""
import glob
import json
import os

import numpy as np
import pandas as pd

import make_report as M
from make_report import COLS, MD_COLS, LOWER_BETTER, DISPLAY, BASE, OURS, RES, REP, PT, load, load_csv, table, save, plt

OURS_L = "main/xylem_L_ft"
COLS.update({"f_1cm": "F@1cm", "f_2cm": "F@2cm", "f_5cm": "F@5cm", "tbr_2cm": "TBR@2cm", "skel_f_5cm": "Sk-F@5cm",
             "skel_f_10cm": "Sk-F@10cm", "path_score": "Path", "vis": "Vis."})
MD_COLS.update({k: COLS[k] for k in ("f_1cm", "f_2cm", "f_5cm", "tbr_2cm", "skel_f_5cm", "skel_f_10cm", "path_score", "vis")})


def _key_df(df):
    df = df.copy()
    df["key"] = df.get("sensor", pd.Series([""] * len(df))).astype(str) + "/" + df["name"].astype(str)
    return df.set_index("key")


# ------------------------------------------------------------------ R7 significance
def t_significance(set_name="syn_seen", metrics=("f1", "f_2cm", "tbr", "skel_f", "path_score", "rad_mare"), n_boot=2000):
    from scipy.stats import wilcoxon
    ours = load_csv(OURS_L, set_name)
    if ours is None or "name" not in ours:
        return
    ours = _key_df(ours)
    rng = np.random.default_rng(0)
    md = [f"### Paired significance vs XylemNet-L ({set_name}): mean difference (ours - baseline) [95% bootstrap CI], Wilcoxon p", "",
          "| Baseline | " + " | ".join(MD_COLS.get(m, m) for m in metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    tex = [r"\begin{table*}[t]\centering\small\caption{Paired comparison with XylemNet-L on " + set_name.replace("_", r"\_")
           + r": mean per-sample difference (ours$-$baseline) with 95\% bootstrap CI; $^{*}$: Wilcoxon $p<0.01$.}"
           r"\label{tab:significance_" + set_name + r"}\resizebox{\textwidth}{!}{\begin{tabular}{l" + "c" * len(metrics)
           + r"}\toprule",
           "Baseline & " + " & ".join(COLS.get(m, m.replace("_", r"\_")) for m in metrics) + r"\\\midrule"]
    for m in BASE:
        b = load_csv(f"main/{m}", set_name)
        if b is None or "name" not in b:
            continue
        b = _key_df(b)
        common = ours.index.intersection(b.index)
        cells_md, cells_tex = [], []
        for k in metrics:
            if k not in ours or k not in b:
                cells_md.append("--"); cells_tex.append("--"); continue
            d = (ours.loc[common, k] - b.loc[common, k]).to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if len(d) < 5:
                cells_md.append("--"); cells_tex.append("--"); continue
            boots = [np.mean(rng.choice(d, len(d))) for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            try:
                p = wilcoxon(d).pvalue
            except ValueError:
                p = 1.0
            sc = 100 if k not in ("rad_mare",) else 1
            txt = f"{sc * d.mean():+.2f} [{sc * lo:+.2f},{sc * hi:+.2f}]"
            cells_md.append(txt + (" *" if p < 0.01 else "") + f" (p={p:.1e})")
            cells_tex.append(txt + ("$^{*}$" if p < 0.01 else ""))
        md.append(f"| {DISPLAY[m]} | " + " | ".join(cells_md) + " |")
        tex.append(DISPLAY[m].replace("_", r"\_") + " & " + " & ".join(cells_tex) + r"\\")
    tex += [r"\bottomrule\end{tabular}}\end{table*}"]
    open(os.path.join(REP, f"tab_significance_{set_name}.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(PT, f"tab_significance_{set_name}.tex"), "w", encoding="utf-8").write("\n".join(tex))


# ------------------------------------------------------------------ R3 absolute thresholds
def t_metric_abs():
    rows = M.main_rows()
    rows = rows[:-3] + [(f"main/{m}@qsm", DISPLAY[m] + " +QSM") for m in ("pointr", "adapointr", "snowflakenet",
                                                                         "seedformer")] + rows[-3:]
    COLS.update({"f_1cm": "F@1cm", "f_2cm": "F@2cm", "f_5cm": "F@5cm", "tbr_2cm": "TBR@2cm", "skel_f_5cm": "Sk-F@5cm",
                 "skel_f_10cm": "Sk-F@10cm", "path_score": "Path", "vis": "Vis."})
    MD_COLS.update({k: v for k, v in COLS.items() if k in ("f_1cm", "f_2cm", "f_5cm", "tbr_2cm", "skel_f_5cm", "skel_f_10cm",
                                                           "path_score")})
    table("tab_metric_abs", rows, ["f_1cm", "f_2cm", "f_5cm", "tbr_2cm", "skel_f_5cm", "skel_f_10cm", "path_score"],
          "Metric-scale evaluation (thresholds in centimetres, independent of tree size) on SynTree seen species. "
          "TBR@2cm: recall of twig surface ($r<1$\\,cm) within 2\\,cm. Path: APLS-style topology score. "
          "+QSM: the baseline completion re-rendered with cylinders on its skeleton, as for our method.", "syn_seen")


def f_threshold_curves():
    ts = [0.5, 1, 2, 3, 5, 10]
    meths = [(f"main/{m}", DISPLAY[m]) for m in ("geobin", "pointr", "adapointr", "snowflakenet", "seedformer", "smarttree",
                                                 "diffts")] + OURS
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    ok = False
    for set_name, ax in (("syn_seen", axs[0]), ("syn_unseen", axs[1])):
        for e, d in meths:
            m = load(e, set_name)
            if not m or "f_1cm" not in m:
                continue
            ok = True
            ax.plot(ts, [100 * m.get(f"f_{t:g}cm", np.nan) for t in ts], "o-", ms=3, label=d,
                    lw=2.2 if "ours" in d else 1.1)
        ax.set_xscale("log")
        ax.set_xticks(ts)
        ax.set_xticklabels([str(t) for t in ts])
        ax.set_xlabel("distance threshold [cm]")
        ax.set_ylabel("F-score [%]")
        ax.set_title({"syn_seen": "seen species", "syn_unseen": "unseen species"}[set_name], fontsize=9)
        ax.grid(alpha=0.3)
    if ok:
        axs[0].legend(fontsize=6)
        save(fig, "fig_threshold_curves")
    else:
        plt.close(fig)


# ------------------------------------------------------------------ R8 visibility
def f_visibility(set_name="syn_seen"):
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    meths = [(f"main/{m}", DISPLAY[m]) for m in ("geobin", "pointr", "snowflakenet", "smarttree", "diffts")] + OURS[2:]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
    ok = False
    for e, d in meths:
        df = load_csv(e, set_name)
        if df is None or "vis" not in df:
            continue
        ok = True
        df["bin"] = pd.cut(df["vis"], bins)
        g = df.groupby("bin", observed=False)
        xs = [0.5 * (a + b) for a, b in zip(bins[:-1], bins[1:])]
        for ax, k in zip(axs, ("f1", "skel_f", "tbr")):
            if k in df:
                ax.plot(xs, 100 * g[k].mean().to_numpy(), "o-", ms=3, label=d, lw=2.2 if "ours" in d else 1.1)
    if not ok:
        plt.close(fig)
        return
    for ax, k in zip(axs, ("F@1%", "Skeleton F@1%", "TBR")):
        ax.set_xlabel("visibility (fraction of the surface covered by the scan)")
        ax.set_ylabel(k + " [%]")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6)
    save(fig, "fig_visibility")
    # histogram of visibility per sensor (dataset statistic)
    df = load_csv(OURS_L, set_name)
    if df is not None and "vis" in df and "sensor" in df:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for s_, g in df.groupby("sensor"):
            ax.hist(g["vis"], bins=20, alpha=0.5, label=s_)
        ax.set_xlabel("visibility")
        ax.legend(fontsize=7)
        save(fig, "fig_visibility_hist")


# ------------------------------------------------------------------ R6 real forest
FOREST_SETS = [("weiser_uls_off", "ULS leaf-off"), ("weiser_uls_on", "ULS leaf-on"), ("weiser_als", "ALS"),
               ("weiser_tls1", "single TLS")]


def forest_rows():
    rows = [("main/circlefit", "Circle fit @1.3 m (forestry)"), ("main/geobin", DISPLAY["geobin"])]
    rows += [(f"main/{m}", DISPLAY[m]) for m in BASE[1:]]
    rows += [("main/xylem_S_ft", "XylemNet-S (ours)"), ("real/xylem_S_bsda_forest", "XylemNet-S + BSDA (ours)"),
             ("main/xylem_M_ft", "XylemNet-M (ours)"), ("main/xylem_L_ft", "XylemNet-L (ours)"),
             ("real/xylem_L_bsda_forest", "XylemNet-L + BSDA (ours)")]
    return rows


def t_forest():
    for key, unit, label in (("dbh", "cm", "DBH (stem-localised circle fit on the observed points, fallback to the "
                                           "reconstructed radius; identical post-processing for all methods)"),
                             ("dbhmodel", "cm", "DBH from the reconstructed radius only (no fitting)"),
                             ("height", "m", "Height")):
        cols = []
        for s_, nm in FOREST_SETS:
            for stat in ("rmse", "bias", "coverage"):
                if stat == "coverage" and key == "height":
                    continue
                c = f"{s_}:{key}_{stat}"
                cols.append(c)
                COLS[c] = f"{nm} {stat.upper() if stat != 'coverage' else 'cov.'}"
                MD_COLS[c] = COLS[c]
                if stat == "rmse":
                    LOWER_BETTER.add(c)

        def getter(exp, key=key, unit=unit):
            out = {}
            for s_, _ in FOREST_SETS:
                m = load(exp, s_) or {}
                for stat in ("rmse", "bias", "coverage"):
                    k = f"{key}_{stat}_{unit}" if stat != "coverage" else f"{key}_coverage"
                    if k in m:
                        out[f"{s_}:{key}_{stat}"] = m[k] if stat != "coverage" else m[k]
            return out
        rows = forest_rows() if key != "dbhmodel" else [r for r in forest_rows() if r[0] != "main/circlefit"]
        table(f"tab_forest_{key}", rows, cols,
              f"Real forest trees (Weiser et al. 2022, central Europe, held-out half): {label} error [{unit}] against field "
              f"inventory / TLS references. cov.: fraction of trees with an estimate. BSDA uses only unlabelled scans.",
              getter=getter)


def f_dbh_scatter():
    meths = [("main/circlefit", "Circle fit"), ("main/geobin", "GeoBin-QSM"), ("main/smarttree", "Smart-Tree*"),
             (OURS_L, "XylemNet-L"), ("real/xylem_L_bsda_forest", "XylemNet-L + BSDA")]
    sets = [("weiser_uls_off", "ULS leaf-off"), ("weiser_tls1", "single TLS")]
    fig, axs = plt.subplots(len(sets), len(meths), figsize=(2.6 * len(meths), 2.6 * len(sets)), squeeze=False)
    ok = False
    for i, (s_, sn) in enumerate(sets):
        for j, (e, d) in enumerate(meths):
            ax = axs[i, j]
            df = load_csv(e, s_)
            if df is None or "dbh_pred_cm" not in df:
                ax.axis("off")
                continue
            ok = True
            df = df[np.isfinite(df["dbh_pred_cm"]) & np.isfinite(df["dbh_ref_cm"])]
            ax.scatter(df["dbh_ref_cm"], df["dbh_pred_cm"], s=4, alpha=0.6)
            lim = [0, max(10, float(np.nanmax(df[["dbh_ref_cm", "dbh_pred_cm"]].to_numpy())) if len(df) else 100)]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            rmse = np.sqrt(np.mean((df["dbh_pred_cm"] - df["dbh_ref_cm"]) ** 2)) if len(df) else np.nan
            ax.set_title(f"{d}\n{sn}: RMSE {rmse:.1f} cm", fontsize=7)
            ax.set_xlabel("reference DBH [cm]", fontsize=7)
            ax.set_ylabel("predicted DBH [cm]", fontsize=7)
    if ok:
        save(fig, "fig_dbh_scatter")
    else:
        plt.close(fig)


# ------------------------------------------------------------------ R1 / R2 prior validity
def t_prior_validity():
    rows = [("abl/full", "Full (pipe band [0.9,1.35])"), ("abl/no_pipe", "w/o pipe-model loss"),
            ("abl/no_bot", "w/o all botanical losses"), ("abl/pipe_exact", "strict Leonardo [0.95,1.05]"),
            ("abl/bot_0.03", r"$\lambda_{bot}=0.03$"), ("abl/bot_0.3", r"$\lambda_{bot}=0.3$")]

    def getter(exp):
        a = load(exp, "syn_seen") or {}
        b = load(exp, "syn_altr") or {}
        return {"s_rad": a.get("rad_mare"), "s_skf": a.get("skel_f"), "s_f1": a.get("f1"),
                "a_rad": b.get("rad_mare"), "a_skf": b.get("skel_f"), "a_f1": b.get("f1"), "a_vol": b.get("vol_err_pct")}
    COLS.update(s_rad="SynTree R-MARE", s_skf="SynTree Sk-F", s_f1="SynTree F@1\\%", a_rad="AltR R-MARE", a_skf="AltR Sk-F",
                a_f1="AltR F@1\\%", a_vol="AltR Vol-err[\\%]")
    MD_COLS.update({k: COLS[k].replace("\\%", "%") for k in ("s_rad", "s_skf", "s_f1", "a_rad", "a_skf", "a_f1", "a_vol")})
    LOWER_BETTER.update({"s_rad", "a_rad", "a_vol"})
    table("tab_prior_validity", rows, ["s_f1", "s_skf", "s_rad", "a_f1", "a_skf", "a_rad", "a_vol"],
          "Do the botanical priors bias the reconstruction when the tree violates them? SynTree-AltR: same test trees with "
          "radii from a path-length taper (children inherit the parent radius, $\\sum r_c^2\\approx 1.8\\,r_p^2$).",
          getter=getter)
    # Synthetic-Trees (SpeedTree radii, not pipe-model generated)
    rows = [("ext/st_xylem", "XylemNet (with botanical losses)"), ("ext/st_xylem_nobot", "XylemNet w/o botanical losses")]
    table("tab_prior_speedtree", rows, ["a_f1", "a_skel_f", "a_rad", "b_f1", "b_skel_f", "b_rad"],
          "Botanical priors on SpeedTree trees (Synthetic-Trees), whose radii are not generated by the pipe model.",
          getter=lambda e: {"a_f1": (load(e, "st_test") or {}).get("f1"), "a_skel_f": (load(e, "st_test") or {}).get("skel_f"),
                            "a_rad": (load(e, "st_test") or {}).get("rad_mare"),
                            "b_f1": (load(e, "st_foliage_test") or {}).get("f1"),
                            "b_skel_f": (load(e, "st_foliage_test") or {}).get("skel_f"),
                            "b_rad": (load(e, "st_foliage_test") or {}).get("rad_mare")})


# ------------------------------------------------------------------ R11 sensitivity
def f_sensitivity():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    ok = False
    lam = [(0.0, "abl/no_bot"), (0.03, "abl/bot_0.03"), (0.1, "abl/full"), (0.3, "abl/bot_0.3")]
    for k, lab in (("f1", "F@1%"), ("skel_f", "Sk-F"), ("rad_mare", "R-MARE")):
        for set_name, ls in (("syn_seen", "-"), ("syn_altr", "--")):
            ys = [(load(e, set_name) or {}).get(k, np.nan) for _, e in lam]
            if np.all(np.isnan(ys)):
                continue
            ok = True
            axs[0].plot([l for l, _ in lam], ys, "o" + ls, ms=3, label=f"{lab} ({set_name})")
    axs[0].set_xlabel(r"$\lambda_{bot}$")
    axs[0].set_xscale("symlog", linthresh=0.03)
    thr = [0.3, 0.5, 0.7, 0.9]
    for k, lab in (("f1", "F@1%"), ("skel_f", "Sk-F"), ("path_score", "Path")):
        ys = [(load("abl/full", "syn_seen", f"thr{t}") or {}).get(k, np.nan) for t in thr]
        if not np.all(np.isnan(ys)):
            ok = True
            axs[1].plot(thr, ys, "o-", ms=3, label=lab)
    axs[1].set_xlabel(r"node confidence threshold $\tau_e$")
    nms = [(0.002, "nms0.002"), (0.003, "default"), (0.006, "nms0.006")]
    for k, lab in (("f1", "F@1%"), ("skel_f", "Sk-F"), ("len_err_pct", "length err/100")):
        ys = [(load("abl/full", "syn_seen", t) or {}).get(k, np.nan) for _, t in nms]
        ys = [y / 100 if k == "len_err_pct" else y for y in ys]
        if not np.all(np.isnan(ys)):
            ok = True
            axs[2].plot([n for n, _ in nms], ys, "o-", ms=3, label=lab)
    axs[2].set_xlabel("NMS radius (normalised)")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    if ok:
        save(fig, "fig_sensitivity")
    else:
        plt.close(fig)


# ------------------------------------------------------------------ R14 sim-to-real source
def t_sim2real():
    rows = [("abl/full@sim2real", "trained on SynTree"), ("ext/mix_xylem", "trained on SynTree + Synthetic-Trees")]

    def getter(exp):
        e, tag = exp.split("@") if "@" in exp else (exp, "default")
        out = {}
        for s_, k, c in (("st_test", "skel_f", "st_skf"), ("st_foliage_test", "skel_f", "stf_skf"),
                         ("branch3d_test", "skel_f", "b3d_skf"), ("branch3d_test", "path_score", "b3d_path"),
                         ("weiser_uls_off", "dbh_rmse_cm", "w_uls"), ("weiser_tls1", "dbh_rmse_cm", "w_tls")):
            m = load(e, s_, tag) or {}
            out[c] = m.get(k)
        return out
    COLS.update(st_skf="ST Sk-F", stf_skf="ST+fol. Sk-F", b3d_skf="BRANCH3D Sk-F", b3d_path="BRANCH3D Path",
                w_uls="Forest ULS DBH-RMSE", w_tls="Forest TLS DBH-RMSE")
    MD_COLS.update({k: COLS[k] for k in ("st_skf", "stf_skf", "b3d_skf", "b3d_path", "w_uls", "w_tls")})
    LOWER_BETTER.update({"w_uls", "w_tls"})
    table("tab_sim2real", rows, ["st_skf", "stf_skf", "b3d_skf", "b3d_path", "w_uls", "w_tls"],
          "Choice of the simulation source for sim-to-real transfer (same architecture and schedule, zero-shot).",
          getter=getter)


# ------------------------------------------------------------------ R4 hyper-parameters (appendix)
def t_hparams():
    import sys
    sys.path.insert(0, M.ROOT)
    from xylem.experiments import OFFICIAL_OPT, BUDGETS
    B = BUDGETS["full"]
    md = ["### Training hyper-parameters (all methods: batch 16, cosine schedule, identical data stream)", "",
          "| Method | lr | weight decay | iterations |", "|---|---|---|---|"]
    tex = [r"\begin{table}[t]\centering\small\caption{Optimiser settings. Baselines use the learning rate and weight decay of "
           r"their official configurations.}\label{tab:hparams}\begin{tabular}{lccc}\toprule Method & lr & wd & iterations\\\midrule"]
    for m, (lr, wd) in OFFICIAL_OPT.items():
        it = B["it_super"] if m == "xylem" else B["it_base"]
        name = DISPLAY.get(m, m) + (" (supernet)" if m == "xylem" else "")
        md.append(f"| {name} | {lr:g} | {wd:g} | {it} |")
        tex.append(f"{name} & {lr:g} & {wd:g} & {it}" + r"\\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    open(os.path.join(REP, "tab_hparams.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(PT, "tab_hparams.tex"), "w", encoding="utf-8").write("\n".join(tex))


# ------------------------------------------------------------------ R12 qualitative with error colours, failures
def _err_colors(pts, gt, scale, cap_cm=3.0):
    from scipy.spatial import cKDTree
    if len(gt) == 0:
        return "k"
    d, _ = cKDTree(gt).query(pts)
    return np.clip(d * scale * 100 / cap_cm, 0, 1)


def _is_worst(f):
    d = np.load(f)
    return "worst" in d.files and bool(d["worst"])


def f_qualitative_err(set_name="syn_seen", n=3):
    meths = [("main/pointr", "PoinTr"), ("main/snowflakenet", "SnowflakeNet"), ("main/smarttree", "Smart-Tree*"),
             ("main/diffts", "DiffTS*"), (OURS_L, "XylemNet-L")]
    d_ours = os.path.join(RES, OURS_L, "eval", f"pred__default__{set_name}")
    files = sorted(f for f in glob.glob(os.path.join(d_ours, "*.npz")) if not _is_worst(f))
    if not files:
        return
    files = files[:: max(1, len(files) // n)][:n]
    ncol = 2 + len(meths)
    fig, axs = plt.subplots(2 * len(files), ncol, figsize=(1.9 * ncol, 3.6 * len(files)), squeeze=False)
    for i, f in enumerate(files):
        base = os.path.basename(f)
        g = np.load(f)
        sc = float(g["scale"]) if "scale" in g.files else 15.0
        for row, (ix, iz, view) in enumerate(((0, 2, "side"), (0, 1, "top"))):
            ax = axs[2 * i + row, 0]
            ax.scatter(g["inp"][:, ix], g["inp"][:, iz], s=0.1, c="tab:green", rasterized=True)
            ax.set_title("Input" if i == 0 and row == 0 else "", fontsize=7)
            ax = axs[2 * i + row, 1]
            ax.scatter(g["gt"][:, ix], g["gt"][:, iz], s=0.05, c="k", rasterized=True)
            ax.set_title("GT" if i == 0 and row == 0 else "", fontsize=7)
            for j, (e, dname) in enumerate(meths):
                ax = axs[2 * i + row, 2 + j]
                pf = os.path.join(RES, e, "eval", f"pred__default__{set_name}", base)
                if os.path.exists(pf):
                    p = np.load(pf)
                    c = _err_colors(p["dense"], g["gt"], sc)
                    ax.scatter(p["dense"][:, ix], p["dense"][:, iz], s=0.05, c=c, cmap="turbo", vmin=0, vmax=1,
                               rasterized=True)
                if i == 0 and row == 0:
                    ax.set_title(dname, fontsize=7)
            for ax in axs[2 * i + row]:
                ax.set_aspect("equal")
                ax.axis("off")
    fig.text(0.5, 0.005, "colour: distance to the ground-truth surface (blue 0 cm -> red >= 3 cm); rows: side / top view",
             ha="center", fontsize=7)
    save(fig, "fig_qualitative_err")


def f_failures(set_name="syn_seen", n=4):
    d_ours = os.path.join(RES, OURS_L, "eval", f"pred__default__{set_name}")
    files = [f for f in glob.glob(os.path.join(d_ours, "*.npz")) if _is_worst(f)]
    if not files:
        return
    files = sorted(files, key=lambda f: float(np.load(f)["metric"]))[:n]
    fig, axs = plt.subplots(len(files), 3, figsize=(6, 2.6 * len(files)), squeeze=False)
    for i, f in enumerate(files):
        g = np.load(f)
        M._draw(axs[i, 0], g["inp"], title=f"input ({os.path.basename(f).split('__')[0]})", color="tab:green")
        M._draw(axs[i, 1], g["gt"], (g["g_pos"], g["g_rad"], g["g_par"]), "GT")
        M._draw(axs[i, 2], g["dense"], (g["p_pos"], g["p_rad"], g["p_par"]) if len(g["p_pos"]) else None,
                f"XylemNet-L (F={100 * float(g['metric']) if 'metric' in g.files else float('nan'):.0f})")
    save(fig, "fig_failures")


def run_all():
    for fn in (t_significance, lambda: t_significance("syn_unseen"), t_metric_abs, f_threshold_curves, f_visibility,
               t_forest, f_dbh_scatter, t_prior_validity, f_sensitivity, t_sim2real, t_hparams, f_qualitative_err,
               f_failures):
        try:
            fn()
        except Exception as ex:
            import traceback
            print("[report_extra] skipped", getattr(fn, "__name__", "fn"), ":", repr(ex)[:300])
            traceback.print_exc()
