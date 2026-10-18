"""Registry of *every* experiment in the paper as a dependency graph of jobs.

Groups
  main   : 11 baselines + XylemNet supernet -> search -> L/M/S (cut) -> short fine-tune
  lms    : one-supernet-cut-into-three vs three networks trained from scratch
  seeds  : seed variance (ours + two strongest baselines)
  abl    : component ablations (standalone L architecture) + supernet KD ablation
  data   : data-efficiency (10% / 25% of the training trees)
  ext    : in-domain training on Synthetic-Trees (SpeedTree)
  real   : BRANCH3D real orchard trees (zero-shot, botanical self-supervised adaptation, supervised FT)
  robust : noise / outliers / density (evaluation of main models on the robustness sets)
  eff    : efficiency benchmark, ONNX export for Jetson
  report : tables (LaTeX + Markdown) and figures
"""
import json
import sys

PY = sys.executable

BUDGETS = {
    "smoke": dict(data="data_smoke", syn_budget="smoke", it_base=60, it_super=80, it_abl=40, it_ext=40, it_ada=20,
                  it_ft=20, bs=4, workers=2, val_every=40, val_limit=8, eval_limit=12, search_n=4, evo_gens=1,
                  evo_pop=2, sec_scale=0.001, seeds=(1,), fracs=(0.25,)),
    "medium": dict(data="data", syn_budget="full", it_base=20000, it_super=30000, it_abl=10000, it_ext=10000,
                   it_ada=3000, it_ft=3000, bs=16, workers=8, val_every=2000, val_limit=160, eval_limit=0,
                   search_n=200, evo_gens=2, evo_pop=25, sec_scale=1.0, seeds=(1, 2), fracs=(0.1, 0.25)),
    "full": dict(data="data", syn_budget="full", it_base=40000, it_super=60000, it_abl=20000, it_ext=20000,
                 it_ada=5000, it_ft=5000, bs=16, workers=8, val_every=2000, val_limit=160, eval_limit=0,
                 search_n=300, evo_gens=3, evo_pop=30, sec_scale=1.0, seeds=(1, 2), fracs=(0.1, 0.25)),
}

BASELINES = ["pcn", "foldingnet", "topnet", "grnet", "pointr", "adapointr", "snowflakenet", "seedformer",
             "pointattn", "smarttree", "diffts"]
# rough seconds / iteration at batch 16 on a 24 GB GPU (used only for scheduling priority)
SEC_IT = dict(pcn=0.15, foldingnet=0.2, topnet=0.1, grnet=0.25, pointr=0.3, adapointr=0.35, snowflakenet=0.35,
              seedformer=0.45, pointattn=0.25, smarttree=0.12, diffts=0.2, xylem=0.45, xylem_super=1.1)
ROBUST_SETS = ["robust/noise_0.0", "robust/noise_0.01", "robust/noise_0.02", "robust/noise_0.04", "robust/noise_0.08",
               "robust/outlier_0.0", "robust/outlier_0.02", "robust/outlier_0.05", "robust/outlier_0.1",
               "robust/density_1.0", "robust/density_0.5", "robust/density_0.25", "robust/density_0.1"]
MAIN_SETS = ["syn_seen", "syn_unseen", "st_test", "st_foliage_test", "branch3d_test", "syn_altr"]
WEISER_SETS = ["weiser_uls_off", "weiser_uls_on", "weiser_als", "weiser_tls1"]
# official optimiser settings of each baseline (lr, weight decay), taken from the configs released with the
# original code (PoinTr: cfgs/PCN_models/*.yaml, SeedFormer: codes/train_pcn.py, PointAttN: cfgs/PointAttN.yaml)
OFFICIAL_OPT = dict(pcn=(1e-4, 0.0), foldingnet=(1e-4, 0.0), topnet=(1e-3, 0.0), grnet=(1e-4, 0.0),
                    pointr=(5e-4, 5e-4), adapointr=(1e-4, 5e-4), snowflakenet=(1e-3, 0.0), seedformer=(1e-3, 0.0),
                    pointattn=(1e-4, 0.0), smarttree=(5e-4, 0.05), diffts=(5e-4, 0.05), xylem=(5e-4, 0.05))


def results_root(budget):
    """Separate result trees per budget so a smoke/medium run can never be mistaken for the full one."""
    return "results" if budget == "full" else f"results_{budget}"


class Registry:
    def __init__(self, budget, have_st=True, have_b3d=True):
        self.B = BUDGETS[budget]
        self.R = results_root(budget)
        self.budget = budget
        self.jobs = {}
        self.have_st, self.have_b3d = have_st, have_b3d

    def add(self, name, cmd, deps=(), gpu=True, hours=0.1, group="misc"):
        self.jobs[name] = dict(name=name, cmd=[str(c) for c in cmd], deps=[d for d in deps if d], gpu=gpu,
                               hours=hours, group=group)
        return name

    # ------------------------------------------------------------------ helpers
    def train(self, exp, method, iters, deps=(), group="main", **kw):
        B = self.B
        if method in OFFICIAL_OPT and "lr" not in kw:
            kw["lr"], kw["wd"] = OFFICIAL_OPT[method]
        cmd = [PY, "-m", "xylem.engine.train", "--exp", exp, "--method", method, "--iters", iters,
               "--bs", B["bs"], "--workers", B["workers"], "--val_every", B["val_every"], "--val_limit", B["val_limit"],
               "--data", B["data"], "--warmup", max(5, min(1000, iters // 20))]
        for k, v in kw.items():
            if isinstance(v, bool):
                if v:
                    cmd.append("--" + k)
            elif isinstance(v, (dict, list)):
                cmd += ["--" + k, json.dumps(v)]
            else:
                cmd += ["--" + k, v]
        sec = SEC_IT.get("xylem_super" if kw.get("supernet") else method, 0.3)
        return self.add("train:" + exp, cmd, deps, True, iters * sec / 3600 * B["sec_scale"], group)

    def evaluate(self, exp, sets, deps=(), tag="default", group="main", **kw):
        sets = [s for s in sets if (self.have_st or not s.startswith("st_")) and (self.have_b3d or not s.startswith("branch3d"))]
        cmd = [PY, "-m", "xylem.engine.evaluate", "--exp", exp, "--sets", ",".join(sets), "--tag", tag,
               "--data", self.B["data"]]
        if self.B["eval_limit"]:
            cmd += ["--limit", self.B["eval_limit"]]
        for k, v in kw.items():
            cmd += ["--" + k] if v in ("", True) else ["--" + k, v]
        return self.add(f"eval:{exp}:{tag}", cmd, deps, True, 0.04 * len(sets) * self.B["sec_scale"], group)

    # ------------------------------------------------------------------ build
    def build(self):
        B = self.B
        data = B["data"]
        prep = self.add("data:syntree", [PY, "-m", "xylem.data.build_syntree", "--root", data, "--budget", B["syn_budget"]],
                        gpu=False, hours=0.5, group="data")
        prep_ext = self.add("data:external", [PY, "tools/prepare_external.py", "--data", data], gpu=False, hours=0.3,
                            group="data")
        prep_forest = self.add("data:weiser", [PY, "tools/prepare_weiser.py", "--build_data", data], gpu=False,
                               hours=0.4, group="data")
        dd = [prep, prep_ext]
        all_sets = MAIN_SETS + WEISER_SETS + ROBUST_SETS
        # classical forestry baseline (DBH by circle fit, height = max z) on the real forest sets
        t = self.train("main/circlefit", "circlefit", 1, [prep_forest])
        self.evaluate("main/circlefit", WEISER_SETS, [t, prep_forest])
        # ---------------- main comparison ----------------
        main_exps = []
        for m in ["geobin"] + BASELINES:
            t = self.train(f"main/{m}", m, B["it_base"], dd)
            self.evaluate(f"main/{m}", all_sets, [t, prep_forest])
            main_exps.append(f"main/{m}")
            if m in ("pcn", "foldingnet", "topnet", "grnet", "pointr", "adapointr", "snowflakenet", "seedformer",
                     "pointattn"):
                # Baseline+QSM: the completion is skeletonised and re-rendered with cylinders (as for ours)
                self.evaluate(f"main/{m}", ["syn_seen", "syn_unseen", "st_test", "st_foliage_test"], [t], tag="qsm",
                              dense_source="graph")
        sn = "main/xylem_supernet"
        t_sn = self.train(sn, "xylem", B["it_super"], dd, supernet=True)
        search = self.add("search:" + sn, [PY, "-m", "xylem.engine.search", "--exp", sn, "--data", data,
                                           "--n_random", B["search_n"], "--evo_gens", B["evo_gens"],
                                           "--evo_pop", B["evo_pop"]], [t_sn], True, 0.8 * B["sec_scale"], "main")
        subnets = f"{self.R}/{sn}/subnets.json"
        for v in ("L", "M", "S"):
            self.evaluate(f"{sn}_{v}", all_sets, [search, prep_forest], group="main")
            ft = self.train(f"main/xylem_{v}_ft", "xylem", B["it_ft"], [search], group="main",
                            fixed_cfg=f"@{subnets}:{v}", init=f"{self.R}/{sn}_{v}/ckpt_best.pt", lr=1e-4, bot_start=0.0)
            self.evaluate(f"main/xylem_{v}_ft", all_sets, [ft, prep_forest])
            main_exps += [f"{sn}_{v}", f"main/xylem_{v}_ft"]
            # ---------------- one supernet vs. three separately trained networks ----------------
            sc = self.train(f"lms/xylem_{v}_scratch", "xylem", B["it_base"], [search], group="lms",
                            fixed_cfg=f"@{subnets}:{v}")
            self.evaluate(f"lms/xylem_{v}_scratch", ["syn_seen", "syn_unseen"], [sc], group="lms")
        # S distilled from L as an independent network (is the supernet better than plain KD?)
        t = self.train("lms/xylem_S_kd", "xylem", B["it_base"], [search, "train:main/xylem_L_ft"], group="lms",
                       fixed_cfg=f"@{subnets}:S", teacher=f"{self.R}/main/xylem_L_ft")
        self.evaluate("lms/xylem_S_kd", ["syn_seen", "syn_unseen"], [t], group="lms")
        # supernet sub-network sweep for the Pareto figure is part of search.csv
        # ---------------- seeds ----------------
        for s in B["seeds"]:
            t = self.train(f"seeds/xylem_L_seed{s}", "xylem", B["it_base"], [search], group="seeds",
                           fixed_cfg=f"@{subnets}:L", seed=s)
            self.evaluate(f"seeds/xylem_L_seed{s}", ["syn_seen", "syn_unseen"], [t], group="seeds")
            for m in ("pointr", "snowflakenet", "smarttree"):
                t = self.train(f"seeds/{m}_seed{s}", m, B["it_base"], dd, group="seeds", seed=s)
                self.evaluate(f"seeds/{m}_seed{s}", ["syn_seen", "syn_unseen"], [t], group="seeds")
        # seed 0 of every entry = the corresponding main / lms run
        # ---------------- ablations ----------------
        ABL = {
            "full": {},
            "no_tube": dict(model_kw=dict(use_tube=False)),
            "no_growth": dict(model_kw=dict(use_growth=False)),
            "no_anchor": dict(model_kw=dict(use_anchor=False)),
            "no_learned_q": dict(model_kw=dict(use_learned_q=False)),
            "no_field": dict(model_kw=dict(use_point_head=False, use_anchor=False)),
            "no_bot": dict(loss_w=dict(bot=0.0)),
            "no_pipe": dict(loss_w=dict(bot_pipe=0.0)),
            "no_rootflow": dict(loss_w=dict(flow=0.0, geo=0.0, p_flow=0.0, p_geo=0.0, bot=0.0), topology="mst"),
            "single_order": dict(model_kw=dict(n_perm=1)),
            "no_leaf_field": dict(loss_w=dict(field_on_leaves=0.0)),
            "bot_0.03": dict(loss_w=dict(bot=0.03)),
            "bot_0.3": dict(loss_w=dict(bot=0.3)),
            "pipe_exact": dict(loss_w=dict(pipe_lo=0.95, pipe_hi=1.05)),
            "cd_only": dict(loss_w=dict(radius=0.0, flow=0.0, geo=0.0, exist=0.0, seg=0.0, p_flow=0.0, p_geo=0.0,
                                        p_rad=0.0, p_med=0.0, bot=0.0), topology="mst"),
        }
        for name, kw in ABL.items():
            t = self.train(f"abl/{name}", "xylem", B["it_abl"], dd, group="abl", **kw)
            self.evaluate(f"abl/{name}", ["syn_seen", "syn_unseen", "robust/noise_0.04", "syn_altr"], [t], group="abl")
        self.evaluate("abl/full", ["syn_seen", "syn_unseen"], ["train:abl/full"], tag="mst", group="abl", topology="mst")
        self.evaluate("abl/full", ["syn_seen", "syn_unseen"], ["train:abl/full"], tag="nomerge", group="abl", no_merge=True)
        self.evaluate("abl/full", ["syn_seen", "syn_unseen"], ["train:abl/full"], tag="nodesonly", group="abl",
                      skel_source="nodes")
        self.evaluate("abl/full", ["syn_seen", "syn_unseen"], ["train:abl/full"], tag="nodetubes", group="abl",
                      dense_source="nodes")
        self.evaluate("abl/full", ["syn_seen", "syn_unseen", "weiser_uls_off", "weiser_tls1"],
                      ["train:abl/full", prep_forest], tag="norefine", group="abl", no_refine=True)
        # decoding hyper-parameter sensitivity (evaluation only, same checkpoint)
        for thr in (0.3, 0.5, 0.7, 0.9):
            self.evaluate("abl/full", ["syn_seen"], ["train:abl/full"], tag=f"thr{thr}", group="abl", node_thr=thr)
        for nd in (0.002, 0.006):
            self.evaluate("abl/full", ["syn_seen"], ["train:abl/full"], tag=f"nms{nd}", group="abl", nms_dist=nd)
        # supernet with / without in-place distillation; evaluate the extreme sub-networks
        from .models.xylemnet import MAX_CFG, MIN_CFG
        mid = dict(MAX_CFG, width=0.75, d3=4, dec_layers=2, dec_width=0.75, n_in=3072, n_anchor=192)
        for kd in ("kd", "nokd"):
            e = f"abl/supernet_{kd}"
            t = self.train(e, "xylem", int(B["it_abl"] * 1.5), dd, group="abl", supernet=True, no_kd=(kd == "nokd"))
            for tag, c in (("max", MAX_CFG), ("mid", mid), ("min", MIN_CFG)):
                self.evaluate(e, ["syn_seen"], [t], tag=tag, group="abl", cfg=json.dumps(c))
        # ---------------- data efficiency ----------------
        for frac in B["fracs"]:
            for m in ("xylem", "pointr", "snowflakenet"):
                e = f"data/{m}_frac{frac}"
                t = self.train(e, m, B["it_abl"], dd, group="data", train_frac=frac)
                self.evaluate(e, ["syn_seen"], [t], group="data")
        # ---------------- Synthetic-Trees in-domain ----------------
        if self.have_st:
            for m in ("xylem", "pointr", "snowflakenet", "smarttree", "diffts", "seedformer"):
                e = f"ext/st_{m}"
                t = self.train(e, m, B["it_ext"], dd, group="ext", source="synthetic_trees")
                self.evaluate(e, ["st_test", "st_foliage_test"], [t], group="ext")
            # botanical priors on data whose radii do NOT come from the pipe model (SpeedTree)
            t = self.train("ext/st_xylem_nobot", "xylem", B["it_ext"], dd, group="ext", source="synthetic_trees",
                           loss_w=dict(bot=0.0))
            self.evaluate("ext/st_xylem_nobot", ["st_test", "st_foliage_test"], [t], group="ext")
            # simulation source for sim-to-real: SynTree vs SynTree + SpeedTree (same schedule as abl/full)
            t = self.train("ext/mix_xylem", "xylem", B["it_abl"], dd, group="ext", source="syntree+synthetic_trees")
            self.evaluate("ext/mix_xylem", ["syn_seen", "st_test", "st_foliage_test", "branch3d_test"] + WEISER_SETS,
                          [t, prep_forest], group="ext")
            self.evaluate("abl/full", ["st_test", "st_foliage_test", "branch3d_test"] + WEISER_SETS,
                          ["train:abl/full", prep_forest], tag="sim2real", group="ext")
            self.train("ext/st_geobin", "geobin", 1, dd, group="ext", source="synthetic_trees")
            self.evaluate("ext/st_geobin", ["st_test", "st_foliage_test"], ["train:ext/st_geobin"], group="ext")
        # ---------------- real BRANCH3D ----------------
        if self.have_b3d:
            L = "main/xylem_L_ft"
            t = self.train("real/xylem_bsda", "xylem", B["it_ada"], ["train:" + L], group="real",
                           fixed_cfg=f"@{subnets}:L", init=f"{self.R}/{L}/ckpt_best.pt", lr=5e-5, bot_start=0.0,
                           target="branch3d")
            self.evaluate("real/xylem_bsda", ["branch3d_test"], [t], group="real")
            # same adaptation with the two-sided observation term (nodes / surface farther than 0.1 tree sizes from
            # every observed point are penalised); the one-sided run above lets decoder nodes drift off the scan
            t = self.train("real/xylem_bsda_sup", "xylem", B["it_ada"], ["train:" + L], group="real",
                           fixed_cfg=f"@{subnets}:L", init=f"{self.R}/{L}/ckpt_best.pt", lr=5e-5, bot_start=0.0,
                           target="branch3d", target_margin=0.1)
            self.evaluate("real/xylem_bsda_sup", ["branch3d_test"], [t], group="real")
            t = self.train("real/xylem_sft", "xylem", B["it_ada"], ["train:" + L], group="real",
                           fixed_cfg=f"@{subnets}:L", init=f"{self.R}/{L}/ckpt_best.pt", lr=5e-5, bot_start=0.0,
                           source="branch3d")
            self.evaluate("real/xylem_sft", ["branch3d_test"], [t], group="real")
            t = self.train("real/xylem_S_bsda", "xylem", B["it_ada"], ["train:main/xylem_S_ft"], group="real",
                           fixed_cfg=f"@{subnets}:S", init=f"{self.R}/main/xylem_S_ft/ckpt_best.pt", lr=5e-5,
                           bot_start=0.0, target="branch3d")
            self.evaluate("real/xylem_S_bsda", ["branch3d_test"], [t], group="real")
            for m in ("diffts", "smarttree", "pointr", "snowflakenet"):
                t = self.train(f"real/{m}_sft", m, B["it_ada"], ["train:main/" + m], group="real",
                               init=f"{self.R}/main/{m}/ckpt_best.pt", lr=5e-5, source="branch3d")
                self.evaluate(f"real/{m}_sft", ["branch3d_test"], [t], group="real")
            t = self.train("real/diffts_scratch", "diffts", B["it_ext"], dd, group="real", source="branch3d")
            self.evaluate("real/diffts_scratch", ["branch3d_test"], [t], group="real")
        # ---------------- real forest (Weiser et al.): label-free botanical adaptation ----------------
        for v in ("L", "S"):
            e = f"real/xylem_{v}_bsda_forest"
            t = self.train(e, "xylem", B["it_ada"], [f"train:main/xylem_{v}_ft", prep_forest], group="real",
                           fixed_cfg=f"@{subnets}:{v}", init=f"{self.R}/main/xylem_{v}_ft/ckpt_best.pt", lr=5e-5,
                           bot_start=0.0, target="weiser")
            self.evaluate(e, WEISER_SETS, [t], group="real")
        # ---------------- efficiency / deployment ----------------
        eff_exps = [e for e in main_exps if not e.endswith("geobin")]
        deps = [f"train:{e}" for e in eff_exps if f"train:{e}" in self.jobs] + [search]
        bench = self.add("bench:efficiency", [PY, "-m", "xylem.engine.benchmark", "--exps", ",".join(eff_exps)],
                         deps, True, 0.3 * B["sec_scale"], "eff")
        onnx = self.add("deploy:onnx", [PY, "deploy/export_onnx.py", "--all"],
                        ["train:main/xylem_S_ft", "train:main/xylem_M_ft", "train:main/xylem_L_ft", search], True, 0.1, "eff")
        # ---------------- report ----------------
        self.add("report", [PY, "tools/make_report.py"], [j for j in self.jobs if j.startswith("eval:")] + [bench, onnx],
                 False, 0.2, "report")
        return self.jobs
