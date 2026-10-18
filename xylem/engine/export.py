"""Physically cut a sub-network out of the elastic supernet.

The standalone network is instantiated with max-dims = the sub-network's active dims and every
parameter is the leading slice of the supernet parameter; outputs are numerically identical
(checked by `verify`).  The result is written as a normal experiment directory so that
evaluate.py / benchmark / ONNX export treat it like any other model.
"""
import json
import os

import torch

from ..methods import XylemMethod


def slice_state(big_sd, small_sd):
    out = {}
    for k, v in small_sd.items():
        b = big_sd[k]
        if b.shape == v.shape:
            out[k] = b.clone()
        else:
            out[k] = b[tuple(slice(0, s) for s in v.shape)].clone()
    return out


def export_subnet(super_method, cfg):
    kw = dict(super_method.net.a)
    for k in ("dims", "depths", "dec_dim", "dec_layers", "n_learned", "mlp_ratio", "grow", "n_anchor"):
        kw.pop(k, None)
    sub = XylemMethod(model_kw=kw, loss_w=super_method.loss_w, fixed_cfg=cfg, topology=super_method.topology)
    sub.load_state_dict(slice_state(super_method.state_dict(), sub.state_dict()), strict=True)
    return sub, kw


@torch.no_grad()
def verify(super_method, sub, cfg, dev="cuda"):
    super_method.eval().to(dev)
    sub.eval().to(dev)
    x = torch.rand(2, cfg["n_in"], 3, device=dev) * torch.tensor([1, 1, 1.0], device=dev) - torch.tensor([0.5, 0.5, 0], device=dev)
    a = super_method(x, cfg)
    b = sub(x)
    return max(float((a["nodes"][-1]["pos"] - b["nodes"][-1]["pos"]).abs().max()),
               float((a["dense"] - b["dense"]).abs().max()))


def save_as_experiment(sub, kw, cfg, out_dir, parent_args, info=None):
    os.makedirs(out_dir, exist_ok=True)
    a = dict(parent_args)
    a.update(method="xylem", fixed_cfg=json.dumps(cfg), model_kw=json.dumps(kw), supernet=False,
             exported_from=parent_args.get("exp"))
    json.dump(a, open(os.path.join(out_dir, "train_args.json"), "w"), indent=1)
    torch.save(dict(model=sub.state_dict()), os.path.join(out_dir, "ckpt_best.pt"))
    json.dump(dict(info or {}, cfg=cfg), open(os.path.join(out_dir, "subnet.json"), "w"), indent=1)
    open(os.path.join(out_dir, "done.flag"), "w").write("exported\n")
