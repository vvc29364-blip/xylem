"""Botanical self-supervised domain adaptation (BSDA) -- label-free objective on real scans.

For an unlabelled real scan the network is asked to
  (1) explain what it observes: every point that the network itself classifies as wood must lie
      on the reconstructed tubular surface (one-sided Chamfer, weighted by the wood probability),
  (2) be botanically plausible: flow/geodesic consistency, monotone radii, pipe-model band.
  (3) [two-sided variant, --target_margin > 0] stay supported by the scan: every generated node and
      surface sample farther than `margin` (fraction of the tree size) from all observed points is
      penalised with a hinge.  Completion close to the observations is free; nodes that drift away
      from the tree are not.  Without (3) the one-sided term (1) cannot see such nodes, which is how
      the first L adaptation on BRANCH3D failed (decoder nodes ~400 tree sizes from the scan).
No completion target, radius or skeleton label is needed.  During adaptation the labelled
synthetic batches are kept (see train.py --target), which prevents the trivial solutions.
"""
import torch

from ..losses import DEFAULT_W, botanical_losses
from ..ops import gather, nn_idx


def target_loss(method, tb, w_obs=1.0, w_bot=0.2, margin=0.0, w_sup=1.0):
    out = method(tb["inp"])
    n = out["n_in"]
    x = tb["inp"][:, :n]
    wood = out["point"]["seg"].detach().sigmoid() if "point" in out else torch.ones_like(x[..., 0])
    j, _ = nn_idx(x, out["dense"])
    d = (x - gather(out["dense"], j)).norm(dim=-1)
    l_obs = (d * wood).sum() / wood.sum().clamp_min(1.0)
    lb, lg = botanical_losses(out["nodes"][-1], DEFAULT_W)
    tot = w_obs * l_obs + w_bot * lb
    logs = {"t_obs": l_obs.detach(), "t_bot": lb.detach()}
    if margin > 0:
        sup = 0.0
        for y in (out["nodes"][-1]["pos"], out["dense"]):
            k, _ = nn_idx(y, x)
            dy = (y - gather(x, k)).norm(dim=-1)
            sup = sup + torch.relu(dy - margin).mean()
        tot = tot + w_sup * sup
        logs["t_sup"] = sup.detach()
    return tot, logs
