"""Pure-PyTorch point cloud ops (no custom CUDA extensions -> runs anywhere, exports to ONNX).

Chamfer distance uses the *index trick*: nearest neighbours are searched without autograd in
memory-bounded chunks, then distances are recomputed on the gathered pairs with autograd.
The gradient is identical to the classic CUDA Chamfer kernel.
"""
import math

import torch
import torch.nn.functional as F


def gather(x, idx):
    """x (B,N,C), idx (B,...) -> (B,...,C)"""
    B = x.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(x, 1, flat.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return out.reshape(*idx.shape, x.shape[-1])


def _chunk(nq, nk, B, budget=2 ** 27):
    return max(1, min(nq, budget // max(1, nk * B)))


@torch.no_grad()
def knn_idx(q, x, k, exclude_self=False):
    """q (B,M,3), x (B,N,3) -> idx (B,M,k) of nearest points in x (sorted)."""
    B, M, _ = q.shape
    N = x.shape[1]
    k_eff = min(k + int(exclude_self), N)
    cs = _chunk(M, N, B)
    out = []
    for s in range(0, M, cs):
        d = torch.cdist(q[:, s:s + cs].float(), x.float())
        idx = d.topk(k_eff, dim=-1, largest=False).indices
        out.append(idx)
    idx = torch.cat(out, 1)
    if exclude_self:
        idx = idx[..., 1:]
    return idx


_TRITON_OK = None


@torch.no_grad()
def nn_idx(q, x):
    """nearest neighbour index + euclidean distance (fused Triton kernel if available)."""
    global _TRITON_OK
    if q.is_cuda and _TRITON_OK is not False:
        try:
            from .ops_triton import HAS_TRITON, nn_triton
            if HAS_TRITON:
                r = nn_triton(q, x)
                _TRITON_OK = True
                return r
            _TRITON_OK = False
        except Exception as ex:  # compilation problems -> permanent fallback
            print("[ops] triton NN disabled:", ex)
            _TRITON_OK = False
    return _nn_idx_torch(q, x)


@torch.no_grad()
def _nn_idx_torch(q, x):
    B, M, _ = q.shape
    N = x.shape[1]
    cs = _chunk(M, N, B)
    idx, dist = [], []
    for s in range(0, M, cs):
        d = torch.cdist(q[:, s:s + cs].float(), x.float())
        m = d.min(-1)
        idx.append(m.indices)
        dist.append(m.values)
    return torch.cat(idx, 1), torch.cat(dist, 1)


def chamfer(a, b, mode="l1", reduce=True):
    """Symmetric Chamfer distance.  mode 'l1' = mean euclidean (PoinTr CD-L1 without /2 averaging
    is (d1+d2)/2 here), 'l2' = mean squared.  Returns (loss, d_ab, d_ba) per-point distances."""
    i_ab, _ = nn_idx(a, b)
    i_ba, _ = nn_idx(b, a)
    d_ab = ((a - gather(b, i_ab)) ** 2).sum(-1)
    d_ba = ((b - gather(a, i_ba)) ** 2).sum(-1)
    if mode == "l1":
        d_ab = torch.sqrt(d_ab + 1e-12)
        d_ba = torch.sqrt(d_ba + 1e-12)
    if reduce:
        return 0.5 * (d_ab.mean() + d_ba.mean()), d_ab, d_ba
    return 0.5 * (d_ab.mean(1) + d_ba.mean(1)), d_ab, d_ba


def one_sided(a, b, mode="l1"):
    """mean distance from each point of a to b."""
    i_ab, _ = nn_idx(a, b)
    d = ((a - gather(b, i_ab)) ** 2).sum(-1)
    return torch.sqrt(d + 1e-12) if mode == "l1" else d


def density_aware_chamfer(a, b, alpha=1000, n_lambda=1.0):
    """DCD (Wu et al., NeurIPS 2021) with squared distances."""
    i_ab, _ = nn_idx(a, b)
    i_ba, _ = nn_idx(b, a)
    d_ab = ((a - gather(b, i_ab)) ** 2).sum(-1)
    d_ba = ((b - gather(a, i_ba)) ** 2).sum(-1)
    B = a.shape[0]
    cnt_b = torch.zeros(B, b.shape[1], device=a.device).scatter_add_(1, i_ab, torch.ones_like(d_ab))
    cnt_a = torch.zeros(B, a.shape[1], device=a.device).scatter_add_(1, i_ba, torch.ones_like(d_ba))
    w_ab = 1.0 / (torch.gather(cnt_b, 1, i_ab) ** n_lambda + 1e-6)
    w_ba = 1.0 / (torch.gather(cnt_a, 1, i_ba) ** n_lambda + 1e-6)
    l1 = (1 - torch.exp(-alpha * d_ab) * w_ab).mean(1)
    l2 = (1 - torch.exp(-alpha * d_ba) * w_ba).mean(1)
    return 0.5 * (l1 + l2).mean()


@torch.no_grad()
def fps(x, n):
    """Farthest point sampling. x (B,N,3) -> idx (B,n)."""
    B, N, _ = x.shape
    n = min(n, N)
    idx = torch.zeros(B, n, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), float("inf"), device=x.device)
    far = torch.zeros(B, dtype=torch.long, device=x.device)
    ar = torch.arange(B, device=x.device)
    xf = x.float()
    for i in range(n):
        idx[:, i] = far
        c = xf[ar, far].unsqueeze(1)
        d = ((xf - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


# ----------------------------------------------------------------------------------------
# serialization (space-filling curve order, PTv3-style) -- only integer div/mod (ONNX friendly)
# ----------------------------------------------------------------------------------------
PERMS = [(0, 1, 2), (1, 0, 2), (2, 0, 1), (0, 2, 1)]


def morton_code(pos, bits=10, perm=(0, 1, 2)):
    """pos (B,N,3) -> int64 Morton code in a per-sample normalised grid (vectorised over bits)."""
    lo = pos.amin(1, keepdim=True)
    hi = pos.amax(1, keepdim=True)
    q = ((pos - lo) / (hi - lo).clamp_min(1e-6) * (2 ** bits - 1)).long().clamp(0, 2 ** bits - 1)
    q = q[..., list(perm)]
    p2 = 2 ** torch.arange(bits, device=pos.device, dtype=torch.long)                  # (bits,)
    bitv = torch.div(q.unsqueeze(-1), p2, rounding_mode="floor") % 2                    # (B,N,3,bits)
    w = (2 ** (3 * torch.arange(bits, device=pos.device, dtype=torch.long))).view(1, 1, 1, bits)         * torch.tensor([4, 2, 1], device=pos.device, dtype=torch.long).view(1, 1, 3, 1)
    return (bitv * w).sum((-1, -2))


def serialize(pos, perm=(0, 1, 2)):
    """Returns order (B,N) and inverse (B,N)."""
    order = torch.argsort(morton_code(pos, perm=perm), dim=1)
    inv = torch.argsort(order, dim=1)
    return order, inv


def sinusoidal(x, n_freq=6):
    """x (...,d) -> (..., d*2*n_freq) Fourier features."""
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    y = x.unsqueeze(-1) * f
    return torch.cat([y.sin(), y.cos()], -1).flatten(-2)


def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def orthonormal_basis(t):
    """t (...,3) unit -> u, v (...,3) orthonormal to t."""
    helper = torch.where((t[..., 2:3].abs() < 0.9), torch.tensor([0.0, 0.0, 1.0], device=t.device, dtype=t.dtype),
                         torch.tensor([1.0, 0.0, 0.0], device=t.device, dtype=t.dtype))
    u = normalize(torch.cross(t, helper.expand_as(t), dim=-1))
    v = torch.cross(t, u, dim=-1)
    return u, v
