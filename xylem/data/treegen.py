"""Procedural botanical tree generator (SynTree).

Three growth engines, all producing a skeleton graph (pos, radius, parent) in metres:
  * space colonisation (Runions et al. 2007) inside a species-specific crown envelope
    -> sympodial broad-leaf trees (oak, maple, birch, poplar, willow, apple, acacia, shrubs)
  * monopodial whorl model (L-system like) -> conifers (spruce, pine)
Radii follow the pipe model with a species-specific exponent (2.0-3.0), rescaled to an
allometric DBH.  Leaves are generated on the fly from the skeleton (see leaves.py) so the
stored trees are tiny (~50 KB).
"""
import numpy as np
from scipy.spatial import cKDTree

from .skeleton import pipe_radii, topo_order, children_lists

# --------------------------------------------------------------------------------------
# species presets.  Values are (low, high) ranges sampled uniformly per tree.
# --------------------------------------------------------------------------------------
SPECIES = {
    "oak":     dict(engine="sc", H=(8, 20), crown_base=(0.2, 0.35), crown_w=(0.7, 1.0), shape="ellipsoid",
                    tropism=(0.0, 0.1), n_attr=(1500, 3000), pipe=(2.2, 2.6), tip_r=(0.004, 0.007),
                    leaf=dict(size=(0.05, 0.09), density=(4, 7), kind="broad")),
    "maple":   dict(engine="sc", H=(8, 18), crown_base=(0.25, 0.4), crown_w=(0.45, 0.7), shape="ellipsoid",
                    tropism=(0.05, 0.2), n_attr=(1500, 3000), pipe=(2.2, 2.6), tip_r=(0.004, 0.006),
                    leaf=dict(size=(0.06, 0.1), density=(4, 7), kind="broad")),
    "birch":   dict(engine="sc", H=(10, 22), crown_base=(0.3, 0.5), crown_w=(0.3, 0.45), shape="ellipsoid",
                    tropism=(0.15, 0.3), n_attr=(1200, 2500), pipe=(2.4, 2.9), tip_r=(0.003, 0.005),
                    leaf=dict(size=(0.03, 0.05), density=(5, 8), kind="broad")),
    "poplar":  dict(engine="sc", H=(14, 26), crown_base=(0.12, 0.25), crown_w=(0.2, 0.3), shape="cylinder",
                    tropism=(0.1, 0.25), n_attr=(2500, 4000), pipe=(2.3, 2.7), tip_r=(0.004, 0.006),
                    leaf=dict(size=(0.04, 0.07), density=(4, 7), kind="broad")),
    "apple":   dict(engine="sc", H=(3, 5), crown_base=(0.15, 0.3), crown_w=(0.6, 0.9), shape="vase",
                    tropism=(0.0, 0.15), n_attr=(1000, 2000), pipe=(2.0, 2.4), tip_r=(0.004, 0.006),
                    leaf=dict(size=(0.05, 0.08), density=(3, 6), kind="broad")),
    "spruce":  dict(engine="whorl", H=(10, 25), crown_base=(0.05, 0.2), crown_w=(0.32, 0.45), droop=(0.1, 0.4),
                    whorl_k=(4, 6), pipe=(2.3, 2.8), tip_r=(0.004, 0.006),
                    leaf=dict(size=(0.02, 0.03), density=(6, 10), kind="needle")),
    "pine":    dict(engine="whorl", H=(12, 25), crown_base=(0.45, 0.65), crown_w=(0.35, 0.5), droop=(-0.2, 0.1),
                    whorl_k=(3, 5), pipe=(2.3, 2.8), tip_r=(0.005, 0.007),
                    leaf=dict(size=(0.03, 0.05), density=(6, 10), kind="needle")),
    "shrub":   dict(engine="sc", H=(2, 4), crown_base=(0.0, 0.05), crown_w=(0.8, 1.2), shape="hemisphere",
                    tropism=(0.05, 0.2), n_attr=(1000, 2000), pipe=(2.0, 2.4), tip_r=(0.003, 0.004),
                    stems=(3, 7), leaf=dict(size=(0.03, 0.05), density=(3, 6), kind="broad")),
    # --- held-out species (never seen in training) ---
    "willow":  dict(engine="sc", H=(8, 15), crown_base=(0.2, 0.35), crown_w=(0.6, 0.9), shape="hemisphere",
                    tropism=(-0.45, -0.25), n_attr=(1500, 3000), pipe=(2.2, 2.6), tip_r=(0.003, 0.005),
                    leaf=dict(size=(0.04, 0.07), density=(5, 8), kind="broad")),
    "acacia":  dict(engine="sc", H=(6, 12), crown_base=(0.45, 0.6), crown_w=(0.8, 1.2), shape="umbrella",
                    tropism=(0.0, 0.1), n_attr=(1200, 2500), pipe=(2.1, 2.5), tip_r=(0.004, 0.006),
                    leaf=dict(size=(0.02, 0.04), density=(5, 8), kind="broad")),
}
SEEN_SPECIES = ["oak", "maple", "birch", "poplar", "apple", "spruce", "pine", "shrub"]
UNSEEN_SPECIES = ["willow", "acacia"]
ALL_SPECIES = SEEN_SPECIES + UNSEEN_SPECIES


def _u(rng, lohi):
    lo, hi = lohi
    return lo + (hi - lo) * rng.random()


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------------------
# crown envelopes (return points uniformly inside)
# --------------------------------------------------------------------------------------
def sample_envelope(shape, n, z0, z1, R, rng):
    out = []
    h = z1 - z0
    while sum(len(o) for o in out) < n:
        m = 4 * n
        p = np.stack([rng.uniform(-R, R, m), rng.uniform(-R, R, m), rng.uniform(z0, z1, m)], 1)
        rr = np.hypot(p[:, 0], p[:, 1])
        zn = (p[:, 2] - z0) / h  # 0..1
        if shape == "ellipsoid":
            ok = (rr / R) ** 2 + ((zn - 0.5) / 0.5) ** 2 <= 1
        elif shape == "cylinder":
            ok = (rr <= R * (1 - 0.5 * zn ** 4)) & (zn > 0)
        elif shape == "cone":
            ok = rr <= R * (1 - zn)
        elif shape == "hemisphere":
            ok = (rr / R) ** 2 + zn ** 2 <= 1
        elif shape == "umbrella":
            ok = (rr / R) ** 2 + ((zn - 0.75) / 0.25) ** 2 <= 1
            ok |= ((rr / (0.5 * R)) ** 2 + ((zn - 0.55) / 0.3) ** 2 <= 1) & (rng.random(m) < 0.25)
        elif shape == "vase":  # open-centre orchard tree: hollow middle
            ok = ((rr / R) ** 2 + ((zn - 0.5) / 0.5) ** 2 <= 1) & (rr > 0.35 * R * (1 - zn) )
        else:
            raise ValueError(shape)
        out.append(p[ok])
    return np.concatenate(out)[:n]


# --------------------------------------------------------------------------------------
# space colonisation
# --------------------------------------------------------------------------------------
def space_colonization(rng, H, crown_base, crown_w, shape, tropism, n_attr, stems=1, max_nodes=9000):
    D = H / rng.uniform(55, 75)                 # growth step
    d_inf = D * rng.uniform(10, 16)             # influence distance
    d_kill = D * rng.uniform(1.6, 2.4)          # kill distance
    z0 = crown_base * H
    R = crown_w * H * 0.5
    attr = sample_envelope(shape, n_attr, z0, H, R, rng)
    lean = np.array([rng.normal(0, 0.05), rng.normal(0, 0.05), 1.0])
    pos = [np.zeros(3)]
    parent = [-1]
    # --- trunk(s) ---
    if stems > 1:
        for s in range(stems):
            ang = 2 * np.pi * s / stems + rng.normal(0, 0.3)
            d = _unit(np.array([np.cos(ang) * rng.uniform(0.4, 0.9), np.sin(ang) * rng.uniform(0.4, 0.9), 1.0]))
            cur = 0
            for _ in range(int(rng.integers(3, 7))):
                d = _unit(d + rng.normal(0, 0.08, 3))
                pos.append(pos[cur] + D * d)
                parent.append(cur)
                cur = len(pos) - 1
    else:
        cur = 0
        n_trunk = max(2, int(z0 / D))
        d = _unit(lean)
        for _ in range(n_trunk):
            d = _unit(d + rng.normal(0, 0.04, 3) + np.array([0, 0, 0.2]))
            pos.append(pos[cur] + D * d)
            parent.append(cur)
            cur = len(pos) - 1
    pos = np.asarray(pos)
    parent = list(parent)
    trop = np.array([0, 0, tropism])
    stall = 0
    for it in range(400):
        if len(attr) == 0 or len(pos) > max_nodes:
            break
        kd = cKDTree(pos)
        d, idx = kd.query(attr, distance_upper_bound=d_inf)
        valid = np.isfinite(d)
        if not valid.any():
            # no node in range: extend the highest node upward towards the crown (trunk still short)
            stall += 1
            if stall > 30:
                break
            top = int(np.argmax(pos[:, 2]))
            tgt = attr[np.argmin(np.linalg.norm(attr - pos[top], axis=1))]
            nd = _unit(tgt - pos[top])
            pos = np.vstack([pos, pos[top] + D * nd])
            parent.append(top)
            continue
        stall = 0
        vi = idx[valid]
        dirs = _unit(attr[valid] - pos[vi])
        acc = np.zeros_like(pos)
        np.add.at(acc, vi, dirs)
        growers = np.unique(vi)
        gd = _unit(_unit(acc[growers]) + trop + rng.normal(0, 0.06, (len(growers), 3)))
        new = pos[growers] + D * gd
        # avoid duplicates (node growing into existing node)
        dd, _ = kd.query(new)
        ok = dd > 0.3 * D
        if not ok.any():
            break
        new, growers = new[ok], growers[ok]
        pos = np.vstack([pos, new])
        parent.extend(growers.tolist())
        # kill attractors
        kd2 = cKDTree(new)
        dk, _ = kd2.query(attr, distance_upper_bound=d_kill)
        attr = attr[~np.isfinite(dk)]
    return pos, np.asarray(parent, dtype=np.int64), D


# --------------------------------------------------------------------------------------
# monopodial whorl model for conifers
# --------------------------------------------------------------------------------------
def whorl_tree(rng, H, crown_base, crown_w, droop, whorl_k, max_nodes=9000):
    D = H / rng.uniform(100, 130)
    pos = [np.zeros(3)]
    parent = [-1]
    trunk = [0]
    d = _unit(np.array([rng.normal(0, 0.03), rng.normal(0, 0.03), 1.0]))
    n_trunk = int(H / D)
    for _ in range(n_trunk):
        d = _unit(d + rng.normal(0, 0.02, 3) + np.array([0, 0, 0.3]))
        pos.append(pos[trunk[-1]] + D * d)
        parent.append(trunk[-1])
        trunk.append(len(pos) - 1)
    Lmax = crown_w * H * 0.5
    whorl_dz = H * rng.uniform(0.025, 0.04)
    z = crown_base * H
    phase = rng.uniform(0, 2 * np.pi)

    def grow_branch(start, direction, length, level):
        nonlocal pos, parent
        n = max(1, int(length / D))
        cur = start
        dirv = _unit(direction)
        lat_every = int(rng.integers(1, 3))
        for s in range(n):
            if len(pos) > max_nodes:
                return
            # gravitational bending grows along the branch
            dirv = _unit(dirv + np.array([0, 0, -droop * 0.08 * (s / n)]) + rng.normal(0, 0.05, 3))
            pos.append(pos[cur] + D * dirv)
            parent.append(cur)
            cur = len(pos) - 1
            if level < 2 and s > 0 and s % lat_every == 0 and s < n - 1 and rng.random() < 0.8:
                rem = (n - s) * D
                side = _unit(np.cross(dirv, np.array([0, 0, 1.0])) * (1 if (s // lat_every) % 2 else -1)
                             + 0.6 * dirv + rng.normal(0, 0.1, 3))
                grow_branch(cur, side, rem * rng.uniform(0.3, 0.55), level + 1)

    while z < H * 0.97:
        ti = trunk[min(len(trunk) - 1, int(z / D))]
        rel = (H - z) / max(H * (1 - crown_base), 1e-6)
        k = int(rng.integers(whorl_k[0], whorl_k[1] + 1))
        for j in range(k):
            ang = phase + 2 * np.pi * j / k + rng.normal(0, 0.25)
            elev = np.deg2rad(rng.uniform(-10, 35)) - droop * (1 - rel) * 0.5
            dirv = np.array([np.cos(ang) * np.cos(elev), np.sin(ang) * np.cos(elev), np.sin(elev)])
            length = Lmax * rel ** rng.uniform(0.8, 1.2) * rng.uniform(0.75, 1.15)
            if length > 1.5 * D and rng.random() > 0.12:   # some whorl branches died
                grow_branch(ti, dirv, length, 0)
        phase += rng.uniform(0.3, 1.0)
        z += whorl_dz * rng.uniform(0.8, 1.2)
    return np.asarray(pos), np.asarray(parent, dtype=np.int64), D


# --------------------------------------------------------------------------------------
def _smooth(pos, parent, iters=2, lam=0.35):
    ch = children_lists(parent)
    for _ in range(iters):
        new = pos.copy()
        for n in range(1, len(pos)):
            if len(ch[n]) == 1:
                new[n] = (1 - lam) * pos[n] + lam * 0.5 * (pos[parent[n]] + pos[ch[n][0]])
        pos = new
    return pos


def _prune_tiny(pos, parent, min_nodes=2):
    """Remove twig stubs made of a single node hanging off a junction (SC artefacts)."""
    ch = children_lists(parent)
    remove = np.zeros(len(pos), bool)
    for n in range(1, len(pos)):
        if not ch[n] and len(ch[parent[n]]) >= 2 and np.random.random() < 0.5:
            remove[n] = True
    keep = ~remove
    new_id = -np.ones(len(pos), dtype=np.int64)
    new_id[keep] = np.arange(keep.sum())
    np_ = parent[keep]
    np_ = np.where(np_ >= 0, new_id[np.maximum(np_, 0)], -1)
    return pos[keep], np_


def generate_tree(species, seed):
    """Generate a single tree.  Returns dict(pos, radius, parent, species, height, D, pipe_exp, leaf params)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2 ** 32))
    sp = SPECIES[species]
    H = _u(rng, sp["H"])
    cb = _u(rng, sp["crown_base"])
    cw = _u(rng, sp["crown_w"])
    if sp["engine"] == "sc":
        pos, parent, D = space_colonization(
            rng, H, cb, cw, sp["shape"], _u(rng, sp["tropism"]), int(_u(rng, sp["n_attr"])),
            stems=int(rng.integers(sp["stems"][0], sp["stems"][1] + 1)) if "stems" in sp else 1)
    else:
        pos, parent, D = whorl_tree(rng, H, cb, cw, _u(rng, sp["droop"]), sp["whorl_k"])
    pos, parent = _prune_tiny(pos, parent)
    pos = _smooth(pos, parent)
    e = _u(rng, sp["pipe"])
    r_tip = _u(rng, sp["tip_r"])
    order = topo_order(parent)
    r = pipe_radii(parent, r_tip * rng.uniform(0.8, 1.25, len(parent)), e, order)
    # allometric rescale of the trunk: DBH [m] ~ 0.009 * H^1.25 (+-20%), keep tips >= 1.5 mm
    Hreal = float(pos[:, 2].max())
    target_r0 = 0.5 * 0.009 * Hreal ** 1.25 * rng.uniform(0.8, 1.2)
    if species in ("shrub",):
        target_r0 = 0.5 * 0.02 * Hreal * rng.uniform(0.8, 1.2)
    # power-law rescale keeps the tips thin while matching the base radius
    gamma = np.log(target_r0 / r_tip) / max(np.log(r[0] / r_tip), 1e-6)
    gamma = float(np.clip(gamma, 0.5, 1.5))
    r = r_tip * (r / r_tip) ** gamma
    r = np.maximum(r, 0.0015)
    # a slight butt swell at the base
    z = pos[:, 2]
    r = r * (1 + 0.25 * np.exp(-z / 0.3))
    leaf = sp["leaf"]
    return dict(pos=pos.astype(np.float32), radius=r.astype(np.float32), parent=parent.astype(np.int32),
                species=species, height=np.float32(Hreal), D=np.float32(D), pipe_exp=np.float32(e),
                leaf_size=np.float32(_u(rng, leaf["size"])), leaf_density=np.float32(_u(rng, leaf["density"])),
                leaf_kind=leaf["kind"], seed=np.int64(seed))
