"""Stream-download the Smart-Tree 'Synthetic-Trees' dataset (Dobbs et al. 2023) and convert
each tree on the fly into a compact npz (skeleton graph + downsampled points).

The original Dropbox archive is > 150 GB (3.4 M float64 points + RGB per tree).  We parse the
ZIP stream sequentially (entries are STORED with data descriptors; every entry is itself an
.npz = zip with known sizes), so nothing big ever touches the disk.

Output: data/synthetic_trees/{branches,branches_foliage}/<species>/<name>.npz with
    pos (K,3) radius (K,) parent (K,)   -- skeleton graph, z-up, metres
    points (<=120k,3) float32, cls (<=120k,) int8 (0 wood, 1 leaf as in the source class_l)
and split.json copied verbatim.
"""
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile

import numpy as np

URL = "https://www.dropbox.com/sh/dkp3sgw6wpdiaam/AAAIRy8liOpy-y9jM6KCiNpNa?dl=1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "synthetic_trees")
MAX_PTS = 120000


class Stream:
    def __init__(self, resp):
        self.r = resp
        self.buf = bytearray()
        self.eof = False

    def fill(self, n):
        while len(self.buf) < n and not self.eof:
            chunk = self.r.read(8 << 20)
            if not chunk:
                self.eof = True
                break
            self.buf += chunk
        return len(self.buf) >= n

    def read(self, n):
        self.fill(n)
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def skip_to(self, sig):
        """Discard bytes until `sig` is at the front.  Returns False at EOF."""
        while True:
            i = self.buf.find(sig)
            if i >= 0:
                del self.buf[:i]
                return True
            keep = len(self.buf) - len(sig) + 1
            if keep > 0:
                del self.buf[:keep]
            if self.eof or not self.fill(len(self.buf) + (8 << 20)):
                if self.eof:
                    return False


def to_graph(sk_xyz, sk_r, bid, bpar, bnum):
    """Convert Synthetic-Trees branch lists to (pos, radius, parent), z-up."""
    pos = sk_xyz[:, [0, 2, 1]].astype(np.float64)   # y-up -> z-up
    pos[:, 1] *= -1                                   # keep right-handed
    rad = sk_r.reshape(-1).astype(np.float64)
    parent = -np.ones(len(pos), dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(bnum)[:-1]])
    first_of = {}
    span = {}
    for b, s, n in zip(bid, start, bnum):
        span[int(b)] = (int(s), int(n))
        first_of[int(b)] = int(s)
        if n > 1:
            parent[s + 1:s + n] = np.arange(s, s + n - 1)
    for b, p in zip(bid, bpar):
        s, n = span[int(b)]
        if n == 0 or int(p) not in span:
            continue
        ps, pn = span[int(p)]
        if pn == 0:
            continue
        d = np.linalg.norm(pos[ps:ps + pn] - pos[s], axis=1)
        parent[s] = ps + int(np.argmin(d))
    # keep the largest connected arborescence rooted at the lowest root
    roots = np.nonzero(parent < 0)[0]
    root = roots[np.argmin(pos[roots, 2])]
    ch = [[] for _ in range(len(pos))]
    for i, p in enumerate(parent):
        if p >= 0:
            ch[p].append(i)
    keep = np.zeros(len(pos), bool)
    st = [root]
    while st:
        n = st.pop()
        keep[n] = True
        st.extend(ch[n])
    new = -np.ones(len(pos), dtype=np.int64)
    new[keep] = np.arange(keep.sum())
    par = parent[keep]
    par = np.where(par >= 0, new[np.maximum(par, 0)], -1)
    # root first
    order = np.argsort(par >= 0, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    par = par[order]
    par = np.where(par >= 0, inv[np.maximum(par, 0)], -1)
    pos = pos[keep][order]
    rad = rad[keep][order]
    base = pos[0].copy()
    pos -= base
    return pos, rad, par, base


def convert(name, blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    g = lambda k: np.load(z.open(k + ".npy"))
    xyz = g("xyz")
    cls = g("class_l") if "class_l.npy" in z.namelist() else np.zeros(len(xyz), np.int64)
    pos, rad, par, base = to_graph(g("skeleton_xyz"), g("skeleton_radii"), g("branch_id"),
                                   g("branch_parent_id"), g("branch_num_elements"))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(xyz), size=min(MAX_PTS, len(xyz)), replace=False)
    pts = xyz[sel][:, [0, 2, 1]]
    pts[:, 1] *= -1
    pts -= base
    dst = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, pos=pos.astype(np.float32), radius=rad.astype(np.float32), parent=par.astype(np.int32),
                        points=pts.astype(np.float32), cls=cls[sel].astype(np.int8))


def main():
    os.makedirs(OUT, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    s = Stream(urllib.request.urlopen(req, timeout=120))
    n_done = 0
    while s.skip_to(b"PK\x03\x04"):
        if not s.fill(30):
            break
        hdr = s.read(30)
        flag, = struct.unpack("<H", hdr[6:8])
        csz, = struct.unpack("<I", hdr[18:22])
        fl, el = struct.unpack("<HH", hdr[26:30])
        name = s.read(fl).decode("utf-8", "ignore").lstrip("/")
        s.read(el)
        if name.endswith("/"):
            continue
        if name.endswith(".npz"):
            dst = os.path.join(OUT, name)
            # inner npz: sequential inner entries (sizes known) until the inner EOCD record
            blob = bytearray()
            while True:
                s.fill(4)
                sig = bytes(s.buf[:4])
                if sig == b"PK\x03\x04":
                    h = s.read(30)
                    isz, = struct.unpack("<I", h[18:22])
                    ifl, iel = struct.unpack("<HH", h[26:30])
                    nm_extra = s.read(ifl + iel)
                    if isz == 0xFFFFFFFF:  # zip64 inner entry: size in extra field
                        ex = nm_extra[ifl:]
                        k = 0
                        while k + 4 <= len(ex):
                            hid, hsz = struct.unpack("<HH", ex[k:k + 4])
                            if hid == 1:
                                isz = struct.unpack("<Q", ex[k + 4:k + 12])[0]
                                break
                            k += 4 + hsz
                    blob += h + nm_extra + s.read(isz)
                elif sig in (b"PK\x01\x02", b"PK\x06\x06", b"PK\x06\x07", b"PK\x05\x06"):
                    # central directory till EOCD
                    s.skip_to(b"PK\x05\x06")
                    s.fill(22)
                    e = s.read(22)
                    clen, = struct.unpack("<H", e[20:22])
                    s.read(clen)
                    break
                else:
                    break
            if os.path.exists(dst):
                continue
            try:
                # rebuild a valid zip from local entries only
                convert(name, _rebuild_zip(bytes(blob)))
                n_done += 1
                print(f"[{n_done}] {name}", flush=True)
            except Exception as ex:
                print(f"FAILED {name}: {ex}", flush=True)
        elif name.endswith(".json") or name.endswith(".txt"):
            s.fill(1 << 20)
            raw = bytes(s.buf)
            end = min([i for i in (raw.find(b"PK\x07\x08"), raw.find(b"PK\x03\x04")) if i >= 0] or [len(raw)])
            with open(os.path.join(OUT, os.path.basename(name)), "wb") as f:
                f.write(raw[:end])
    print("finished, trees converted:", n_done)


def _rebuild_zip(local_entries):
    """Parse local file entries (stored) and write a fresh in-memory zip."""
    out = io.BytesIO()
    zf = zipfile.ZipFile(out, "w", zipfile.ZIP_STORED)
    k = 0
    while k + 30 <= len(local_entries) and local_entries[k:k + 4] == b"PK\x03\x04":
        h = local_entries[k:k + 30]
        comp, = struct.unpack("<H", h[8:10])
        csz, = struct.unpack("<I", h[18:22])
        fl, el = struct.unpack("<HH", h[26:30])
        nm = local_entries[k + 30:k + 30 + fl].decode()
        ex = local_entries[k + 30 + fl:k + 30 + fl + el]
        if csz == 0xFFFFFFFF:
            j = 0
            while j + 4 <= len(ex):
                hid, hsz = struct.unpack("<HH", ex[j:j + 4])
                if hid == 1:
                    csz = struct.unpack("<Q", ex[j + 4:j + 12])[0]
                    break
                j += 4 + hsz
        d0 = k + 30 + fl + el
        data = local_entries[d0:d0 + csz]
        if comp == 8:
            import zlib
            data = zlib.decompress(data, -15)
        zf.writestr(nm, data)
        k = d0 + csz
    zf.close()
    return out.getvalue()


if __name__ == "__main__":
    sys.exit(main())
