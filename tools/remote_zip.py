"""Read single members of a huge remote ZIP (HTTP range requests, ZIP64 aware).

Used to pull only the individual-tree point clouds and inventory tables out of the
5-18 GB per-plot archives of Weiser et al. (2022), PANGAEA 942856.
"""
import struct
import urllib.request
import zlib


def _get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "Tree3DRecon/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Tree3DRecon/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


class RemoteZip:
    def __init__(self, url):
        self.url = url
        self.size = _size(url)
        tail = _get(url, max(0, self.size - 66000), self.size - 1)
        e = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[e + 12:e + 20])
        n = struct.unpack("<H", tail[e + 10:e + 12])[0]
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n == 0xFFFF:
            l = tail.rfind(b"PK\x06\x07")
            z64_off = struct.unpack("<Q", tail[l + 8:l + 16])[0]
            z = _get(url, z64_off, z64_off + 56)
            n, cd_size, cd_off = struct.unpack("<QQQ", z[32:56])
        cd = _get(url, cd_off, cd_off + cd_size - 1)
        self.entries = {}
        k = 0
        while k + 46 <= len(cd) and cd[k:k + 4] == b"PK\x01\x02":
            comp, = struct.unpack("<H", cd[k + 10:k + 12])
            csz, usz = struct.unpack("<II", cd[k + 20:k + 28])
            fl, el, cl = struct.unpack("<HHH", cd[k + 28:k + 34])
            off, = struct.unpack("<I", cd[k + 42:k + 46])
            name = cd[k + 46:k + 46 + fl].decode("utf-8", "ignore")
            extra = cd[k + 46 + fl:k + 46 + fl + el]
            j = 0
            while j + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[j:j + 4])
                if hid == 1:
                    vals = extra[j + 4:j + 4 + hsz]
                    q = 0
                    if usz == 0xFFFFFFFF:
                        usz = struct.unpack("<Q", vals[q:q + 8])[0]; q += 8
                    if csz == 0xFFFFFFFF:
                        csz = struct.unpack("<Q", vals[q:q + 8])[0]; q += 8
                    if off == 0xFFFFFFFF:
                        off = struct.unpack("<Q", vals[q:q + 8])[0]
                j += 4 + hsz
            self.entries[name] = (comp, csz, usz, off)
            k += 46 + fl + el + cl

    def names(self):
        return list(self.entries)

    def read(self, name):
        comp, csz, usz, off = self.entries[name]
        h = _get(self.url, off, off + 29)
        fl, el = struct.unpack("<HH", h[26:30])
        data = _get(self.url, off + 30 + fl + el, off + 30 + fl + el + csz - 1)
        if comp == 8:
            data = zlib.decompress(data, -15)
        return data
