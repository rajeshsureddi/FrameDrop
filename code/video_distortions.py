"""Video distortion bank (SDR, up to 4K) for synthetic VQA training data.

Video, not image, in three ways:
  1. Time-varying severity. Every op has an envelope over seconds:
       const | interval (onset/offset) | ramp | osc (pumping/hunting) | walk (drift)
     Bipolar ops (exposure, white balance, hue) can swing both ways over time.
  2. Correct temporal statistics. Sensor noise is fresh every frame; fixed-pattern
     noise is frozen while the scene moves; low-frequency fields (haze density,
     flare position, blur direction) drift smoothly; shutter blur and ghosting
     use real neighbouring frames, not a synthetic kernel.
  3. Video-only distortions: lighting flicker, rolling-shutter flicker bands,
     exposure pumping, AWB drift, focus hunting, temporal ghosting, shimmering
     from aliasing in motion.

Design guarantees (tested in test_video_distortions.py):
  * severity 0 is exactly the identity; severity ladders that share a plan's
    parameters are monotone (free ranking pairs, no labels needed)
  * crop-consistent: a 4K frame cropped after distortion equals the crop
    distorted alone (pass `offset` and `full_size`, plus Plan.margin() pixels of
    context). This lets Tile-MAE distort native-resolution tiles on the fly.
  * deterministic and seekable: noise uses a counter-based hash of
    (seed, source frame index, absolute pixel), so any frame range renders
    identically and held frames (applied later as index remaps) repeat their noise.
  * pixel sizes are defined at 2160p and scale with frame height.

Input: SDR BT.709 video decoded to RGB in [0,1] (uint8/uint16/float). HDR is
out of scope; linear-light ops use the sRGB/BT.709 transfer approximation.

Not here on purpose:
  * compression blocking / mosquito noise / codec colour bleeding -> real encodes
    (Step 2 ladders); never fake codec artifacts with JPEG/DCT tricks
  * frame holds / judder -> drops.py index remaps, applied after this module
  * "legibility loss": an outcome of other distortions, not an operator

CLI preview:
  python video_distortions.py in.mp4 out.mp4 --op flicker --severity 0.6 --seconds 5
  python video_distortions.py in.mp4 out.mp4 --random --seed 3
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import cv2
import numpy as np

REF_H = 2160.0
STAGES = ("scene", "optics", "sensor", "isp", "scale", "output")
KR, KB = 0.2126, 0.0722


# ============================================================ low-level helpers
def lin(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def unlin(y):
    y = np.maximum(y, 0.0)
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * np.power(y, 1 / 2.4) - 0.055).astype(np.float32)


def luma(x):
    return KR * x[..., 0] + (1 - KR - KB) * x[..., 1] + KB * x[..., 2]


def to_ycc(x):
    y = luma(x)
    return y, (x[..., 2] - y) / (2 * (1 - KB)), (x[..., 0] - y) / (2 * (1 - KR))


def from_ycc(y, cb, cr):
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / (1 - KR - KB)
    return np.stack([r, g, b], -1).astype(np.float32)


def blur(x, sigma):
    if sigma < 0.05:
        return x
    return cv2.GaussianBlur(np.ascontiguousarray(x, np.float32), (0, 0), float(sigma),
                            borderType=cv2.BORDER_REFLECT)


def _smooth(f):
    return f * f * (3 - 2 * f)


def _hash(seed, t, c, x, y):
    """Counter-based 32-bit hash; x, y broadcastable integer arrays (absolute px)."""
    with np.errstate(over="ignore"):
        x = np.asarray(x, np.int64).astype(np.uint32)
        y = np.asarray(y, np.int64).astype(np.uint32)
        k = np.uint32((int(seed) * 0x9E3779B1 ^ int(t) * 0xCB1AB31F ^ int(c) * 0x165667B1)
                      & 0xFFFFFFFF)
        h = (x * np.uint32(0x8DA6B343)) ^ (y * np.uint32(0xD8163841)) ^ k
        h = h ^ (h >> np.uint32(16))
        h = h * np.uint32(0x7FEB352D)
        h = h ^ (h >> np.uint32(15))
        h = h * np.uint32(0x846CA68B)
        h = h ^ (h >> np.uint32(16))
    return h


def _u01(h):
    return ((h >> np.uint32(8)).astype(np.float32) + 0.5) * np.float32(1 / 16777216)


def gauss_field(C, c):
    """Per-pixel N(0,1), fresh per frame (keyed by source frame index)."""
    x, y = C.xa.astype(np.int64)[None, :], C.ya.astype(np.int64)[:, None]
    u1 = _u01(_hash(C.seed, C.t, 2 * c, x, y))
    u2 = _u01(_hash(C.seed, C.t, 2 * c + 1, x, y))
    return np.sqrt(-2 * np.log(u1)) * np.cos(np.float32(2 * np.pi) * u2)


def vnoise(xa, ya, cell, seed, c=0, t=0.0, tcell=0.0):
    """Smooth value noise in [-1,1] over absolute pixel coords (and time, in frames)."""
    gx, gy = xa / cell, ya / cell
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = _smooth(gx - ix)[None, :], _smooth(gy - iy)[:, None]
    kx = np.arange(ix.min(), ix.max() + 2)
    ky = np.arange(iy.min(), iy.max() + 2)
    lx, ly = (ix - ix.min())[None, :], (iy - iy.min())[:, None]

    def plane(it):
        G = _u01(_hash(seed, it, c, kx[None, :], ky[:, None])) * 2 - 1
        a, b = G[ly, lx], G[ly, lx + 1]
        cc, d = G[ly + 1, lx], G[ly + 1, lx + 1]
        return ((a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy).astype(np.float32)

    if tcell <= 0:
        return plane(0)
    gt = t / tcell
    it = int(math.floor(gt))
    ft = _smooth(gt - it)
    return plane(it) if ft == 0 else plane(it) * (1 - ft) + plane(it + 1) * ft


def walk1d(ts, cell, seed, c=99):
    """Smooth random walk in [0,1] over time (seconds)."""
    out = np.empty(len(ts), np.float32)
    for i, t in enumerate(ts):
        g = t / cell
        k = math.floor(g)
        f = _smooth(g - k)
        a = float(_u01(_hash(seed, k, c, 0, 0)))
        b = float(_u01(_hash(seed, k + 1, c, 0, 0)))
        out[i] = a * (1 - f) + b * f
    return out


def _remap(img, mapx, mapy, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    mx = np.ascontiguousarray(np.broadcast_to(mapx, (mapy.shape[0], mapx.shape[1])), np.float32)
    my = np.ascontiguousarray(np.broadcast_to(mapy, (mapy.shape[0], mapx.shape[1])), np.float32)
    return cv2.remap(np.ascontiguousarray(img, np.float32), mx, my, interp,
                     borderMode=cv2.BORDER_REPLICATE)


def _grid_resample(x, f, C, fetch, up):
    """Resample on an ABSOLUTE grid with pitch f (crop-consistent).
    fetch: interpolation for reading samples at abs positions j*f
    up:    'linear' (bilinear back-up) or 'block' (nearest, block-aligned)."""
    xa, ya = C.xa, C.ya
    jx = np.arange(math.ceil(xa[0] / f), math.floor(xa[-1] / f) + 1)
    jy = np.arange(math.ceil(ya[0] / f), math.floor(ya[-1] / f) + 1)
    if len(jx) < 2 or len(jy) < 2:
        return x
    coarse = _remap(x, (jx * f - xa[0])[None, :], (jy * f - ya[0])[:, None], fetch)
    if up == "block":
        mx = (np.floor(xa / f) - jx[0])[None, :]
        my = (np.floor(ya / f) - jy[0])[:, None]
        return _remap(coarse, mx, my, cv2.INTER_NEAREST)
    return _remap(coarse, (xa / f - jx[0])[None, :], (ya / f - jy[0])[:, None], cv2.INTER_LINEAR)


# ================================================================ context/plan
@dataclass
class Ctx:
    xa: np.ndarray          # (w,) absolute x of each column
    ya: np.ndarray          # (h,) absolute y of each row
    W: int
    H: int
    scale: float            # H / 2160: pixel sizes below are at 2160p
    fps: float
    seed: int = 0
    t: int = 0              # absolute SOURCE frame index
    ts: float = 0.0         # seconds


@dataclass
class Envelope:
    kind: str = "const"     # const | interval | ramp | osc | walk
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    seed: int = 0

    def __call__(self, ts):
        ts = np.asarray(ts, np.float64)
        if self.kind == "const":
            return np.ones(len(ts), np.float32)
        if self.kind == "interval":            # a=start, b=end, c=ramp seconds
            r = max(self.c, 1e-3)
            up = _smooth(np.clip((ts - self.a) / r, 0, 1))
            dn = _smooth(np.clip((self.b - ts) / r, 0, 1))
            return (up * dn).astype(np.float32)
        if self.kind == "ramp":                # a=start, b=end, c<0 -> decreasing
            e = _smooth(np.clip((ts - self.a) / max(self.b - self.a, 1e-3), 0, 1))
            return (1 - e if self.c < 0 else e).astype(np.float32)
        if self.kind == "osc":                 # a=Hz, b=depth, c=phase
            return ((1 - self.b) + self.b * (0.5 + 0.5 * np.sin(2 * np.pi * self.a * ts + self.c))
                    ).astype(np.float32)
        if self.kind == "walk":                # a=cell seconds, b=floor
            return (self.b + (1 - self.b) * walk1d(ts, self.a, self.seed)).astype(np.float32)
        raise ValueError(self.kind)


@dataclass
class Support:
    kind: str = "global"    # global | blob (soft ellipse drifting over time)
    cx: float = 0.5
    cy: float = 0.5
    rx: float = 0.3
    ry: float = 0.3
    vx: float = 0.0         # normalized units per second
    vy: float = 0.0
    soft: float = 0.5

    def mask(self, C):
        if self.kind == "global":
            return None
        u = (C.xa / C.W)[None, :]
        v = (C.ya / C.H)[:, None]
        cx, cy = self.cx + self.vx * C.ts, self.cy + self.vy * C.ts
        d = np.sqrt(((u - cx) / self.rx) ** 2 + ((v - cy) / self.ry) ** 2)
        e = np.clip((d - (1 - self.soft)) / self.soft, 0, 1)
        return (1 - _smooth(e)).astype(np.float32)[..., None]


@dataclass
class Op:
    name: str
    severity: float                         # 0..1, monotone strength
    P: dict
    seed: int
    envelope: Envelope = field(default_factory=Envelope)
    support: Support = field(default_factory=Support)
    bipolar: bool = False

    def strength(self, ts):
        e = self.envelope(ts)
        return (self.severity * (2 * e - 1) if self.bipolar else self.severity * e).astype(np.float32)


@dataclass
class Plan:
    ops: list
    fps: float
    seed: int = 0

    def sorted_ops(self):
        # stage order, then registry order inside a stage (e.g. denoise before
        # sharpening) -> the same plan always renders the same way
        order = list(REGISTRY)
        return sorted(self.ops, key=lambda o: (STAGES.index(REGISTRY[o.name].stage),
                                               order.index(o.name)))

    def margin(self, H):
        """Spatial context (px at this resolution) a crop needs on each side."""
        sc = H / REF_H
        return int(math.ceil(sum(REGISTRY[o.name].margin(o.P, sc) for o in self.ops))) + 2

    def margin_t(self):
        """Previous source frames needed before the first frame you keep."""
        return max([REGISTRY[o.name].margin_t(o.P) for o in self.ops] + [0])

    def with_severity(self, sev):
        """Same parameters/envelopes/seeds, scaled severity -> ranking ladders."""
        return replace(self, ops=[replace(o, severity=float(np.clip(o.severity * sev, 0, 1)))
                                  for o in self.ops])


# ================================================================= operators
# Each fn(x, s, P, C) -> y with x (h,w,3) float32 SDR RGB in [0,1].
# s is the current strength (0..1, or -1..1 for bipolar ops). fn(x, 0, ...) == x.

def f_haze(x, s, P, C):
    v = (C.ya / C.H)[:, None]
    n = vnoise(C.xa, C.ya, P["cell"] * C.H, C.seed, 11, C.t, P["tcell"] * C.fps)
    depth = np.clip(P["gy"] * (1 - v) + (1 - P["gy"]) * 0.5 + 0.35 * n, 0, 1)
    tr = np.exp(-P["beta"] * s * depth)[..., None]
    return x * tr + np.asarray(P["A"], np.float32) * (1 - tr)


def f_flicker(x, s, P, C):
    ph = 2 * np.pi * P["hz"] * C.ts + P["phase"]
    if P["mode"] == "rolling":           # LED/mains bands rolling through the frame
        ph = ph - 2 * np.pi * P["bands"] * (C.ya / C.H)[:, None, None]
    g = 1 + s * P["amp"] * np.sin(ph)
    return unlin(lin(x) * g)


def f_flare(x, s, P, C):
    sx = (P["x0"] + P["vx"] * C.ts + 0.015 * math.sin(2 * np.pi * 0.3 * C.ts)) * C.W
    sy = (P["y0"] + P["vy"] * C.ts + 0.010 * math.cos(2 * np.pi * 0.23 * C.ts)) * C.H
    X, Y, R = C.xa[None, :], C.ya[:, None], float(C.H)
    dx, dy = (X - sx) / R, (Y - sy) / R
    d = np.sqrt(dx * dx + dy * dy)
    glare = (P["veil"] * np.exp(-(d / 0.45) ** 2) + 1.5 * np.exp(-d / 0.02)
             + 0.6 * np.exp(-(dy / 0.006) ** 2) * np.exp(-np.abs(dx) / 0.6))
    add = glare[..., None] * np.asarray(P["tint"], np.float32)
    for (tt, rad, col, a) in P["ghosts"]:
        gx, gy = sx + (C.W / 2 - sx) * tt, sy + (C.H / 2 - sy) * tt
        dd = np.sqrt((X - gx) ** 2 + (Y - gy) ** 2) / (rad * R)
        disc = a / (1 + np.exp(np.clip((dd - 1) * 14, -50, 50)))
        add = add + disc[..., None] * np.asarray(col, np.float32)
    return unlin(lin(x) + s * P["gain"] * add)


def f_bloom(x, s, P, C):
    xl = lin(x)
    hi = np.maximum(xl - P["thr"], 0)
    return unlin(xl + s * P["gain"] * blur(hi, P["sigma"] * C.scale))


def f_ca(x, s, P, C):
    k = s * P["k"]
    cx, cy = C.W / 2, C.H / 2
    out = x.copy()
    for ch, sc in ((0, 1 + k), (2, 1 - k)):
        mx = (cx + (C.xa - cx) / sc - C.xa[0])[None, :]
        my = (cy + (C.ya - cy) / sc - C.ya[0])[:, None]
        out[..., ch] = _remap(x[..., ch], mx, my)
    return out


def f_fringe(x, s, P, C):
    b = np.clip((luma(x) - 0.82) / 0.18, 0, 1)
    halo = np.clip(blur(b, P["sigma"] * C.scale) - b, 0, 1)
    a = np.clip(s * P["gain"] * halo, 0, 1)[..., None]
    return x * (1 - a) + np.asarray(P["col"], np.float32) * a


def f_defocus(x, s, P, C):
    return blur(x, abs(s) * P["sigma"] * C.scale)


def f_shake(x, s, P, C):
    L = abs(s) * P["L"] * C.scale
    if L < 1.0:
        return x
    th = P["th0"] + 0.6 * math.sin(2 * np.pi * P["f1"] * C.ts + P["p1"]) \
        + 0.3 * math.sin(2 * np.pi * P["f2"] * C.ts + P["p2"])
    n = int(math.ceil(L)) | 1
    big = np.zeros((n * 4, n * 4), np.uint8)
    c, r = n * 2, L * 2
    p1 = (int(round(c - r * math.cos(th))), int(round(c - r * math.sin(th))))
    p2 = (int(round(c + r * math.cos(th))), int(round(c + r * math.sin(th))))
    cv2.line(big, p1, p2, 255, 2, cv2.LINE_AA)
    k = cv2.resize(big.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
    k /= k.sum() + 1e-8
    return cv2.filter2D(np.ascontiguousarray(x), -1, k, borderType=cv2.BORDER_REFLECT)


def f_shutter(X, st, P, C):
    """Temporal: average the last N source frames in linear light (long shutter /
    frame blending). Needs real motion; uses Plan.margin_t() frames of history."""
    L = lin(X)
    Y = X.copy()
    for i in range(len(X)):
        n = 1 + int(round(abs(st[i]) * (P["N"] - 1)))
        if n > 1:
            Y[i] = unlin(L[max(0, i - n + 1):i + 1].mean(0))
    return Y


def f_exposure(x, s, P, C):
    return unlin(lin(x) * np.float32(2.0 ** (s * P["ev"])))


def f_luma_noise(x, s, P, C):
    xl = lin(x)
    yl = luma(xl)
    sig = np.sqrt(P["shot"] * abs(s) * np.maximum(yl, 0) + (P["read"] * abs(s)) ** 2)
    n = gauss_field(C, 1)
    return unlin(xl + (sig * n)[..., None])


def f_chroma_noise(x, s, P, C):
    y, cb, cr = to_ycc(x)
    cell = P["cell"] * C.scale
    ncb = vnoise(C.xa, C.ya, cell, C.seed, 21, C.t, 1.0)    # tcell 1 frame: fresh each frame
    ncr = vnoise(C.xa, C.ya, cell, C.seed, 22, C.t, 1.0)
    a = abs(s) * P["amp"]
    return from_ycc(y, cb + a * ncb, cr + a * ncr)


def f_fpn(x, s, P, C):
    """Fixed-pattern noise: frozen in time (t=0), so the scene slides under it.
    Column offsets + per-pixel gain (PRNU), modulated by a large-scale field
    (non-uniformity)."""
    S = replace(C, t=0)
    col = _u01(_hash(C.seed, 0, 31, C.xa.astype(np.int64)[None, :], 0)) * 2 - 1
    prnu = gauss_field(S, 32)
    amp = 0.3 + 0.7 * (0.5 + 0.5 * vnoise(C.xa, C.ya, P["cell"] * C.H, C.seed, 33))
    xl = lin(x)
    a = abs(s) * amp
    out = xl * (1 + (a * P["prnu"] * prnu)[..., None]) + (a * P["col"] * col)[..., None]
    return unlin(out)


def f_demosaic(x, s, P, C):
    """RGGB mosaic on the ABSOLUTE pixel grid + bilinear demosaic -> zipper,
    false colour and colour moire on fine detail (shimmers when it moves)."""
    py = (C.ya.astype(np.int64) % 2)[:, None]
    px = (C.xa.astype(np.int64) % 2)[None, :]
    masks = [((py == 0) & (px == 0)), ((py + px) % 2 == 1), ((py == 1) & (px == 1))]
    masks = [np.broadcast_to(m, x.shape[:2]).astype(np.float32) for m in masks]
    mos = sum(x[..., c] * masks[c] for c in range(3))
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    out = np.empty_like(x)
    for c in range(3):
        num = cv2.filter2D(mos * masks[c], -1, K, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(masks[c], -1, K, borderType=cv2.BORDER_REFLECT)
        out[..., c] = np.where(masks[c] > 0, mos, num / np.maximum(den, 1e-6))
    return x + abs(s) * (out - x)


def f_wb(x, s, P, C):
    lg = s * P["mag"] * np.asarray(P["dir"], np.float32)       # (log R/G, log B/G)
    g = np.exp(np.array([lg[0], 0.0, lg[1]], np.float32))
    g /= KR * g[0] + (1 - KR - KB) * g[1] + KB * g[2]            # keep luminance
    return unlin(lin(x) * g)


def f_saturation(x, s, P, C):
    y, cb, cr = to_ycc(x)
    k = max(0.0, 1 + s * P["k"])
    return from_ycc(y, cb * k, cr * k)


def f_hue(x, s, P, C):
    y, cb, cr = to_ycc(x)
    a = s * P["deg"] * np.pi / 180
    ca, sa = math.cos(a), math.sin(a)
    return from_ycc(y, ca * cb - sa * cr, sa * cb + ca * cr)


def f_oversmooth(x, s, P, C):
    """Heavy noise reduction: smooths flat/texture regions, keeps strong edges
    (waxy, plastic look; fine texture vanishes)."""
    sig = abs(s) * P["sigma"] * C.scale
    b = blur(x, sig)
    Y = luma(x)
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    g = blur(np.sqrt(gx * gx + gy * gy), max(sig * 0.5, 0.5))
    w = np.exp(-(g / P["tau"]) ** 2)[..., None]
    return x + w * (b - x)


def f_ghost(X, st, P, C):
    """Temporal: recursive blend with the previous output (over-strong temporal
    noise reduction / slow display response) -> trailing ghosts behind motion."""
    L = lin(X)
    Y = L.copy()
    for i in range(1, len(X)):
        a = abs(st[i]) * P["a"]
        Y[i] = (1 - a) * L[i] + a * Y[i - 1]
    return unlin(Y)


def f_halo(x, s, P, C):
    b = blur(x, P["sigma"] * C.scale)
    return x + abs(s) * P["gain"] * (x - b)


def f_ringing(x, s, P, C):
    tau = P["tau"] * C.scale
    r = int(math.ceil(3 * tau))
    n = np.arange(-r, r + 1, dtype=np.float32)
    k = np.sinc(n / (2 * tau)) * (0.5 + 0.5 * np.cos(np.pi * n / (r + 1)))
    k = (k / k.sum()).astype(np.float32)
    lp = cv2.sepFilter2D(np.ascontiguousarray(x), -1, k, k, borderType=cv2.BORDER_REFLECT)
    return x + abs(s) * (lp - x)


def f_contrast(x, s, P, C):
    return 0.45 + (x - 0.45) * (1 - abs(s) * P["c"])


def f_dynrange(x, s, P, C):
    lo, hi = abs(s) * P["lo"], 1 - abs(s) * P["hi"]
    return lo + (hi - lo) * x


def f_shadow_clip(x, s, P, C):
    b = abs(s) * P["b"]
    return (x - b) / (1 - b)


def f_shadow_comp(x, s, P, C):
    k = P["knee"]
    return np.where(x < k, k * np.power(np.clip(x / k, 0, 1), 1 + abs(s) * P["g"]), x)


def f_highlight_clip(x, s, P, C):
    k = P["knee"]
    return np.where(x > k, k + (x - k) * (1 + abs(s) * P["g"]), x)


def f_rescale(x, s, P, C):
    f = 1 + abs(s) * (P["f"] - 1)
    if f < 1.02:
        return x
    xb = blur(x, 0.5 * math.sqrt(f * f - 1))
    return _grid_resample(xb, f, C, cv2.INTER_LINEAR, "linear")


def f_aliasing(x, s, P, C):
    f = 1 + abs(s) * (P["f"] - 1)
    if f < 1.02:
        return x
    return _grid_resample(x, f, C, cv2.INTER_NEAREST, "linear")


def f_pixelate(x, s, P, C):
    # power-of-two blocks on an absolute grid are NESTED partitions, so the error
    # can only grow with severity (non-nested sizes, e.g. 3 then 4, can shrink it)
    k = 2 ** int(round(abs(s) * math.log2(P["k"])))
    if k < 2:
        return x
    # block MEAN (box anchored at each block's top-left), read on the absolute grid
    xb = cv2.boxFilter(np.ascontiguousarray(x), -1, (k, k), anchor=(0, 0),
                       borderType=cv2.BORDER_REFLECT)
    return _grid_resample(xb, k, C, cv2.INTER_NEAREST, "block")


def _quant(v, s, bits_removed, lo=0.0):
    """Nested uniform quantiser: remove round(s*bits) bits from an 8-bit grid,
    reconstruct at cell centres -> error grows monotonically with severity."""
    b = 8 - int(round(abs(s) * bits_removed))
    if b >= 8:
        return v
    step = (1.0 / 2 ** b)
    return (np.floor((v - lo) / step) + 0.5) * step + lo


def f_posterize(x, s, P, C):
    return _quant(np.clip(x, 0, 1 - 1e-6), s, P["bits"])


def f_chroma_bleed(x, s, P, C):
    y, cb, cr = to_ycc(x)
    sig = abs(s) * P["sigma"] * C.scale
    dx = abs(s) * P["dx"] * C.scale
    mx = (np.arange(len(C.xa), dtype=np.float32) - dx)[None, :]
    my = np.arange(len(C.ya), dtype=np.float32)[:, None]
    cb = _remap(blur(cb, sig), mx, my)
    cr = _remap(blur(cr, sig), mx, my)
    return from_ycc(y, cb, cr)


def f_chroma_quant(x, s, P, C):
    y, cb, cr = to_ycc(x)
    return from_ycc(y, _quant(cb, s, P["bits"], -0.5), _quant(cr, s, P["bits"], -0.5))


# ================================================================== registry
@dataclass
class Spec:
    fn: Callable
    stage: str
    family: str                         # tone | detail | color | temporal
    terms: Callable                     # (P, sign) -> lexicon terms
    sample: Callable                    # rng -> P
    margin: Callable = lambda P, sc: 0.0
    margin_t: Callable = lambda P: 0
    temporal: bool = False              # fn works on the whole clip window
    bipolar: bool = False               # may swing both ways over time
    local: bool = True                  # may be restricted to a drifting region
    env_w: tuple = (0.55, 0.15, 0.10, 0.10, 0.10)   # const interval ramp osc walk
    group: str = ""                     # mutually exclusive within a plan


T = lambda *a: (lambda P, sg: a)
U = lambda r, lo, hi: float(r.uniform(lo, hi))
NO_ENV = (1.0, 0, 0, 0, 0)

REGISTRY: dict[str, Spec] = {
    # ---------------- scene / optics
    "haze": Spec(f_haze, "scene", "tone", T("haze veiling", "low contrast"),
                 lambda r: dict(beta=U(r, 0.8, 2.5), gy=U(r, 0.3, 1.0), cell=U(r, 0.3, 0.8),
                                tcell=U(r, 2, 8), A=[U(r, .85, .95)] * 2 + [U(r, .9, 1)]),
                 env_w=(0.5, 0.2, 0.2, 0, 0.1)),
    "flicker": Spec(f_flicker, "sensor", "temporal",
                    lambda P, sg: ("rolling banding", "flicker") if P["mode"] == "rolling" else ("flicker",),
                    lambda r: dict(mode=r.choice(["global", "rolling"]), hz=float(np.exp(U(r, np.log(0.5), np.log(12)))),
                                   bands=U(r, 1, 6), amp=U(r, 0.15, 0.4), phase=U(r, 0, 6.28)),
                    local=False, env_w=(0.8, 0.2, 0, 0, 0), group="flicker"),
    "flare": Spec(f_flare, "optics", "tone", T("flare", "low contrast"),
                  lambda r: dict(x0=U(r, 0.05, 0.95), y0=U(r, 0.05, 0.5), vx=U(r, -0.04, 0.04),
                                 vy=U(r, -0.02, 0.02), veil=U(r, 0.05, 0.25), gain=U(r, 0.6, 1.5),
                                 tint=[1.0, U(r, .85, 1), U(r, .6, .9)],
                                 ghosts=[(U(r, 1.2, 2.4), U(r, .03, .12), list(r.uniform(.2, 1, 3)), U(r, .05, .25))
                                         for _ in range(int(r.integers(2, 5)))]),
                  local=False, env_w=(0.5, 0.3, 0.2, 0, 0)),
    "bloom": Spec(f_bloom, "optics", "tone", T("light bloom", "halo"),
                  lambda r: dict(thr=U(r, 0.5, 0.85), sigma=U(r, 10, 30), gain=U(r, 1, 3)),
                  margin=lambda P, sc: 3 * P["sigma"] * sc),
    "chromatic_aberration": Spec(f_ca, "optics", "color", T("chromatic aberration", "color fringing"),
                                 lambda r: dict(k=U(r, 0.0008, 0.003)),
                                 margin=lambda P, sc: P["k"] * 2300 * sc + 2, local=False),
    "purple_fringing": Spec(f_fringe, "optics", "color", T("color fringing"),
                            lambda r: dict(sigma=U(r, 2, 6), gain=U(r, 2, 5), col=[.55, .15, .85]),
                            margin=lambda P, sc: 3 * P["sigma"] * sc + 2),
    "defocus": Spec(f_defocus, "optics", "detail",
                    lambda P, sg: ("blur softness", "detail loss"),
                    lambda r: dict(sigma=U(r, 2, 8)), margin=lambda P, sc: 3 * P["sigma"] * sc,
                    bipolar=False, env_w=(0.45, 0.15, 0.1, 0.2, 0.1), group="blur"),
    "shake_blur": Spec(f_shake, "optics", "detail", T("motion blur", "detail loss"),
                       lambda r: dict(L=U(r, 8, 40), th0=U(r, 0, 3.14), f1=U(r, .2, 1.5), f2=U(r, .5, 3),
                                      p1=U(r, 0, 6.28), p2=U(r, 0, 6.28)),
                       margin=lambda P, sc: P["L"] * sc / 2 + 2, group="blur"),
    "shutter_blur": Spec(f_shutter, "sensor", "temporal", T("motion blur", "ghosting"),
                         lambda r: dict(N=int(r.integers(3, 9))), margin_t=lambda P: P["N"] - 1,
                         temporal=True, group="blur"),
    # ---------------- sensor
    "exposure": Spec(f_exposure, "sensor", "tone",
                     lambda P, sg: ("underexposure", "shadow clipping", "tonal detail loss")
                     if sg < 0 else ("overexposure", "highlight clipping"),
                     lambda r: dict(ev=float(r.choice([-1, 1])) * U(r, 0.7, 2.5)),
                     bipolar=True, env_w=(0.5, 0.15, 0.1, 0.15, 0.1)),
    "luma_noise": Spec(f_luma_noise, "sensor", "detail", T("luma noise", "temporal noise"),
                       lambda r: dict(shot=U(r, 0.002, 0.02), read=U(r, 0.004, 0.03))),
    "chroma_noise": Spec(f_chroma_noise, "sensor", "detail", T("chroma noise", "temporal noise"),
                         lambda r: dict(cell=U(r, 2, 8), amp=U(r, 0.02, 0.08))),
    "fixed_pattern_noise": Spec(f_fpn, "sensor", "detail", T("noise non-uniformity", "fixed pattern noise"),
                                lambda r: dict(prnu=U(r, 0.05, 0.3), col=U(r, 0.003, 0.02), cell=U(r, 0.2, 0.6))),
    "demosaic": Spec(f_demosaic, "sensor", "color", T("false color", "color aliasing"),
                     lambda r: dict(), margin=lambda P, sc: 2),
    # ---------------- ISP
    "white_balance": Spec(f_wb, "isp", "color", T("color cast"),
                          lambda r: (lambda a: dict(dir=[math.cos(a), math.sin(a)], mag=U(r, 0.1, 0.45)))(U(r, 0, 6.28)),
                          bipolar=True, env_w=(0.5, 0.1, 0.1, 0.05, 0.25)),
    "saturation": Spec(f_saturation, "isp", "color",
                       lambda P, sg: ("color desaturation", "muddy color") if P["k"] * sg < 0 else ("oversaturation",),
                       lambda r: dict(k=float(r.choice([-0.9, 1.5])) * U(r, 0.6, 1.0)), group="sat"),
    "hue_shift": Spec(f_hue, "isp", "color", T("color fidelity loss", "color cast"),
                      lambda r: dict(deg=float(r.choice([-1, 1])) * U(r, 8, 30)),
                      bipolar=True, env_w=(0.6, 0.1, 0.1, 0.1, 0.1)),
    "over_smoothing": Spec(f_oversmooth, "isp", "detail", T("over-smoothing", "detail loss"),
                           lambda r: dict(sigma=U(r, 2, 6), tau=U(r, 0.15, 0.6)),
                           margin=lambda P, sc: 3 * P["sigma"] * sc + 4),
    "ghosting": Spec(f_ghost, "isp", "temporal", T("ghosting", "motion blur"),
                     lambda r: dict(a=U(r, 0.5, 0.9)),
                     margin_t=lambda P: min(30, int(math.ceil(math.log(0.02) / math.log(P["a"])))),
                     temporal=True),
    "halo": Spec(f_halo, "isp", "tone", T("halo"),
                 lambda r: dict(sigma=U(r, 4, 14), gain=U(r, 0.8, 2.5)),
                 margin=lambda P, sc: 3 * P["sigma"] * sc, group="sharp"),
    "ringing": Spec(f_ringing, "isp", "detail", T("ringing", "blur softness"),
                    lambda r: dict(tau=U(r, 1.5, 4)), margin=lambda P, sc: 3 * P["tau"] * sc + 2,
                    group="sharp"),
    "low_contrast": Spec(f_contrast, "isp", "tone", T("low contrast", "tonal separation loss"),
                         lambda r: dict(c=U(r, 0.4, 0.75)), group="range"),
    "dynamic_range": Spec(f_dynrange, "isp", "tone", T("dynamic range loss", "low contrast"),
                          lambda r: dict(lo=U(r, 0.05, 0.2), hi=U(r, 0.05, 0.2)), group="range"),
    "shadow_clip": Spec(f_shadow_clip, "isp", "tone", T("shadow clipping", "tonal detail loss"),
                        lambda r: dict(b=U(r, 0.08, 0.25)), group="shadow"),
    "shadow_compression": Spec(f_shadow_comp, "isp", "tone", T("shadow compression", "tonal detail loss"),
                               lambda r: dict(knee=U(r, 0.25, 0.45), g=U(r, 0.8, 2.5)), group="shadow"),
    "highlight_clip": Spec(f_highlight_clip, "isp", "tone", T("highlight clipping"),
                           lambda r: dict(knee=U(r, 0.55, 0.8), g=U(r, 0.5, 2.0))),
    # ---------------- scaling
    "rescale": Spec(f_rescale, "scale", "detail", T("detail loss", "blur softness"),
                    lambda r: dict(f=U(r, 1.5, 4.0)), margin=lambda P, sc: 2 * P["f"] + 4, group="scale"),
    "aliasing": Spec(f_aliasing, "scale", "detail", T("pixelation / aliasing", "shimmering"),
                     lambda r: dict(f=U(r, 1.5, 4.0)), margin=lambda P, sc: 2 * P["f"] + 2, group="scale"),
    "pixelation": Spec(f_pixelate, "scale", "detail", T("pixelation / aliasing", "detail loss"),
                       lambda r: dict(k=float(r.choice([4, 8, 16, 32]))), margin=lambda P, sc: 3 * P["k"] + 2,
                       group="scale"),
    # ---------------- output quantisation / chroma
    "posterize": Spec(f_posterize, "output", "tone", T("tonal separation loss", "banding"),
                      lambda r: dict(bits=U(r, 2.5, 5.0))),
    "chroma_bleed": Spec(f_chroma_bleed, "output", "color", T("color bleeding"),
                         lambda r: dict(sigma=U(r, 2, 10), dx=U(r, 0, 4)),
                         margin=lambda P, sc: (3 * P["sigma"] + P["dx"]) * sc + 2),
    "chroma_quantize": Spec(f_chroma_quant, "output", "color", T("color separation loss"),
                            lambda r: dict(bits=U(r, 3, 5.5))),
}

# Terms whose meaning comes from the TIME course, not the operator.
TEMPORAL_TERMS = {
    ("exposure", "osc"): "exposure pumping", ("exposure", "walk"): "exposure drift",
    ("white_balance", "osc"): "white balance hunting", ("white_balance", "walk"): "white balance drift",
    ("defocus", "osc"): "focus hunting", ("defocus", "walk"): "focus breathing",
    ("hue_shift", "walk"): "color drift",
}
CODEC_TERMS = ("compression blocking", "mosquito noise", "color bleeding")   # from real encodes
OUTCOME_ONLY = ("legibility loss",)


# ================================================================== sampling
def _sample_envelope(r, w, duration):
    kinds = ("const", "interval", "ramp", "osc", "walk")
    w = np.asarray(w, float) / sum(w)
    k = kinds[int(r.choice(5, p=w))]
    D = max(duration, 0.5)
    if k == "interval":
        a = U(r, 0, 0.6 * D)
        return Envelope(k, a, a + U(r, 0.2 * D, 0.7 * D), U(r, 0.05, 0.5))
    if k == "ramp":
        a = U(r, 0, 0.3 * D)
        return Envelope(k, a, a + U(r, 0.3 * D, 0.7 * D), float(r.choice([-1, 1])))
    if k == "osc":
        return Envelope(k, float(np.exp(U(r, np.log(0.2), np.log(3)))), U(r, 0.5, 1.0), U(r, 0, 6.28))
    if k == "walk":
        return Envelope(k, U(r, 0.3, 2.0), U(r, 0.0, 0.5), seed=int(r.integers(2**31)))
    return Envelope()


def sample_op(r, name, duration, severity=None, p_local=0.15):
    sp = REGISTRY[name]
    env = _sample_envelope(r, sp.env_w, duration)
    bip = sp.bipolar and env.kind in ("osc", "walk") and r.random() < 0.5
    if bip:
        env = replace(env, b=1.0) if env.kind == "osc" else replace(env, b=0.0)
    sup = Support()
    if sp.local and r.random() < p_local:
        sup = Support("blob", U(r, .2, .8), U(r, .2, .8), U(r, .12, .4), U(r, .12, .4),
                      U(r, -.05, .05), U(r, -.05, .05), U(r, .3, .8))
    sev = U(r, 0.15, 1.0) if severity is None else float(severity)
    return Op(name, sev, sp.sample(r), int(r.integers(2**31)), env, sup, bip)


def sample_plan(r, fps, duration, n_ops=(1, 3), allow=None, exclude=()):
    """Random plan: 1-3 ops from distinct conflict groups, canonical stage order."""
    names = [n for n in REGISTRY if (allow is None or n in allow) and n not in exclude]
    k = int(r.integers(n_ops[0], n_ops[1] + 1))
    chosen, groups = [], set()
    for n in r.permutation(names):
        g = REGISTRY[n].group
        if g and g in groups:
            continue
        chosen.append(str(n))
        if g:
            groups.add(g)
        if len(chosen) == k:
            break
    return Plan([sample_op(r, n, duration) for n in chosen], fps, int(r.integers(2**31)))


# ================================================================== rendering
def render(frames, frame_idx, plan, full_size=None, offset=(0, 0)):
    """frames: (T,h,w,3) SDR RGB (uint8/uint16/float in [0,1]) of SOURCE frames
    frame_idx[i] (absolute source index, sets time and noise). For crops pass the
    full frame size (W,H) and the crop's top-left offset, with Plan.margin(H)
    pixels of context on each side and Plan.margin_t() frames of history."""
    X = np.asarray(frames)
    if X.dtype == np.uint8:
        X = X.astype(np.float32) / 255
    elif X.dtype == np.uint16:
        X = X.astype(np.float32) / 65535
    else:
        X = X.astype(np.float32, copy=True)
    Tn, h, w, _ = X.shape
    W, H = full_size or (w, h)
    fidx = np.asarray(frame_idx, np.int64)
    ts = fidx / plan.fps
    base = Ctx(np.arange(w, dtype=np.float32) + offset[0], np.arange(h, dtype=np.float32) + offset[1],
               W, H, H / REF_H, plan.fps)
    for op in plan.sorted_ops():
        sp = REGISTRY[op.name]
        st = op.strength(ts)
        if sp.temporal:
            C = replace(base, seed=op.seed)
            Y = sp.fn(X, st, op.P, C)
            for i in range(Tn):
                m = op.support.mask(replace(C, t=int(fidx[i]), ts=float(ts[i])))
                X[i] = Y[i] if m is None else X[i] + m * (Y[i] - X[i])
            continue
        for i in range(Tn):
            s = float(st[i])
            if abs(s) < 1e-6:
                continue
            C = replace(base, seed=op.seed, t=int(fidx[i]), ts=float(ts[i]))
            y = sp.fn(X[i], s, op.P, C)
            m = op.support.mask(C)
            X[i] = y if m is None else X[i] + m * (y - X[i])
        np.clip(X, 0, 1, out=X)
    return np.clip(X, 0, 1)


# =================================================================== labels
def describe(plan, frame_idx):
    """Structured labels for one rendered window (all derived from the plan)."""
    ts = np.asarray(frame_idx) / plan.fps
    out, terms = [], set()
    for op in plan.sorted_ops():
        sp = REGISTRY[op.name]
        st = op.strength(ts)
        active = np.abs(st) > 0.02
        signs = sorted({int(np.sign(v)) for v in st[active]}) or [1]
        tm = set()
        for sg in signs:
            tm.update(sp.terms(op.P, sg))
        extra = TEMPORAL_TERMS.get((op.name, op.envelope.kind))
        if extra:
            tm.add(extra)
        peak = float(np.abs(st).max()) if len(st) else 0.0
        on = np.flatnonzero(active)
        out.append(dict(
            op=op.name, family=sp.family, stage=sp.stage, severity=round(op.severity, 3),
            peak=round(peak, 3),
            level="none" if peak < 0.02 else "slight" if peak < 0.35 else "moderate" if peak < 0.7 else "strong",
            envelope=op.envelope.kind, bipolar=op.bipolar, support=op.support.kind,
            onset_s=float(ts[on[0]]) if len(on) else None,
            offset_s=float(ts[on[-1]]) if len(on) else None,
            active_frac=round(float(active.mean()), 3) if len(st) else 0.0,
            strength=st, terms=sorted(tm)))
        if peak >= 0.02:
            terms.update(tm)
    return dict(ops=out, terms=sorted(terms))


# ====================================================================== CLI
def _cli():
    import argparse
    import subprocess
    ap = argparse.ArgumentParser(description="Preview a video distortion")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--op", default=None, choices=sorted(REGISTRY))
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--severity", type=float, default=0.7)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,avg_frame_rate", "-of", "csv=p=0", a.src],
                        capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(pr[0]), int(pr[1])
    num, den = pr[2].split("/")
    fps = float(num) / float(den)
    r = np.random.default_rng(a.seed)
    if a.random or not a.op:
        plan = sample_plan(r, fps, a.seconds)
    else:
        plan = Plan([sample_op(r, a.op, a.seconds, a.severity)], fps, a.seed)
        plan.ops[0].envelope = Envelope()
    print(describe(plan, np.arange(int(a.seconds * fps)))["terms"],
          [o.name for o in plan.ops])
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-t", str(a.seconds), "-i", a.src,
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                            "-crf", "10", "-pix_fmt", "yuv420p", a.dst], stdin=subprocess.PIPE)
    mt, buf, t = plan.margin_t(), [], 0
    while True:
        raw = dec.stdout.read(w * h * 3)
        if len(raw) < w * h * 3:
            break
        buf.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
        buf = buf[-(mt + 1):]
        idx = np.arange(t - len(buf) + 1, t + 1)
        y = render(np.stack(buf), idx, plan)[-1]
        enc.stdin.write((y * 255 + 0.5).astype(np.uint8).tobytes())
        t += 1
    enc.stdin.close()
    enc.wait()


if __name__ == "__main__":
    _cli()
