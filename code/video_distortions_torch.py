"""PyTorch (GPU) port of video_distortions.py: same API, registry, parameters
and hash functions, so a plan renders the same on either backend.

Everything that is not per-pixel work is taken from the numpy module itself
(plans, ops, envelopes, supports, parameter sampling, margins, describe), so
sampling and labels are identical by construction. Only the operators and
render() are reimplemented:

  * images are float32 tensors (h, w, 3) in [0, 1], channels last as in numpy
  * cv2 BORDER_REFLECT (symmetric, fedcba|abcdef) is reproduced by index
    mirroring; Gaussian kernels follow cv2.getGaussianKernel / GaussianBlur
    (ksize = round(8 sigma + 1) | 1 for float images)
  * cv2.remap is reproduced including its fixed-point coordinates (1/32 px for
    INTER_LINEAR, round-to-nearest for INTER_NEAREST) and BORDER_REPLICATE
  * the counter-based hash is computed in int64 with 32-bit masking, which is
    bit-identical to the numpy uint32 version
  * remap/resample grids that depend only on coordinates are computed with the
    same numpy expressions as the reference and uploaded

render(frames, frame_idx, plan, full_size=None, offset=(0, 0), device=None)
accepts numpy or torch frames (uint8/uint16/float) and returns the same type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import video_distortions as vd
from video_distortions import (CODEC_TERMS, KB, KR, OUTCOME_ONLY, REF_H, STAGES, TEMPORAL_TERMS,  # noqa: F401
                               Envelope, Op, Plan, Spec, Support, describe, sample_op, sample_plan)

M32 = 0xFFFFFFFF


class _exact_fp32:
    """Inside render(): disable TF32 (on Ampere+ cuDNN convolutions default to
    TF32, a 10-bit mantissa: ~2e-4 error on high-contrast blur input) and use
    deterministic cuDNN algorithms. Global settings are restored afterwards so
    training code keeps its own choice."""

    def __enter__(self):
        b = torch.backends
        self.prev = (b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32, b.cudnn.deterministic, b.cudnn.benchmark)
        b.cudnn.allow_tf32 = False
        b.cuda.matmul.allow_tf32 = False
        b.cudnn.deterministic = True      # bit-identical re-renders from a plan JSON
        b.cudnn.benchmark = False

    def __exit__(self, *exc):
        b = torch.backends
        (b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32, b.cudnn.deterministic, b.cudnn.benchmark) = self.prev


# ============================================================ low-level helpers
def lin(x):
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def unlin(y):
    y = y.clamp_min(0.0)
    return torch.where(y <= 0.0031308, y * 12.92, 1.055 * torch.pow(y, 1 / 2.4) - 0.055)


def luma(x):
    return KR * x[..., 0] + (1 - KR - KB) * x[..., 1] + KB * x[..., 2]


def to_ycc(x):
    y = luma(x)
    return y, (x[..., 2] - y) / (2 * (1 - KB)), (x[..., 0] - y) / (2 * (1 - KR))


def from_ycc(y, cb, cr):
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / (1 - KR - KB)
    return torch.stack([r, g, b], -1)


def _smooth(f):
    return f * f * (3 - 2 * f)


def _mirror(n, lo, hi, device):
    """Indices for positions lo..hi-1 of a length-n axis under cv2.BORDER_REFLECT."""
    i = torch.arange(lo, hi, device=device)
    if n == 1:
        return torch.zeros_like(i)
    j = torch.remainder(i, 2 * n)
    return torch.where(j >= n, 2 * n - 1 - j, j)


def _as_nchw(x):
    """(h,w) or (h,w,c) -> (1,c,h,w); returns the tensor and an inverse."""
    if x.dim() == 2:
        return x[None, None], lambda y: y[0, 0]
    return x.permute(2, 0, 1)[None], lambda y: y[0].permute(1, 2, 0)


def _pad_reflect(t, top, bottom, left, right):
    """Symmetric padding of a (1,c,h,w) tensor (cv2.BORDER_REFLECT)."""
    h, w = t.shape[-2:]
    if top or bottom:
        t = t.index_select(2, _mirror(h, -top, h + bottom, t.device))
    if left or right:
        t = t.index_select(3, _mirror(w, -left, w + right, t.device))
    return t


def _sep_conv(x, kx, ky):
    """Separable correlation with 1-D kernels kx (along x) and ky (along y),
    centred anchors, cv2.BORDER_REFLECT. kx/ky: 1-D float32 numpy arrays."""
    t, back = _as_nchw(x)
    c = t.shape[1]
    rx, ry = len(kx) // 2, len(ky) // 2
    wx = torch.as_tensor(np.asarray(kx, np.float32), device=t.device).view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    wy = torch.as_tensor(np.asarray(ky, np.float32), device=t.device).view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    t = F.conv2d(_pad_reflect(t, 0, 0, rx, rx), wx, groups=c)
    t = F.conv2d(_pad_reflect(t, ry, ry, 0, 0), wy, groups=c)
    return back(t)


def _filter2d(x, k):
    """cv2.filter2D(x, -1, k, borderType=BORDER_REFLECT): 2-D correlation, centred anchor."""
    t, back = _as_nchw(x)
    c = t.shape[1]
    kh, kw = k.shape
    w = torch.as_tensor(np.asarray(k, np.float32), device=t.device).view(1, 1, kh, kw).repeat(c, 1, 1, 1)
    t = F.conv2d(_pad_reflect(t, kh // 2, kh - 1 - kh // 2, kw // 2, kw - 1 - kw // 2), w, groups=c)
    return back(t)


_GK = {}


def _gauss_kernel(sigma):
    """cv2.getGaussianKernel for a float image in GaussianBlur(ksize=(0,0))."""
    n = int(round(sigma * 4 * 2 + 1)) | 1      # cvRound(sigma*4*2+1)|1
    key = (n, float(sigma))
    if key not in _GK:
        _GK[key] = cv2.getGaussianKernel(n, float(sigma), cv2.CV_32F).ravel()
    return _GK[key]


def blur(x, sigma):
    if sigma < 0.05:
        return x
    k = _gauss_kernel(sigma)
    return _sep_conv(x, k, k)


def _box_tl(x, k):
    """cv2.boxFilter(x, -1, (k,k), anchor=(0,0), BORDER_REFLECT): mean of src[y..y+k-1, x..x+k-1]."""
    t, back = _as_nchw(x)
    t = F.avg_pool2d(_pad_reflect(t, 0, k - 1, 0, k - 1), k, stride=1)
    return back(t)


def _sobel(Y, dx):
    a, b = np.array([-1, 0, 1], np.float32), np.array([1, 2, 1], np.float32)
    return _sep_conv(Y, a, b) if dx else _sep_conv(Y, b, a)


def _hash(seed, t, c, x, y):
    """Bit-identical to video_distortions._hash; x, y int64 tensors (or ints), broadcastable."""
    k = (int(seed) * 0x9E3779B1 ^ int(t) * 0xCB1AB31F ^ int(c) * 0x165667B1) & M32
    x = (x if torch.is_tensor(x) else torch.tensor(int(x))) & M32
    y = (y if torch.is_tensor(y) else torch.tensor(int(y), device=x.device)) & M32
    h = ((x * 0x8DA6B343) & M32) ^ ((y * 0xD8163841) & M32) ^ k
    h = h ^ (h >> 16)
    h = (h * 0x7FEB352D) & M32
    h = h ^ (h >> 15)
    h = (h * 0x846CA68B) & M32
    h = h ^ (h >> 16)
    return h


def _u01(h):
    return ((h >> 8).to(torch.float32) + 0.5) * np.float32(1 / 16777216)


def gauss_field(C, c):
    x, y = C.xa.long()[None, :], C.ya.long()[:, None]
    u1 = _u01(_hash(C.seed, C.t, 2 * c, x, y))
    u2 = _u01(_hash(C.seed, C.t, 2 * c + 1, x, y))
    return torch.sqrt(-2 * torch.log(u1)) * torch.cos(np.float32(2 * np.pi) * u2)


def vnoise(xa, ya, cell, seed, c=0, t=0.0, tcell=0.0):
    gx, gy = xa / cell, ya / cell
    ix, iy = torch.floor(gx).long(), torch.floor(gy).long()
    fx, fy = _smooth(gx - ix)[None, :], _smooth(gy - iy)[:, None]
    x0, y0 = int(ix.min()), int(iy.min())
    kx = torch.arange(x0, int(ix.max()) + 2, device=xa.device)
    ky = torch.arange(y0, int(iy.max()) + 2, device=xa.device)
    lx, ly = (ix - x0)[None, :], (iy - y0)[:, None]

    def plane(it):
        G = _u01(_hash(seed, it, c, kx[None, :], ky[:, None])) * 2 - 1
        a, b = G[ly, lx], G[ly, lx + 1]
        cc, d = G[ly + 1, lx], G[ly + 1, lx + 1]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy

    if tcell <= 0:
        return plane(0)
    gt = t / tcell
    it = int(math.floor(gt))
    ft = float(_smooth(gt - it))
    return plane(it) if ft == 0 else plane(it) * (1 - ft) + plane(it + 1) * ft


def _remap(img, mapx, mapy, interp=cv2.INTER_LINEAR):
    """cv2.remap with float32 maps, BORDER_REPLICATE. mapx/mapy: numpy arrays or
    tensors broadcastable to (H', W'). LINEAR uses cv2's 1/32 px fixed point."""
    dev = img.device
    mx = torch.as_tensor(np.asarray(mapx, np.float32) if not torch.is_tensor(mapx) else mapx,
                         device=dev, dtype=torch.float32)
    my = torch.as_tensor(np.asarray(mapy, np.float32) if not torch.is_tensor(mapy) else mapy,
                         device=dev, dtype=torch.float32)
    Ho, Wo = my.shape[0], mx.shape[1]
    mx, my = mx.expand(Ho, Wo), my.expand(Ho, Wo)
    h, w = img.shape[:2]
    flat = img.reshape(h * w, -1)
    if interp == cv2.INTER_NEAREST:
        ix = torch.round(mx).long().clamp(0, w - 1)
        iy = torch.round(my).long().clamp(0, h - 1)
        out = flat[(iy * w + ix).reshape(-1)]
        return out.reshape(Ho, Wo, *img.shape[2:])
    X, Y = torch.round(mx * 32).long(), torch.round(my * 32).long()
    ix, iy = torch.div(X, 32, rounding_mode="floor"), torch.div(Y, 32, rounding_mode="floor")
    ax = ((X - ix * 32).to(torch.float32) / 32).reshape(-1, 1)
    ay = ((Y - iy * 32).to(torch.float32) / 32).reshape(-1, 1)
    x0, x1 = ix.clamp(0, w - 1), (ix + 1).clamp(0, w - 1)
    y0, y1 = iy.clamp(0, h - 1), (iy + 1).clamp(0, h - 1)
    g = lambda yy, xx: flat[(yy * w + xx).reshape(-1)]
    out = (g(y0, x0) * ((1 - ax) * (1 - ay)) + g(y0, x1) * (ax * (1 - ay))
           + g(y1, x0) * ((1 - ax) * ay) + g(y1, x1) * (ax * ay))
    return out.reshape(Ho, Wo, *img.shape[2:])


def _grid_resample(x, f, C, fetch, up):
    """Same absolute-grid resample as the numpy version; grids built with the
    reference's numpy expressions."""
    xa, ya = C.xa_np, C.ya_np
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


# ================================================================ context
@dataclass
class Ctx:
    xa: torch.Tensor        # (w,) absolute x of each column (float32, on device)
    ya: torch.Tensor        # (h,)
    xa_np: np.ndarray       # the same coordinates as numpy float32 (for grids)
    ya_np: np.ndarray
    W: int
    H: int
    scale: float
    fps: float
    seed: int = 0
    t: int = 0
    ts: float = 0.0


def _mask(sup, C):
    if sup.kind == "global":
        return None
    u = (C.xa / C.W)[None, :]
    v = (C.ya / C.H)[:, None]
    cx, cy = sup.cx + sup.vx * C.ts, sup.cy + sup.vy * C.ts
    d = torch.sqrt(((u - cx) / sup.rx) ** 2 + ((v - cy) / sup.ry) ** 2)
    e = ((d - (1 - sup.soft)) / sup.soft).clamp(0, 1)
    return (1 - _smooth(e))[..., None]


def _vec(v, dev):
    return torch.as_tensor(np.asarray(v, np.float32), device=dev)


# ================================================================= operators
def f_haze(x, s, P, C):
    v = (C.ya / C.H)[:, None]
    n = vnoise(C.xa, C.ya, P["cell"] * C.H, C.seed, 11, C.t, P["tcell"] * C.fps)
    depth = (P["gy"] * (1 - v) + (1 - P["gy"]) * 0.5 + 0.35 * n).clamp(0, 1)
    tr = torch.exp(-P["beta"] * s * depth)[..., None]
    return x * tr + _vec(P["A"], x.device) * (1 - tr)


def f_flicker(x, s, P, C):
    ph = 2 * np.pi * P["hz"] * C.ts + P["phase"]
    if P["mode"] == "rolling":
        ph = ph - 2 * np.pi * P["bands"] * (C.ya / C.H)[:, None, None]
        g = 1 + s * P["amp"] * torch.sin(ph)
    else:
        g = 1 + s * P["amp"] * float(np.sin(np.float32(ph)))
    return unlin(lin(x) * g)


def f_flare(x, s, P, C):
    sx = (P["x0"] + P["vx"] * C.ts + 0.015 * math.sin(2 * np.pi * 0.3 * C.ts)) * C.W
    sy = (P["y0"] + P["vy"] * C.ts + 0.010 * math.cos(2 * np.pi * 0.23 * C.ts)) * C.H
    X, Y, R = C.xa[None, :], C.ya[:, None], float(C.H)
    dx, dy = (X - sx) / R, (Y - sy) / R
    d = torch.sqrt(dx * dx + dy * dy)
    glare = (P["veil"] * torch.exp(-(d / 0.45) ** 2) + 1.5 * torch.exp(-d / 0.02)
             + 0.6 * torch.exp(-(dy / 0.006) ** 2) * torch.exp(-dx.abs() / 0.6))
    add = glare[..., None] * _vec(P["tint"], x.device)
    for (tt, rad, col, a) in P["ghosts"]:
        gx, gy = sx + (C.W / 2 - sx) * tt, sy + (C.H / 2 - sy) * tt
        dd = torch.sqrt((X - gx) ** 2 + (Y - gy) ** 2) / (rad * R)
        disc = a / (1 + torch.exp(((dd - 1) * 14).clamp(-50, 50)))
        add = add + disc[..., None] * _vec(col, x.device)
    return unlin(lin(x) + s * P["gain"] * add)


def f_bloom(x, s, P, C):
    xl = lin(x)
    hi = (xl - P["thr"]).clamp_min(0)
    return unlin(xl + s * P["gain"] * blur(hi, P["sigma"] * C.scale))


def f_ca(x, s, P, C):
    k = s * P["k"]
    cx, cy = C.W / 2, C.H / 2
    out = x.clone()
    for ch, sc in ((0, 1 + k), (2, 1 - k)):
        mx = (cx + (C.xa_np - cx) / sc - C.xa_np[0])[None, :]
        my = (cy + (C.ya_np - cy) / sc - C.ya_np[0])[:, None]
        out[..., ch] = _remap(x[..., ch].contiguous(), mx, my)
    return out


def f_fringe(x, s, P, C):
    b = ((luma(x) - 0.82) / 0.18).clamp(0, 1)
    halo = (blur(b, P["sigma"] * C.scale) - b).clamp(0, 1)
    a = (s * P["gain"] * halo).clamp(0, 1)[..., None]
    return x * (1 - a) + _vec(P["col"], x.device) * a


def f_defocus(x, s, P, C):
    return blur(x, abs(s) * P["sigma"] * C.scale)


def _shake_kernel(L, th):
    n = int(math.ceil(L)) | 1
    big = np.zeros((n * 4, n * 4), np.uint8)
    c, r = n * 2, L * 2
    p1 = (int(round(c - r * math.cos(th))), int(round(c - r * math.sin(th))))
    p2 = (int(round(c + r * math.cos(th))), int(round(c + r * math.sin(th))))
    cv2.line(big, p1, p2, 255, 2, cv2.LINE_AA)
    k = cv2.resize(big.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
    k /= k.sum() + 1e-8
    return k


def f_shake(x, s, P, C):
    L = abs(s) * P["L"] * C.scale
    if L < 1.0:
        return x
    th = P["th0"] + 0.6 * math.sin(2 * np.pi * P["f1"] * C.ts + P["p1"]) \
        + 0.3 * math.sin(2 * np.pi * P["f2"] * C.ts + P["p2"])
    return _filter2d(x, _shake_kernel(L, th))


def _lin_frames(X):
    """lin() frame by frame into one buffer (no full-clip temporaries)."""
    L = torch.empty_like(X)
    for i in range(len(X)):
        L[i] = lin(X[i])
    return L


def f_shutter(X, st, P, C):
    L = _lin_frames(X)
    Y = X.clone()
    for i in range(len(X)):
        n = 1 + int(round(abs(st[i]) * (P["N"] - 1)))
        if n > 1:
            Y[i] = unlin(L[max(0, i - n + 1):i + 1].mean(0))
    return Y


def f_exposure(x, s, P, C):
    return unlin(lin(x) * float(np.float32(2.0 ** (s * P["ev"]))))


def f_luma_noise(x, s, P, C):
    xl = lin(x)
    yl = luma(xl)
    sig = torch.sqrt(P["shot"] * abs(s) * yl.clamp_min(0) + (P["read"] * abs(s)) ** 2)
    n = gauss_field(C, 1)
    return unlin(xl + (sig * n)[..., None])


def f_chroma_noise(x, s, P, C):
    y, cb, cr = to_ycc(x)
    cell = P["cell"] * C.scale
    ncb = vnoise(C.xa, C.ya, cell, C.seed, 21, C.t, 1.0)
    ncr = vnoise(C.xa, C.ya, cell, C.seed, 22, C.t, 1.0)
    a = abs(s) * P["amp"]
    return from_ycc(y, cb + a * ncb, cr + a * ncr)


def f_fpn(x, s, P, C):
    S = replace(C, t=0)
    col = _u01(_hash(C.seed, 0, 31, C.xa.long()[None, :], 0)) * 2 - 1
    prnu = gauss_field(S, 32)
    amp = 0.3 + 0.7 * (0.5 + 0.5 * vnoise(C.xa, C.ya, P["cell"] * C.H, C.seed, 33))
    xl = lin(x)
    a = abs(s) * amp
    out = xl * (1 + (a * P["prnu"] * prnu)[..., None]) + (a * P["col"] * col)[..., None]
    return unlin(out)


def f_demosaic(x, s, P, C):
    py = (C.ya.long() % 2)[:, None]
    px = (C.xa.long() % 2)[None, :]
    h, w = x.shape[:2]
    masks = [((py == 0) & (px == 0)), ((py + px) % 2 == 1), ((py == 1) & (px == 1))]
    masks = [m.expand(h, w).to(torch.float32) for m in masks]
    mos = sum(x[..., c] * masks[c] for c in range(3))
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    out = torch.empty_like(x)
    for c in range(3):
        num = _filter2d(mos * masks[c], K)
        den = _filter2d(masks[c], K)
        out[..., c] = torch.where(masks[c] > 0, mos, num / den.clamp_min(1e-6))
    return x + abs(s) * (out - x)


def f_wb(x, s, P, C):
    lg = s * P["mag"] * np.asarray(P["dir"], np.float32)
    g = np.exp(np.array([lg[0], 0.0, lg[1]], np.float32))
    g /= KR * g[0] + (1 - KR - KB) * g[1] + KB * g[2]
    return unlin(lin(x) * _vec(g, x.device))


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
    sig = abs(s) * P["sigma"] * C.scale
    b = blur(x, sig)
    Y = luma(x)
    gx, gy = _sobel(Y, True), _sobel(Y, False)
    g = blur(torch.sqrt(gx * gx + gy * gy), max(sig * 0.5, 0.5))
    w = torch.exp(-(g / P["tau"]) ** 2)[..., None]
    return x + w * (b - x)


def f_ghost(X, st, P, C):
    # recursive blend in place (Y[i] only needs L[i] and the output Y[i-1]),
    # then unlin frame by frame: same arithmetic as the numpy version, ~3x less memory
    Y = _lin_frames(X)
    for i in range(1, len(X)):
        a = abs(st[i]) * P["a"]
        Y[i] = (1 - a) * Y[i] + a * Y[i - 1]
    for i in range(len(X)):
        Y[i] = unlin(Y[i])
    return Y


def f_halo(x, s, P, C):
    b = blur(x, P["sigma"] * C.scale)
    return x + abs(s) * P["gain"] * (x - b)


def _ring_kernel(tau):
    r = int(math.ceil(3 * tau))
    n = np.arange(-r, r + 1, dtype=np.float32)
    k = np.sinc(n / (2 * tau)) * (0.5 + 0.5 * np.cos(np.pi * n / (r + 1)))
    return (k / k.sum()).astype(np.float32)


def f_ringing(x, s, P, C):
    k = _ring_kernel(P["tau"] * C.scale)
    return x + abs(s) * (_sep_conv(x, k, k) - x)


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
    return torch.where(x < k, k * torch.pow((x / k).clamp(0, 1), 1 + abs(s) * P["g"]), x)


def f_highlight_clip(x, s, P, C):
    k = P["knee"]
    return torch.where(x > k, k + (x - k) * (1 + abs(s) * P["g"]), x)


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
    k = 2 ** int(round(abs(s) * math.log2(P["k"])))
    if k < 2:
        return x
    return _grid_resample(_box_tl(x, k), k, C, cv2.INTER_NEAREST, "block")


def _quant(v, s, bits_removed, lo=0.0):
    b = 8 - int(round(abs(s) * bits_removed))
    if b >= 8:
        return v
    step = (1.0 / 2 ** b)
    return (torch.floor((v - lo) / step) + 0.5) * step + lo


def f_posterize(x, s, P, C):
    return _quant(x.clamp(0, 1 - 1e-6), s, P["bits"])


def f_chroma_bleed(x, s, P, C):
    y, cb, cr = to_ycc(x)
    sig = abs(s) * P["sigma"] * C.scale
    dx = abs(s) * P["dx"] * C.scale
    mx = (np.arange(len(C.xa_np), dtype=np.float32) - dx)[None, :]
    my = np.arange(len(C.ya_np), dtype=np.float32)[:, None]
    cb = _remap(blur(cb, sig), mx, my)
    cr = _remap(blur(cr, sig), mx, my)
    return from_ycc(y, cb, cr)


def f_chroma_quant(x, s, P, C):
    y, cb, cr = to_ycc(x)
    return from_ycc(y, _quant(cb, s, P["bits"], -0.5), _quant(cr, s, P["bits"], -0.5))


# ================================================================== registry
_FN = {
    "haze": f_haze, "flicker": f_flicker, "flare": f_flare, "bloom": f_bloom,
    "chromatic_aberration": f_ca, "purple_fringing": f_fringe, "defocus": f_defocus,
    "shake_blur": f_shake, "shutter_blur": f_shutter, "exposure": f_exposure,
    "luma_noise": f_luma_noise, "chroma_noise": f_chroma_noise, "fixed_pattern_noise": f_fpn,
    "demosaic": f_demosaic, "white_balance": f_wb, "saturation": f_saturation, "hue_shift": f_hue,
    "over_smoothing": f_oversmooth, "ghosting": f_ghost, "halo": f_halo, "ringing": f_ringing,
    "low_contrast": f_contrast, "dynamic_range": f_dynrange, "shadow_clip": f_shadow_clip,
    "shadow_compression": f_shadow_comp, "highlight_clip": f_highlight_clip, "rescale": f_rescale,
    "aliasing": f_aliasing, "pixelation": f_pixelate, "posterize": f_posterize,
    "chroma_bleed": f_chroma_bleed, "chroma_quantize": f_chroma_quant,
}
assert list(_FN) == list(vd.REGISTRY), "registry order must match video_distortions.REGISTRY"
# same Spec objects (stage, family, sampling, margins, flags), torch operator
REGISTRY: dict[str, Spec] = {k: replace(v, fn=_FN[k]) for k, v in vd.REGISTRY.items()}


# ================================================================== rendering
def render(frames, frame_idx, plan, full_size=None, offset=(0, 0), device=None):
    """Same contract as video_distortions.render. frames: (T,h,w,3) numpy or
    torch (uint8/uint16/float in [0,1]); returns float32 in [0,1] of the same
    type (numpy in -> numpy out; torch in -> torch on its device)."""
    as_numpy = not torch.is_tensor(frames)
    if device is None:
        device = frames.device if not as_numpy else ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.array(frames) if as_numpy else frames, device=device)
    if X.dtype == torch.uint8:
        X = X.to(torch.float32) / 255
    elif X.dtype == torch.uint16 or (as_numpy and np.asarray(frames).dtype == np.uint16):
        X = X.to(torch.float32) / 65535
    else:
        X = X.to(torch.float32).clone()
    Tn, h, w, _ = X.shape
    W, H = full_size or (w, h)
    fidx = np.asarray(frame_idx, np.int64)
    ts = fidx / plan.fps
    xa_np = np.arange(w, dtype=np.float32) + offset[0]
    ya_np = np.arange(h, dtype=np.float32) + offset[1]
    base = Ctx(torch.as_tensor(xa_np, device=device), torch.as_tensor(ya_np, device=device),
               xa_np, ya_np, W, H, H / REF_H, plan.fps)
    with torch.no_grad(), _exact_fp32():
        for op in plan.sorted_ops():
            sp = REGISTRY[op.name]
            st = op.strength(ts)
            if sp.temporal:
                C = replace(base, seed=op.seed)
                Y = sp.fn(X, st, op.P, C)
                for i in range(Tn):
                    m = _mask(op.support, replace(C, t=int(fidx[i]), ts=float(ts[i])))
                    X[i] = Y[i] if m is None else X[i] + m * (Y[i] - X[i])
                continue
            for i in range(Tn):
                s = float(st[i])
                if abs(s) < 1e-6:
                    continue
                C = replace(base, seed=op.seed, t=int(fidx[i]), ts=float(ts[i]))
                y = sp.fn(X[i], s, op.P, C)
                m = _mask(op.support, C)
                X[i] = y if m is None else X[i] + m * (y - X[i])
            X.clamp_(0, 1)
        X = X.clamp(0, 1)
    return X.cpu().numpy() if as_numpy else X
