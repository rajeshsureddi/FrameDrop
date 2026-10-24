"""Shared helpers for Step 3 (tile store and loaders).

Colour: decoded yuv420p (limited range, BT.709) is converted to RGB exactly as
in the Step 2 capture renderer (colour pipeline v1): chroma 2x bilinear
upsample with half-pixel centres (align_corners=False, edge clamp), BT.709
YCbCr -> RGB, clamp to [0,1]. Tiles are cut from the yuv planes with one
chroma sample of context on each side, so a crop converts to the same pixels
as the full frame would.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

KR, KB = 0.2126, 0.0722
W4K, H4K = 3840, 2160
FPS = 60
TILE_CORE = 256
TILE_MARGIN = 64
TILE_FULL = TILE_CORE + 2 * TILE_MARGIN          # 384
STORE_STEP = 6                                   # every 6th source frame (10 per second)
TILES_PER_FRAME = 16


# ------------------------------------------------------------------ colour
def _up2(c, axis):
    """2x linear upsample along axis with half-pixel centres and edge clamp
    (== F.interpolate(bilinear, align_corners=False) for a 2x factor)."""
    n = c.shape[axis]
    prev = np.take(c, np.r_[0, np.arange(n - 1)], axis=axis)
    nxt = np.take(c, np.r_[np.arange(1, n), n - 1], axis=axis)
    even = 0.75 * c + 0.25 * prev
    odd = 0.75 * c + 0.25 * nxt
    out = np.stack([even, odd], axis=axis + 1)
    shp = list(c.shape)
    shp[axis] = 2 * n
    return out.reshape(shp)


def yuv_planes(buf, w, h):
    Y = np.frombuffer(buf, np.uint8, w * h).reshape(h, w)
    U = np.frombuffer(buf, np.uint8, w * h // 4, w * h).reshape(h // 2, w // 2)
    V = np.frombuffer(buf, np.uint8, w * h // 4, w * h * 5 // 4).reshape(h // 2, w // 2)
    return Y, U, V


def yuv_crop_to_rgb(Y, U, V, x, y, cw, ch):
    """RGB uint8 (ch, cw, 3) of the region [x, x+cw) x [y, y+ch) of a yuv420p
    frame. x, y, cw, ch must be even. Bit-identical to converting the whole
    frame and cropping."""
    hc, wc = U.shape
    cx0, cy0, ncw, nch = x // 2, y // 2, cw // 2, ch // 2
    ix = np.clip(np.arange(cx0 - 1, cx0 + ncw + 1), 0, wc - 1)
    iy = np.clip(np.arange(cy0 - 1, cy0 + nch + 1), 0, hc - 1)
    c = np.stack([U[np.ix_(iy, ix)], V[np.ix_(iy, ix)]], 0).astype(np.float32)
    # clamp-at-edge of the full plane is reproduced by the clipped indices
    c = _up2(_up2(c, 1), 2)[:, 2:-2, 2:-2]
    c = (c - 128.0) / 224.0
    yy = (Y[y:y + ch, x:x + cw].astype(np.float32) - 16.0) / 219.0
    cb, cr = c[0], c[1]
    r = yy + 2 * (1 - KR) * cr
    b = yy + 2 * (1 - KB) * cb
    g = (yy - KR * r - KB * b) / (1 - KR - KB)
    rgb = np.stack([r, g, b], -1)
    return np.clip(np.rint(np.clip(rgb, 0, 1) * 255.0), 0, 255).astype(np.uint8)


_LUT = None


def _yuv444_lut():
    """(Y, U, V) -> RGB for all 2^24 triples, computed with _yuv444_to_rgb_float
    itself, so the lookup is bit-identical to the direct conversion (48 MB)."""
    global _LUT
    if _LUT is None:
        y, u, v = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8),
                              np.arange(256, dtype=np.uint8), indexing="ij")
        _LUT = _yuv444_to_rgb_float(y.ravel(), u.ravel(), v.ravel())
    return _LUT


def yuv444_to_rgb(Y, U, V):
    """Low-res frames already scaled to 4:4:4 by ffmpeg (area filter).
    Uses the exact lookup table for large inputs (10x faster, identical output)."""
    if Y.size < 1 << 20:
        return _yuv444_to_rgb_float(Y, U, V)
    idx = (Y.astype(np.uint32) << 16) | (U.astype(np.uint32) << 8) | V.astype(np.uint32)
    return _yuv444_lut()[idx]


def _yuv444_to_rgb_float(Y, U, V):
    yy = (Y.astype(np.float32) - 16.0) / 219.0
    cb = (U.astype(np.float32) - 128.0) / 224.0
    cr = (V.astype(np.float32) - 128.0) / 224.0
    r = yy + 2 * (1 - KR) * cr
    b = yy + 2 * (1 - KB) * cb
    g = (yy - KR * r - KB * b) / (1 - KR - KB)
    return np.clip(np.rint(np.clip(np.stack([r, g, b], -1), 0, 1) * 255.0), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ decode
def decode_cmd(path, threads=2, trim=None):
    """Raw yuv420p frames in stream order, one per coded frame (passthrough:
    no CFR duplication/dropping on irregular timestamps)."""
    vf = []
    if trim is not None:
        vf = ["-vf", f"trim=start_frame={trim[0]}:end_frame={trim[1]},setpts=N/({FPS}*TB),format=yuv420p"]
    return ["ffmpeg", "-v", "error", "-nostdin", "-threads", str(threads), "-i", path,
            "-fps_mode", "passthrough", *vf, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]


def iter_frames(cmd, w=W4K, h=H4K):
    fsz = w * h * 3 // 2
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=fsz)
    try:
        while True:
            b = p.stdout.read(fsz)
            if len(b) < fsz:
                break
            yield b
    finally:
        p.stdout.close()
        err = p.stderr.read().decode(errors="replace")
        rc = p.wait()
        if rc not in (0, None) and not err:
            err = f"rc={rc}"
        iter_frames.last_err = err


iter_frames.last_err = ""


def load_json(p):
    with open(p) as f:
        return json.load(f)
