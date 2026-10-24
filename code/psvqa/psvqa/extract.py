"""Feature extraction.

--mode full   : spatial features for EVERY source frame. Use for training
                encodes: drop patterns are applied later by re-indexing, so one
                extraction serves any number of patterns.
--mode policy : spatial features only where sample_mask() says (the real-time
                path). Use for the test set and for timing.

Output per video: <out>/<stem>.npz with
  z (T,1152) fp16, thumbs (T,72,128) uint8, spat (T,4096) fp16,
  valid (T,) bool, fps, sec_per_frame
"""
import argparse
import os
import time
import numpy as np
import torch

from .common import SAMPLE_EVERY, is_dup_pair
from .encoders import DenseEncoder, SpatialEncoder, to_tensor
from .video import decode, fragments


def extract(path, dense, spatial, device, mode="policy", batch=32):
    zs, thumbs, spat_rows, spat_idx = [], [], [], []
    low_buf, frag_buf, frag_idx = [], [], []
    prev_frame, prev_th, prev_dup, uniq = None, None, False, 0
    added = set()
    fps = 60.0
    t0 = time.perf_counter()

    def flush_low():
        if low_buf:
            zs.append(dense(to_tensor(low_buf, device)).half().cpu().numpy())
            low_buf.clear()

    def flush_frag():
        if frag_buf:
            spat_rows.append(spatial(to_tensor(frag_buf, device)).half().cpu().numpy())
            spat_idx.extend(frag_idx)
            frag_buf.clear()
            frag_idx.clear()

    for t, low, th, fr, fps in decode(path):
        low_buf.append(low)
        thumbs.append(th)
        if mode == "full":
            frag_buf.append(fragments(fr, seed=t))
            frag_idx.append(t)
        else:
            is_dup = t > 0 and is_dup_pair(prev_th, th)
            # identical rule to common.sample_mask()
            if t == 0 or (not is_dup and (uniq := uniq + 1) % SAMPLE_EVERY == 0):
                frag_buf.append(fragments(fr, seed=t)); frag_idx.append(t); added.add(t)
            elif is_dup and not prev_dup and (t - 1) not in added:
                # a freeze starts: the held image is the previous frame
                frag_buf.append(fragments(prev_frame, seed=t - 1))
                frag_idx.append(t - 1); added.add(t - 1)
            prev_dup = is_dup
        prev_frame, prev_th = fr, th
        if len(low_buf) >= batch:
            flush_low()
        if len(frag_buf) >= batch:
            flush_frag()
    flush_low()
    flush_frag()
    elapsed = time.perf_counter() - t0

    T = len(thumbs)
    spat = np.zeros((T, spatial.dim), np.float16)
    valid = np.zeros(T, bool)
    if spat_rows:
        rows = np.concatenate(spat_rows)
        spat[spat_idx] = rows
        valid[spat_idx] = True
    return dict(z=np.concatenate(zs), thumbs=np.stack(thumbs), spat=spat,
                valid=valid, fps=fps, sec_per_frame=elapsed / max(T, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["full", "policy"], default="policy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dense = DenseEncoder().to(a.device)
    spatial = SpatialEncoder().to(a.device)
    for v in a.videos:
        dst = os.path.join(a.out, os.path.splitext(os.path.basename(v))[0] + ".npz")
        if os.path.exists(dst):
            continue
        r = extract(v, dense, spatial, a.device, a.mode, a.batch)
        np.savez_compressed(dst, **r)
        print(f"{os.path.basename(v)}: T={len(r['thumbs'])} "
              f"spatial={r['valid'].sum()} {1.0 / r['sec_per_frame']:.1f} fps")


if __name__ == "__main__":
    main()
