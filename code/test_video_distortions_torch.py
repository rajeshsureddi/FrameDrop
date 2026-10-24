"""Parity tests: video_distortions_torch.py (GPU) vs video_distortions.py (numpy).
python test_video_distortions_torch.py            (needs a CUDA GPU; falls back to CPU torch)

  1. per op, severity 1, const envelope, global support: max |torch - numpy| < 2e-3
  2. per op, 3 fully random samples (random envelopes, bipolar swings, local
     blob supports, random severities, 24-frame windows): max |diff| < 2e-3
  3. 40 random 2-3-op plans (sample_plan): max |diff| < 2e-3. Plans containing a
     quantiser (posterize, chroma_quantize, pixelation) after other ops can flip a
     few boundary pixels by one step on 1e-7 input differences; for those the
     criterion is: < 1e-4 of pixels above 2e-3 (reported separately)
  4. identity at severity 0 (torch): exactly the input
  5. crop consistency (torch): full-frame render cropped == crop rendered with
     Plan.margin() context, < 2e-3
  6. seekability (torch): non-temporal ops give identical frames whatever window
     they are rendered in
  6b. bit-exact re-render: rendering the same plan twice gives identical output
  7. 4K: one full 3840x2160 frame per op (real content), max |diff| < 2e-3;
     quantisers: < 1e-4 of pixels above 2e-3 (GPU kernels fuse multiply-adds, so
     the luma/chroma split differs from numpy in the last bit and a value sitting
     exactly on a step boundary can land one step away)
"""
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_distortions as vd          # noqa: E402
import video_distortions_torch as vt    # noqa: E402

TOL = 2e-3
QUANT = {"posterize", "chroma_quantize", "pixelation"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print("  FAIL", msg)


def frames(w, h, t, src=None):
    inp = ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=60"] if src is None else ["-i", src]
    raw = subprocess.run(["ffmpeg", "-v", "error", *inp, "-frames:v", str(t), "-vf", f"scale={w}:{h}",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def diff(F, idx, plan, **kw):
    a = vd.render(F, idx, plan, **kw)
    b = vt.render(F, idx, plan, device=DEV, **kw)
    return np.abs(a - b)


def main():
    print(f"device {DEV}, {len(vt.REGISTRY)} ops")
    check(list(vt.REGISTRY) == list(vd.REGISTRY), "registry names/order differ")
    W, H = 640, 360
    F = frames(W, H, 24)
    idx = np.arange(len(F))

    print("1. per op, severity 1, const envelope")
    for name in vd.REGISTRY:
        op = vd.sample_op(np.random.default_rng(1), name, 1.0, severity=1.0)
        op.envelope, op.support, op.bipolar = vd.Envelope(), vd.Support(), False
        d = diff(F[:6], idx[:6], vd.Plan([op], 60.0, 0)).max()
        check(d < TOL, f"{name}: const max diff {d:.2e}")
        print(f"  {name:22s} {d:.1e}")

    print("2. per op, random envelopes / supports / severities")
    worst = {}
    for name in vd.REGISTRY:
        for k in range(3):
            r = np.random.default_rng(100 + k)
            op = vd.sample_op(r, name, 0.4, p_local=0.5)
            d = diff(F, idx, vd.Plan([op], 60.0, k)).max()
            worst[name] = max(worst.get(name, 0), d)
            check(d < TOL, f"{name}: sampled #{k} ({op.envelope.kind}, {op.support.kind}, "
                           f"bipolar={op.bipolar}) max diff {d:.2e}")
    print("  worst:", ", ".join(f"{k}={v:.0e}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:6]))

    print("3. 40 random 2-3-op plans")
    qmax = 0.0
    for k in range(40):
        plan = vd.sample_plan(np.random.default_rng(1000 + k), 60.0, 0.4, n_ops=(2, 3))
        names = [o.name for o in plan.sorted_ops()]
        d = diff(F[:12], idx[:12], plan)
        q = any(n in QUANT for n in names[1:])
        if q:
            frac = float((d > TOL).mean())
            qmax = max(qmax, frac)
            check(frac < 1e-4, f"plan {k} {names}: {frac:.1e} of pixels > {TOL} (quantiser)")
        else:
            check(d.max() < TOL, f"plan {k} {names}: max diff {d.max():.2e}")
    print(f"  done (quantiser plans: worst fraction of pixels > {TOL}: {qmax:.1e})")

    print("4-6. identity, crop consistency, seekability (torch)")
    for name in vd.REGISTRY:
        op = vd.sample_op(np.random.default_rng(1), name, 1.0, severity=1.0)
        op.envelope, op.support, op.bipolar = vd.Envelope(), vd.Support(), False
        plan = vd.Plan([op], 60.0, 0)
        x = F[:4].astype(np.float32) / 255
        y0 = vt.render(x, idx[:4], plan.with_severity(0.0), device=DEV)
        # same bound as test_video_distortions.py (temporal ops run the lin/unlin round trip)
        check(np.abs(y0 - x).max() < 1e-6, f"{name}: severity 0 not identity")
        m = plan.margin(H)
        cx0, cy0, cw, ch = 200, 100, 160, 120
        full = vt.render(F[:3], idx[:3], plan, device=DEV)
        crop = vt.render(F[:3, cy0 - m:cy0 + ch + m, cx0 - m:cx0 + cw + m], idx[:3], plan,
                         full_size=(W, H), offset=(cx0 - m, cy0 - m), device=DEV)
        d = np.abs(full[:, cy0:cy0 + ch, cx0:cx0 + cw] - crop[:, m:m + ch, m:m + cw]).max()
        check(d < TOL, f"{name}: crop mismatch {d:.2e} (margin {m})")
        if not vd.REGISTRY[name].temporal:
            a1 = vt.render(F[5:8], idx[5:8], plan, device=DEV)
            a2 = vt.render(F[3:8], idx[3:8], plan, device=DEV)[2:]
            check(np.abs(a1 - a2).max() < 1e-6, f"{name}: not seekable")

    print("6b. bit-exact re-render (same device)")
    for k in range(10):
        plan = vd.sample_plan(np.random.default_rng(5000 + k), 60.0, 0.4, n_ops=(1, 3))
        a = vt.render(F[:6], idx[:6], plan, device=DEV)
        b = vt.render(F[:6], idx[:6], plan, device=DEV)
        check(np.array_equal(a, b), f"plan {k}: two renders differ ({np.abs(a - b).max():.1e})")

    print("7. 4K full frame per op")
    src = os.environ.get("VD_TEST_4K", "/scratch/09997/rs64952/FrameDrop-VQA/data_extracted/Inter4K/60fps/UHD/1.mp4")
    F4 = frames(3840, 2160, 3, src if os.path.exists(src) else None)
    t0 = time.time()
    for name in vd.REGISTRY:
        op = vd.sample_op(np.random.default_rng(7), name, 1.0, severity=1.0)
        op.envelope, op.support, op.bipolar = vd.Envelope(), vd.Support(), False
        n = 3 if vd.REGISTRY[name].temporal else 1
        d = diff(F4[:n], np.arange(n), vd.Plan([op], 60.0, 0))
        if name in QUANT:          # FMA-level input differences flip exact step boundaries
            frac = float((d > TOL).mean())
            check(frac < 1e-4, f"{name}: 4K {frac:.1e} of pixels > {TOL}")
            print(f"  {name:22s} 4K max {d.max():.1e}, pixels > {TOL}: {frac:.1e}")
        else:
            check(d.max() < TOL, f"{name}: 4K max diff {d.max():.2e}")
    print(f"  done in {time.time() - t0:.0f}s")

    print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
