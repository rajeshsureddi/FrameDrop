"""Checks for video_distortions.py (numpy + OpenCV only).
python test_video_distortions.py [frames.rgb W H]   (default: ffmpeg testsrc2)"""
import subprocess, sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_distortions as vd

def load():
    if len(sys.argv) == 4:
        w, h = int(sys.argv[2]), int(sys.argv[3]); raw = open(sys.argv[1], "rb").read()
    else:
        w, h = 640, 360
        raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=60",
                              "-t", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)

F = load(); T, H, W, _ = F.shape
idx = np.arange(T); r = np.random.default_rng(0)
x = F[:24].astype(np.float32) / 255
fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("  FAIL", msg)

print(f"{len(vd.REGISTRY)} ops, test video {W}x{H}x{T}")
for name in vd.REGISTRY:
    t0 = time.perf_counter()
    op = vd.sample_op(np.random.default_rng(1), name, 1.0, severity=1.0)
    op.envelope = vd.Envelope(); op.support = vd.Support(); op.bipolar = False
    plan = vd.Plan([op], 60.0, 0)
    # 1. identity at zero severity
    y0 = vd.render(x[:4], idx[:4], plan.with_severity(0.0))
    check(np.abs(y0 - x[:4]).max() < 1e-6, f"{name}: severity 0 not identity")
    # 2. monotone ladder (mean abs error vs clean, over a window with history)
    errs = []
    for s in (0.25, 0.5, 0.75, 1.0):  # severities share all parameters
        y = vd.render(x, idx[:24], plan.with_severity(s))
        errs.append(float(np.abs(y[12:] - x[12:]).mean()))
    check(all(b >= a * 0.999 for a, b in zip(errs, errs[1:])) and errs[-1] > 1e-4,
          f"{name}: ladder not monotone {np.round(errs, 5)}")
    # 3. crop consistency
    m = plan.margin(H); cx0, cy0, cw, ch = 200, 100, 160, 120
    full = vd.render(x[:3], idx[:3], plan)
    crop = vd.render(x[:3, cy0 - m:cy0 + ch + m, cx0 - m:cx0 + cw + m], idx[:3], plan,
                     full_size=(W, H), offset=(cx0 - m, cy0 - m))
    d = np.abs(full[:, cy0:cy0 + ch, cx0:cx0 + cw] - crop[:, m:m + ch, m:m + cw]).max()
    check(d < 2e-3, f"{name}: crop mismatch {d:.4f} (margin {m})")
    # 4. determinism / seekability
    a1 = vd.render(x[5:8], idx[5:8], plan); a2 = vd.render(x[3:8], idx[3:8], plan)[2:]
    if not vd.REGISTRY[name].temporal:
        check(np.abs(a1 - a2).max() < 1e-6, f"{name}: not seekable")
    print(f"  {name:22s} err@1={errs[-1]:.4f} crop={d:.1e} margin={m:3d}px "
          f"{(time.perf_counter() - t0):.2f}s")

# 5. temporal statistics
def frame_diff(name):
    op = vd.sample_op(np.random.default_rng(2), name, 1.0, severity=1.0)
    op.envelope = vd.Envelope(); op.support = vd.Support()
    still = np.repeat(x[:1], 3, 0)                       # static scene
    y = vd.render(still, idx[:3], vd.Plan([op], 60.0))
    return float(np.abs(y[1] - y[0]).mean())
check(frame_diff("luma_noise") > 1e-3, "luma noise should change every frame")
check(frame_diff("fixed_pattern_noise") < 1e-6, "fixed-pattern noise must be frozen")
check(frame_diff("flicker") > 1e-4, "flicker must vary over time")

# 6. envelopes / bipolar / labels
plan = vd.sample_plan(r, 60.0, 4.0, n_ops=(3, 3))
lab = vd.describe(plan, np.arange(240))
check(len(lab["ops"]) == 3, "plan size")
op = vd.Op("exposure", 1.0, dict(ev=2.0), 1, vd.Envelope("osc", 1.0, 1.0, 0.0), vd.Support(), True)
st = op.strength(np.linspace(0, 2, 120))
check(st.min() < -0.9 and st.max() > 0.9, "bipolar osc should swing both ways")
lab = vd.describe(vd.Plan([op], 60.0), np.arange(120))
check({"underexposure", "overexposure", "exposure pumping"} <= set(lab["terms"]), f"terms {lab['terms']}")
iv = vd.Op("defocus", 1.0, dict(sigma=4), 1, vd.Envelope("interval", 0.5, 1.0, 0.1))
lab = vd.describe(vd.Plan([iv], 60.0), np.arange(120))["ops"][0]
check(0.45 < lab["onset_s"] < 0.6 and 0.9 < lab["offset_s"] < 1.01, f"interval timing {lab}")

# 7. lexicon coverage (the DriveIQA vocabulary)
LEX = ["low contrast", "haze veiling", "shadow clipping", "highlight clipping", "underexposure", "halo",
       "flare", "light bloom", "tonal separation loss", "dynamic range loss", "tonal detail loss",
       "overexposure", "shadow compression", "detail loss", "blur softness", "over-smoothing",
       "chroma noise", "luma noise", "motion blur", "noise non-uniformity", "legibility loss",
       "pixelation / aliasing", "temporal noise", "compression blocking", "ringing", "ghosting",
       "color desaturation", "color cast", "muddy color", "color fringing", "chromatic aberration",
       "false color", "color aliasing", "color separation loss", "color bleeding", "oversaturation",
       "color fidelity loss"]
cov = set()
for n, sp in vd.REGISTRY.items():
    P = sp.sample(np.random.default_rng(5))
    for sg in (-1, 1):
        cov.update(sp.terms(P, sg))
for n, sp in vd.REGISTRY.items():          # sign-dependent terms with both parameter signs
    for seed in range(20):
        cov.update(sp.terms(sp.sample(np.random.default_rng(seed)), 1))
cov |= set(vd.TEMPORAL_TERMS.values()) | set(vd.CODEC_TERMS) | set(vd.OUTCOME_ONLY)
missing = [t for t in LEX if t not in cov]
check(not missing, f"lexicon terms without a generator: {missing}")
print("lexicon coverage:", len(LEX) - len(missing), "/", len(LEX))

# 8. COMBINED plans: random 2-3 op plans with random envelopes/support/bipolar
print("combined plans:")
rc = np.random.default_rng(123)
n_plans, worst_crop, nonmono = 40, 0.0, []
for k in range(n_plans):
    plan = vd.sample_plan(rc, 60.0, 0.4, n_ops=(2, 3))
    names = "+".join(o.name for o in plan.sorted_ops())
    # 8a identity at zero severity
    y0 = vd.render(x[:4], idx[:4], plan.with_severity(0.0))
    check(np.abs(y0 - x[:4]).max() < 1e-6, f"combo {names}: severity 0 not identity")
    # 8b monotone ladder of the whole plan (all ops scaled together)
    errs = [float(np.abs(vd.render(x, idx[:24], plan.with_severity(s))[12:] - x[12:]).mean())
            for s in (0.25, 0.5, 0.75, 1.0)]
    if not all(b >= a * 0.98 for a, b in zip(errs, errs[1:])):
        nonmono.append((names, np.round(errs, 4)))
    # 8c crop consistency with summed margins
    m = plan.margin(H); mt = plan.margin_t()
    cx0, cy0, cw, ch = 220, 110, 128, 96
    if cy0 - m >= 0 and cx0 - m >= 0 and cy0 + ch + m <= H and cx0 + cw + m <= W:
        sl = slice(0, mt + 3)
        full = vd.render(x[sl], idx[sl], plan)
        crop = vd.render(x[sl, cy0 - m:cy0 + ch + m, cx0 - m:cx0 + cw + m], idx[sl], plan,
                         full_size=(W, H), offset=(cx0 - m, cy0 - m))
        d = float(np.abs(full[-1, cy0:cy0 + ch, cx0:cx0 + cw] - crop[-1, m:m + ch, m:m + cw]).max())
        worst_crop = max(worst_crop, d)
        check(d < 5e-3, f"combo {names}: crop mismatch {d:.4f} (margin {m})")
    # 8d order independence of input listing (rendering uses canonical stage order)
    rev = vd.Plan(list(reversed(plan.ops)), plan.fps, plan.seed)
    check(np.abs(vd.render(x[:2], idx[:2], rev) - vd.render(x[:2], idx[:2], plan)).max() < 1e-6,
          f"combo {names}: depends on listing order")
    # 8e labels cover every op
    lab = vd.describe(plan, idx[:24])
    check(len(lab["ops"]) == len(plan.ops), f"combo {names}: labels missing ops")
print(f"  {n_plans} plans, worst crop error {worst_crop:.1e}, "
      f"non-monotone ladders: {len(nonmono)}")
for n, e in nonmono:
    print("   ", n, e)
check(len(nonmono) <= n_plans // 10, "too many non-monotone combined ladders")

# 9. combination with frame holds (drops applied AFTER distortion, as in the study)
plan = vd.Plan([vd.sample_op(np.random.default_rng(7), "luma_noise", 1, 1.0),
                vd.sample_op(np.random.default_rng(8), "saturation", 1, 0.8)], 60.0)
for o in plan.ops: o.envelope = vd.Envelope(); o.support = vd.Support()
y = vd.render(x[:12], idx[:12], plan)
hold = np.array([0, 1, 2, 3, 3, 3, 3, 7, 8, 9, 10, 11])        # 3 missed updates after frame 3
shown = y[hold]
check(np.abs(shown[4] - shown[3]).max() == 0, "held frame must repeat exactly (incl. noise)")
check(np.abs(shown[7] - shown[6]).mean() > 1e-3, "catch-up frame must change")

print("ALL PASSED" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
