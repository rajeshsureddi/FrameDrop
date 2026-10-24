"""Checks for step3_video_loader: tiles bit-exact vs a full decode, low-res vs
cv2 INTER_AREA of the full RGB frame, drop remap consistency, pair types."""
import sys, os, time, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import step3_common as c3, step3_video_loader as vl
from psvqa import drops as dr
M = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/encodes_manifest.csv"
PL = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step3/capture_plans.json"
ds = vl.VideoDataset(M, splits=("train",), plans_json=PL, seed=3)
if len(sys.argv) > 1: ds.hw_device = int(sys.argv[1]); print("NVDEC device", ds.hw_device)
rng = np.random.default_rng(5)
sid = "LMM-10K__12170220_5s"
t0 = time.time(); s = ds.single_sample(sid, rng); print("single sample %.1fs" % (time.time() - t0), s["meta"]["encode"], s["meta"]["drop_family"], s["meta"]["n_dropped"], s["low"].shape, s["tiles"].shape)
# independent full decode of the same encode
v = [x for x in ds.by_src[sid] if os.path.basename(x["path"]) == s["meta"]["encode"]][0]
fr = list(c3.iter_frames(c3.decode_cmd(v["path"], 8)))
a = s["meta"]["start"]; idx = dr.drops_to_index(s["meta"]["drop_mask"])
bad = 0; lowdiff = []
for k, d in enumerate(s["tile_ticks"]):
    ef = a + idx[d] - v["start"]
    Y, U, V = c3.yuv_planes(fr[ef], 3840, 2160)
    full = c3.yuv_crop_to_rgb(Y, U, V, 0, 0, 3840, 2160)
    for j, (r, c) in enumerate(s["cells"][k]):
        bad += int((s["tiles"][k, j] != full[r*238:r*238+256, c*256:c*256+256]).any())
    ref = cv2.resize(full, (384, 216), interpolation=cv2.INTER_AREA)
    lowdiff.append(np.abs(ref.astype(int) - s["low"][d].astype(int)).mean())
print("tile mismatches", bad, "of", s["tiles"].shape[0] * 32, "| low-res vs cv2 area mean abs diff %.3f max-frame %.3f" % (np.mean(lowdiff), np.max(lowdiff)))
# remap: displayed tick d shows source tick idx[d]
eq = all((s["low"][d] == s["low"][idx[d]]).all() for d in range(240) if idx[d] != d)
print("held ticks repeat earlier frame exactly:", eq)
for pt in ("drops", "rate", "severity"):
    ds.pair_types = (pt,)
    t0 = time.time(); p = ds.pair_sample(sid, rng)
    if p is None: print(pt, "None", dict(ds.stats)); continue
    print(pt, "%.1fs" % (time.time() - t0), [m["encode"] for m in p["meta"]], [m["n_dropped"] for m in p["meta"]], "diff %.2f" % p["mean_abs_diff"])
print(dict(ds.stats))
# low-res cache path == on-the-fly path, incl. an orig (source-span) window
LR = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step3/lowres_smoke"
S1 = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step1/manifest.csv"
d2 = vl.VideoDataset(M, splits=("train",), tiles=False, step1_manifest=S1, lowres_dir=LR, seed=3)
o = [x for x in d2.by_src[sid] if x["kind"] == "orig"][0]
for a in (o["start"], o["start"] + 37):
    c = d2._low_only(o, a); d2.lowres_dir = None; f = d2._low_only(o, a); d2.lowres_dir = LR
    print("orig window", a, "cache == on-the-fly:", bool((c == f).all()), c.shape)
print(dict(d2.stats))
