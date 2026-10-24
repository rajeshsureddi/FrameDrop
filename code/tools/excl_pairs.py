"""Render dataset-frame / study-frame pairs behind each dense exclusion match.
python tools/excl_pairs.py <step1 out dir> <review dir>   (frames only)"""
import csv, glob, json, os, subprocess, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_manifest import DENSE_EVERY, FLAT_STD, TH, TW, highpass
out, rev = sys.argv[1], sys.argv[2]
os.makedirs(rev, exist_ok=True)
ncc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
recs = [r for r in (json.load(open(f)) for f in glob.glob(os.path.join(out, "probe_cache_seg5", "*.json"))) if "dense_tiny" in r]
ex = {os.path.basename(r["path"]): r for r in recs if r["dataset"] == "__EXCLUDE__"}
by_path = {r["path"]: r for r in recs}
def frame(path, idx, fps):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{idx}),scale=384:216",
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    b = np.frombuffer(r.stdout, np.uint8)
    return b.reshape(216, 384, 3).copy() if len(b) == 384 * 216 * 3 else np.zeros((216, 384, 3), np.uint8)
rows = [r for r in csv.DictReader(open(os.path.join(out, "manifest.csv"))) if "dense:" in r["excl_match"]]
lines = []
for r in rows:
    d = by_path[r["path"]]
    for m in r["excl_match"].split("|"):
        if not m.startswith("dense:"):
            continue
        _, name, cnt = m.split(":")
        e = ex[name]
        def z(rr):
            v = np.array([highpass(np.frombuffer(bytes.fromhex(t), np.uint8).astype(np.float32)
                          .reshape(TH // 4, TW // 4)).ravel() for t in rr["dense_tiny"]], np.float32)
            v -= v.mean(1, keepdims=True); return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
        C = z(d) @ z(e).T
        best = []
        for k, sd in enumerate(d["dense_std"]):
            j = int(C[k].argmax())
            if sd >= FLAT_STD and e["dense_std"][j] >= FLAT_STD and C[k, j] >= ncc:
                best.append((-round(float(C[k, j]), 3), k * DENSE_EVERY, j, sd))
        best.sort()
        tiles = []
        for dist, fi, fj, sd in best[:3]:
            a, b = frame(r["path"], fi, 60), frame(e["path"], fj, 60)
            t = np.concatenate([a, np.full((216, 6, 3), 255, np.uint8), b], 1)
            cv2.putText(t, f"ncc {-dist}  data f{fi} (std {sd})  study f{fj}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(t)
        img = np.concatenate(tiles, 0)
        fn = f"{r['dataset']}_{os.path.basename(r['path'])[:-4]}__{name[:-4]}.jpg"
        cv2.imwrite(os.path.join(rev, fn), img)
        lines.append(f"{fn}: {cnt} frames, best ncc {[-b[0] for b in best[:5]]}")
print("\n".join(lines))
