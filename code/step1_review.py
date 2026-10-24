"""Step 1.5: spot-check images for the human review of the Step 1 filters.

For each filter in FLAGS, picks 10 flagged and 10 kept clips (fixed seed) and
renders one panel per clip:
  * mid-clip thumbnail (480x270)
  * 30-frame duplicate pattern around mid-clip: red = confirmed repeat (128x72
    suspect AND 960x540 mean < 0.25, max <= 2), orange = 128x72 suspect only;
    bar height = mean |diff| at 128x72
  * whole-scan motion trace with detected cuts (magenta) and the usable spans
    (green bands); the active area (letterbox crop) is drawn in yellow
Panels are tiled into <flag>_flagged.jpg / <flag>_kept.jpg; review_index.csv
lists every sampled clip with its stats and dup pattern as text.

python step1_review.py --manifest .../step1/manifest.csv --out .../step1/step1_review
Reads frames only; thresholds are not changed here.
"""
import argparse
import csv
import os
import subprocess

import json

import cv2
import numpy as np

from build_manifest import DUP_MAX, DUP_MEAN, HH, HI_DUP_MAX, HI_DUP_MEAN, HW, TH, TW

FLAGS = ["upconverted_cadence", "native_freeze", "low_bpp", "no_usable_span", "static", "letterbox"]
PW, PH = 480, 270            # thumbnail
BAR_H, TRACE_H, TXT_H = 40, 60, 64


def thumb(path, t):
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path,
                        "-frames:v", "1", "-vf", f"scale={PW}:{PH}:flags=area,format=bgr24",
                        "-f", "rawvideo", "-"], capture_output=True)
    b = np.frombuffer(r.stdout, np.uint8)
    return b[:PW * PH * 3].reshape(PH, PW, 3).copy() if len(b) >= PW * PH * 3 else np.zeros((PH, PW, 3), np.uint8)


def luma(path, max_seconds, w=TW, h=TH):
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-t", str(max_seconds), "-i", path,
                        "-an", "-sn", "-fps_mode", "passthrough",
                        "-vf", f"scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    b = np.frombuffer(r.stdout, np.uint8)
    n = len(b) // (w * h)
    return b[:n * w * h].reshape(n, h, w)


def hi_diffs(path, max_seconds):
    fr = luma(path, max_seconds, HW, HH)
    hm, hx = np.zeros(len(fr)), np.zeros(len(fr))
    for t in range(1, len(fr)):
        d = np.abs(fr[t].astype(np.int16) - fr[t - 1].astype(np.int16))
        hm[t], hx[t] = d.mean(), d.max()
    return hm, hx


def panel(row, max_seconds):
    fr = luma(row["path"], max_seconds)
    n = len(fr)
    motion = np.zeros(n, np.float32)
    dup = np.zeros(n, bool)
    for t in range(1, n):
        d = np.abs(fr[t].astype(np.int16) - fr[t - 1].astype(np.int16))
        motion[t] = d.mean()
        dup[t] = d.mean() < DUP_MEAN and d.max() <= DUP_MAX
    hm, hx = hi_diffs(row["path"], max_seconds)
    m = min(len(hm), n)
    conf = np.zeros(n, bool)
    conf[:m] = dup[:m] & (hm[:m] < HI_DUP_MEAN) & (hx[:m] <= HI_DUP_MAX)
    mid = n // 2
    lo = max(1, mid - 15)
    win = range(lo, min(n, lo + 30))
    fps = float(row["fps"] or 60)
    img = np.full((PH + BAR_H + TRACE_H + TXT_H, PW, 3), 255, np.uint8)
    img[:PH] = thumb(row["path"], mid / fps)
    ax = json.loads(row.get("active_xywh") or "[]")
    W0, H0 = int(row["width"] or 3840), int(row["height"] or 2160)
    if len(ax) == 4 and (ax[2] < W0 or ax[3] < H0):
        sx, sy = PW / W0, PH / H0
        cv2.rectangle(img, (int(ax[0] * sx), int(ax[1] * sy)),
                      (int((ax[0] + ax[2]) * sx) - 1, int((ax[1] + ax[3]) * sy) - 1), (0, 230, 255), 2)
    # dup pattern bar: cell colour = dup/new, cell height = mean |diff| (capped at 8)
    cw = PW / 30
    y0 = PH
    for k, t in enumerate(win):
        x0, x1 = int(k * cw), int((k + 1) * cw) - 1
        col = (40, 40, 220) if conf[t] else (0, 150, 255) if dup[t] else (60, 170, 60)
        hh = int(min(motion[t], 8.0) / 8.0 * (BAR_H - 4)) + 3
        cv2.rectangle(img, (x0, y0 + BAR_H - hh), (x1, y0 + BAR_H - 1), col, -1)
    # whole-scan trace
    y0 = PH + BAR_H
    cv2.rectangle(img, (0, y0), (PW - 1, y0 + TRACE_H - 1), (235, 235, 235), -1)
    for a_, b_ in json.loads(row.get("spans") or "[]"):
        if n:
            cv2.rectangle(img, (int(a_ / n * PW), y0), (int(min(n, b_) / n * PW) - 1, y0 + TRACE_H - 1),
                          (200, 240, 200), -1)
    med = float(np.median(motion[1:])) if n > 2 else 0.0
    thr = max(25.0, 8.0 * med + 1.0)
    top = max(float(motion.max()) if n else 1.0, thr, 1.0)
    pts = [(int(t / max(n - 1, 1) * (PW - 1)), y0 + TRACE_H - 1 - int(motion[t] / top * (TRACE_H - 2)))
           for t in range(n)]
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, (90, 90, 90), 1)
    for t in np.nonzero(motion > thr)[0]:
        x = int(t / max(n - 1, 1) * (PW - 1))
        cv2.line(img, (x, y0), (x, y0 + TRACE_H - 1), (200, 0, 200), 2)
    for t in np.nonzero(dup)[0]:
        x = int(t / max(n - 1, 1) * (PW - 1))
        img[y0 + TRACE_H - 4:y0 + TRACE_H - 1, x] = (40, 40, 220) if conf[t] else (0, 150, 255)
    # text
    name = f"{row['dataset']}/{os.path.basename(row['path'])}"
    l1 = f"{name}  keep={row['keep']}"
    l2 = f"flags: {row['flags'] or '-'}"
    l3 = (f"bpp {float(row['bpp'] or 0):.4f}  fps {row['fps']}  dup lo/conf {row['dup_frac_lo']}/"
          f"{row['dup_frac']}  run {row['longest_repeat']}  lag {row['cadence_lag']}/{row['cadence_strength']}")
    l4 = (f"cuts {row['n_cuts']}  spans {len(json.loads(row.get('spans') or '[]'))} "
          f"({row['usable_frames']} fr)  held {row['held_frames']}  medmot {med:.2f}")
    y = PH + BAR_H + TRACE_H + 14
    for s in (l1, l2, l3, l4):
        cv2.putText(img, s[:78], (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        y += 15
    pattern = "".join("R" if conf[t] else "s" if dup[t] else "." for t in win)
    return img, pattern, med


def sheet(panels, cols=5):
    if not panels:
        return None
    h, w = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    out = np.full((rows * (h + 6), cols * (w + 6), 3), 128, np.uint8)
    for k, p in enumerate(panels):
        r, c = divmod(k, cols)
        out[r * (h + 6):r * (h + 6) + h, c * (w + 6):c * (w + 6) + w] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))
    kept = [r for r in rows if r["keep"] == "True"]
    index = []
    for flag in FLAGS:
        flagged = [r for r in rows if flag in r["flags"].split("|")]
        for kind, pool in (("flagged", flagged), ("kept", kept)):
            rng = np.random.default_rng([a.seed, FLAGS.index(flag), kind == "kept"])
            pick = [pool[i] for i in sorted(rng.choice(len(pool), min(a.n, len(pool)), replace=False))]
            panels = []
            for r in pick:
                img, pat, med = panel(r, a.max_seconds)
                panels.append(img)
                index.append(dict(filter=flag, sample=kind, dataset=r["dataset"],
                                  clip=os.path.basename(r["path"]), keep=r["keep"], flags=r["flags"],
                                  bpp=r["bpp"], fps=r["fps"], dup_frac=r["dup_frac"],
                                  longest_repeat=r["longest_repeat"], cadence_lag=r["cadence_lag"],
                                  cadence_strength=r["cadence_strength"], n_cuts=r["n_cuts"],
                                  spans=r["spans"], usable_frames=r["usable_frames"],
                                  held_frames=r["held_frames"], active_xywh=r["active_xywh"],
                                  median_motion=round(med, 3), dup_pattern_30=pat))
            s = sheet(panels)
            if s is not None:
                cv2.imwrite(os.path.join(a.out, f"{flag}_{kind}.jpg"), s, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"{flag:22s} {kind:8s} {len(pick):3d} of {len(pool)}", flush=True)
    with open(os.path.join(a.out, "review_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()) if index else ["filter"])
        w.writeheader()
        w.writerows(index)


if __name__ == "__main__":
    main()
