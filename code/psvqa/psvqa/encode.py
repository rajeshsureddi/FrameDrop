"""Make compressed training versions of pristine clips + a manifest.

Bitrates are drawn log-uniformly over a broad range per resolution (not tied
to any RD-curve protocol), with random GOP length and encoder, all in
low-latency mode. The pristine clip is included as the top rank.

  python -m psvqa.encode clips/*.mp4 --out encodes/ --n 5 --encoders libx264 hevc_nvenc
"""
import argparse
import csv
import os
import subprocess

import numpy as np


def encoder_args(enc, kbps, gop):
    rate = ["-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps // 2}k", "-g", str(gop)]
    if enc == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p1", "-tune", "ull", "-rc", "cbr", "-bf", "0"] + rate
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", "-rc", "cbr", "-bf", "0"] + rate
    if enc == "libx265":
        return ["-c:v", "libx265", "-preset", "ultrafast", "-tune", "zerolatency",
                "-x265-params", "bframes=0"] + rate
    return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-bf", "0"] + rate


def height_of(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=height", "-of", "csv=p=0", path], capture_output=True, text=True)
    return int(out.stdout.strip() or 2160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5, help="encodes per clip")
    ap.add_argument("--encoders", nargs="+", default=["libx264"])
    ap.add_argument("--cache_dir", default="caches_train",
                    help="where extract --mode full will write (for the manifest)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for clip in a.clips:
        stem = os.path.splitext(os.path.basename(clip))[0]
        lo, hi = (1500, 60000) if height_of(clip) > 1500 else (500, 20000)
        rates = sorted(int(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for _ in range(a.n))
        for rank, kbps in enumerate(rates):
            enc = str(rng.choice(a.encoders))
            gop = int(rng.choice([30, 60, 120, 240, 600]))
            dst = os.path.join(a.out, f"{stem}__{enc}_{kbps}k.mp4")
            if not os.path.exists(dst):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-an"]
                               + encoder_args(enc, kbps, gop) + [dst], check=True)
            rows.append(dict(source_id=stem, video=dst, rank=rank,
                             cache=os.path.join(a.cache_dir, os.path.basename(dst)[:-4] + ".npz")))
        rows.append(dict(source_id=stem, video=clip, rank=a.n,
                         cache=os.path.join(a.cache_dir, stem + ".npz")))
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "video", "rank", "cache", "mos", "dataset"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(rows)} rows -> {os.path.join(a.out, 'manifest.csv')}")


if __name__ == "__main__":
    main()
