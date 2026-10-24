"""Study design table (one-time, for review): video -> base clip, drop condition, bitrate.

The study's design is recorded only in its file names, so this is the single place that reads them. Metric
code reads the table (vqa_data/study_design.csv), never the names; models never see either.

    python step10_study_design.py      # writes the table and prints counts per condition for review
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA"
GT = f"{ROOT}/mos_datasets/framedrop_vqa_subjective_study/ground_truth/mos_visibility_per_video.csv"
OUT = f"{ROOT}/vqa_data/study_design.csv"
COLS = ("video_filename", "base_clip", "drop_type", "drop_level", "compressed", "kbps")


def design(name):
    m = re.search(r"consecutive_drop_(\d+)frames", name)
    r = re.search(r"random_drop_([\d.]+)pct", name)
    k = re.search(r"_(\d+)kbps_", name)
    if m:
        dt, lv = "consecutive", f"{int(m.group(1))}frames"
    elif r:
        dt, lv = "random", f"{float(r.group(1)):g}pct"
    else:
        dt, lv = "none", ""
    return dict(video_filename=name, base_clip=name.split("_")[0].removesuffix(".mp4"), drop_type=dt, drop_level=lv,
                compressed=int(bool(k)), kbps=int(k.group(1)) if k else "")


def main():
    with open(GT) as f:
        names = [r["video_filename"] for r in csv.DictReader(f)]
    rows = [design(n) for n in names]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(OUT, len(rows), "videos,", len({r["base_clip"] for r in rows}), "base clips")
    print("drop condition:", sorted(Counter((r["drop_type"], r["drop_level"]) for r in rows).items()))
    print("compressed:", Counter(r["compressed"] for r in rows), "kbps values:",
          sorted(Counter(r["kbps"] for r in rows if r["compressed"]).items()))
    print("unparsed _drop_ names:", [r["video_filename"] for r in rows
                                     if r["drop_type"] == "none" and "_drop_" in r["video_filename"]][:5])


if __name__ == "__main__":
    main()
