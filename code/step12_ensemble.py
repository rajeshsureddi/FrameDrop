"""Step 12: score-level ensemble of models (label-free z-score of Q and V over the scored set, then mean).

    python step12_ensemble.py --name ens_d1_v2_90 --members <scores dir> <scores dir> --out <dir>
Each member dir holds scores.csv (step7_eval score or step10_eval score output); only clips scored by every member
are kept. Writes <out>/scores.csv with the same columns as the first member (Q, V replaced). No labels are read.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tabs = [{r["clip"]: r for r in csv.DictReader(open(f"{d}/scores.csv"))} for d in a.members]
    clips = sorted(set.intersection(*(set(t) for t in tabs)))
    out = {c: dict(tabs[0][c]) for c in clips}
    for k in ("Q", "V"):
        z = []
        for t in tabs:
            x = np.array([float(t[c][k]) for c in clips])
            z.append((x - x.mean()) / x.std())
        for c, v in zip(clips, np.mean(z, axis=0)):
            out[c][k] = f"{v:.6f}"
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tabs[0][clips[0]]))
        w.writeheader()
        w.writerows(out[c] for c in clips)
    print(f"{len(clips)} clips ({[len(t) for t in tabs]} per member) -> {a.out}/scores.csv")


if __name__ == "__main__":
    main()
