"""Step 10 report check: do KoNViD-1k clips also appear in LSVQ train (both drawn from YFCC100M)?

LSVQ renames its YFCC clips, so names can't be matched. Per clip: 64-bit difference hashes
of the 32x18 luma at 3 relative times (20 / 50 / 80% of the duration). A KoNViD clip
matches an LSVQ-train clip if all three hashes are within Hamming distance <= 6.

    python step10_overlap.py hash --workers 4
    python step10_overlap.py match
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step7_eval import DATASETS  # noqa: E402

OUT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/overlap"


def dhash(path):
    try:
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                 capture_output=True, text=True).stdout.strip())
        hs = []
        for f in (0.2, 0.5, 0.8):
            p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{d * f:.3f}", "-i", path, "-frames:v", "1",
                                "-vf", "scale=9:8:flags=area,format=gray", "-f", "rawvideo", "-"], capture_output=True)
            a = np.frombuffer(p.stdout, np.uint8).reshape(8, 9).astype(int)
            hs.append(int("".join("1" if x else "0" for x in (a[:, 1:] > a[:, :-1]).ravel()), 2))
        return path, hs, d
    except Exception as e:                       # noqa: BLE001
        return path, None, str(e)[-100:]


def hash_all(a):
    os.makedirs(OUT, exist_ok=True)
    for ds in ("konvid1k", "lsvq_train"):
        dst = f"{OUT}/{ds}.json"
        if os.path.exists(dst):
            continue
        rows = [r[0].strip() for r in csv.reader(open(DATASETS[ds]["labels"]))]
        if ds == "lsvq_train":
            rows = [r for r in rows if r.startswith("yfcc")]          # LSVQ's YFCC part (KoNViD is YFCC too)
        paths = [os.path.join(DATASETS[ds]["videos"], r) for r in rows]
        with Pool(a.workers) as p:
            res = p.map(dhash, paths, chunksize=16)
        json.dump(res, open(dst, "w"))
        print(ds, len(res), sum(r[1] is None for r in res), "failed", flush=True)


def match(a):
    K = [r for r in json.load(open(f"{OUT}/konvid1k.json")) if r[1]]
    L = [r for r in json.load(open(f"{OUT}/lsvq_train.json")) if r[1]]
    kh = np.array([r[1] for r in K], dtype=np.uint64)
    lh = np.array([r[1] for r in L], dtype=np.uint64)
    pop = np.vectorize(lambda x: bin(int(x)).count("1"))
    hits = []
    for i in range(len(K)):
        d = np.stack([pop(np.bitwise_xor(lh[:, j], kh[i, j])) for j in range(3)], 1)
        for k in np.flatnonzero(d.max(1) <= 6):
            hits.append(dict(konvid=os.path.basename(K[i][0]), lsvq=L[k][0].split("videos/")[-1], hamming=d[k].tolist(),
                             dur_konvid=K[i][2], dur_lsvq=L[k][2]))
    json.dump(dict(n_konvid=len(K), n_lsvq_yfcc=len(L), matches=hits), open(f"{OUT}/matches.json", "w"), indent=1)
    print(f"{len(K)} KoNViD x {len(L)} LSVQ-train YFCC clips: {len(hits)} matches "
          f"({len({h['konvid'] for h in hits})} KoNViD clips)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hash", "match"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dict(hash=hash_all, match=match)[a.cmd](a)
