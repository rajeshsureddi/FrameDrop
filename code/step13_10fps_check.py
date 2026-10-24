"""Step 13 (10-21): model-free 10 fps check on DropDev / DropDev-hard (no study data, no model).

A 10 fps sampler reads every 6th displayed frame of a 60 fps clip. For each pair, map
the sampled display ticks through each member's drop index map and test whether both
members show the same source frames. If they do, the pair cannot be told apart at
10 fps. Averaged over the 6 sampling phases. Index maps only: for re-encoded pairs
this ignores the encoder's noise.

    DROPDEV=hard python step13_10fps_check.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "psvqa"))
from psvqa import drops  # noqa: E402
from step10_dropdev import OUT, load_plan, mask_from  # noqa: E402

STEP = 6                                             # 60 fps -> 10 fps

P = load_plan()
C = {c["clip"]: c for c in P["clips"]}
res = defaultdict(list)
for p in P["pairs"]:
    b, w = C[p["better"]], C[p["worse"]]
    T = b["T"]
    ib = np.asarray(drops.drops_to_index(mask_from(T, b["drop"])))
    iw = np.asarray(drops.drops_to_index(mask_from(T, w["drop"])))
    full_same = np.array_equal(ib, iw)
    same = [np.array_equal(ib[ph::STEP], iw[ph::STEP]) for ph in range(STEP)]
    for key in (p["type"], f"{p['type']}/{p['mode']}", "all"):
        res[key].append((full_same, float(np.mean(same))))
out = {k: dict(n=len(v), identical_at_60fps=float(np.mean([a for a, _ in v])),
               identical_at_10fps=float(np.mean([s for _, s in v]))) for k, v in sorted(res.items())}
os.makedirs(f"{OUT}/eval", exist_ok=True)
json.dump(out, open(f"{OUT}/eval/step13_10fps_check.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:28s} n={v['n']:6d}  same@60={v['identical_at_60fps']:.3f}  same@10={v['identical_at_10fps']:.3f}")
