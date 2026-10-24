"""Step 7 supplementary analysis of the frozen model's study scores (no new labels):
  within-content Q vs MOS (median SROCC over the 35 contents)
  V and Q monotonicity in the drop amount on real clips: for the same content and
  encode (only the drop setting differs), more drops (5 < 15 < 25 frames;
  15 < 30 < 60 %) should give higher V and lower Q; no-drop reference counts as least
  V vs MOS (rank correlation, sign expected negative)

    python step7_extra.py [--out DIR_WITH_scores.csv]   (default step7/study)
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from step7_eval import OUT
from step7_metrics import content_of, read_study_labels
from step7_eval import DATASETS


def drop_key(name):
    m = re.search(r"_(consecutive_drop_(\d+)frames|random_drop_([\d.]+)pct)_drop_", name)
    if not m:
        return re.sub(r"_3840x2160_60fps\.mp4$", "", name), None, 0.0
    base = name[:m.start()]
    kind = "consecutive" if m.group(2) else "random"
    return base, kind, float(m.group(2) or m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT}/study")
    od = ap.parse_args().out
    with open(f"{od}/scores.csv") as f:
        sc = {r["clip"]: (float(r["Q"]), float(r["V"])) for r in csv.DictReader(f)}
    lab = read_study_labels(DATASETS["study"]["labels"])
    names = sorted(set(sc) & set(lab))
    res = {}
    per = defaultdict(list)
    for n in names:
        per[content_of(n)].append(n)
    rs = [spearmanr([sc[n][0] for n in v], [lab[n]["mos"] for n in v])[0] for v in per.values() if len(v) > 5]
    res["within_content_Q_SROCC"] = dict(median=float(np.median(rs)), mean=float(np.mean(rs)), min=float(np.min(rs)),
                                         max=float(np.max(rs)), contents=len(rs))
    V = np.array([sc[n][1] for n in names])
    Q = np.array([sc[n][0] for n in names])
    mos = np.array([lab[n]["mos"] for n in names])
    res["V_vs_MOS_SROCC"] = float(spearmanr(V, mos)[0])
    groups = defaultdict(dict)                      # (base, kind) -> amount -> name ; refs under amount 0
    refs = {}
    for n in names:
        base, kind, amt = drop_key(n)
        if kind is None:
            refs[base] = n
        else:
            groups[(base, kind)][amt] = n
    for kind in ("consecutive", "random"):
        v_ok = q_ok = mos_ok = tot = 0
        for (base, k), d in groups.items():
            if k != kind:
                continue
            items = sorted(d.items())
            if base in refs:
                items = [(0.0, refs[base])] + items
            for (a1, n1), (a2, n2) in combinations(items, 2):      # a1 < a2: n2 has more drops
                tot += 1
                v_ok += sc[n2][1] > sc[n1][1]
                q_ok += sc[n2][0] < sc[n1][0]
                mos_ok += lab[n2]["mos"] < lab[n1]["mos"]
        res[f"more_drops_{kind}"] = dict(pairs=tot, V_higher=v_ok / tot, Q_lower=q_ok / tot, MOS_lower=mos_ok / tot)
    res["refs_found"] = len(refs)
    with open(f"{od}/metrics_extra.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
