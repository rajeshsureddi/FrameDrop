"""Step 17: Protocol A (step10_protA.py splits + regressors, unchanged) on v4 / GAMIVAL NSS / v4+GAMIVAL NSS,
with the 20 splits run in parallel (joblib). Median PLCC/SROCC/KROCC/RMSE over the 20 splits.
  python step17_protA.py  ->  vqa_data/step17_gamival_concat/summary.json
  python step17_protA.py --feat name=file.npz [--feat ...] --out DIR    (other feature sets, e.g. Step 18 multi-scale)
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step10_protA as P  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

D = f"{P.ROOT}/step17_gamival_concat"
FEATS = dict(v4=f"{P.ROOT}/step15_v4/eval/v4/study/pooled_features.npz",
             gamival_nss=f"{D}/gamival_nss.npz", v4_gamival_nss=f"{D}/v4_gamival_nss.npz")


def one(X, y, groups, k, tr, te):
    out = {}
    for r, (mdl, kw) in P.regressors(groups[tr]).items():
        mdl.fit(X[tr], y[tr], **kw)
        out[r] = P.scores(mdl.predict(X[te]), y[te])
    return k, out


def main():
    global D, FEATS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", action="append", default=[])
    ap.add_argument("--out", default=D)
    a = ap.parse_args()
    if a.feat:
        FEATS = dict(f.split("=", 1) for f in a.feat)
    D = a.out
    os.makedirs(D, exist_ok=True)
    gt = {r["video_filename"]: float(r["mos"]) for r in csv.DictReader(open(P.GT))}
    names = sorted(gt)
    groups = P.base_clips(names)
    y = np.array([gt[n] for n in names])
    splits = list(GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=0).split(names, groups=groups))
    res, rows = {}, []
    for m, path in FEATS.items():
        pf = np.load(path)
        fx = dict(zip(pf["clips"].tolist(), pf["x"]))
        X = np.stack([fx[n] for n in names]).astype(np.float64)
        per = Parallel(n_jobs=20)(delayed(one)(X, y, groups, k, tr, te) for k, (tr, te) in enumerate(splits, 1))
        per = dict(per)
        res[m] = dict(feature_dim=int(X.shape[1]))
        for r in per[1]:
            v = [per[k][r] for k in range(1, 21)]
            res[m][r] = dict(median={q: float(np.median([s[q] for s in v])) for q in P.METRICS},
                             mean_SROCC=float(np.mean([s["SROCC"] for s in v])), split_01={q: v[0][q] for q in P.METRICS})
            rows += [dict(model=m, regressor=r, split=k, **{q: per[k][r][q] for q in P.METRICS}) for k in range(1, 21)]
            print(m, r, {q: round(x, 3) for q, x in res[m][r]["median"].items()}, flush=True)
    json.dump(res, open(f"{D}/summary.json", "w"), indent=1)
    with open(f"{D}/per_split_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
