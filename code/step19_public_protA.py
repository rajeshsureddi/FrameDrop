"""Step 19 (user 10-23: "can you do the same with other baselines such as on public datasets"): frozen pooled features
(native, half resolution, native+half) + the Protocol A regressors (ridge / SVR grid-search / SVR fixed, step10_protA) on
the public MOS sets, on the Step 14 splits (10 random 70:10:20 per dataset): fit on train+val, test once, median
PLCC/SROCC over the 10 splits (the hand-crafted rows of HFVQA Table 1 are this kind of protocol: features + SVR).
Models: v3 (LSVQ-trained; never saw KoNViD / LIVE-VQC / YT-UGC) and v4 on KoNViD / LIVE-VQC only (v4 trained on YT-UGC).
  python step19_public_protA.py  ->  vqa_data/step19_public/summary.json, per_split_metrics.csv
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step10_protA as P  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

D = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
O = f"{D}/step19_public"
NATIVE = {("v3", "konvid1k"): f"{D}/step9_e2e/eval/v3_e2e/konvid1k", ("v3", "livevqc"): f"{D}/step15_v4/eval/v3/livevqc",
          ("v3", "ytugc"): f"{O}/eval/v3/ytugc", ("v4", "konvid1k"): f"{D}/step15_v4/eval/v4/konvid1k",
          ("v4", "livevqc"): f"{D}/step15_v4/eval/v4/livevqc"}


def feats(path):
    z = np.load(f"{path}/pooled_features.npz")
    return dict(zip(z["clips"].tolist(), z["x"]))


def labels(ds):
    with open(f"{D}/eval/labels/{ds}/" + ("labels_FAST-VQA-and-FasterVQA.txt" if ds != "ytugc" else "labels.txt")) as f:
        return {os.path.basename(r[0].strip()): float(r[-1]) for r in csv.reader(f) if r}


def one(X, y, k, tr, te):
    out = {}
    for r, (mdl, kw) in P.regressors(np.arange(len(tr))).items():   # one video per content: GroupKFold == KFold
        mdl.fit(X[tr], y[tr], **kw)
        out[r] = P.scores(mdl.predict(X[te]), y[te])
    return k, out


def main():
    res, rows = {}, []
    for (m, ds), npath in NATIVE.items():
        lab = labels(ds)
        fn, fh = feats(npath), feats(f"{O}/eval/{m}/{ds}_half")
        sets = {"native": None, "half": None, "native+half": None}
        names = sorted(set(fn) & set(fh) & set(lab))
        Xn = np.stack([fn[n] for n in names]).astype(np.float64)
        Xh = np.stack([fh[n] for n in names]).astype(np.float64)
        sets = {"native": Xn, "half": Xh, "native+half": np.concatenate([Xn, Xh], 1)}
        idx = {n: i for i, n in enumerate(names)}
        y = np.array([lab[n] for n in names])
        splits = []
        for s in range(10):
            sp = json.load(open(f"{D}/step14_finetune/splits/{ds}_split{s}.json"))
            tr = [idx[os.path.basename(p)] for p in sp["train"] + sp["val"] if os.path.basename(p) in idx]
            te = [idx[os.path.basename(p)] for p in sp["test"] if os.path.basename(p) in idx]
            splits.append((np.array(tr), np.array(te)))
        key = f"{m}/{ds}"
        res[key] = dict(n_videos=len(names), n_test=[len(te) for _, te in splits])
        for fs, X in sets.items():
            per = dict(Parallel(n_jobs=10)(delayed(one)(X, y, k, tr, te) for k, (tr, te) in enumerate(splits)))
            res[key][fs] = {r: {q: float(np.median([per[k][r][q] for k in range(10)])) for q in ("PLCC", "SROCC")}
                            for r in per[0]}
            rows += [dict(model=m, dataset=ds, features=fs, regressor=r, split=k, PLCC=per[k][r]["PLCC"],
                          SROCC=per[k][r]["SROCC"]) for k in range(10) for r in per[k]]
            print(key, fs, X.shape[1], {r: (round(v["PLCC"], 3), round(v["SROCC"], 3)) for r, v in res[key][fs].items()},
                  flush=True)
    json.dump(res, open(f"{O}/summary.json", "w"), indent=1)
    with open(f"{O}/per_split_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
