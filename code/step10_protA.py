"""Protocol A on the baselines' 20 splits (protocol matching; logged in study_eval_log.csv).

Splits: GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=0) over base clips (study_design.csv), as in the baselines' splits.csv; checked against its published facts (355-391 test videos,
each clip in test 0-8 times, Godofwar1 / Monsterhunterworldiceborne2 never in test).
Per split: fit on train only (ridge: RidgeCV; SVR: C/gamma by GroupKFold-5 inside train, plus the earlier
fixed SVR C=1, eps=0.1), predict test, PLCC/SROCC/KROCC/RMSE on raw predictions; report the median over 20
splits and split 1 alone (VSFA's test set). Features: frozen pooled fusion features of each model's single
study scoring run (no new inference). GT: mos_visibility_per_video.csv.

    python step10_protA.py
    python step10_protA.py --model d1=<study pooled_features.npz> --out <dir>    (one more model, own output dir)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))   # scikit-learn (private copy)
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
GT = "/scratch/09997/rs64952/FrameDrop-VQA/mos_datasets/framedrop_vqa_subjective_study/ground_truth/mos_visibility_per_video.csv"
OUT = f"{ROOT}/step10_protA"
MODELS = dict(v2=f"{ROOT}/frozen_v2_20261017/eval/v2_mos_pairs/study/pooled_features.npz",
              v3=f"{ROOT}/frozen_v3_20261018/eval/study/pooled_features.npz")
METRICS = ("PLCC", "SROCC", "KROCC", "RMSE")


def base_clips(names):
    with open(f"{ROOT}/study_design.csv") as f:                    # explicit design table (step10_study_design.py)
        d = {r["video_filename"]: r["base_clip"] for r in csv.DictReader(f)}
    return np.array([d[n] for n in names])


def regressors(groups_tr):
    svr_cv = GridSearchCV(make_pipeline(StandardScaler(), SVR()),
                          {"svr__C": [0.1, 1, 10, 100], "svr__gamma": ["scale", 1e-3, 1e-2], "svr__epsilon": [0.05, 0.1, 0.2]},
                          cv=GroupKFold(5), scoring="neg_root_mean_squared_error", n_jobs=1)
    return {"ridge": (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))), {}),
            "svr_cv": (svr_cv, dict(groups=groups_tr)),
            "svr_fixed": (make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)), {})}


def scores(p, y):
    return dict(PLCC=float(pearsonr(p, y)[0]), SROCC=float(spearmanr(p, y)[0]), KROCC=float(kendalltau(p, y)[0]),
                RMSE=float(np.sqrt(np.mean((p - y) ** 2))))


def main():
    global OUT, MODELS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=pooled_features.npz (replaces the default v2/v3)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.model:
        MODELS = dict(m.split("=", 1) for m in a.model)
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    with open(GT) as f:
        gt = {r["video_filename"]: float(r["mos"]) for r in csv.DictReader(f)}
    names = sorted(gt)
    groups = base_clips(names)
    y = np.array([gt[n] for n in names])
    splits = list(GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=0).split(names, groups=groups))
    te_sizes = [len(te) for _, te in splits]
    times = {g: 0 for g in np.unique(groups)}
    for _, te in splits:
        for g in np.unique(groups[te]):
            times[g] += 1
    check = dict(videos=len(names), clips=len(times), test_videos=[min(te_sizes), max(te_sizes)],
                 times_in_test=[min(times.values()), max(times.values())],
                 never_in_test=sorted(g for g, t in times.items() if t == 0))
    print(json.dumps(check), flush=True)
    with open(f"{OUT}/splits_regenerated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_filename", "base_clip"] + [f"split_{k:02d}" for k in range(1, 21)])
        lab = np.full((len(names), 20), "train", dtype=object)
        for k, (_, te) in enumerate(splits):
            lab[te, k] = "test"
        for i, n in enumerate(names):
            w.writerow([n, groups[i]] + lab[i].tolist())
    res, rows = dict(split_check=check, gt=GT), []
    for m, path in MODELS.items():
        pf = np.load(path)
        fx = dict(zip(pf["clips"].tolist(), pf["x"]))
        missing = [n for n in names if n not in fx]
        if missing:
            raise SystemExit(f"{m}: {len(missing)} study videos without features, e.g. {missing[:3]}")
        X = np.stack([fx[n] for n in names]).astype(np.float64)
        per = {}
        for k, (tr, te) in enumerate(splits, 1):
            for r, (mdl, kw) in regressors(groups[tr]).items():
                mdl.fit(X[tr], y[tr], **kw)
                s = scores(mdl.predict(X[te]), y[te])
                if r == "svr_cv":
                    s["best_params"] = mdl.best_params_
                per.setdefault(r, []).append(s)
                rows.append(dict(model=m, regressor=r, split=k, **{q: s[q] for q in METRICS}))
            print(m, k, {r: round(v[-1]["SROCC"], 4) for r, v in per.items()}, flush=True)
        res[m] = {r: dict(median={q: float(np.median([s[q] for s in v])) for q in METRICS},
                          split_01={q: v[0][q] for q in METRICS},
                          svr_cv_params=[s.get("best_params") for s in v] if r == "svr_cv" else None,
                          feature_dim=int(X.shape[1])) for r, v in per.items()}
    with open(f"{OUT}/per_split_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(res, f, indent=1)
    for m in MODELS:
        for r, v in res[m].items():
            print(m, r, "median", {q: round(x, 3) for q, x in v["median"].items()},
                  "split1", {q: round(x, 3) for q, x in v["split_01"].items()})


if __name__ == "__main__":
    main()
