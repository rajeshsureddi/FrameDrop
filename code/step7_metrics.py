"""Step 7 metrics (run only on the frozen final model's scores).

Protocol B (headline, zero-shot)
  Q vs MOS: SROCC, KROCC, PLCC and RMSE after the 4-parameter logistic fit,
            plus unmapped PLCC
  V vs mean visibility grade (0..3): SROCC/KROCC, logistic PLCC, SROCC vs P(noticed)=1-p_not;
            AUC noticed (P(noticed) > 0.5), AUC clearly (p_clearly >= 0.5); within-content SROCC
  per subset (from the explicit design table study_design.csv, never a model input):
            uncompressed+drops, consecutive 5/15/25, random 15/30/60 %,
            bitrate region (by kbps tercile), no-drop references
Protocol A: ridge / SVR on frozen pooled fusion features, 20 random 80/20
  splits grouped by source content; median SROCC / PLCC on the held-out folds.

    python step7_metrics.py --dataset study
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))   # scikit-learn (private copy)
from step4_mos_eval import metrics as q_metrics  # noqa: E402  (logistic-4 mapping, SROCC/KROCC/PLCC/RMSE)
from step7_eval import DATASETS, OUT, clip_name  # noqa: E402


def auc(score, pos):
    """ROC AUC via the rank-sum statistic."""
    pos = np.asarray(pos, bool)
    r = spearmanr_rank(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def spearmanr_rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


DESIGN = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/study_design.csv"   # step10_study_design.py (reviewed)


def read_design():
    """video -> explicit design row; metric code never parses file names."""
    with open(DESIGN) as f:
        return {r["video_filename"]: r for r in csv.DictReader(f)}


def subsets_of(d):
    s = [f"{d['drop_type']}_{d['drop_level']}" if d["drop_type"] != "none" else "no_drops"]
    if d["compressed"] == "0":
        s.append("uncompressed" + ("+drops" if d["drop_type"] != "none" else ""))
    return s


VIS_COLS = ("p_not_noticeable", "p_barely_noticeable", "p_slightly_noticeable", "p_clearly_noticeable")   # grades 0..3


def read_study_labels(path):
    """Explicit schema (mos_visibility_per_video.csv): video_filename, mos, sos, the four p_ columns.
    vis = mean visibility grade 0..3; p_noticed = 1 - p_not_noticeable. No column guessing."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    need = ("video_filename", "mos") + VIS_COLS
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    out = {}
    for r in rows:
        p = np.array([float(r[c]) for c in VIS_COLS])
        if abs(p.sum() - 1) > 1e-3 or (p < 0).any():
            raise SystemExit(f"{r['video_filename']}: visibility probabilities {p} do not form a distribution")
        out[r["video_filename"]] = dict(mos=float(r["mos"]), vis=float(p @ np.arange(4)), p_noticed=float(1 - p[0]),
                                        p_clearly=float(p[3]))
    return out


def protocol_a(X, y, groups, n=20, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    res = {"ridge": [], "svr": []}
    for _ in range(n):
        test_g = set(rng.choice(ug, max(1, int(round(0.2 * len(ug)))), replace=False))
        te = np.array([g in test_g for g in groups])
        for name, mdl in (("ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))),
                          ("svr", make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)))):
            mdl.fit(X[~te], y[~te])
            p = mdl.predict(X[te])
            res[name].append((spearmanr(p, y[te])[0], pearsonr(p, y[te])[0]))
    return {k: dict(SROCC_median=float(np.median([a for a, _ in v])), PLCC_median=float(np.median([b for _, b in v])),
                    splits=n, test_groups=int(round(0.2 * len(ug)))) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="study", choices=list(DATASETS))
    ap.add_argument("--out", default=None, help="dir holding scores.csv (default step7/<dataset>)")
    a = ap.parse_args()
    od = a.out or f"{OUT}/{a.dataset}"
    with open(f"{od}/scores.csv") as f:
        sc = {r["clip"]: r for r in csv.DictReader(f)}
    if a.dataset == "study":
        lab = read_study_labels(DATASETS["study"]["labels"])
    else:
        with open(DATASETS[a.dataset]["labels"]) as f:
            lab = {clip_name(a.dataset, r[0]): dict(mos=float(r[-1]), vis=None) for r in csv.reader(f)}
    names = sorted(set(sc) & set(lab))
    res = dict(dataset=a.dataset, clips_scored=len(sc), clips_labelled=len(lab), clips_matched=len(names),
               unmatched_scores=sorted(set(sc) - set(lab))[:10], unscored_labels=sorted(set(lab) - set(sc))[:10])
    Q = np.array([float(sc[n]["Q"]) for n in names])
    V = np.array([float(sc[n]["V"]) for n in names])
    mos = np.array([lab[n]["mos"] for n in names])
    res["Q_vs_MOS"] = q_metrics(Q, mos)
    vis = np.array([lab[n]["vis"] if lab[n]["vis"] is not None else np.nan for n in names])
    if a.dataset == "study":                         # visibility (fixed before 10d): V zero-shot vs human ratings
        pn = np.array([lab[n]["p_noticed"] for n in names])
        pc = np.array([lab[n]["p_clearly"] for n in names])
        base = np.array([read_design()[n]["base_clip"] for n in names])
        within = [spearmanr(V[base == b], vis[base == b])[0] for b in np.unique(base)]
        res["V_vs_visibility"] = dict(
            SROCC=float(spearmanr(V, vis)[0]), KROCC=float(kendalltau(V, vis)[0]),
            logistic=q_metrics(V, vis), SROCC_p_noticed=float(spearmanr(V, pn)[0]),
            AUC_noticed=auc(V, pn > 0.5), AUC_clearly=auc(V, pc >= 0.5),
            SROCC_within_content_mean=float(np.nanmean(within)), n=len(names),
            noticed=int((pn > 0.5).sum()), clearly=int((pc >= 0.5).sum()))
        res["Q_vs_visibility_SROCC"] = float(spearmanr(Q, vis)[0])
    if a.dataset == "study":
        des = read_design()
        subs = {}
        sets = {}
        for i, n in enumerate(names):
            for s in subsets_of(des[n]):
                sets.setdefault(s, []).append(i)
        kb = np.array([int(des[n]["kbps"]) if des[n]["kbps"] else -1 for n in names])
        valid_kb = kb[kb > 0]
        if len(valid_kb):
            t1, t2 = np.quantile(valid_kb, [1 / 3, 2 / 3])
            for i, k in enumerate(kb):
                if k > 0:
                    sets.setdefault("bitrate_" + ("low" if k <= t1 else "mid" if k <= t2 else "high"), []).append(i)
        for s, idx in sorted(sets.items()):
            idx = np.array(idx)
            if len(idx) < 10:
                continue
            r = dict(n=len(idx), Q_SROCC=float(spearmanr(Q[idx], mos[idx])[0]), Q_PLCC_unmapped=float(pearsonr(Q[idx], mos[idx])[0]))
            if np.isfinite(vis[idx]).sum() >= 10:
                ok = idx[np.isfinite(vis[idx])]
                r["V_SROCC"] = float(spearmanr(V[ok], vis[ok])[0])
                r["V_mean"], r["vis_mean"] = float(V[ok].mean()), float(vis[ok].mean())
                r["V_AUC_noticed"] = auc(V[ok], pn[ok] > 0.5)
            subs[s] = r
        res["per_subset"] = subs
        pf = np.load(f"{od}/pooled_features.npz")
        fx = dict(zip(pf["clips"].tolist(), pf["x"]))
        X = np.stack([fx[n] for n in names])
        res["protocol_A"] = protocol_a(X, mos, np.array([des[n]["base_clip"] for n in names]))
        res["n_contents"] = int(len(set(des[n]["base_clip"] for n in names)))
    with open(f"{od}/metrics.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
