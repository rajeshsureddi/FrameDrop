"""Accuracy + AUC of V for majority-vote frame-drop visibility (user 10-23: "instead of ranking, we measure accuracy").
Label = majority-vote grade 0..3 (step15_vis_majority.labels). V is continuous, so grade cut-points are learned:
  - cv: grouped 5-fold over the 35 source contents; monotone cut-points fit on 4 folds (max accuracy), applied to the
    5th -> one out-of-fold prediction per video, accuracy over all 1823; mean +- sd over 10 fold seeds.
  - oracle: cut-points fit on all 1823 (optimistic upper bound).
  - chance: always predict the most frequent class.
Tasks: 4-class grade, binary noticed (grade >= 1), binary clearly (grade 3). AUC needs no cut-points (all 1823).
  python step15_vis_accuracy.py   ->  vqa_data/step15_v4/eval/vis_accuracy.json
"""
import csv
import json

import numpy as np

from step7_metrics import auc, read_design
from step15_vis_majority import D, MODELS, labels


def fit_cuts(s, y, k):
    """Monotone cut-points (k-1) on score s maximizing accuracy for ordinal labels 0..k-1 (DP over sorted scores)."""
    o = np.argsort(s, kind="stable")
    ss, yy = s[o], y[o]
    n = len(ss)
    best = np.zeros((n + 1, k))               # best[i, c]: max correct over first i items, item i-1 in class <= c
    back = np.zeros((n + 1, k), int)
    for i in range(1, n + 1):
        for c in range(k):
            stay = best[i - 1, c] + (yy[i - 1] == c)
            if c and best[i, c - 1] >= stay:
                best[i, c], back[i, c] = best[i, c - 1], back[i, c - 1]
            else:
                best[i, c], back[i, c] = stay, c
    cls = np.empty(n, int)                    # recover the class of each sorted item, then cuts between classes
    c = k - 1
    for i in range(n, 0, -1):
        c = back[i, c]
        cls[i - 1] = c
    cuts = []
    for g in range(1, k):
        hi = ss[cls >= g]
        lo = ss[cls < g]
        if len(hi) and len(lo):
            cuts.append((lo.max() + hi.min()) / 2)
        else:
            cuts.append(np.inf if not len(hi) else -np.inf)
    return np.array(cuts)


def predict(s, cuts):
    return np.searchsorted(cuts, s, side="right")


def grouped_folds(groups, k, seed):
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    return np.array([fold_of[g] for g in groups])


def evaluate(s, y, groups, k_cls, seeds=10, folds=5):
    cv = []
    for seed in range(seeds):
        f = grouped_folds(groups, folds, seed)
        pred = np.empty_like(y)
        for i in range(folds):
            cuts = fit_cuts(s[f != i], y[f != i], k_cls)
            pred[f == i] = predict(s[f == i], cuts)
        rec = [(pred[y == c] == c).mean() for c in range(k_cls)]
        cv.append([(pred == y).mean(), np.mean(rec)] + rec)
    cv = np.array(cv)
    oracle = predict(s, fit_cuts(s, y, k_cls))
    return dict(acc_cv=float(cv[:, 0].mean()), acc_cv_sd=float(cv[:, 0].std()),
                bal_acc_cv=float(cv[:, 1].mean()), recall_cv=cv[:, 2:].mean(0).tolist(), acc_oracle=float((oracle == y).mean()),
                acc_chance=float(np.bincount(y, minlength=k_cls).max() / len(y)))


def main():
    lab = labels()
    des = read_design()
    res = {}
    for name, path in MODELS.items():
        sc = {r["clip"]: r for r in csv.DictReader(open(path))}
        names = sorted(lab)
        y = np.array([lab[n][0] for n in names])
        g = np.array([des[n]["base_clip"] for n in names])
        r = {}
        for head in ("V", "Q"):
            s = np.array([float(sc[n][head]) for n in names])
            if head == "Q":
                s = -s                                   # lower quality <-> more visible drops
            r[head] = {"grade4": evaluate(s, y, g, 4),
                       "noticed": evaluate(s, (y >= 1).astype(int), g, 2) | dict(AUC=auc(s, y >= 1)),
                       "clearly": evaluate(s, (y == 3).astype(int), g, 2) | dict(AUC=auc(s, y == 3)),
                       "AUC_ordinal_mean": float(np.mean([auc(s, y >= t) for t in (1, 2, 3)]))}
        res[name] = r
    out = f"{D}/step15_v4/eval/vis_accuracy.json"
    json.dump(res, open(out, "w"), indent=1)
    for name, r in res.items():
        for head in ("V", "Q"):
            x = r[head]
            print(f"{name:24s} {head}{'(-)' if head == 'Q' else '   '} "
                  f"4cls cv {x['grade4']['acc_cv']:.3f}±{x['grade4']['acc_cv_sd']:.3f} bal {x['grade4']['bal_acc_cv']:.3f} "
                  f"oracle {x['grade4']['acc_oracle']:.3f} chance {x['grade4']['acc_chance']:.3f} | "
                  f"noticed cv {x['noticed']['acc_cv']:.3f} bal {x['noticed']['bal_acc_cv']:.3f} chance {x['noticed']['acc_chance']:.3f} AUC {x['noticed']['AUC']:.3f} | "
                  f"clearly cv {x['clearly']['acc_cv']:.3f} bal {x['clearly']['bal_acc_cv']:.3f} chance {x['clearly']['acc_chance']:.3f} AUC {x['clearly']['AUC']:.3f} | "
                  f"ordAUC {x['AUC_ordinal_mean']:.3f}")


if __name__ == "__main__":
    main()
