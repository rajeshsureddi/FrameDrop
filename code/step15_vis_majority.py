"""V (and Q) vs majority-vote frame-drop visibility label (user 10-23: "take majority vote among the four we have as label").
Label = argmax of the 4 subject-fraction columns (grade 0..3). Ties (117/1823) -> tied grade closest to the mean grade;
also reported with tied videos excluded. Re-scores existing study predictions only (no inference).
  python step15_vis_majority.py   ->  vqa_data/step15_v4/eval/vis_majority.json
"""
import csv
import json

import numpy as np
from scipy.stats import kendalltau, spearmanr

from step7_metrics import VIS_COLS, auc

D = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
GT = "/scratch/09997/rs64952/FrameDrop-VQA/mos_datasets/framedrop_vqa_subjective_study/ground_truth/mos_visibility_per_video.csv"
MODELS = {"d1 (eval 8)": f"{D}/step10_cand/eval/d1/study/scores.csv",
          "ens(d1,v3_90) (eval 9)": f"{D}/step12/eval/ens_d1_v3_90/study/scores.csv",
          "v3": f"{D}/step15_v4/eval/v3/study/scores.csv",
          "v4": f"{D}/step15_v4/eval/v4/study/scores.csv"}


def labels():
    out = {}
    for r in csv.DictReader(open(GT)):
        p = np.array([float(r[c]) for c in VIS_COLS])
        top = np.flatnonzero(np.isclose(p, p.max(), atol=1e-4))
        mean = p @ np.arange(4)
        out[r["video_filename"]] = (int(top[np.argmin(np.abs(top - mean))]), len(top) > 1)
    return out


def metrics(s, y):
    m = dict(n=len(y), SROCC=float(spearmanr(s, y)[0]), KROCC=float(kendalltau(s, y)[0]),
             AUC_noticed=float(auc(s, y >= 1)), AUC_clearly=float(auc(s, y == 3)),
             AUC_0_vs_3=float(auc(s[(y == 0) | (y == 3)], y[(y == 0) | (y == 3)] == 3)),
             mean_by_grade=[float(s[y == g].mean()) for g in range(4)])
    return m


def main():
    lab = labels()
    res = {"label": "majority vote (argmax of p_ columns); ties -> grade closest to mean grade",
           "counts": np.bincount([v[0] for v in lab.values()], minlength=4).tolist(),
           "ties": int(sum(v[1] for v in lab.values())), "models": {}}
    for name, path in MODELS.items():
        sc = {r["clip"]: r for r in csv.DictReader(open(path))}
        names = sorted(set(sc) & set(lab))
        assert len(names) == len(lab) == 1823, (name, len(names))
        y = np.array([lab[n][0] for n in names])
        tie = np.array([lab[n][1] for n in names])
        r = {}
        for head in ("V", "Q"):
            s = np.array([float(sc[n][head]) for n in names])
            r[head] = dict(all=metrics(s, y), no_ties=metrics(s[~tie], y[~tie]))
        res["models"][name] = r
    out = f"{D}/step15_v4/eval/vis_majority.json"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("counts", "ties")}))
    for name, r in res["models"].items():
        for head in ("V", "Q"):
            a, b = r[head]["all"], r[head]["no_ties"]
            print(f"{name:24s} {head}  SROCC {a['SROCC']:.3f} KROCC {a['KROCC']:.3f} AUCnot {a['AUC_noticed']:.3f} "
                  f"AUCclr {a['AUC_clearly']:.3f} AUC0v3 {a['AUC_0_vs_3']:.3f} | no-ties SROCC {b['SROCC']:.3f} | "
                  f"mean by grade {' '.join(f'{v:+.2f}' for v in a['mean_by_grade'])}")


if __name__ == "__main__":
    main()
