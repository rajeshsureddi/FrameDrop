"""Frame-drop visibility for the baselines with per-video outputs on disk (user 10-23).
  - InternVL3-8B / InternVideo3-8B: the study prompt asked "Are there any frame drops?" with the same 4 labels
    (A clearly / B slightly / C barely / D none) -> scored directly, no fitting (zero-shot).
  - DOVER, FasterVQA (and the VLMs' MOS rating): continuous quality score, negated -> same CV cut-point protocol as
    step15_vis_accuracy.py.
  python step15_vis_baselines.py  ->  vqa_data/step15_v4/eval/vis_baselines.json
"""
import csv
import json
import os

import numpy as np

from step7_metrics import auc, read_design
from step15_vis_accuracy import evaluate
from step15_vis_majority import D, labels

B = "/scratch/09997/rs64952/FrameDrop-VQA/baselines"
CHOICE = {"D": 0, "C": 1, "B": 2, "A": 3}


def vlm(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        res = r["results"][0] if "results" in r else r
        out[os.path.basename(r.get("video", res.get("video", "")))] = (res.get("choice"), res.get("rating"))
    return out


def main():
    lab = labels()
    des = read_design()
    names = sorted(lab)
    y = np.array([lab[n][0] for n in names])
    g = np.array([des[n]["base_clip"] for n in names])
    res = {}
    for name, path in (("InternVL3-8B", f"{B}/InternVL3/results/internvl3_cot_118f_ALL.jsonl"),
                       ("InternVideo3-8B", f"{B}/InternVideo3/results/internvideo3_cot_60fps_ALL.jsonl")):
        v = vlm(path)
        miss = [n for n in names if n not in v or v[n][0] not in CHOICE]
        p = np.array([CHOICE.get(v.get(n, (None,))[0], -1) for n in names])
        ok = p >= 0
        rec = [float((p[ok & (y == c)] == c).mean()) for c in range(4)]
        res[name + " (direct answer)"] = dict(
            n_unparsed=len(miss), pred_counts=np.bincount(p[ok], minlength=4).tolist(),
            grade4_acc=float((p == y).mean()), grade4_recall=rec,
            noticed_acc=float(((p >= 1) == (y >= 1)).mean()), clearly_acc=float(((p == 3) == (y == 3)).mean()),
            noticed_recall=[float(((p >= 1) == False)[y == 0].mean()), float((p >= 1)[y >= 1].mean())],
            clearly_recall=[float((p != 3)[y != 3].mean()), float((p == 3)[y == 3].mean())],
            AUC_noticed=auc(p, y >= 1), AUC_clearly=auc(p, y == 3))
        r = np.array([float(v[n][1]) if n in v and v[n][1] is not None else np.nan for n in names])
        if np.isnan(r).any():
            r[np.isnan(r)] = np.nanmedian(r)
        res[name + " (MOS rating, CV cuts)"] = cont(-r, y, g)
    for name, path, col in (("DOVER", f"{B}/DOVER/results/dover_scores_all.csv", "fused_score"),
                            ("FasterVQA", f"{B}/FasterVQA/results/fastervqa_scores_all.csv", "score")):
        sc = {r["video"]: float(r[col]) for r in csv.DictReader(open(path))}
        assert all(n in sc for n in names), name
        res[name + " (quality score, CV cuts)"] = cont(-np.array([sc[n] for n in names]), y, g)
    json.dump(res, open(f"{D}/step15_v4/eval/vis_baselines.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def cont(s, y, g):
    return dict(grade4=evaluate(s, y, g, 4),
                noticed=evaluate(s, (y >= 1).astype(int), g, 2) | dict(AUC=auc(s, y >= 1)),
                clearly=evaluate(s, (y == 3).astype(int), g, 2) | dict(AUC=auc(s, y == 3)))


if __name__ == "__main__":
    main()
