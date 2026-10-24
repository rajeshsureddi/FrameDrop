"""Zero-shot evaluation on the subjective study (never used for training).

test.csv columns: cache (from `extract --mode policy`), mos,
                  optional: visibility (0-3 mean or level), group (e.g.
                  'uncompressed', 'consecutive-5', 'random-30', ...)

  python -m psvqa.evaluate --test test.csv --head head.pt --predictor predictor.pt
Metrics follow the baseline table: 4-parameter logistic fit on the whole
set for PLCC/RMSE (standard VQA practice), SROCC/KROCC on raw predictions.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

from .common import sample_mask
from .head import PSVQAHead, clip_tensors
from .temporal import per_frame_features, segment_stats, gate_inputs


def logistic4(x, b1, b2, b3, b4):
    return (b1 - b2) / (1 + np.exp(-(x - b3) / (np.abs(b4) + 1e-6))) + b2


def vqa_metrics(pred, mos):
    pred, mos = np.asarray(pred, float), np.asarray(mos, float)
    try:
        p0 = [mos.max(), mos.min(), np.median(pred), pred.std() + 1e-3]
        par, _ = curve_fit(logistic4, pred, mos, p0=p0, maxfev=20000)
        fit = logistic4(pred, *par)
    except RuntimeError:
        fit = np.polyval(np.polyfit(pred, mos, 1), pred)
    return dict(srocc=spearmanr(pred, mos)[0], krocc=kendalltau(pred, mos)[0],
                plcc=pearsonr(fit, mos)[0], rmse=float(np.sqrt(((fit - mos) ** 2).mean())))


def predict_clip(cache_path, net, stats, runner, device):
    c = np.load(cache_path)
    T = len(c["thumbs"])
    z = runner.normalize(c["z"]) if runner else c["z"].astype(np.float32)
    F, dup = per_frame_features(z, c["thumbs"], runner)
    seg = int(round(float(c["fps"])))
    s = dict(idx=np.arange(T), valid=c["valid"], gate=gate_inputs(F, dup),
             tstats=segment_stats(F, dup, seg))
    if not np.array_equal(c["valid"], sample_mask(dup)):
        print(f"warning: {cache_path} sampling differs from policy (was it extracted with --mode full?)")
    sp, g, ts = clip_tensors(s, c["spat"], stats, device)
    with torch.no_grad():
        Q, V, q = net(sp, g, ts, seg)
        lvl = torch.sigmoid(net.vis_logits(V)).sum().item()     # expected level 0-3
    return Q.item(), lvl, float(c["sec_per_frame"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--predictor", default=None)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.head, map_location=a.device, weights_only=False)
    net = PSVQAHead().to(a.device).eval()
    net.load_state_dict(ck["state"])
    runner = None
    if a.predictor:
        from .predictor import PredictorRunner
        runner = PredictorRunner(a.predictor, a.device)

    rows = list(csv.DictReader(open(a.test)))
    for r in rows:
        r["pred"], r["pred_vis"], r["spf"] = predict_clip(r["cache"], net, ck["stats"], runner, a.device)

    mos = [float(r["mos"]) for r in rows]
    pred = [r["pred"] for r in rows]
    print("ALL", {k: round(v, 4) for k, v in vqa_metrics(pred, mos).items()})
    groups = defaultdict(list)
    for r in rows:
        if r.get("group"):
            groups[r["group"]].append(r)
    for gname, rs in sorted(groups.items()):
        if len(rs) >= 10:
            m = vqa_metrics([r["pred"] for r in rs], [float(r["mos"]) for r in rs])
            print(f"  {gname:24s} n={len(rs):4d} srocc={m['srocc']:.3f} plcc={m['plcc']:.3f}")
    if rows and rows[0].get("visibility", "") != "":
        v = np.array([float(r["visibility"]) for r in rows])
        pv = np.array([r["pred_vis"] for r in rows])
        print(f"visibility: srocc={spearmanr(pv, v)[0]:.4f}")
        try:
            from sklearn.metrics import roc_auc_score
            print(f"visibility noticed-vs-not AUC={roc_auc_score(v >= 0.5, pv):.4f}")
        except (ImportError, ValueError):
            pass
    spf = np.median([r["spf"] for r in rows])
    print(f"extraction speed (median): {1.0 / spf:.1f} frames/s")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
