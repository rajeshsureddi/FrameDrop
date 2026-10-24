"""Step 4 checks on held-out (valid split) tiles, for a trained Tile-MAE:

  1. reconstruction error rises with severity     (val_clean: stored versions,
     no on-the-fly ops; severity 0 = source/clean ladder, 0.5 / 1.0 = capture)
  2. pair-ranking accuracy > 90%, per pair type, per op family, per codec
     (val_pairs; inference mode = no mask)
  3. no feature collapse: per-dim std of pooled tile features > 0.1

    python step4_eval.py --ckpt vqa_data/step4/runs/s_mae/final_avg4.pt
writes <ckpt dir>/eval_<ckpt name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_common as c3  # noqa: E402
import step4_model as M  # noqa: E402
import step4_train as T  # noqa: E402


def severity_of(encode_id):
    m = re.search(r"_sev(\d{3})_", encode_id)
    return int(m.group(1)) / 100 if m else 0.0


def codec_of(enc_pair):
    a, b = enc_pair
    return a if a == b else f"{a or 'orig'}|{b or 'orig'}"


def table(keys, ok, viol=None, min_n=20):
    out = {}
    for k in sorted(set(keys)):
        m = np.array([x == k for x in keys])
        if m.sum() >= min_n:
            out[k] = dict(n=int(m.sum()), acc=float(ok[m].mean()))
            if viol is not None:
                out[k]["viol"] = float(viol[m].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="s")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda", a.gpu)
    model = M.TileMAE(a.variant).to(dev)
    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    model.load_state_dict(sd)
    plans = c3.load_json(f"{T.ROOT}/step3/capture_plans.json")
    res = dict(ckpt=a.ckpt)

    # 1. rec error vs severity (each tile of each stored-version pair)
    x, meta = T.load_val("clean")
    r, s, rec, f = T.evaluate(model, x, meta, dev)
    sev = np.array([[severity_of(v) for v in p] for p in meta["versions"]]).ravel()
    kind = np.array([["capture" if "_cap_" in v else ("orig" if v == "orig" or "orig" in v else "clean")
                      for v in p] for p in meta["versions"]]).ravel()
    recf = rec.numpy().ravel()
    by = {}
    for k in ("orig", "clean"):
        if (kind == k).any():
            by[k] = dict(n=int((kind == k).sum()), rec=float(recf[kind == k].mean()))
    for sv in (0.5, 1.0):
        m = (kind == "capture") & (sev == sv)
        if m.any():
            by[f"capture_sev{sv}"] = dict(n=int(m.sum()), rec=float(recf[m].mean()))
    # paired: same capture plan & rung, 0.5 vs 1.0 inside one val pair (severity pairs)
    pr = [(i, severity_of(p[0]), severity_of(p[1])) for i, p in enumerate(meta["versions"])
          if "_cap_" in p[0] and "_cap_" in p[1] and severity_of(p[0]) != severity_of(p[1])]
    if pr:
        rr = rec.numpy()
        hi_gt = [rr[i, 0 if s0 > s1 else 1] > rr[i, 1 if s0 > s1 else 0] for i, s0, s1 in pr]
        by["paired_sev1.0_gt_sev0.5_frac"] = dict(n=len(pr), frac=float(np.mean(hi_gt)))
    res["rec_vs_severity"] = by

    # 2. ranking accuracy (val_pairs)
    x, meta = T.load_val("pairs")
    r, s, rec, f = T.evaluate(model, x, meta, dev)
    res["val_pairs_summary"] = r
    order = np.array(meta["order"])
    o = order == 1
    d = (s[:, 0] - s[:, 1]).numpy()
    ok, viol = (d > 0)[o], (d < T.MARGIN)[o]
    ptype = [p for p, k in zip(meta["ptype"], o) if k]
    fam = []
    for p, pid, ops in zip(meta["ptype"], meta["plan_id"], meta["ops"]):
        if p in ("capture_rate", "severity") and pid in plans:
            fam.append("+".join(sorted(op[0] for op in plans[pid]["ops"])))
        elif p == "otf_severity":
            fam.append("otf:" + "+".join(sorted(ops.split("+"))))
        else:
            fam.append(p)
    fam = [x_ for x_, k in zip(fam, o) if k]
    # single-op families (each op counted in every pair that contains it)
    op_keys, op_ok = [], []
    for fm, g in zip(fam, ok):
        for op in fm.replace("otf:", "").split("+"):
            if op and op not in ("rate", "none"):
                op_keys.append(op)
                op_ok.append(g)
    codec = [codec_of(e) for e, k in zip(meta["encoders"], o) if k]
    res["rank_acc_overall"] = float(ok.mean())
    res["rank_by_ptype"] = table(ptype, ok, viol)
    res["rank_by_op"] = table(op_keys, np.array(op_ok))
    res["rank_by_codec"] = table(codec, ok, viol)

    # 3. feature collapse (pooled, no mask, val_pairs tiles)
    sd_ = f.std(0).numpy()
    res["feat_std"] = dict(min=float(sd_.min()), median=float(np.median(sd_)),
                           frac_dims_above_0_1=float((sd_ > 0.1).mean()), dims=int(len(sd_)))
    res["checks"] = dict(
        rec_rises_with_severity=bool(by.get("capture_sev1.0", {}).get("rec", 0) > by.get("capture_sev0.5", {}).get("rec", 1)
                                     > min(by.get("clean", {}).get("rec", 9), by.get("orig", {}).get("rec", 9))),
        rank_acc_above_0_9=bool(res["rank_acc_overall"] > 0.9),
        feat_std_all_dims_above_0_1=bool(sd_.min() > 0.1))
    dst = os.path.join(os.path.dirname(a.ckpt), "eval_" + os.path.basename(a.ckpt).replace(".pt", ".json"))
    with open(dst, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(dict(checks=res["checks"], rank=res["rank_acc_overall"], rec=by, feat=res["feat_std"]), indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
