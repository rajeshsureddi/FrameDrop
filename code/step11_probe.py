"""Backbone-size probe (diagnostic, not a Step 10 candidate; no study data): does a larger spatial ViT help?

Same tiles for every backbone: step7_eval.ClipDS views (fixed CELL_SEED; native-resolution 256x256 tiles,
<= 32 per tick, a tick every 15 native frames). Per tile features from
  ours_v1   Tile-MAE ViT-S, frozen_v1 Stage 1        (e1.features, 384-d)
  ours_v3   Tile-MAE ViT-S, frozen_v3 Stage 1
  dino_s / dino_b / dino_l   DINOv2 ViT-S/14, B/14, L/14 (facebook/dinov2-*), tile resized 256 -> 224,
            ImageNet normalisation, [CLS, mean patch token]
Clip feature = mean and std over all tiles of the clip. Ridge (StandardScaler + RidgeCV) fitted on a fixed
random 6,000-clip subset of LSVQ train90 (seed 0), tested on LSVQ val10 (2,806) and KoNViD-1k (cross-dataset).

Decision rule (fixed before any number): B or L is worth a Step 11 run only if its val10 SROCC beats dino_s by
>= 0.02 with the paired bootstrap 95% CI (1,000 resamples over clips) excluding 0, and its KoNViD SROCC is not
lower than dino_s's. Spatial only: this probe says nothing about frame drops.

    python step11_probe.py extract --part 0 --parts 3 --gpu 0
    python step11_probe.py fit
    python step11_probe.py speed --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_features as F6  # noqa: E402
from step7_eval import DATASETS, ClipDS, clip_name  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
OUT = f"{ROOT}/step11_probe"
TRAIN90 = f"{ROOT}/eval/labels/lsvq/train90_seed0.txt"
N_TRAIN = 6000
HF = "/scratch/09997/rs64952/.cache/huggingface"
DINO = dict(dino_s="facebook/dinov2-small", dino_b="facebook/dinov2-base", dino_l="facebook/dinov2-large")
OURS = dict(ours_v1=(f"{ROOT}/frozen_v1_20261017/stage1_s_mae_final_avg4.pt", f"{ROOT}/frozen_v1_20261017/stage2_v_mae_final_avg4.pt"),
            ours_v3=(f"{ROOT}/frozen_v3_20261018/s1.pt", f"{ROOT}/frozen_v3_20261018/s2.pt"))
MODELS = tuple(OURS) + tuple(DINO)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def items():
    """(set, dataset, path, key, mos); keys and MOS come from the label files, matched by key."""
    out = []
    with open(TRAIN90) as f:
        rows = list(csv.reader(f))
    pick = np.random.default_rng(0).choice(len(rows), N_TRAIN, replace=False)
    for i in sorted(pick):
        out.append(("train", "lsvq_val10", rows[i][0].strip(), float(rows[i][-1])))
    for s, ds in (("val10", "lsvq_val10"), ("konvid", "konvid1k")):
        with open(DATASETS[ds]["labels"]) as f:
            out += [(s, ds, r[0].strip(), float(r[-1])) for r in csv.reader(f)]
    res = []
    for s, ds, rel, mos in out:
        key = clip_name(ds, rel)
        res.append(dict(set=s, ds=ds, path=os.path.join(DATASETS[ds]["videos"], rel), key=f"{s}__{key}", mos=mos))
    return res


def load_models(dev):
    os.environ.setdefault("HF_HOME", HF)
    from transformers import AutoModel
    m = {}
    for k, (s1, s2) in OURS.items():
        m[k] = F6.load_encoders(s1, s2, dev)[0].eval()
    for k, rep in DINO.items():
        m[k] = AutoModel.from_pretrained(rep).to(dev).eval()
    return m


@torch.no_grad()
def tile_feats(models, tiles_u8, dev, bs=256):
    """(N, 256, 256, 3) uint8 -> {model: (N, d) float32}"""
    t = torch.as_tensor(tiles_u8).to(dev).permute(0, 3, 1, 2)
    out = {k: [] for k in models}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(t), bs):
            x = t[i:i + bs]
            for k, m in models.items():
                if k in OURS:
                    out[k].append(m.features(x).float())
                else:
                    xi = F.interpolate(x.float() / 255, size=224, mode="bilinear", antialias=True, align_corners=False)
                    h = m(pixel_values=(xi - MEAN.to(dev)) / STD.to(dev)).last_hidden_state
                    out[k].append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float())
    return {k: torch.cat(v) for k, v in out.items()}


def extract(a):
    dev = torch.device("cuda", a.gpu)
    od = f"{OUT}/feats"
    os.makedirs(od, exist_ok=True)
    its = [it for it in items()[a.part::a.parts] if not os.path.exists(f"{od}/{it['key']}.npz")]
    print(f"part {a.part}/{a.parts}: {len(its)} clips to extract", flush=True)
    models = load_models(dev)
    ds = ClipDS([it["path"] for it in its], "lsvq_val10")      # dataset name only selects the 4K branch
    dl = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=2)
    t0 = time.time()
    for k, (i, s, err) in enumerate(dl):
        it = its[i]
        if s is None:
            open(f"{od}/{it['key']}.err", "w").write(err)
            continue
        tiles = np.asarray(s["tiles"]).reshape(-1, 256, 256, 3)
        f = tile_feats(models, tiles, dev)
        vec = {m: torch.cat([v.mean(0), v.std(0)]).cpu().numpy() for m, v in f.items()}
        np.savez(f"{od}/.{it['key']}.tmp.npz", n_tiles=len(tiles), **vec)
        os.replace(f"{od}/.{it['key']}.tmp.npz", f"{od}/{it['key']}.npz")
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(its)} {(k + 1) / (time.time() - t0):.2f} clips/s", flush=True)
    print("FINISHED extract", flush=True)


def fit(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    its = [it for it in items() if os.path.exists(f"{OUT}/feats/{it['key']}.npz")]
    n_all = len(items())
    X = {m: {} for m in MODELS}
    for it in its:
        z = np.load(f"{OUT}/feats/{it['key']}.npz")
        for m in MODELS:
            X[m][it["key"]] = z[m]
    by = {s: [it for it in its if it["set"] == s] for s in ("train", "val10", "konvid")}
    res = dict(clips_with_features=len(its), clips_total=n_all, per_set={s: len(v) for s, v in by.items()})
    preds = {}
    for m in MODELS:
        Xtr = np.stack([X[m][it["key"]] for it in by["train"]])
        ytr = np.array([it["mos"] for it in by["train"]])
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))).fit(Xtr, ytr)
        r = dict(dim=int(Xtr.shape[1]), alpha=float(mdl[-1].alpha_))
        for s in ("val10", "konvid"):
            p = mdl.predict(np.stack([X[m][it["key"]] for it in by[s]]))
            y = np.array([it["mos"] for it in by[s]])
            preds[(m, s)] = (p, y)
            r[s] = dict(SROCC=float(spearmanr(p, y)[0]), PLCC=float(pearsonr(p, y)[0]), n=len(y))
        res[m] = r
        print(m, json.dumps(r), flush=True)
    rng = np.random.default_rng(0)
    p0, y = preds[("dino_s", "val10")]
    idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    for m in ("dino_b", "dino_l", "ours_v1", "ours_v3"):
        p1, _ = preds[(m, "val10")]
        d = [spearmanr(p1[i], y[i])[0] - spearmanr(p0[i], y[i])[0] for i in idx]
        lo, hi = np.percentile(d, [2.5, 97.5])
        gap = res[m]["val10"]["SROCC"] - res["dino_s"]["val10"]["SROCC"]
        res[f"{m}_vs_dino_s"] = dict(val10_SROCC_diff=float(gap), ci95=[float(lo), float(hi)],
                                     konvid_SROCC_diff=res[m]["konvid"]["SROCC"] - res["dino_s"]["konvid"]["SROCC"])
        if m in ("dino_b", "dino_l"):
            res[f"{m}_vs_dino_s"]["passes_rule"] = bool(gap >= 0.02 and lo > 0 and res[f"{m}_vs_dino_s"]["konvid_SROCC_diff"] >= 0)
    json.dump(res, open(f"{OUT}/probe_results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.endswith("_vs_dino_s")}, indent=1))


@torch.no_grad()
def speed(a):
    """tiles/s of each backbone on the 1,280 tiles of a 10 s 4K60 clip (bf16), as in step7 'model' timing."""
    dev = torch.device("cuda", a.gpu)
    models = load_models(dev)
    tiles = np.random.default_rng(0).integers(0, 256, (1280, 256, 256, 3), dtype=np.uint8)
    res = {}
    for k in MODELS:
        one = {k: models[k]}
        tile_feats(one, tiles[:256], dev)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(3):
            t0 = time.time()
            tile_feats(one, tiles, dev)
            torch.cuda.synchronize(dev)
            ts.append(time.time() - t0)
        t = float(np.median(ts))
        res[k] = dict(params_M=sum(p.numel() for p in models[k].parameters()) / 1e6, tiles_per_s=1280 / t,
                      clip_seconds_10s_4k60=t)
        print(k, json.dumps(res[k]), flush=True)
    res["gpu"] = torch.cuda.get_device_name(dev)
    json.dump(res, open(f"{OUT}/speed.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit", "speed"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(extract=extract, fit=fit, speed=speed)[a.cmd](a)


if __name__ == "__main__":
    main()
