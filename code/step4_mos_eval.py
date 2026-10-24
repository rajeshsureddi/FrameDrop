"""Step 4 true validation on public MOS data (evaluation only: never used for
training, tuning or checkpoint selection).

Clips are used as is (native resolution and frame rate, see step3_video_loader
.mos_native). Clip score = mean Stage 1 score-head output over all native
256x256 tiles of frames sampled every 0.25 s (no mask). Reported:
  SROCC, KROCC     on raw scores
  PLCC, RMSE       after the standard 4-parameter logistic mapping score -> MOS
                   (VQEG practice; fitted on the evaluated clips themselves)

    python step4_mos_eval.py --ckpt runs/s_mae/ckpt_0005000.pt --dataset konvid1k
writes <ckpt dir>/mos_<dataset>_<ckpt>.json  (+ per-clip scores .csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_video_loader as vl  # noqa: E402
import step4_model as M  # noqa: E402

BASE = "/scratch/09997/rs64952/FrameDrop-VQA"
DATASETS = {
    # labels: FAST-VQA/DOVER mirror of the official KoNViD-1k MOS (official server's TLS cert expired)
    "konvid1k": dict(videos=f"{BASE}/mos_datasets/KonViD",
                     labels=f"{BASE}/vqa_data/eval/labels/konvid1k/labels_FAST-VQA-and-FasterVQA.txt"),
}


def read_labels(name):
    d = DATASETS[name]
    items = []
    with open(d["labels"]) as f:
        for row in csv.reader(f):
            rel, mos = row[0].strip(), float(row[-1])
            items.append(dict(path=os.path.join(d["videos"], rel), mos=mos, dataset=name))
    return items


class Clips(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = dict(self.items[i])
        try:
            o = vl.mos_native(it, tiles=True, max_tiles=None, threads=1)
            return i, torch.from_numpy(o["tiles"].reshape(-1, 256, 256, 3)), ""
        except Exception as e:                       # report, never silently drop
            return i, torch.zeros(0, 256, 256, 3, dtype=torch.uint8), str(e)[-300:]


def logistic4(x, b1, b2, b3, b4):
    return (b1 - b2) / (1 + np.exp(-(x - b3) / np.abs(b4))) + b2


def metrics(score, mos):
    srocc = spearmanr(score, mos)[0]
    krocc = kendalltau(score, mos)[0]
    sgn = 1.0 if srocc >= 0 else -1.0
    x = sgn * score
    p0 = [mos.max(), mos.min(), float(np.mean(x)), float(np.std(x)) or 1.0]
    try:
        beta, _ = curve_fit(logistic4, x, mos, p0=p0, maxfev=20000)
        pred = logistic4(x, *beta)
        fit = "logistic4"
    except RuntimeError:
        a, b = np.polyfit(x, mos, 1)
        pred, beta, fit = a * x + b, [a, b], "linear (logistic fit failed)"
    return dict(SROCC=float(srocc), KROCC=float(krocc), PLCC=float(pearsonr(pred, mos)[0]),
                RMSE=float(np.sqrt(np.mean((pred - mos) ** 2))), PLCC_raw=float(pearsonr(score, mos)[0]),
                mapping=fit, beta=[float(b) for b in beta], n=int(len(mos)), mos_range=[float(mos.min()), float(mos.max())])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="s")
    ap.add_argument("--dataset", default="konvid1k", choices=list(DATASETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda", a.gpu)
    model = M.TileMAE(a.variant).to(dev).eval()
    ck = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    items = read_labels(a.dataset)
    if a.limit:
        items = items[:a.limit]
    dl = DataLoader(Clips(items), batch_size=None, num_workers=a.workers, prefetch_factor=4)
    t0 = time.time()
    scores, feats_std, errs = np.full(len(items), np.nan), [], {}
    for i, tiles, err in dl:
        if err:
            errs[items[i]["path"]] = err
            continue
        s = []
        for j in range(0, len(tiles), a.bs):
            x = tiles[j:j + a.bs].to(dev, non_blocking=True).permute(0, 3, 1, 2)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lat = model.encode(model.normalize(x))
                s.append(model.score(lat).float())
        scores[i] = float(torch.cat(s).mean())
    ok = ~np.isnan(scores)
    mos = np.array([it["mos"] for it in items])
    res = dict(ckpt=a.ckpt, step=ck.get("step"), dataset=a.dataset, clips=len(items), evaluated=int(ok.sum()),
               failed=errs, seconds=round(time.time() - t0, 1),
               protocol="as-is native res/fps; mean score-head over all native tiles every 0.25 s; no mask",
               **metrics(scores[ok], mos[ok]))
    base = os.path.join(os.path.dirname(a.ckpt), f"mos_{a.dataset}_" + os.path.basename(a.ckpt).replace(".pt", ""))
    with open(base + ".json", "w") as f:
        json.dump(res, f, indent=1)
    with open(base + ".csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["path", "mos", "score"])
        for it, sc in zip(items, scores):
            w.writerow([it["path"], it["mos"], sc])
    print(json.dumps({k: v for k, v in res.items() if k not in ("failed", "beta")}), flush=True)
    if errs:
        print(f"{len(errs)} clips failed, see json", flush=True)


if __name__ == "__main__":
    main()
