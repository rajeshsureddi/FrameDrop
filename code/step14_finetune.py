"""Step 14 (10-21): per-dataset fine-tuning, HFVQA protocol (arXiv 2609.16946, Table 1 right half).

Start from an LSVQ-trained end-to-end model (v3 by default) and fine-tune it on one public
MOS dataset. Each dataset gets 10 random train/val/test splits (70:10:20, seeds 0-9), fixed and
saved before any training. Median test PLCC/SROCC over the splits is reported.

Study (user 10-21, "fine-tune on the study splits too, to compare against GAMIVAL"): the baselines' 20
GroupShuffleSplit(test_size=.2, random_state=0) splits (step10_protA splits_regenerated.csv). Each split's
val = ~10% of its *train* base clips (whole contents, seed = split), never its test clips. Trained-protocol
result only; zero-shot models never see the study.

Schedule (as HFVQA): encoders frozen for the first 4 epochs (fusion only), then everything
for 8 more (12 epochs). Loss = L_mos of v2/v3 (MSE on train-split z-scored MOS + 1 - PLCC).
The checkpoint with the best val SROCC (evaluated after every epoch) is scored on test once.
Clips are scored whole and deterministically (step7_eval.ClipDS: fixed tile cells, all frames).

    python step14_finetune.py splits --dataset konvid1k
    python step14_finetune.py train --dataset konvid1k --split 0 --init v3 --gpu 0
    python step14_finetune.py summary --dataset konvid1k --init v3
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step9_e2e as E9  # noqa: E402
import step7_eval as S7  # noqa: E402
from step4_train import lr_at  # noqa: E402

ROOT = E9.ROOT
OUT = f"{ROOT}/step14_finetune"
INITS = dict(
    v3=f"{ROOT}/frozen_v3_20261018",                 # v3 e2e, trained on all of LSVQ train
    d1=f"{ROOT}/step10_cand/runs/d1",                # d1 (DINOv2-S chain), trained on LSVQ train90
)
N_SPLITS = dict(konvid1k=10, livevqc=10, ytugc=10, study=20)
PROTA_SPLITS = f"{ROOT}/step10_protA/splits_regenerated.csv"
EPOCHS, FROZEN_EPOCHS = 12, 4


def items(dataset):
    if dataset == "study":
        with open(S7.DATASETS["study"]["labels"]) as f:
            return [(os.path.join(S7.DATASETS["study"]["videos"], r["video_filename"]), float(r["mos"]))
                    for r in csv.DictReader(f)]
    with open(S7.DATASETS[dataset]["labels"]) as f:
        return [(os.path.join(S7.DATASETS[dataset]["videos"], r[0].strip()), float(r[-1])) for r in csv.reader(f)]


def split_file(dataset, s):
    return f"{OUT}/splits/{dataset}_split{s}.json"


def make_study_splits():
    rows = list(csv.DictReader(open(PROTA_SPLITS)))
    vdir = S7.DATASETS["study"]["videos"]
    for s in range(N_SPLITS["study"]):
        if os.path.exists(split_file("study", s)):
            continue
        col = f"split_{s + 1:02d}"
        tr = [r for r in rows if r[col] == "train"]
        bases = sorted({r["base_clip"] for r in tr})
        vb = set(np.random.default_rng(s).choice(bases, max(1, round(0.1 * len(bases))), replace=False).tolist())
        sp = dict(train=[os.path.join(vdir, r["video_filename"]) for r in tr if r["base_clip"] not in vb],
                  val=[os.path.join(vdir, r["video_filename"]) for r in tr if r["base_clip"] in vb],
                  test=[os.path.join(vdir, r["video_filename"]) for r in rows if r[col] == "test"],
                  seed=s, protA_split=col, val_bases=sorted(vb))
        json.dump(sp, open(split_file("study", s), "w"), indent=0)
    print(f"study: {N_SPLITS['study']} splits (protA split_01..20) in {OUT}/splits", flush=True)


def make_splits(a):
    os.makedirs(f"{OUT}/splits", exist_ok=True)
    if a.dataset == "study":
        return make_study_splits()
    paths = sorted(p for p, _ in items(a.dataset))
    n = len(paths)
    for s in range(N_SPLITS[a.dataset]):
        if os.path.exists(split_file(a.dataset, s)):
            continue                                  # fixed once written
        perm = np.random.default_rng(s).permutation(n)
        n_tr, n_va = int(round(0.7 * n)), int(round(0.1 * n))
        sp = dict(train=[paths[i] for i in perm[:n_tr]], val=[paths[i] for i in perm[n_tr:n_tr + n_va]],
                  test=[paths[i] for i in perm[n_tr + n_va:]], seed=s)
        json.dump(sp, open(split_file(a.dataset, s), "w"), indent=0)
    print(f"{a.dataset}: {N_SPLITS[a.dataset]} splits of {n} clips in {OUT}/splits", flush=True)


def stage(paths, dataset, root):
    """copy videos to node-local disk (Lustre stalls the loaders) -> {original path: local path}"""
    import fcntl
    vdir = S7.DATASETS[dataset]["videos"]
    m = {}
    os.makedirs(root, exist_ok=True)
    lock = open(f"{root}/.stage_{dataset}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)                   # one copier per node; the other workers wait, then find the files
    for p in paths:
        q = os.path.join(root, dataset, os.path.relpath(p, vdir))
        if not os.path.exists(q) or os.path.getsize(q) != os.path.getsize(p):
            os.makedirs(os.path.dirname(q), exist_ok=True)
            shutil.copyfile(p, f"{q}.part{os.getpid()}")      # workers on one node may stage the same file
            os.replace(f"{q}.part{os.getpid()}", q)
        m[p] = q
    fcntl.flock(lock, fcntl.LOCK_UN)
    return m


def correlations(pred, mos, mu=None, sd=None):
    """raw predictions (no logistic mapping, as for trained methods); RMSE of mu + sd * Q on the MOS scale"""
    pred, mos = np.asarray(pred, float), np.asarray(mos, float)
    r = dict(SROCC=float(stats.spearmanr(pred, mos)[0]), PLCC=float(stats.pearsonr(pred, mos)[0]),
             KROCC=float(stats.kendalltau(pred, mos)[0]), n=int(len(mos)))
    if mu is not None:
        r["RMSE"] = float(np.sqrt(np.mean((mu + sd * pred - mos) ** 2)))
    return r


@torch.no_grad()
def predict(model, paths, dataset, dev, workers):
    """whole-clip Q for each path (deterministic tiles), in the order of paths"""
    model.eval()
    out = np.full(len(paths), np.nan)
    dl = DataLoader(S7.ClipDS(paths, dataset), batch_size=None, num_workers=workers, prefetch_factor=2)
    for i, s, err in dl:
        if s is None:
            print("ERR", paths[i], err[-200:], flush=True)
            continue
        low = np.asarray(s["low"])
        low = low[:(low.shape[0] // 2) * 2]
        Q, _ = model.clips([E9.to_dev(low, dev)], [E9.to_dev(s["tiles"], dev)],
                           [E9.to_dev(np.asarray(s["cells"]).astype(np.int64), dev)])
        out[i] = float(Q[0])
    model.train()
    return out


def set_encoders_trainable(model, on):
    for e in (model.e1, model.e2):
        for n_, p in e.named_parameters():
            p.requires_grad_(on and n_.split(".")[0] in ("patch_embed", "cls_token", "pos_embed", "blocks", "norm"))


def train(a):
    dev = torch.device("cuda", a.gpu)
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed)
    sp = json.load(open(split_file(a.dataset, a.split)))
    mos_of = dict(items(a.dataset))
    if a.stage:                                        # same clips, read from node-local copies
        loc = stage(sp["train"] + sp["val"] + sp["test"], a.dataset, a.stage)
        sp = {k: [loc[p] for p in v] if k in ("train", "val", "test") else v for k, v in sp.items()}
        mos_of = {loc[p]: m for p, m in mos_of.items() if p in loc}
    run = f"{OUT}/runs/{a.init}/{a.dataset}/split{a.split}"
    if os.path.exists(f"{run}/result.json"):
        print("already done:", run, flush=True)
        return
    os.makedirs(run, exist_ok=True)
    tr = [(p, mos_of[p]) for p in sp["train"]]
    mu, sd = float(np.mean([m for _, m in tr])), float(np.std([m for _, m in tr]))
    E9.MOS_MEAN, E9.MOS_SD = mu, sd                    # L_mos z-scores with the train split's statistics
    d = INITS[a.init]
    model = E9.E2E(f"{d}/s1.pt", f"{d}/s2.pt", f"{d}/fusion.pt").to(dev)
    opt = torch.optim.AdamW(E9.param_groups(model, a.lr_enc, a.lr_fusion, a.llrd, a.wd), lr=a.lr_fusion)
    spe = -(-len(tr) // a.clips)                       # steps per epoch
    total, warm = EPOCHS * spe, spe
    json.dump(dict(args=vars(a), n_train=len(tr), n_val=len(sp["val"]), n_test=len(sp["test"]), mos_mean=mu,
                   mos_sd=sd, steps_per_epoch=spe, total_steps=total), open(f"{run}/config.json", "w"), indent=1)
    dl = iter(DataLoader(E9.LSVQWindows(tr, a.tiles, a.seed + 1000 * a.split), batch_size=None,
                         num_workers=a.workers, prefetch_factor=3))
    log = open(f"{run}/train_log.jsonl", "a")
    best, step, t0 = -2.0, 0, time.time()
    for ep in range(1, EPOCHS + 1):
        set_encoders_trainable(model, ep > FROZEN_EPOCHS)
        params = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
        acc = []
        for _ in range(spe):
            step += 1
            f = lr_at(step, total, 1.0, warm)
            for g in opt.param_groups:
                g["lr"] = g["lr_base"] * f
            loss, parts = E9.mos_step(model, [next(dl) for _ in range(a.clips)], dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            acc.append(float(loss))
            if step % 20 == 0:
                print(f"step {step}/{total} loss {np.mean(acc[-20:]):.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        pv = predict(model, sp["val"], a.dataset, dev, a.pred_workers or a.workers)
        ok = ~np.isnan(pv)
        r = dict(epoch=ep, step=step, loss=float(np.mean(acc)), frozen=ep <= FROZEN_EPOCHS,
                 val=correlations(pv[ok], [mos_of[p] for p, k in zip(sp["val"], ok) if k]),
                 min=round((time.time() - t0) / 60, 1))
        log.write(json.dumps(r) + "\n")
        log.flush()
        print(json.dumps(r), flush=True)
        if r["val"]["SROCC"] > best:
            best = r["val"]["SROCC"]
            torch.save(dict(model=model.state_dict(), epoch=ep), f"{run}/best.pt")
    model.load_state_dict(torch.load(f"{run}/best.pt", map_location="cpu")["model"])
    pt = predict(model, sp["test"], a.dataset, dev, a.pred_workers or a.workers)
    ok = ~np.isnan(pt)
    with open(f"{run}/test_scores.csv", "w") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "Q", "mos"])
        w.writerows([os.path.basename(p), q, mos_of[p]] for p, q in zip(sp["test"], pt))
    res = dict(dataset=a.dataset, split=a.split, init=a.init, best_epoch=int(torch.load(f"{run}/best.pt",
               map_location="cpu")["epoch"]), best_val_SROCC=best, n_test_missing=int((~ok).sum()),
               test=correlations(pt[ok], [mos_of[p] for p, k in zip(sp["test"], ok) if k], mu, sd))
    json.dump(res, open(f"{run}/result.json", "w"), indent=1)
    os.remove(f"{run}/best.pt")                        # 10 splits x datasets: keep only scores and results
    print("RESULT", json.dumps(res), flush=True)
    os._exit(0)


def summary(a):
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/{a.init}/{a.dataset}/split*/result.json"))]
    if not rs:
        print("no results", flush=True)
        return
    med = {k: float(np.median([r["test"][k] for r in rs])) for k in ("SROCC", "PLCC", "KROCC", "RMSE")
           if all(k in r["test"] for r in rs)}
    out = dict(dataset=a.dataset, init=a.init, splits_done=len(rs), n_splits=N_SPLITS[a.dataset], median=med,
               per_split=[dict(split=r["split"], **{k: r["test"][k] for k in ("SROCC", "PLCC")},
                               best_epoch=r["best_epoch"]) for r in rs])
    json.dump(out, open(f"{OUT}/runs/{a.init}/{a.dataset}/summary.json", "w"), indent=1)
    print(f"{a.init} {a.dataset}: {len(rs)}/{N_SPLITS[a.dataset]} splits, median SROCC {med['SROCC']:.4f} "
          f"PLCC {med['PLCC']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["splits", "train", "summary"])
    ap.add_argument("--dataset", required=True, choices=list(N_SPLITS))
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--init", default="v3", choices=list(INITS))
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-fusion", type=float, default=3e-5)
    ap.add_argument("--llrd", type=float, default=0.75)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pred-workers", type=int, default=0, help="val/test loader workers (0 = --workers); runs next to the live train loader")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="/tmp/rs64952_step14", help="node-local copy of the videos ('' = read Lustre)")
    a = ap.parse_args()
    dict(splits=make_splits, train=train, summary=summary)[a.cmd](a)


if __name__ == "__main__":
    main()
