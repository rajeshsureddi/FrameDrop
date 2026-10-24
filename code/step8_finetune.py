"""Step 8: fine-tune Stage 3 (fusion + Q/V heads) on LSVQ train MOS -> model v2.

User instruction 2026-10-17: fine-tune on LSVQ separately (v1 is frozen and backed
up in vqa_data/frozen_v1_20261017), then evaluate on LSVQ test_1080p, KoNViD-1k and
the study. Stage 1/2 encoders stay frozen (tokens are precomputed by step7_eval.py).

Init: v1 fusion (step6/runs/fusion/best.pt).
Batches alternate between two sources (one dataset per batch, METHOD_PLAN L_mos):
  MOS   16 LSVQ train clips: L_mos = MSE(Q, z) + (1 - PLCC(Q, z)) in the batch,
        z = LSVQ-train z-scored MOS (plan Step 6 L_mos)
  pairs 16 synthetic pairs from step6/train: L_rank + L_drop + 0.1 L_inv (as in v1),
        so V and the drop / rate / severity orderings are kept
        (--pair-every 0 disables them: MOS-only ablation)
All 28,056 LSVQ train clips are trained on (user, 2026-10-17), so there is no MOS
holdout: the model is the final step of a fixed schedule (final.pt). Validation
during training logs only the synthetic valid pairs (step6/valid) and the running
batch PLCC. LSVQ test / test_1080p, KoNViD and the study are never used here
(--holdout N restores an N-clip LSVQ-train holdout with best.pt selection on it).

    python step8_finetune.py --name v2_mos_pairs --gpu 0
    python step8_finetune.py --name v2_mos_only --pair-every 0 --gpu 1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as S3  # noqa: E402
import step6_train as T6  # noqa: E402
from step4_train import lr_at, param_groups  # noqa: E402
import step7_eval as E7  # noqa: E402
from step7_eval import DATASETS, clip_name, token_dir  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
OUT = f"{ROOT}/step8_lsvq"
V1 = f"{ROOT}/frozen_v1_20261017/stage3_fusion_best.pt"


def lsvq_train_split(n_hold=0, seed=0):
    """(train, holdout) lists of (token path, mos) for LSVQ train clips with tokens."""
    with open(DATASETS["lsvq_train"]["labels"]) as f:
        lab = [(clip_name("lsvq_train", r[0]), float(r[-1])) for r in csv.reader(f)]
    td = token_dir("lsvq_train")
    lab = [(f"{td}/{n}.npz", m) for n, m in lab]
    rng = random.Random(seed)
    rng.shuffle(lab)
    hold = lab[:n_hold]                                   # fixed by seed, independent of which tokens exist
    have = lambda L: [x for x in L if os.path.exists(x[0])]  # noqa: E731
    return have(lab[n_hold:]), have(hold)


def load_clip(path):
    z = np.load(path)
    K, steps = z["spatial"].shape[0], z["motion"].shape[0]
    K = min(K, -(-steps * 2 // 15))
    return (torch.from_numpy(z["spatial"][:K]), torch.from_numpy(z["cells"][:K].astype(np.int64)),
            torch.from_numpy(z["motion"]))


class ClipSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, m = self.items[i]
        return load_clip(p), m


def loop(dl):
    while True:
        yield from dl


def clip_q(model, clip, dev):
    sp, ce, mo = (t[None].to(dev, non_blocking=True) for t in clip)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Q, V, _, _ = model(sp.float(), ce, mo.float())
    return Q[0].float(), V[0].float()


def l_mos(q, z):
    qc, zc = q - q.mean(), z - z.mean()
    plcc = (qc * zc).sum() / (qc.norm() * zc.norm() + 1e-8)
    mse = ((q - z) ** 2).mean()
    return mse + (1 - plcc), dict(l_mse=float(mse), plcc_batch=float(plcc))


@torch.no_grad()
def eval_mos(model, hold_dl, dev):
    model.eval()
    qs, ms = [], []
    for clips, mos in hold_dl:
        for c, m in zip(clips, mos):
            qs.append(float(clip_q(model, c, dev)[0]))
            ms.append(float(m))
    model.train()
    return dict(hold_SROCC=float(spearmanr(qs, ms)[0]), hold_PLCC_raw=float(pearsonr(qs, ms)[0]), hold_n=len(qs))


def train(a):
    dev = torch.device("cuda", a.gpu)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if a.tok_root:
        E7.TOK_ROOT = a.tok_root                            # token_dir() of other encoders
    run_dir = f"{a.out}/runs/{a.name}"
    os.makedirs(run_dir, exist_ok=True)
    tr, hold = lsvq_train_split(a.holdout)
    mos_all = np.array([m for _, m in tr])
    mu, sd = float(mos_all.mean()), float(mos_all.std())
    print(f"LSVQ train {len(tr)} clips, holdout {len(hold)}; MOS mean {mu:.2f} sd {sd:.2f}", flush=True)
    coll = lambda b: ([x[0] for x in b], [x[1] for x in b])  # noqa: E731
    tr_dl = loop(DataLoader(ClipSet(tr), batch_size=a.clips, shuffle=True, drop_last=True, num_workers=a.workers,
                            collate_fn=coll, persistent_workers=True, prefetch_factor=4))
    hold_dl = DataLoader(ClipSet(hold), batch_size=32, num_workers=a.workers, collate_fn=coll) if hold else None
    pool = val_pairs = None
    if a.pair_every:
        files = sorted(f for f in glob.glob(f"{a.pairs_root}/train/pairs_*.npz") if ".tmp" not in f)
        pool = T6.ShardPool(files, a.pool, a.seed)
        val_pairs = [T6.load_shard(p) for p in sorted(glob.glob(f"{a.pairs_root}/valid/pairs_0*.npz")) if ".tmp" not in p]

    model = S3.Fusion(drop_path=a.drop_path, n_q=a.n_q).to(dev)
    sd0 = torch.load(a.init, map_location="cpu")["model"]
    if a.n_q > 1 and S3.n_q_of(sd0) == 1:                   # Step 20: start as the same function as the init
        sd0 = S3.expand_queries(sd0, a.n_q, seed=a.seed)
    model.load_state_dict(sd0)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.999))
    step0, best = 0, -1.0
    if os.path.exists(f"{run_dir}/last.pt"):                  # resume
        ck = torch.load(f"{run_dir}/last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0, best = ck["step"], ck["best"]
        print(f"resumed at step {step0}, best {best:.4f}", flush=True)
    warm = int(0.05 * a.steps)
    log = open(f"{run_dir}/train_log.jsonl", "a")
    json.dump(dict(args=vars(a), lsvq_train=len(tr), holdout=len(hold), mos_mean=mu, mos_sd=sd), open(f"{run_dir}/config.json", "w"), indent=1)
    if step0 == 0:
        r = eval_mos(model, hold_dl, dev) if hold_dl else {}
        if val_pairs:
            r.update(T6.evaluate(model, val_pairs, dev))
        r.update(step=0, kind="val")
        log.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
    acc, n, t0 = defaultdict(float), defaultdict(int), time.time()
    for step in range(step0 + 1, a.steps + 1):
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, a.steps, a.lr, warm)
        if a.pair_every and step % a.pair_every == 0:
            b = pool.batch(a.pairs)
            Q, V = T6.forward_pairs(model, b, dev)
            loss, parts = T6.losses(Q, V, b["ptype"], dev)
            kind = "pairs"
            if (step // a.pair_every) % a.rotate == 0:
                pool.rotate()
        else:
            clips, mos = next(tr_dl)
            q = torch.stack([clip_q(model, c, dev)[0] for c in clips])
            z = (torch.tensor(mos, device=dev, dtype=torch.float32) - mu) / sd
            loss, parts = l_mos(q, z)
            kind = "mos"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in parts.items():
            acc[k] += v
            n[k] += 1
        acc[f"loss_{kind}"] += float(loss)
        n[f"loss_{kind}"] += 1
        if step % a.log_every == 0:
            rec = dict(step=step, lr=opt.param_groups[0]["lr"], **{k: v / n[k] for k, v in acc.items()},
                       a=float(torch.sigmoid(model.a_logit)), steps_per_s=a.log_every / (time.time() - t0),
                       time=time.strftime("%m-%d %H:%M:%S"))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            acc, n, t0 = defaultdict(float), defaultdict(int), time.time()
        if step % a.val_every == 0 or step == a.steps:
            r = eval_mos(model, hold_dl, dev) if hold_dl else {}
            if val_pairs:
                r.update(T6.evaluate(model, val_pairs, dev))
            r.update(step=step, kind="val")
            sel = (r["hold_SROCC"] + r["hold_PLCC_raw"]) / 2 if hold_dl else -1.0
            r["select"] = sel
            if hold_dl and sel > best:
                best = sel
                torch.save(dict(model=model.state_dict(), step=step, val=r, mos_mean=mu, mos_sd=sd), f"{run_dir}/best.pt")
                r["best"] = True
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best), f"{run_dir}/last.pt")
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(json.dumps(r), flush=True)
    torch.save(dict(model=model.state_dict(), step=a.steps, mos_mean=mu, mos_sd=sd), f"{run_dir}/final.pt")
    print("FINISHED best", best, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v2_mos_pairs")
    ap.add_argument("--init", default=V1)
    ap.add_argument("--out", default=OUT, help="runs/ are written under it")
    ap.add_argument("--pairs-root", default=f"{ROOT}/step6", help="synthetic pair tokens (train/ valid/)")
    ap.add_argument("--tok-root", default=None, help="LSVQ token root of other encoders (step7_eval --tok-root)")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--holdout", type=int, default=0, help="LSVQ-train clips held out for selection (0 = train on all)")
    ap.add_argument("--clips", type=int, default=16, help="LSVQ clips per MOS batch")
    ap.add_argument("--pairs", type=int, default=16, help="synthetic pairs per pair batch")
    ap.add_argument("--pair-every", type=int, default=2, help="every k-th step is a pair batch; 0 = MOS only")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--rotate", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-q", type=int, default=1, help="Step 20: pooling queries per head (1 = original fusion)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
