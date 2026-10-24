"""Step 15 (10-22): v4 = v3 + extra MOS data (user: "yes, start the v4 training with the extra MOS data").

Starts from frozen v3 (s1/s2/fusion) and continues end-to-end training with the v3 recipe, but the MOS
batches come from three datasets instead of LSVQ alone:
  lsvq     LSVQ train, all 28,056 clips (as v3)
  ytugc    YouTube-UGC, train part of step14 split 0 (684 clips). Its val (98) is monitored, its test (195)
           is never seen. Consequence: v4 cannot be run through Step 14 on ytugc (v3 keeps that result).
  gaming   LIVE-YT-Gaming split=train (430 clips, 39 games). split=val (109, 15 whole games) is monitored;
           the 61 clips of the 5 study titles (split=excluded_study_title) are never read.
Never trained on: LSVQ test / 1080p, KoNViD-1k, LIVE-VQC (Protocol 1 cross-dataset sets), the study.

Each MOS batch is drawn from ONE dataset (chosen with --weights by an rng seeded with the step, identical on
all ranks) and z-scored with that dataset's train mean/sd, so MSE + (1 - PLCC) never mixes MOS scales.
Every --pair-every-th step is a synthetic drop/bitrate pair batch, exactly as v3. Model = final step of the
fixed schedule; the monitored val numbers are logged, never used to pick a checkpoint.

  python step15_v4.py train --name v4 --world 2 --rank r --gpu r --dist-url tcp://localhost:29515
  python step15_v4.py export --name v4
  python step15_v4.py evalmos --name v4      (held-out MOS sets: ytugc split0 test, gaming val; also --name v3)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step9_e2e as E9  # noqa: E402
import step10_train as S10  # noqa: E402
import step14_finetune as S14  # noqa: E402
import step7_eval as S7  # noqa: E402
from step4_train import lr_at  # noqa: E402

ROOT = E9.ROOT
OUT = f"{ROOT}/step15_v4"
V3 = f"{ROOT}/frozen_v3_20261018"
GAMING_DIR = f"{S7.BASE}/mos_datasets/LIVE-VQC/LIVE YouTube Gaming/unpacked/mp4"
GAMING_LABELS = f"{ROOT}/eval/labels/livegaming/labels.csv"
S7.DATASETS.setdefault("livegaming", dict(videos=GAMING_DIR, labels=GAMING_LABELS))   # for stage() / ClipDS


def gaming_items(split):
    with open(GAMING_LABELS) as f:
        return [(os.path.join(GAMING_DIR, r["file"]), float(r["mos"])) for r in csv.DictReader(f) if r["split"] == split]


def ytugc_items(part):
    mos_of = dict(S14.items("ytugc"))
    return [(p, mos_of[p]) for p in json.load(open(S14.split_file("ytugc", 0)))[part]]


def train(a):
    dev = torch.device("cuda", a.gpu)
    torch.cuda.set_device(dev)
    W, rank = a.world, a.rank
    if W > 1:
        import datetime
        dist.init_process_group("nccl", init_method=a.dist_url, rank=rank, world_size=W,
                                timeout=datetime.timedelta(hours=2))
        assert a.clips % W == 0 and a.pairs % W == 0, "clips and pairs must split evenly over ranks"
    main_rank = rank == 0
    torch.manual_seed(a.seed)
    run_dir = f"{OUT}/runs/{a.name}"
    os.makedirs(run_dir, exist_ok=True)

    data = dict(lsvq=E9.lsvq_train_items(), ytugc=ytugc_items("train"), gaming=gaming_items("train"))
    val = dict(ytugc=ytugc_items("val"), gaming=gaming_items("val"))
    if a.stage:                                        # node-local copies (Lustre stalls the loaders); LSVQ as v3
        for k, ds in (("ytugc", "ytugc"), ("gaming", "livegaming")):
            loc = S14.stage([p for p, _ in data[k] + val[k]], ds, a.stage)
            data[k] = [(loc[p], m) for p, m in data[k]]
            val[k] = [(loc[p], m) for p, m in val[k]]
    stats = {k: (float(np.mean([m for _, m in v])), float(np.std([m for _, m in v]))) for k, v in data.items()}
    names = list(a.weights)
    wts = np.array([a.weights[k] for k in names], float)
    wts /= wts.sum()
    E9.MOS_MEAN, E9.MOS_SD = 0.0, 1.0                  # items carry per-dataset z-scores

    model = E9.E2E(f"{V3}/s1.pt", f"{V3}/s2.pt", f"{V3}/fusion.pt").to(dev)
    model.train()
    opt = torch.optim.AdamW(E9.param_groups(model, a.lr_enc, a.lr_fusion, a.llrd, a.wd), lr=a.lr_fusion)
    step0 = 0
    if os.path.exists(f"{run_dir}/last.pt"):
        ck = torch.load(f"{run_dir}/last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        print(f"resumed at step {step0}", flush=True)
    params = [p for g in opt.param_groups for p in g["params"]]
    if W > 1:
        for p in model.parameters():
            dist.broadcast(p.data, 0)
        torch.manual_seed(a.seed + 7919 * rank)
    rseed = a.seed + step0 + 1_000_003 * rank
    dls = {k: iter(DataLoader(E9.LSVQWindows(data[k], a.tiles, rseed + 17 * i), batch_size=None,
                              num_workers=a.workers[k], prefetch_factor=3)) for i, k in enumerate(names)}
    pairs = E9.PairBuffer(a.tiles, a.pair_workers, a.buffer, rseed) if a.pair_every else None
    vfiles = sorted(glob.glob(f"{E9.OUT}/valid_pairs/pair_*.npz"))
    if main_rank:
        json.dump(dict(args=vars(a), n_train={k: len(v) for k, v in data.items()},
                       n_val={k: len(v) for k, v in val.items()}, mos_stats=stats, valid_pairs=len(vfiles)),
                  open(f"{run_dir}/config.json" if step0 == 0 else f"{run_dir}/config_resume{step0}.json", "w"), indent=1)
    log = open(f"{run_dir}/train_log.jsonl" if main_rank else os.devnull, "a")

    def validate(step):
        r = dict(step=step, kind="val", **(E9.eval_pairs(model, vfiles, dev) if vfiles else {}))
        for k, ds in (("ytugc", "ytugc"), ("gaming", "livegaming")):
            pv = S14.predict(model, [p for p, _ in val[k]], ds, dev, a.val_workers)
            ok = ~np.isnan(pv)
            r[f"{k}_val"] = S14.correlations(pv[ok], [m for (_, m), o in zip(val[k], ok) if o])
        model.train()
        log.write(json.dumps(r) + "\n")
        log.flush()
        print(json.dumps(r), flush=True)

    if step0 == 0 and main_rank:
        validate(0)
    if pairs:
        while not pairs.ready(a.pairs // W * 4):
            time.sleep(5)
    warm = int(0.05 * a.steps)
    acc, n, t0 = defaultdict(float), defaultdict(int), time.time()
    for step in range(step0 + 1, a.steps + 1):
        f = lr_at(step, a.steps, 1.0, warm)
        for g in opt.param_groups:
            g["lr"] = g["lr_base"] * f
        if pairs and step % a.pair_every == 0:
            pb = pairs.batch(a.pairs // W)
            Q, V = E9.pair_forward(model, pb, dev)
            ptypes = [p["ptype"] for p in pb]
            if W > 1:
                Q, V = S10.gather(Q, W), S10.gather(V, W)
                allp = [None] * W
                dist.all_gather_object(allp, ptypes)
                ptypes = [t for r in allp for t in r]
            loss, parts = S10.pair_losses(Q, V, ptypes, dev, 1.0)
            parts["q_acc_train"] = float((Q[:, 0] > Q[:, 1]).float().mean())
            kind = "pairs"
        else:
            k = names[np.random.default_rng([a.seed, step]).choice(len(names), p=wts)]   # same on every rank
            items = [next(dls[k]) for _ in range(a.clips // W)]
            mu, sd = stats[k]
            for s in items:
                s["mos"] = (s["mos"] - mu) / sd
            loss, parts = S10.mos_step_dist(model, items, dev, W)
            parts = {f"{k}_{n_}": v for n_, v in parts.items()}
            kind = f"mos_{k}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if W > 1:                                    # sum of per-rank contributions = the one-GPU gradient
            gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat)
            flat /= W
            for p, g in zip(params, torch._utils._unflatten_dense_tensors(flat, gs)):
                p.grad = g
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        for k_, v in parts.items():
            acc[k_] += v
            n[k_] += 1
        acc[f"loss_{kind}"] += float(loss)
        n[f"loss_{kind}"] += 1
        if step % a.log_every == 0 and main_rank:
            rec = dict(step=step, lr_fusion=a.lr_fusion * f, **{k_: v / n[k_] for k_, v in acc.items()},
                       steps_per_s=a.log_every / (time.time() - t0), time=time.strftime("%m-%d %H:%M:%S"))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            acc, n, t0 = defaultdict(float), defaultdict(int), time.time()
        if (step % a.val_every == 0 or step == a.steps) and main_rank:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), f"{run_dir}/last.pt.tmp")
            os.replace(f"{run_dir}/last.pt.tmp", f"{run_dir}/last.pt")
            validate(step)
    if main_rank:
        torch.save(dict(model=model.state_dict(), step=a.steps), f"{run_dir}/final.pt")
    if W > 1:
        dist.barrier()
    print("FINISHED", flush=True)
    os._exit(0)                                       # PairBuffer thread: skip the abort at interpreter exit


def export(a):
    run_dir = f"{OUT}/runs/{a.name}"
    sd = torch.load(f"{run_dir}/final.pt", map_location="cpu")["model"]
    for pre, name in (("e1.", "s1.pt"), ("e2.", "s2.pt"), ("fusion.", "fusion.pt")):
        torch.save(dict(model={k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}), f"{run_dir}/{name}")
    print("exported", run_dir, flush=True)


def evalmos(a):
    """whole-clip Q (step14 predict, 32 tiles) on the held-out MOS sets that v4 never trained on:
    ytugc step14 split0 test (195) and LIVE-YT-Gaming val (109). Same code for v3 and v4."""
    dev = torch.device("cuda", a.gpu)
    d = V3 if a.name == "v3" else f"{OUT}/runs/{a.name}"
    model = E9.E2E(f"{d}/s1.pt", f"{d}/s2.pt", f"{d}/fusion.pt").to(dev)
    sets = dict(ytugc_test=("ytugc", ytugc_items("test")), gaming_val=("livegaming", gaming_items("val")))
    res = dict(model=a.name, weights=d)
    for k, (ds, it) in sets.items():
        paths = [p for p, _ in it]
        if a.stage:
            loc = S14.stage(paths, ds, a.stage)
            paths = [loc[p] for p in paths]
        pv = S14.predict(model, paths, ds, dev, a.val_workers)
        ok = ~np.isnan(pv)
        res[k] = S14.correlations(pv[ok], [m for (_, m), o in zip(it, ok) if o])
        print(k, res[k], flush=True)
    os.makedirs(f"{OUT}/eval/{a.name}", exist_ok=True)
    json.dump(res, open(f"{OUT}/eval/{a.name}/mos_heldout.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "export", "evalmos"])
    ap.add_argument("--name", default="v4")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--weights", type=json.loads, default=dict(lsvq=0.6, ytugc=0.25, gaming=0.15),
                    help="probability that a MOS batch comes from each dataset (~12 epochs of ytugc / gaming)")
    ap.add_argument("--workers", type=json.loads, default=dict(lsvq=12, ytugc=14, gaming=4))
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--pair-every", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-fusion", type=float, default=3e-5)
    ap.add_argument("--llrd", type=float, default=0.75)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--pair-workers", type=int, default=8)
    ap.add_argument("--buffer", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-workers", type=int, default=16)
    ap.add_argument("--stage", default="/tmp/rs64952_step14", help="node-local video copies ('' = read Lustre)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--dist-url", default=None)
    a = ap.parse_args()
    dict(train=train, export=export, evalmos=evalmos)[a.cmd](a)


if __name__ == "__main__":
    main()
