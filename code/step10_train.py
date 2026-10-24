"""Step 10c: end-to-end candidates (v4), trained like v3 (step9_e2e.py) but on LSVQ train90.

All runs hold out the same 10% of LSVQ train (seed 0, eval/labels/lsvq/val10_seed0.txt,
identical to step8_finetune --holdout 2806) for the 10b selection, and start from v1
encoders + the v2_90 fusion (v2 recipe retrained on train90). Outputs: vqa_data/step10_cand/.

  v3_90  v3 recipe on train90 (a component of 10c-5, the v2/v3 average)
  c1     --pair-every 2                              (drop batches every 2nd step)
  c2     --pair-every 2 --drop-weight 3              (c1 + higher drop-pair loss weight)
  c3     --freeze-motion                             (Stage 2 frozen, all layers)
  c4     --consist-weight 1                          (match v2_90's Q differences on drop pairs)

Pair batches: L_rank (drop pairs weighted by --drop-weight) + drop_weight * L_drop +
0.1 L_inv (+ consist_weight * mean((dQ - dQ_v2)^2) over drop pairs, dQ = Q_better - Q_worse,
v2_90 = v1 encoders + v2_90 fusion, frozen). Model = final step (fixed 12k schedule).

    python step10_train.py train --name c1 --pair-every 2
    python step10_train.py export --name c1

Multi-GPU (10-20): --world W --rank r --dist-url tcp://host:port, one process per GPU. Gradient-equivalent to one GPU:
each rank loads clips/W MOS clips and pairs/W pairs; Q/V are all-gathered with autograd, every rank computes the
batch losses (PLCC over the full batch) on the gathered batch, gradients are summed over ranks and divided by W.
Rank 0 alone logs, validates and saves.
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
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_train as T6  # noqa: E402
import step9_e2e as E9  # noqa: E402
from step4_train import lr_at  # noqa: E402
from step7_eval import DATASETS  # noqa: E402

ROOT = E9.ROOT
OUT = f"{ROOT}/step10_cand"
TRAIN90 = f"{ROOT}/eval/labels/lsvq/train90_seed0.txt"
V2_90 = f"{ROOT}/step8_lsvq/runs/v2_90/final.pt"


def train90_items():
    with open(TRAIN90) as f:
        return [(os.path.join(DATASETS["lsvq_train"]["videos"], r[0].strip()), float(r[-1])) for r in csv.reader(f)]


def pair_losses(Q, V, ptype, dev, drop_weight, Qt=None, consist_weight=0.0):
    t = np.array(ptype)
    dr = torch.from_numpy(t == "drops").to(dev)
    rs = ~dr
    w = torch.where(dr, drop_weight, 1.0)
    l_rank = (w * F.relu(T6.MARGIN - (Q[:, 0] - Q[:, 1]))).mean()
    l_drop = F.relu(T6.MARGIN - (V[dr, 1] - V[dr, 0])).mean() if dr.any() else Q.sum() * 0
    l_inv = F.huber_loss(V[rs, 0], V[rs, 1]) if rs.any() else Q.sum() * 0
    loss = l_rank + drop_weight * l_drop + 0.1 * l_inv
    parts = dict(l_rank=float(l_rank), l_drop=float(l_drop), l_inv=float(l_inv))
    if Qt is not None and dr.any():
        l_con = (((Q[dr, 0] - Q[dr, 1]) - (Qt[dr, 0] - Qt[dr, 1])) ** 2).mean()
        loss = loss + consist_weight * l_con
        parts["l_consist"] = float(l_con)
    return loss, parts


def gather(x, world):
    return torch.cat(dnn.all_gather(x.contiguous())) if world > 1 else x


def mos_step_dist(model, items, dev, world):
    """E9.mos_step on the batch gathered over ranks (same loss as one GPU with all clips)."""
    Q, _ = model.clips([E9.to_dev(s["low"], dev) for s in items], [E9.to_dev(s["tiles"], dev) for s in items],
                       [E9.to_dev(s["cells"], dev) for s in items])
    z = (torch.tensor([s["mos"] for s in items], device=dev, dtype=torch.float32) - E9.MOS_MEAN) / E9.MOS_SD
    Q, z = gather(Q, world), gather(z, world).detach()
    qc, zc = Q - Q.mean(), z - z.mean()
    plcc = (qc * zc).sum() / (qc.norm() * zc.norm() + 1e-8)
    mse = ((Q - z) ** 2).mean()
    return mse + (1 - plcc), dict(l_mse=float(mse), plcc_batch=float(plcc))


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
    model = E9.E2E(a.init_s1, f"{E9.V1}/stage2_v_mae_final_avg4.pt", a.init_fusion).to(dev)
    if a.freeze_motion:
        for p in model.e2.parameters():
            p.requires_grad_(False)
    model.train()
    if a.freeze_motion:
        model.e2.eval()
    teacher = None
    if a.consist_weight > 0:
        teacher = E9.E2E(f"{E9.V1}/stage1_s_mae_final_avg4.pt", f"{E9.V1}/stage2_v_mae_final_avg4.pt",
                         a.teacher_fusion).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(E9.param_groups(model, a.lr_enc, a.lr_fusion, a.llrd, a.wd), lr=a.lr_fusion,
                            betas=(0.9, 0.999))
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
    items = train90_items()
    mos_dl = iter(DataLoader(E9.LSVQWindows(items, a.tiles, rseed), batch_size=None, num_workers=a.workers,
                             prefetch_factor=3))
    pairs = E9.PairBuffer(a.tiles, a.pair_workers, a.buffer, rseed) if a.pair_every else None
    vfiles = sorted(glob.glob(f"{E9.OUT}/valid_pairs/pair_*.npz"))
    if main_rank:
        json.dump(dict(args=vars(a), lsvq_train90=len(items), valid_pairs=len(vfiles)),
                  open(f"{run_dir}/config.json" if step0 == 0 else f"{run_dir}/config_resume{step0}.json", "w"), indent=1)
    log = open(f"{run_dir}/train_log.jsonl" if main_rank else os.devnull, "a")
    if step0 == 0 and vfiles and main_rank:
        r = dict(E9.eval_pairs(model, vfiles, dev), step=0, kind="val")
        log.write(json.dumps(r) + "\n")
    if a.freeze_motion:
        model.e2.eval()                              # eval_pairs switches the whole model back to train()
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
                Q, V = gather(Q, W), gather(V, W)
                allp = [None] * W
                dist.all_gather_object(allp, ptypes)
                ptypes = [t for r in allp for t in r]
            Qt = None
            if teacher is not None:
                with torch.no_grad():
                    Qt, _ = E9.pair_forward(teacher, pb, dev)
            loss, parts = pair_losses(Q, V, ptypes, dev, a.drop_weight, Qt, a.consist_weight)
            parts["q_acc_train"] = float((Q[:, 0] > Q[:, 1]).float().mean())
            kind = "pairs"
        else:
            loss, parts = mos_step_dist(model, [next(mos_dl) for _ in range(a.clips // W)], dev, W)
            kind = "mos"
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
        for k, v in parts.items():
            acc[k] += v
            n[k] += 1
        acc[f"loss_{kind}"] += float(loss)
        n[f"loss_{kind}"] += 1
        if step % a.log_every == 0 and main_rank:
            rec = dict(step=step, lr_fusion=a.lr_fusion * f, **{k: v / n[k] for k, v in acc.items()},
                       steps_per_s=a.log_every / (time.time() - t0), time=time.strftime("%m-%d %H:%M:%S"))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            acc, n, t0 = defaultdict(float), defaultdict(int), time.time()
        if (step % a.val_every == 0 or step == a.steps) and main_rank:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), f"{run_dir}/last.pt.tmp")
            os.replace(f"{run_dir}/last.pt.tmp", f"{run_dir}/last.pt")
            if vfiles:
                r = dict(E9.eval_pairs(model, vfiles, dev), step=step, kind="val")
                log.write(json.dumps(r) + "\n")
                log.flush()
                if a.freeze_motion:
                    model.e2.eval()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "export"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--init-s1", default=f"{E9.V1}/stage1_s_mae_final_avg4.pt",
                    help="spatial encoder to start from (d1: Step 11 DINOv2-S Tile-MAE)")
    ap.add_argument("--init-fusion", default=V2_90)
    ap.add_argument("--teacher-fusion", default=V2_90)
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--pair-every", type=int, default=4)
    ap.add_argument("--drop-weight", type=float, default=1.0)
    ap.add_argument("--freeze-motion", action="store_true")
    ap.add_argument("--consist-weight", type=float, default=0.0)
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-fusion", type=float, default=3e-5)
    ap.add_argument("--llrd", type=float, default=0.75)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--pair-workers", type=int, default=10)
    ap.add_argument("--buffer", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--dist-url", default=None)
    a = ap.parse_args()
    dict(train=train, export=export)[a.cmd](a)


if __name__ == "__main__":
    main()
