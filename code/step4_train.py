"""Step 4: train Stage 1 (Tile-MAE) on the Step 3 tile store.

    torchrun --nproc_per_node 3 step4_train.py --name s_mae --init mae ...
    python step4_train.py build-val           (fixed validation pair sets, valid split)

Losses (METHOD_PLAN Step 4):
  L_rec   MSE on masked patches, 774-d targets (per-patch normalized pixels of
          the distorted tile + per-channel patch mean/std)
  L_rank  max(0, 0.1 - (s_better - s_worse)) on ordered pairs, both tiles of a
          pair share one MAE mask; weight 0.1, switched on after 10% of steps
Data: TileDataset (train split only), DiverseBatcher (<= 2 pairs per source per
batch, per GPU; ranks read disjoint source shards). Each rank streams its shards
forever (epoch += 1 when exhausted), so all ranks run the same step count.
Batch diversity: every --dino-every steps, mean pairwise cosine of frozen
DINOv2-small features of the batch tiles (rank 0) is logged.
Validation (rank 0, every --val-every steps) on the fixed valid-split pair set:
ranking accuracy per pair type, rec loss, pooled-feature per-dim std.
Checkpoints every --ckpt-every steps; the final model is the average of the
last 4 checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_tile_loader as tl  # noqa: E402
import step4_model as M  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
TILES = f"{ROOT}/step3/tiles"
OUT = f"{ROOT}/step4"
MAE_S = ("1aurent/vit_small_patch16_224.medmae_in1k_mae", "model.safetensors")
DINO_S = "facebook/dinov2-small"
MARGIN = 0.1


# ------------------------------------------------------------------ validation set
def build_val(a):
    """Fixed pair sets from the valid split (deterministic): 'pairs' with the
    training distribution of pair types and on-the-fly ops, and 'clean' with
    p_ops=0 (stored versions only) for reconstruction error vs severity."""
    os.makedirs(OUT, exist_ok=True)
    shards = tl.list_shards(TILES, ("valid",))
    for name, p_ops, n in (("pairs", 0.5, a.val_pairs), ("clean", 0.0, a.val_clean)):
        dst = f"{OUT}/val_{name}.npz"
        if os.path.exists(dst):
            print("exists", dst)
            continue
        ds = tl.TileDataset(TILES, shards=shards, p_ops=p_ops, seed=12345, open_per_worker=16, items_per_group=2)
        dl = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=4)
        xs, meta, per_src = [], defaultdict(list), defaultdict(int)
        for s in dl:
            if per_src[s["source_id"]] >= max(4, n // len(shards) * 2):   # spread over sources
                continue
            per_src[s["source_id"]] += 1
            xs.append(s["x"])
            for k in ("order", "ptype", "source_id", "ops", "plan_id", "frame"):
                meta[k].append(s[k])
            meta["versions"].append(list(s["versions"]))
            meta["encoders"].append([e or "" for e in s["encoders"]])
            if len(xs) >= n:
                break
        np.savez(dst, x=np.stack(xs), meta=json.dumps(meta))
        print(f"wrote {dst}: {len(xs)} pairs from {len(per_src)} sources", flush=True)


def load_val(name):
    z = np.load(f"{OUT}/val_{name}.npz")
    return z["x"], json.loads(str(z["meta"]))


@torch.no_grad()
def evaluate(model, x, meta, dev, bs=128, seed=0):
    """Ranking accuracy (no mask: inference mode), rec loss (fixed masks),
    pooled-feature per-dim std."""
    model.eval()
    g = torch.Generator(device=dev).manual_seed(seed)
    s_all, rec_all, feats = [], [], []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(dev).permute(0, 1, 4, 2, 3)
        B = xb.shape[0]
        xn = model.normalize(xb.flatten(0, 1))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat = model.encode(xn)
            s_all.append(model.score(lat).float().view(B, 2).cpu())
            feats.append(model.pooled(lat).float().cpu())
            ids_keep, ids_restore, mask = model.random_masking_ids(B, dev, g)
            ids_keep, ids_restore, mask = (t.repeat_interleave(2, 0) for t in (ids_keep, ids_restore, mask))
            pred = model.decode(model.encode(xn, ids_keep), ids_restore).float()
        err = ((pred - model.targets(xn)) ** 2).mean(-1)
        rec_all.append(((err * mask).sum(1) / mask.sum(1)).view(B, 2).cpu())
    model.train()
    s, rec, f = torch.cat(s_all), torch.cat(rec_all), torch.cat(feats)
    order = torch.tensor(meta["order"])
    res = dict(rec=float(rec.mean()), feat_std_min=float(f.std(0).min()), feat_std_mean=float(f.std(0).mean()))
    ok = order == 1
    if ok.any():
        res["rank_acc"] = float((s[ok, 0] > s[ok, 1]).float().mean())
        res["n_ordered"] = int(ok.sum())
        for t in sorted(set(meta["ptype"])):
            m = ok & torch.tensor([p == t for p in meta["ptype"]])
            if m.any():
                res[f"acc_{t}"] = float((s[m, 0] > s[m, 1]).float().mean())
                res[f"viol_{t}"] = float((s[m, 0] - s[m, 1] < MARGIN).float().mean())
    return res, s, rec, f


# ------------------------------------------------------------------ training
def param_groups(model, wd):
    dec, no = [], []
    for n, p in model.named_parameters():
        (no if p.ndim < 2 or n in ("cls_token", "pos_embed", "mask_token") else dec).append(p)
    return [dict(params=dec, weight_decay=wd), dict(params=no, weight_decay=0.0)]


def lr_at(step, total, base, warm):
    if step < warm:
        return base * step / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def infinite_batches(a, rank, world, epoch=0):
    while True:
        ds = tl.TileDataset(TILES, splits=("train",), seed=a.seed, rank=rank, world=world, epoch=epoch,
                            items_per_group=a.items_per_group, open_per_worker=a.open_per_worker)
        dl = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=a.prefetch,
                        persistent_workers=False)
        try:
            for b in tl.DiverseBatcher(dl, a.pairs_per_gpu, max_per_source=2, seed=a.seed + 7919 * epoch + rank):
                if len(b["order"]) == a.pairs_per_gpu:
                    yield b
        except RuntimeError as e:          # tail of the pass: too few sources left for a diverse batch
            print(f"[rank {rank}] epoch {epoch} ended early: {e}", flush=True)
        epoch += 1


def dino_cosine(dino, x):
    """mean pairwise cosine of frozen DINOv2-small features over a batch of tiles."""
    from step3_bench import mean_pairwise_cos
    return mean_pairwise_cos(dino(x))


def save_ckpt(path, model, opt, step, a):
    tmp = path + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, args=vars(a)), tmp)
    os.replace(tmp, path)


def average_last(run_dir, k=4):
    cks = sorted(glob.glob(f"{run_dir}/ckpt_*.pt"))[-k:]
    avg = None
    for c in cks:
        sd = torch.load(c, map_location="cpu")["model"]
        if avg is None:
            avg = {n: v.float().clone() for n, v in sd.items()}
        else:
            for n in avg:
                avg[n] += sd[n].float()
    for n in avg:
        avg[n] /= len(cks)
    torch.save(dict(model=avg, averaged=[os.path.basename(c) for c in cks]), f"{run_dir}/final_avg{len(cks)}.pt")
    return cks


def train(a):
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(a.seed + rank)
    run_dir = f"{OUT}/runs/{a.name}"
    os.makedirs(run_dir, exist_ok=True)

    if a.init == "dinov2":                          # DINOv2 geometry: patch 14 on the 252 centre crop
        model = M.TileMAE(a.variant, img=252, patch=14, layerscale=True, drop_path=a.drop_path).to(dev)
    else:
        model = M.TileMAE(a.variant, drop_path=a.drop_path).to(dev)
    init_info = None
    if a.init == "dinov2" and not glob.glob(f"{run_dir}/ckpt_*.pt"):
        init_info = M.load_dinov2_hf(model, a.init_path or DINO_S)
        init_info["source"] = f"hf://{a.init_path or DINO_S}"
    if a.init == "mae":
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(*MAE_S) if a.init_path is None else a.init_path
        init_info = M.load_timm_vit(model, p)
        init_info["source"] = a.init_path or f"hf://{MAE_S[0]}/{MAE_S[1]}"
    tiles_per_step = 2 * a.pairs_per_gpu * world
    base_lr = a.blr * tiles_per_step / 256
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=base_lr, betas=(0.9, 0.95))
    step = 0
    cks = sorted(glob.glob(f"{run_dir}/ckpt_*.pt"))
    if cks:
        ck = torch.load(cks[-1], map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    if rank == 0:
        with open(f"{run_dir}/config.json", "w") as f:
            json.dump(dict(args=vars(a), world=world, tiles_per_step=tiles_per_step, base_lr=base_lr,
                           init=init_info, resumed_from=cks[-1] if cks else None,
                           params=sum(p.numel() for p in model.parameters())), f, indent=1)
    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    warm = int(a.warmup * a.steps)
    rank_on = int(a.rank_start * a.steps)

    val = load_val("pairs") if rank == 0 and os.path.exists(f"{OUT}/val_pairs.npz") else None
    dino = None
    if rank == 0 and a.dino_every:
        from step3_bench import dino as make_dino
        dino = make_dino(dev)
    log = open(f"{run_dir}/train_log.jsonl", "a") if rank == 0 else None
    acc = defaultdict(float)
    nacc = defaultdict(int)
    t_last, t_data = time.time(), 0.0
    batches = infinite_batches(a, rank, world, epoch=1 if step else 0)   # resume: new data order
    model.train()
    while step < a.steps:
        t0 = time.time()
        b = next(batches)
        t_data += time.time() - t0
        x = b["x"].to(dev, non_blocking=True)                     # (P, 2, 3, 256, 256) uint8
        P = x.shape[0]
        order = b["order"].to(dev)
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, a.steps, base_lr, warm)
        xn = model.normalize(x.flatten(0, 1))
        ids_keep, ids_restore, mask = model.random_masking_ids(P, dev)
        ids_keep, ids_restore, mask = (t.repeat_interleave(2, 0) for t in (ids_keep, ids_restore, mask))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat, pred, s = dmodel(xn, ids_keep, ids_restore)
        tgt = model.targets(xn)
        err = ((pred.float() - tgt) ** 2).mean(-1)
        l_rec = (err * mask).sum() / mask.sum()
        s = s.float().view(P, 2)
        o = order == 1
        diff = s[:, 0] - s[:, 1]
        l_rank = F.relu(MARGIN - diff[o]).mean() if o.any() else s.sum() * 0
        w_rank = a.rank_weight if step >= rank_on else 0.0
        loss = l_rec + w_rank * l_rank
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1

        acc["l_rec"] += float(l_rec)
        acc["l_rank"] += float(l_rank)
        nacc["loss"] += 1
        if o.any():
            dd = diff.detach().float().cpu()
            for i, t in enumerate(b["ptype"]):
                if b["order"][i] == 1:
                    acc[f"viol_{t}"] += float(dd[i] < MARGIN)
                    acc[f"acc_{t}"] += float(dd[i] > 0)
                    nacc[t] += 1
        if rank == 0 and step % a.log_every == 0:
            dt = time.time() - t_last
            rec = dict(step=step, lr=opt.param_groups[0]["lr"], w_rank=w_rank,
                       l_rec=acc["l_rec"] / nacc["loss"], l_rank=acc["l_rank"] / nacc["loss"],
                       tiles_per_s=tiles_per_step * a.log_every / dt, data_wait_frac=t_data / dt,
                       mem_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                       feat_std_train=float(model.pooled(lat.detach()).float().std(0).mean()),
                       time=time.strftime("%m-%d %H:%M:%S"))
            for t in tl.PAIR_TYPES:
                if nacc[t]:
                    rec[f"viol_{t}"] = acc[f"viol_{t}"] / nacc[t]
                    rec[f"acc_{t}"] = acc[f"acc_{t}"] / nacc[t]
            if dino is not None and step % a.dino_every == 0:
                rec["dino_batch_cos"] = dino_cosine(dino, b["x"][:, 0])
                rec["batch_sources"] = len(set(b["source_id"]))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            acc.clear(), nacc.clear()
            t_last, t_data = time.time(), 0.0
        if step % a.ckpt_every == 0 or step == a.steps:
            if rank == 0:
                save_ckpt(f"{run_dir}/ckpt_{step:07d}.pt", model, opt, step, a)
                for old in sorted(glob.glob(f"{run_dir}/ckpt_*.pt"))[:-a.keep_ckpts]:
                    os.remove(old)
        if rank == 0 and val is not None and (step % a.val_every == 0 or step == a.steps):
            r, *_ = evaluate(model, val[0], val[1], dev)
            r.update(step=step, kind="val")
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(json.dumps(r), flush=True)
        if ddp and step % a.log_every == 0:
            dist.barrier()
    if rank == 0:
        cks = average_last(run_dir, 4)
        print("averaged", cks, flush=True)
        print("FINISHED", flush=True)
    if ddp:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train", choices=["train", "build-val"])
    ap.add_argument("--name", default="s_mae")
    ap.add_argument("--variant", default="s", choices=list(M.VARIANTS))
    ap.add_argument("--init", default="mae", choices=["mae", "scratch", "dinov2"])
    ap.add_argument("--init-path", default=None)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--pairs-per-gpu", type=int, default=86, help="3 GPUs x 86 pairs x 2 = 516 tiles/step")
    ap.add_argument("--blr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--rank-start", type=float, default=0.10)
    ap.add_argument("--rank-weight", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--items-per-group", type=int, default=4)
    ap.add_argument("--open-per-worker", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dino-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--keep-ckpts", type=int, default=6)
    ap.add_argument("--val-pairs", type=int, default=8192)
    ap.add_argument("--val-clean", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_val(a) if a.cmd == "build-val" else train(a)


if __name__ == "__main__":
    main()
