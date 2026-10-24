"""Step 5: train Stage 2 (Video-MAE on 384x216 displayed frames).

    torchrun --nproc_per_node 3 step5_train.py --name v_mae
    python step5_train.py build-val          (fixed valid-split windows and drop pairs)
    python step5_train.py eval --ckpt ...    (Step 5 checks)

Losses (METHOD_PLAN Step 5):
  L_vmae  MSE on masked tubelets (normalized pixels + per-channel mean/std)
  L_rank  max(0, 0.1 - (s_P - s_superset(P))) on drop pairs of the same window,
          same mask; weight 0.1, switched on after 10% of steps
Data: clean, capture and 2f post-drop encodes plus source spans (train split),
32-frame sub-windows of 240-tick decodes from the lossless low-res cache;
on-the-fly drops in 50% of the single windows (p_none = 0.5). No time reversal.
A batch of 64 windows (global) = 36 single windows + 14 drop pairs (28 windows),
<= 2 samples per source per batch on each GPU.
Checks: rec error of masked steps at catch-up frames vs normal frames; ranking
accuracy by drop family and by burst length (1, 2-4, 5-15, 16+); score
difference for drops in near-static vs moving windows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_video_loader as vl  # noqa: E402
import step5_model as V  # noqa: E402
from step4_train import average_last, lr_at, param_groups, save_ckpt  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
MANIFEST = f"{ROOT}/step2/encodes_manifest.csv"
STEP1 = f"{ROOT}/step1/manifest.csv"
PLANS = f"{ROOT}/step3/capture_plans.json"
LOWRES = f"{ROOT}/step3/lowres"
OUT = f"{ROOT}/step5"
CKPT = ("MCG-NJU/videomae-small-finetuned-ssv2", "pytorch_model.bin")
MARGIN = 0.1


def make_ds(split, mode, seed, rank=0, world=1):
    base = vl.VideoDataset(MANIFEST, splits=(split,), mode=mode, tiles=False, p_none=0.5,
                           pair_types=("drops",), plans_json=PLANS, seed=seed, rank=rank, world=world,
                           step1_manifest=STEP1, lowres_dir=LOWRES)
    return vl.SubWindows(base, V.T_WIN)


class Mixer:
    """Pull from the single-window and pair loaders through shuffle pools and
    emit (singles, pairs) with at most 2 samples per source per batch."""

    def __init__(self, dl_single, dl_pair, n_single, n_pair, seed, pool=6):
        self.its = [iter(dl_single), iter(dl_pair)]
        self.ns = [n_single, n_pair]
        self.pools = [[], []]
        self.pool = pool
        self.rng = random.Random(seed)

    def __iter__(self):
        while True:
            cnt, out = Counter(), [[], []]
            for k in (0, 1):
                while len(self.pools[k]) < self.pool * self.ns[k]:
                    self.pools[k].append(next(self.its[k]))
                self.rng.shuffle(self.pools[k])
                rest = []
                for s in self.pools[k]:
                    if len(out[k]) < self.ns[k] and cnt[s["source_id"]] < 2:
                        out[k].append(s)
                        cnt[s["source_id"]] += 1
                    else:
                        rest.append(s)
                self.pools[k] = rest
            if len(out[0]) == self.ns[0] and len(out[1]) == self.ns[1]:
                yield out
            else:                                 # rare: not diverse enough yet, put back
                for k in (0, 1):
                    self.pools[k].extend(out[k])


def to_batch(singles, pairs):
    x = [s["low"] for s in singles]
    for p in pairs:
        x += [p["better"], p["worse"]]
    order = torch.tensor([p["order"] for p in pairs])
    return torch.from_numpy(np.stack(x)), order


# ------------------------------------------------------------------ validation set
def runs_of(m):
    """lengths of runs of True in a bool array"""
    out, n = [], 0
    for v in list(m) + [False]:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    return out


def build_val(a):
    os.makedirs(OUT, exist_ok=True)
    for name, mode, n in (("singles", "single", a.val_singles), ("pairs", "pair", a.val_pairs)):
        dst = f"{OUT}/val_{name}.npy"
        if os.path.exists(dst):
            print("exists", dst)
            continue
        dl = DataLoader(make_ds("valid", mode, 777), batch_size=None, num_workers=a.workers, prefetch_factor=2)
        xs, meta, per_src = [], defaultdict(list), Counter()
        for s in dl:
            if per_src[s["source_id"]] >= 3:          # spread over the 365 valid sources
                continue
            if mode == "pair" and not s["order"]:
                continue
            per_src[s["source_id"]] += 1
            if mode == "single":
                xs.append(s["low"])
                meta["drop_mask"].append(s["meta"]["drop_mask"].tolist())
                meta["family"].append(s["meta"]["drop_family"])
                meta["kind"].append(s["meta"]["kind"])
            else:
                xs.append(np.stack([s["better"], s["worse"]]))
                meta["masks"].append([m.tolist() for m in s["masks"]])
                meta["family"].append(s["family"])
                meta["kind"].append(s["kinds"][0])
            meta["source_id"].append(s["source_id"])
            if len(xs) >= n:
                break
        np.save(dst, np.stack(xs))
        with open(dst[:-4] + ".json", "w") as f:
            json.dump(meta, f)
        print(f"wrote {dst}: {len(xs)} from {len(per_src)} sources", flush=True)


def load_val(name):
    return np.load(f"{OUT}/val_{name}.npy", mmap_mode="r"), json.load(open(f"{OUT}/val_{name}.json"))


def burst_bin(n):
    return "1" if n <= 1 else "2-4" if n <= 4 else "5-15" if n <= 15 else "16+"


@torch.no_grad()
def evaluate(model, dev, full=False, bs=16, seed=0):
    model.eval()
    g = torch.Generator(device=dev).manual_seed(seed)
    res = {}
    # pairs: ranking accuracy (no mask) + rec on fixed masks
    x, meta = load_val("pairs")
    s_all = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(dev).flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_all.append(model.score(model.encode(model.normalize(xb))).float().view(-1, 2).cpu())
    s = torch.cat(s_all)
    d = (s[:, 0] - s[:, 1]).numpy()
    res["rank_acc"] = float((d > 0).mean())
    res["viol"] = float((d < MARGIN).mean())
    res["n_pairs"] = len(d)
    if full:
        fam = defaultdict(list)
        burst = defaultdict(list)
        static = defaultdict(list)
        for k, (mk, f) in enumerate(zip(meta["masks"], meta["family"])):
            p, q = np.array(mk[0]), np.array(mk[1])
            fam[f.replace("+superset", "")].append(d[k])
            burst[burst_bin(max(runs_of(q & ~p)) if (q & ~p).any() else 0)].append(d[k])
            xb = np.asarray(x[k, 0], np.int16)
            motion = float(np.abs(np.diff(xb, axis=0)).mean())
            static["static (<1 gray level/frame)" if motion < 1.0 else "moving"].append(d[k])
        res["rank_by_family"] = {k: dict(n=len(v), acc=float(np.mean(np.array(v) > 0))) for k, v in sorted(fam.items())}
        res["rank_by_burst"] = {k: dict(n=len(v), acc=float(np.mean(np.array(v) > 0))) for k, v in sorted(burst.items())}
        res["score_diff_static_vs_moving"] = {k: dict(n=len(v), mean_diff=float(np.mean(v)), acc=float(np.mean(np.array(v) > 0)))
                                              for k, v in static.items()}
    # singles: rec error, catch-up vs normal steps
    x, meta = load_val("singles")
    err_all, mask_all, sm_all = [], [], []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(dev)
        xn = model.normalize(xb)
        ids_keep, ids_restore, mask, sm = model.masking_ids(xb.shape[0], dev, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.decode(model.encode(xn, ids_keep), ids_restore).float()
        err = ((pred - model.targets(xn)) ** 2).mean(-1)                     # (B, 2304)
        err_all.append(err.view(-1, V.GT, V.GH * V.GW).cpu())
        mask_all.append(mask.view(-1, V.GT, V.GH * V.GW).cpu())
    err, mask = torch.cat(err_all), torch.cat(mask_all)
    step_err = (err * mask).sum(-1) / mask.sum(-1).clamp_min(1)              # (N, 16) masked-token error per step
    step_masked = mask.mean(-1) == 1
    res["rec"] = float(((err * mask).sum() / mask.sum()))
    if full:
        cat_e, norm_e = [], []
        for k, dm in enumerate(meta["drop_mask"]):
            dm = np.array(dm, bool)
            catchup = np.zeros(V.T_WIN, bool)
            catchup[1:] = dm[:-1] & ~dm[1:]                  # first shown frame after a held run
            for t in range(V.GT):
                if not step_masked[k, t]:
                    continue
                fr = slice(t * V.TP, (t + 1) * V.TP)
                if catchup[fr].any():
                    cat_e.append(float(step_err[k, t]))
                elif not dm[fr].any() and not catchup[fr].any():
                    norm_e.append(float(step_err[k, t]))
        res["rec_catchup_vs_normal"] = dict(catchup=float(np.mean(cat_e)) if cat_e else None, n_catchup=len(cat_e),
                                            normal=float(np.mean(norm_e)) if norm_e else None, n_normal=len(norm_e))
    model.train()
    return res


# ------------------------------------------------------------------ training
def train(a):
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(a.seed + rank)
    run_dir = f"{OUT}/runs/{a.name}"
    os.makedirs(run_dir, exist_ok=True)
    model = V.VideoMAE(drop_path=a.drop_path).to(dev)
    init_info = None
    if a.init == "videomae":
        from huggingface_hub import hf_hub_download
        init_info = V.load_videomae_hf(model, hf_hub_download(*CKPT))
        init_info["source"] = f"hf://{CKPT[0]}/{CKPT[1]} (CC-BY-NC-4.0)"
    win_per_step = (a.singles + 2 * a.pairs) * world
    base_lr = a.blr * win_per_step / 256
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
            json.dump(dict(args=vars(a), world=world, windows_per_step=win_per_step, base_lr=base_lr,
                           init=init_info, resumed_from=cks[-1] if cks else None), f, indent=1)
    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    warm, rank_on = int(a.warmup * a.steps), int(a.rank_start * a.steps)
    dl_s = DataLoader(make_ds("train", "single", a.seed * 101 + 1, rank, world), batch_size=None,
                      num_workers=a.workers_single, prefetch_factor=4)
    dl_p = DataLoader(make_ds("train", "pair", a.seed * 101 + 2, rank, world), batch_size=None,
                      num_workers=a.workers_pair, prefetch_factor=4)
    batches = iter(Mixer(dl_s, dl_p, a.singles, a.pairs, a.seed + rank))
    has_val = rank == 0 and os.path.exists(f"{OUT}/val_pairs.npy")
    log = open(f"{run_dir}/train_log.jsonl", "a") if rank == 0 else None
    acc, nacc = defaultdict(float), defaultdict(int)
    t_last, t_data = time.time(), 0.0
    model.train()
    while step < a.steps:
        t0 = time.time()
        singles, pairs = next(batches)
        t_data += time.time() - t0
        xb, order = to_batch(singles, pairs)
        xb, order = xb.to(dev, non_blocking=True), order.to(dev)
        n_s, n_p = len(singles), len(pairs)
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, a.steps, base_lr, warm)
        xn = model.normalize(xb)
        # one mask per single window and per pair (both members share it)
        ids_keep, ids_restore, mask, _ = model.masking_ids(n_s + n_p, dev)
        rep = torch.cat([torch.arange(n_s, device=dev), (n_s + torch.arange(n_p, device=dev)).repeat_interleave(2)])
        ids_keep, ids_restore, mask = ids_keep[rep], ids_restore[rep], mask[rep]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat, pred, s = dmodel(xn, ids_keep, ids_restore)
        err = ((pred.float() - model.targets(xn)) ** 2).mean(-1)
        l_rec = (err * mask).sum() / mask.sum()
        sp = s.float()[n_s:].view(n_p, 2)
        o = order == 1
        diff = sp[:, 0] - sp[:, 1]
        l_rank = F.relu(MARGIN - diff[o]).mean() if o.any() else s.sum() * 0
        w_rank = a.rank_weight if step >= rank_on else 0.0
        loss = l_rec + w_rank * l_rank
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        acc["l_rec"] += float(l_rec)
        acc["l_rank"] += float(l_rank)
        nacc["n"] += 1
        if o.any():
            dd = diff.detach()[o]
            acc["acc"] += float((dd > 0).float().sum())
            acc["viol"] += float((dd < MARGIN).float().sum())
            nacc["pairs"] += int(o.sum())
        if rank == 0 and step % a.log_every == 0:
            dt = time.time() - t_last
            rec = dict(step=step, lr=opt.param_groups[0]["lr"], w_rank=w_rank, l_rec=acc["l_rec"] / nacc["n"],
                       l_rank=acc["l_rank"] / nacc["n"], windows_per_s=win_per_step * a.log_every / dt,
                       data_wait_frac=t_data / dt, mem_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                       ordered_pair_frac=nacc["pairs"] / max(1, nacc["n"] * n_p),
                       drop_acc=acc["acc"] / max(1, nacc["pairs"]), drop_viol=acc["viol"] / max(1, nacc["pairs"]),
                       batch_sources=len({x["source_id"] for x in singles + pairs}),
                       time=time.strftime("%m-%d %H:%M:%S"))
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
        if has_val and (step % a.val_every == 0 or step == a.steps):
            r = evaluate(model, dev)
            r.update(step=step, kind="val")
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(json.dumps(r), flush=True)
        if ddp and step % a.log_every == 0:
            dist.barrier()
    if rank == 0:
        print("averaged", average_last(run_dir, 4), flush=True)
        print("FINISHED", flush=True)
    if ddp:
        dist.destroy_process_group()


def run_eval(a):
    dev = torch.device("cuda", a.gpu)
    model = V.VideoMAE().to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    r = evaluate(model, dev, full=True)
    r["ckpt"] = a.ckpt
    dst = os.path.join(os.path.dirname(a.ckpt), "eval_" + os.path.basename(a.ckpt).replace(".pt", ".json"))
    json.dump(r, open(dst, "w"), indent=1)
    print(json.dumps(r, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train", choices=["train", "build-val", "eval"])
    ap.add_argument("--name", default="v_mae")
    ap.add_argument("--init", default="videomae", choices=["videomae", "scratch"])
    ap.add_argument("--ckpt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--singles", type=int, default=12, help="per GPU; 3 GPUs x (12 + 2x5) = 66 windows/step")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--blr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--rank-start", type=float, default=0.10)
    ap.add_argument("--rank-weight", type=float, default=0.1)
    ap.add_argument("--workers-single", type=int, default=6)
    ap.add_argument("--workers-pair", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--keep-ckpts", type=int, default=6)
    ap.add_argument("--val-pairs", type=int, default=768)
    ap.add_argument("--val-singles", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"train": train, "build-val": build_val, "eval": run_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
