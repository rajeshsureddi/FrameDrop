"""Step 9: end-to-end fine-tune on LSVQ train (model v3): Stage 1 + Stage 2 encoders + Stage 3 fusion.

User instruction 2026-10-17. v1 and v2 stay frozen (vqa_data/frozen_v{1,2}_20261017);
this step writes only to vqa_data/step9_e2e/.

Init: encoders from v1 (Stage 1 s_mae, Stage 2 v_mae), fusion from v2_mos_pairs (already
MOS-calibrated, so the encoders start under a sensible head: LP-FT order).
Training input (pixels only, as is: native resolution / frame rate):
  MOS    random window of <= 128 native frames of an LSVQ-train clip; tiles every 15th
         frame, a random subset of --tiles grid cells; L_mos = MSE(Q, z) + (1 - PLCC)
  pairs  synthetic pairs from the Step 2 data (step6_features.PairStream, T = 128) through a
         replay buffer (they decode slower than they are used); the same tile subset for
         both clips; L_rank + L_drop + 0.1 L_inv as in Stage 3
  every --pair-every-th step is a pair batch.
Optimiser: AdamW, cosine, 5% warmup; encoders with layer-wise lr decay (--llrd), fusion
--lr-fusion. Gradient checkpointing on every encoder block. Model = final step (fixed
schedule; no MOS holdout, all LSVQ train is used). A fixed 150-pair synthetic valid set
(raw pixels, step9_e2e/valid_pairs) tracks drop / rank accuracy during training.

    python step9_e2e.py make-valid
    python step9_e2e.py bench --clips 8
    python step9_e2e.py train --name v3_e2e
    python step9_e2e.py export --name v3_e2e      # -> s1.pt, s2.pt, fusion.pt for step7_eval
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_video_loader as vl  # noqa: E402
import step5_model as M2  # noqa: E402
import step6_features as F6  # noqa: E402
import step6_model as S3  # noqa: E402
import step6_train as T6  # noqa: E402
from step4_train import lr_at  # noqa: E402
from step7_eval import DATASETS, decodable_frames  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
OUT = f"{ROOT}/step9_e2e"
V1 = f"{ROOT}/frozen_v1_20261017"
V2_FUSION = f"{ROOT}/frozen_v2_20261017/stage3_fusion_v2_mos_pairs_final.pt"
MOS_MEAN, MOS_SD = 59.78, 10.67           # LSVQ train (as in v2)
T_TRAIN = 128


# ---------------------------------------------------------------- data
def lsvq_train_items():
    with open(DATASETS["lsvq_train"]["labels"]) as f:
        return [(os.path.join(DATASETS["lsvq_train"]["videos"], r[0].strip()), float(r[-1])) for r in csv.reader(f)]


def sub_tiles(tiles, cells, m, rng):
    """the same random m of the tile slots in every tile frame"""
    if tiles.shape[1] <= m:
        return tiles, cells
    pick = np.sort(rng.choice(tiles.shape[1], m, replace=False))
    return tiles[:, pick], cells[:, pick]


class LSVQWindows(IterableDataset):
    def __init__(self, items, n_tiles, seed):
        self.items, self.n_tiles, self.seed = items, n_tiles, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0, os.getpid()])
        while True:
            p, mos = self.items[rng.integers(len(self.items))]
            try:
                pb = vl.probe(p)
                n_all = pb["frames"]
                for attempt in range(2):
                    n = min(T_TRAIN, n_all)
                    n -= n % 2
                    if n < M2.T_WIN:
                        raise ValueError("too short")
                    st = int(rng.integers(0, n_all - n + 1))
                    try:
                        o = vl.mos_native(dict(path=p), start=st, n=n, tiles=True, max_tiles=None,
                                          tile_every_s=F6.TILE_EVERY / pb["fps"], seed=0, threads=2)
                        break
                    except RuntimeError:
                        if attempt:
                            raise
                        n_all = decodable_frames(p)       # container count > decodable frames
                xy = o["tile_xy"]
                cells = np.stack([np.minimum(8, xy[..., 1] // 238), np.minimum(14, xy[..., 0] // 256)], -1)
                tiles, cells = sub_tiles(o["tiles"], cells, self.n_tiles, rng)
                yield dict(low=o["low"], tiles=tiles, cells=cells.astype(np.int64), mos=mos)
            except Exception:
                continue


class PairBuffer:
    """Background thread fills a replay buffer from PairStream; batches sample from it."""

    def __init__(self, n_tiles, workers, cap, seed):
        self.buf = collections.deque(maxlen=cap)
        self.n_tiles, self.rng = n_tiles, np.random.default_rng(seed)
        self.lock, self.added = threading.Lock(), 0
        dl = DataLoader(F6.PairStream("train", 5000 + seed, T_TRAIN), batch_size=None, num_workers=workers,
                        prefetch_factor=2)
        threading.Thread(target=self._fill, args=(dl,), daemon=True).start()

    def _fill(self, dl):
        for s in dl:
            tiles = np.stack([np.asarray(s["better"]["tiles"]), np.asarray(s["worse"]["tiles"])])
            cells = np.asarray(s["cells"]).astype(np.int64)
            if tiles.shape[2] > self.n_tiles:
                pick = np.sort(self.rng.choice(tiles.shape[2], self.n_tiles, replace=False))
                tiles, cells = tiles[:, :, pick], cells[:, pick]
            item = dict(low=np.stack([np.asarray(s["better"]["low"]), np.asarray(s["worse"]["low"])]),
                        tiles=tiles, cells=cells, ptype=s["ptype"])
            with self.lock:
                self.buf.append(item)
                self.added += 1

    def ready(self, n):
        return len(self.buf) >= n

    def batch(self, n_pairs):
        with self.lock:
            idx = self.rng.choice(len(self.buf), n_pairs, replace=False)
            return [self.buf[i] for i in idx]


# ---------------------------------------------------------------- model
def ckpt_blocks(blocks, x):
    for b in blocks:
        x = checkpoint(b, x, use_reentrant=False)
    return x


def tile_features(e1, tiles_u8):
    """(N, 256, 256, 3) uint8 -> (N, d), with grad."""
    xn = e1.normalize(tiles_u8.permute(0, 3, 1, 2))
    x = e1.patch_embed.proj(xn).flatten(2).transpose(1, 2) + e1.pos_embed[:, 1:]
    cls = (e1.cls_token + e1.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
    x = e1.norm(ckpt_blocks(e1.blocks, torch.cat([cls, x], 1)))
    return e1.pooled(x)


def motion_features(e2, chunks_u8):
    """(C, 32, 216, 384, 3) uint8 -> (C, 16, 12, d), with grad."""
    xn = e2.normalize(chunks_u8)
    x = e2.patch_embed(xn).flatten(2).transpose(1, 2) + e2.pos
    lat = e2.norm(ckpt_blocks(e2.blocks, x))
    g = lat.view(lat.shape[0], M2.GT, M2.GH, M2.GW, -1).permute(0, 1, 4, 2, 3).flatten(0, 1)
    r = F.adaptive_avg_pool2d(g, (3, 4))
    return r.view(lat.shape[0], M2.GT, -1, 12).permute(0, 1, 3, 2)


class E2E(torch.nn.Module):
    def __init__(self, s1, s2, fusion):
        super().__init__()
        dev = torch.device("cpu")
        self.e1, self.e2 = F6.load_encoders(s1, s2, dev)
        for e in (self.e1, self.e2):                  # MAE decoders / score heads are not used
            for n_, p in e.named_parameters():
                p.requires_grad_(n_.split(".")[0] in ("patch_embed", "cls_token", "pos_embed", "blocks", "norm"))
        self.fusion = S3.Fusion()
        self.fusion.load_state_dict(torch.load(fusion, map_location="cpu")["model"])

    def clips(self, lows, tiles, cells):
        """lists over clips: low (T, 216, 384, 3) u8, tiles (K, M, 256, 256, 3) u8, cells (K, M, 2)
        (all on device) -> Q, V (n,)"""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            n_t = [t.shape[0] * t.shape[1] for t in tiles]
            sp_all = tile_features(self.e1, torch.cat([t.flatten(0, 1) for t in tiles])).split(n_t)
            starts = [F6.chunk_starts(lo.shape[0]) for lo in lows]
            ch = torch.cat([torch.stack([lo[s:s + M2.T_WIN] for s in st]) for lo, st in zip(lows, starts)])
            mo_all = motion_features(self.e2, ch).split([len(s) for s in starts])
            Qs, Vs = [], []
            for lo, t, ce, sp, f, st in zip(lows, tiles, cells, sp_all, mo_all, starts):
                T = lo.shape[0]
                motion = f.new_zeros(T // 2, 12, f.shape[-1])
                for c, s in enumerate(st):
                    motion[s // 2:s // 2 + M2.GT] = f[c]          # later chunks overwrite the overlap
                K = min(t.shape[0], -(-T // F6.TILE_EVERY))
                Q, V, _, _ = self.fusion(sp.view(t.shape[0], t.shape[1], -1)[None, :K].float(), ce[None, :K],
                                         motion[None].float())
                Qs.append(Q[0])
                Vs.append(V[0])
        return torch.stack(Qs).float(), torch.stack(Vs).float()


def param_groups(model, lr_enc, lr_fusion, llrd, wd):
    groups = []
    for e in (model.e1, model.e2):
        depth = len(e.blocks)
        for n_, p in e.named_parameters():
            if not p.requires_grad:
                continue
            top = n_.split(".")[0]
            lid = int(n_.split(".")[1]) + 1 if top == "blocks" else (depth + 1 if top == "norm" else 0)
            no_wd = p.ndim == 1 or n_.endswith(".bias") or top in ("cls_token", "pos_embed")
            groups.append(dict(params=[p], lr_base=lr_enc * llrd ** (depth + 1 - lid), weight_decay=0.0 if no_wd else wd))
    for n_, p in model.fusion.named_parameters():
        no_wd = p.ndim <= 1 or n_.endswith(".bias")
        groups.append(dict(params=[p], lr_base=lr_fusion, weight_decay=0.0 if no_wd else wd))
    return groups


def to_dev(x, dev):
    return torch.as_tensor(np.asarray(x)).to(dev, non_blocking=True)


def mos_step(model, items, dev):
    Q, _ = model.clips([to_dev(s["low"], dev) for s in items], [to_dev(s["tiles"], dev) for s in items],
                       [to_dev(s["cells"], dev) for s in items])
    z = (torch.tensor([s["mos"] for s in items], device=dev, dtype=torch.float32) - MOS_MEAN) / MOS_SD
    qc, zc = Q - Q.mean(), z - z.mean()
    plcc = (qc * zc).sum() / (qc.norm() * zc.norm() + 1e-8)
    mse = ((Q - z) ** 2).mean()
    return mse + (1 - plcc), dict(l_mse=float(mse), plcc_batch=float(plcc))


def pair_forward(model, pairs, dev):
    lows, tiles, cells = [], [], []
    for p in pairs:
        for k in range(2):
            lows.append(to_dev(p["low"][k], dev))
            tiles.append(to_dev(p["tiles"][k], dev))
            cells.append(to_dev(p["cells"], dev))
    Q, V = model.clips(lows, tiles, cells)
    return Q.view(-1, 2), V.view(-1, 2)


# ---------------------------------------------------------------- valid pairs (raw pixels, fixed)
def make_valid(a):
    od = f"{OUT}/valid_pairs"
    os.makedirs(od, exist_ok=True)
    have = len(glob.glob(f"{od}/pair_*.npz"))
    want = {"drops": 50, "rate": 50, "severity": 50}
    for f in glob.glob(f"{od}/pair_*.npz"):
        want[str(np.load(f)["ptype"])] -= 1
    rng = np.random.default_rng(1)
    dl = DataLoader(F6.PairStream("valid", 777, T_TRAIN), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    k = have
    for s in dl:
        if want.get(s["ptype"], 0) <= 0:
            if sum(want.values()) == 0:
                break
            continue
        want[s["ptype"]] -= 1
        tiles = np.stack([np.asarray(s["better"]["tiles"]), np.asarray(s["worse"]["tiles"])])
        cells = np.asarray(s["cells"]).astype(np.int64)
        pick = np.sort(rng.choice(tiles.shape[2], a.tiles, replace=False))
        np.savez(f"{od}/pair_{k:04d}.npz", low=np.stack([np.asarray(s["better"]["low"]), np.asarray(s["worse"]["low"])]),
                 tiles=tiles[:, :, pick], cells=cells[:, pick], ptype=s["ptype"])
        k += 1
        print(k, s["ptype"], flush=True)
    print("valid pairs:", k, flush=True)


@torch.no_grad()
def eval_pairs(model, files, dev):
    model.eval()
    Qs, Vs, t = [], [], []
    for f in files:
        z = np.load(f)
        Q, V = pair_forward(model, [dict(low=z["low"], tiles=z["tiles"], cells=z["cells"])], dev)
        Qs.append(Q[0].cpu().numpy())
        Vs.append(V[0].cpu().numpy())
        t.append(str(z["ptype"]))
    model.train()
    Q, V, t = np.array(Qs), np.array(Vs), np.array(t)
    r = {f"q_acc_{k}": float((Q[t == k, 0] > Q[t == k, 1]).mean()) for k in T6.TYPES if (t == k).any()}
    dm = t == "drops"
    r["v_acc_drops"] = float((V[dm, 1] > V[dm, 0]).mean())
    r["v_absdiff_rate_sev"] = float(np.abs(V[~dm, 0] - V[~dm, 1]).mean())
    return r


# ---------------------------------------------------------------- train
def build(a, dev):
    model = E2E(f"{V1}/stage1_s_mae_final_avg4.pt", f"{V1}/stage2_v_mae_final_avg4.pt", a.init_fusion).to(dev)
    model.train()
    return model


def bench(a):
    dev = torch.device("cuda", a.gpu)
    model = build(a, dev)
    opt = torch.optim.AdamW([g for g in param_groups(model, 1e-5, 1e-5, a.llrd, 0.05)], lr=1e-5)
    dl = iter(DataLoader(LSVQWindows(lsvq_train_items(), a.tiles, 0), batch_size=None, num_workers=a.workers))
    items = [next(dl) for _ in range(a.clips)]
    for it in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        loss, _ = mos_step(model, items, dev)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        print(f"step {it}: {time.time() - t0:.2f}s  peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
              f"(clips {a.clips}, frames {[s['low'].shape[0] for s in items]}, tiles {items[0]['tiles'].shape[:2]})", flush=True)


def train(a):
    dev = torch.device("cuda", a.gpu)
    torch.manual_seed(a.seed)
    run_dir = f"{OUT}/runs/{a.name}"
    os.makedirs(run_dir, exist_ok=True)
    model = build(a, dev)
    groups = param_groups(model, a.lr_enc, a.lr_fusion, a.llrd, a.wd)
    opt = torch.optim.AdamW(groups, lr=a.lr_fusion, betas=(0.9, 0.999))
    step0 = 0
    if os.path.exists(f"{run_dir}/last.pt"):
        ck = torch.load(f"{run_dir}/last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        print(f"resumed at step {step0}", flush=True)
    items = lsvq_train_items()
    mos_dl = iter(DataLoader(LSVQWindows(items, a.tiles, a.seed + step0), batch_size=None, num_workers=a.workers,
                             prefetch_factor=3))
    pairs = PairBuffer(a.tiles, a.pair_workers, a.buffer, a.seed + step0) if a.pair_every else None
    vfiles = sorted(glob.glob(f"{OUT}/valid_pairs/pair_*.npz"))
    json.dump(dict(args=vars(a), lsvq_train=len(items), valid_pairs=len(vfiles)), open(f"{run_dir}/config.json", "w"), indent=1)
    log = open(f"{run_dir}/train_log.jsonl", "a")
    if step0 == 0 and vfiles:
        r = dict(eval_pairs(model, vfiles, dev), step=0, kind="val")
        log.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
    if pairs:
        while not pairs.ready(a.pairs * 4):
            time.sleep(5)
    warm = int(0.05 * a.steps)
    acc, n, t0 = defaultdict(float), defaultdict(int), time.time()
    for step in range(step0 + 1, a.steps + 1):
        f = lr_at(step, a.steps, 1.0, warm)
        for g in opt.param_groups:
            g["lr"] = g["lr_base"] * f
        if pairs and step % a.pair_every == 0:
            pb = pairs.batch(a.pairs)
            Q, V = pair_forward(model, pb, dev)
            loss, parts = T6.losses(Q, V, [p["ptype"] for p in pb], dev)
            parts["q_acc_train"] = float((Q[:, 0] > Q[:, 1]).float().mean())
            kind = "pairs"
        else:
            loss, parts = mos_step(model, [next(mos_dl) for _ in range(a.clips)], dev)
            kind = "mos"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        for k, v in parts.items():
            acc[k] += v
            n[k] += 1
        acc[f"loss_{kind}"] += float(loss)
        n[f"loss_{kind}"] += 1
        if step % a.log_every == 0:
            rec = dict(step=step, lr_fusion=a.lr_fusion * f, **{k: v / n[k] for k, v in acc.items()},
                       steps_per_s=a.log_every / (time.time() - t0), time=time.strftime("%m-%d %H:%M:%S"))
            if pairs:
                rec.update(buffer=len(pairs.buf), pairs_added=pairs.added)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            acc, n, t0 = defaultdict(float), defaultdict(int), time.time()
        if step % a.val_every == 0 or step == a.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), f"{run_dir}/last.pt.tmp")
            os.replace(f"{run_dir}/last.pt.tmp", f"{run_dir}/last.pt")
            if vfiles:
                r = dict(eval_pairs(model, vfiles, dev), step=step, kind="val")
                log.write(json.dumps(r) + "\n")
                log.flush()
                print(json.dumps(r), flush=True)
    torch.save(dict(model=model.state_dict(), step=a.steps), f"{run_dir}/final.pt")
    print("FINISHED", flush=True)


def export(a):
    """final.pt -> step7_eval-compatible checkpoints {"model": state_dict} for s1 / s2 / fusion."""
    run_dir = f"{OUT}/runs/{a.name}"
    sd = torch.load(f"{run_dir}/{a.ckpt}", map_location="cpu")["model"]
    for pre, name in (("e1.", "s1.pt"), ("e2.", "s2.pt"), ("fusion.", "fusion.pt")):
        torch.save(dict(model={k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}), f"{run_dir}/{name}")
    print("exported", run_dir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-valid", "bench", "train", "export"])
    ap.add_argument("--name", default="v3_e2e")
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--init-fusion", default=V2_FUSION)
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--clips", type=int, default=8, help="LSVQ windows per MOS batch")
    ap.add_argument("--pairs", type=int, default=4, help="synthetic pairs per pair batch")
    ap.add_argument("--pair-every", type=int, default=4, help="every k-th step is a pair batch; 0 = MOS only")
    ap.add_argument("--tiles", type=int, default=32, help="tile slots per tile frame in training")
    ap.add_argument("--lr-enc", type=float, default=2e-5, help="top encoder block; lower blocks x llrd^depth")
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
    a = ap.parse_args()
    {"make-valid": make_valid, "bench": bench, "train": train, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
