"""Step 3 review: loader throughput, GPU utilisation with stand-in Stage 1/2
models, and DINOv2 batch diversity.

  python step3_bench.py tiles  --workers 32 --seconds 120 [--gpu-model]
  python step3_bench.py video  --workers 32 --seconds 180 --what stage3|stage2|pairs [--nvdec 0,1,2] [--gpu-model]
  python step3_bench.py diversity --workers 32 --batches 20

The stand-in models only reproduce the Step 4/5 tensor shapes and FLOPs
(ViT-S encoders, MAE masking, decoders); they are not trained here.
Results are appended as JSON lines to <out>/bench.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_tile_loader as tl  # noqa: E402
import step3_video_loader as vl  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
TILES = f"{ROOT}/step3/tiles"
MANIFEST = f"{ROOT}/step2/encodes_manifest.csv"
PLANS = f"{ROOT}/step3/capture_plans.json"
STEP1 = f"{ROOT}/step1/manifest.csv"
LOWRES = f"{ROOT}/step3/lowres"
OUT = f"{ROOT}/step3/review"


# ------------------------------------------------------------------ stand-in models
def blocks(dim, depth, heads):
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class TileMAE(nn.Module):
    """Step 4 shapes: 256x256 tiles, 16x16 patches, 75% mask, ViT-S enc, dec 256x4."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(768, 384)
        self.pos = nn.Parameter(torch.zeros(1, 256, 384))
        self.enc = blocks(384, 12, 6)
        self.dec_in = nn.Linear(384, 256)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, 256))
        self.dpos = nn.Parameter(torch.zeros(1, 256, 256))
        self.dec = blocks(256, 4, 8)
        self.head = nn.Linear(256, 774)
        self.score = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):                      # x float (N,3,256,256)
        N = x.shape[0]
        p = x.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5).reshape(N, 256, 768)
        keep = torch.rand(N // 2, 256, device=x.device).argsort(1)[:, :64].repeat_interleave(2, 0)  # same mask per pair
        t = self.embed(p) + self.pos
        t = torch.gather(t, 1, keep[..., None].expand(-1, -1, 384))
        h = self.enc(t)
        d = self.mask_tok.expand(N, 256, 256).clone()
        e = self.dec_in(h)
        d = d.to(e.dtype).scatter(1, keep[..., None].expand(-1, -1, 256), e)
        rec = self.head(self.dec(d + self.dpos))
        loss = F.mse_loss(rec[..., :768], p)
        s = self.score(h.mean(1))
        return loss + 0.1 * F.relu(0.1 - (s[0::2] - s[1::2])).mean()


class VideoMAE(nn.Module):
    """Step 5 shapes: 32x216x384, tubelets 2x24x24 -> 2304 tokens, 576 visible."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3456, 384)
        self.pos = nn.Parameter(torch.zeros(1, 2304, 384))
        self.enc = blocks(384, 12, 6)
        self.dec_in = nn.Linear(384, 192)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, 192))
        self.dec = blocks(192, 4, 6)
        self.head = nn.Linear(192, 3462)

    def forward(self, x):                      # x float (N,3,32,216,384)
        N = x.shape[0]
        p = x.reshape(N, 3, 16, 2, 9, 24, 16, 24).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(N, 2304, 3456)
        keep = torch.rand(N, 2304, device=x.device).argsort(1)[:, :576]
        t = torch.gather(self.embed(p) + self.pos, 1, keep[..., None].expand(-1, -1, 384))
        h = self.enc(t)
        d = self.mask_tok.expand(N, 2304, 192).clone()
        e = self.dec_in(h)
        d = d.to(e.dtype).scatter(1, keep[..., None].expand(-1, -1, 192), e)
        rec = self.head(self.dec(d))
        return F.mse_loss(rec[..., :3456], p)


class GpuUtil(threading.Thread):
    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu, self.samples, self.stop = gpu, [], False

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                self.samples.append(float(o.stdout.strip()))
            except Exception:
                pass
            time.sleep(0.5)


def train_step_fn(kind, gpu):
    dev = torch.device(f"cuda:{gpu}")
    model = (TileMAE() if kind == "tile" else VideoMAE()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step(x_uint8):
        x = x_uint8.to(dev, non_blocking=True).float().div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss
    return step, dev


def pure_gpu_rate(kind, gpu, B, n=20):
    step, dev = train_step_fn(kind, gpu)
    shape = (2 * B, 3, 256, 256) if kind == "tile" else (B, 3, 32, 216, 384)
    x = torch.randint(0, 255, shape, dtype=torch.uint8).pin_memory()
    for _ in range(3):
        step(x)
    torch.cuda.synchronize(dev)
    t0 = time.time()
    for _ in range(n):
        step(x)
    torch.cuda.synchronize(dev)
    return n / (time.time() - t0)


def log(rec):
    os.makedirs(OUT, exist_ok=True)
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rec["host"] = os.uname().nodename
    print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, "bench.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ tiles
def bench_tiles(a):
    shards = tl.list_shards(TILES, ("train",))
    ds = tl.TileDataset(TILES, shards=shards, items_per_group=a.items_per_group,
                        open_per_worker=a.open_per_worker, seed=a.seed)
    dl = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=8,
                    persistent_workers=False)
    bt = tl.DiverseBatcher(dl, a.batch, max_per_source=2)
    step = None
    if a.gpu_model:
        pure = pure_gpu_rate("tile", a.gpu, a.batch)
        step, _ = train_step_fn("tile", a.gpu)
        mon = GpuUtil(a.gpu)
    n, t0, per_src, nsrc, ptypes = 0, None, [], [], Counter()
    for b in bt:
        if t0 is None:                          # skip warm-up batch
            t0 = time.time()
            if step:
                mon.start()
            continue
        c = Counter(b["source_id"])
        per_src.append(max(c.values()))
        nsrc.append(len(c))
        ptypes.update(b["ptype"])
        if step:
            step(b["x"].reshape(-1, 3, 256, 256))
        n += 1
        if time.time() - t0 > a.seconds:
            break
    if step:
        torch.cuda.synchronize()
        mon.stop = True
    dt = time.time() - t0
    rec = dict(bench="tiles", workers=a.workers, batch_pairs=a.batch, tiles_per_batch=2 * a.batch,
               shards=len(shards), batches=n, batches_per_s=n / dt, tiles_per_s=2 * a.batch * n / dt,
               max_per_source=max(per_src), mean_distinct_sources=float(np.mean(nsrc)),
               ptypes={k: v / sum(ptypes.values()) for k, v in ptypes.items()})
    if step:
        rec.update(gpu_util_mean=float(np.mean(mon.samples)) if mon.samples else None,
                   pure_gpu_steps_per_s=pure, gpu_model="TileMAE stand-in (ViT-S/16, 75% mask)")
    log(rec)


# ------------------------------------------------------------------ video
def video_collate(batch):
    return batch


def bench_video(a):
    gpus = [int(g) for g in a.nvdec.split(",")] if a.nvdec else None
    if a.what == "stage3":
        ds = vl.VideoDataset(MANIFEST, mode="single", tiles=True, plans_json=PLANS, seed=a.seed,
                             hwaccel_gpus=gpus, threads=a.threads)
    elif a.what == "pairs":
        ds = vl.VideoDataset(MANIFEST, mode="pair", tiles=True, plans_json=PLANS, seed=a.seed,
                             hwaccel_gpus=gpus, threads=a.threads)
    else:
        base = vl.VideoDataset(MANIFEST, mode="single", tiles=False, p_none=0.5, plans_json=PLANS,
                               seed=a.seed, hwaccel_gpus=gpus, threads=a.threads, step1_manifest=STEP1,
                               lowres_dir=None if a.no_cache else LOWRES)
        ds = vl.SubWindows(base, 32)
    dl = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=2)
    step = None
    if a.gpu_model and a.what == "stage2":
        pure = pure_gpu_rate("video", a.gpu, a.batch)
        step, _ = train_step_fn("video", a.gpu)
        mon = GpuUtil(a.gpu)
    n, t0, buf, nb, src = 0, None, [], 0, Counter()
    for s in dl:
        if t0 is None:
            t0 = time.time()
            if step:
                mon.start()
            continue
        n += 1
        src[s["source_id"]] += 1
        if step:
            buf.append(s["low"])
            if len(buf) == a.batch:
                x = torch.from_numpy(np.stack(buf)).permute(0, 4, 1, 2, 3)
                step(x)
                nb += 1
                buf = []
        if time.time() - t0 > a.seconds:
            break
    if step:
        torch.cuda.synchronize()
        mon.stop = True
    dt = time.time() - t0
    rec = dict(bench=f"video_{a.what}", lowres_cache=(a.what == "stage2" and not a.no_cache), workers=a.workers, nvdec=a.nvdec, ffmpeg_threads=a.threads,
               samples=n, samples_per_s=n / dt, distinct_sources=len(src))
    if step:
        rec.update(batch=a.batch, train_steps=nb, steps_per_s=nb / dt, pure_gpu_steps_per_s=pure,
                   gpu_util_mean=float(np.mean(mon.samples)) if mon.samples else None,
                   gpu_model="VideoMAE stand-in (ViT-S, 2x24x24 tubelets, 576 visible)")
    log(rec)


# ------------------------------------------------------------------ diversity
def dino(dev):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]

    @torch.no_grad()
    def feats(x_uint8):                         # (N,3,H,W) uint8
        x = x_uint8.to(dev).float().div(255)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        f = m(pixel_values=(x - mean) / std).pooler_output
        return F.normalize(f, dim=-1)
    return feats


def mean_pairwise_cos(f):
    s = f @ f.T
    n = f.shape[0]
    return float((s.sum() - s.diagonal().sum()) / (n * (n - 1)))


def bench_diversity(a):
    dev = torch.device(f"cuda:{a.gpu}")
    feats = dino(dev)
    shards = tl.list_shards(TILES, ("train",))
    res = {}
    # (1) DiverseBatcher batches (what training sees); first tile of each pair
    ds = tl.TileDataset(TILES, shards=shards, items_per_group=a.items_per_group,
                        open_per_worker=a.open_per_worker, seed=a.seed + 1)
    dl = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=8)
    vals, nsrc = [], []
    for k, b in enumerate(tl.DiverseBatcher(dl, a.batch, max_per_source=2)):
        vals.append(mean_pairwise_cos(feats(b["x"][:, 0])))
        nsrc.append(len(set(b["source_id"])))
        if k + 1 >= a.batches:
            break
    res["tiles_diverse_batcher"] = dict(mean_cos=float(np.mean(vals)), std=float(np.std(vals)),
                                        distinct_sources=float(np.mean(nsrc)))
    # (2) naive: consecutive samples from ONE worker reading 2 tars (no batcher)
    ds2 = tl.TileDataset(TILES, shards=shards, items_per_group=a.items_per_group, open_per_worker=2,
                         seed=a.seed + 2)
    it, vals = iter(ds2), []
    for k in range(min(a.batches, 5)):
        b = tl.collate([next(it) for _ in range(a.batch)])
        vals.append(mean_pairwise_cos(feats(b["x"][:, 0])))
    res["tiles_naive_2_sources"] = dict(mean_cos=float(np.mean(vals)), std=float(np.std(vals)))
    # (3) reference: one tile from each of `batch` different random sources
    rng = np.random.default_rng(a.seed)
    vals = []
    for k in range(min(a.batches, 5)):
        xs = []
        for sh in rng.choice(shards, a.batch, replace=False):
            meta, pngs = next(tl.iter_groups(sh))
            xs.append(tl.center(tl.decode_png(pngs[sorted(pngs)[0]])))
        vals.append(mean_pairwise_cos(feats(torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2))))
    res["tiles_one_per_source_reference"] = dict(mean_cos=float(np.mean(vals)), std=float(np.std(vals)))
    # (4) video windows (Stage 2/3): middle displayed frame of each window
    vds = vl.VideoDataset(MANIFEST, mode="single", tiles=False, plans_json=PLANS, seed=a.seed)
    vdl = DataLoader(vds, batch_size=None, num_workers=a.workers, prefetch_factor=2)
    vals, cur = [], []
    for s in vdl:
        cur.append(torch.from_numpy(s["low"][len(s["low"]) // 2]).permute(2, 0, 1))
        if len(cur) == a.video_batch:
            vals.append(mean_pairwise_cos(feats(torch.stack(cur))))
            cur = []
            if len(vals) >= a.batches:
                break
    res["video_windows"] = dict(mean_cos=float(np.mean(vals)), std=float(np.std(vals)),
                                batch=a.video_batch)
    log(dict(bench="diversity_dinov2_small", batch=a.batch, batches=a.batches, **res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what_bench", choices=["tiles", "video", "diversity"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--batch", type=int, default=256, help="tile pairs (512 tiles) / video windows")
    ap.add_argument("--video-batch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--items-per-group", type=int, default=4)
    ap.add_argument("--open-per-worker", type=int, default=8)
    ap.add_argument("--what", default="stage3", choices=["stage3", "stage2", "pairs"])
    ap.add_argument("--nvdec", default="")
    ap.add_argument("--no-cache", action="store_true", help="stage2: decode 4K on the fly instead of the low-res cache")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu-model", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"tiles": bench_tiles, "video": bench_video, "diversity": bench_diversity}[a.what_bench](a)


if __name__ == "__main__":
    main()
