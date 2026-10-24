"""Step 7: speed of the frozen model on 4K60 video (METHOD_PLAN "Speed": measure, don't infer).

  model   GPU compute only: Stage 1 tiles + Stage 2 motion + fusion for a
          synthetic 10 s 4K60 clip (600 frames, 40 tile frames x 32 tiles),
          host->device copies included, bf16.
  nvdec   NVDEC decode ceiling: ffmpeg -hwaccel cuda, frames kept on the GPU,
          no download (-f null).
  e2e     end to end on real 4K60 clips with the current pipeline: NVDEC decode,
          384x216 downscale + 4K tile crops on the CPU, encoders + fusion on the GPU.
          stream: one clip at a time (single-stream fps);
          parallel: decode in `--workers` processes, one GPU (throughput fps).
Weights don't change timing; fusion uses a fresh Fusion() of the final shape.

    python step7_speed.py --gpu 1 --clips 8 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_video_loader as vl  # noqa: E402
import step6_features as F6  # noqa: E402
import step6_model as S3  # noqa: E402
from step7_eval import OUT, clip_list  # noqa: E402

S1 = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step4/runs/s_mae/final_avg4.pt"
S2 = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step5/runs/v_mae/final_avg4.pt"


def sync(dev):
    torch.cuda.synchronize(dev)


@torch.no_grad()
def run_model(e1, e2, fu, low, tiles, cells, dev):
    sp, mo = F6.encode_view(e1, e2, low, tiles, dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Q, V, _, _ = fu(torch.from_numpy(sp).float()[None].to(dev), torch.as_tensor(cells, dtype=torch.long)[None].to(dev),
                        torch.from_numpy(mo).float()[None].to(dev))
    return float(Q), float(V)


def bench_model(e1, e2, fu, dev, T=600, reps=5):
    rng = np.random.default_rng(0)
    K = T // F6.TILE_EVERY
    low = rng.integers(0, 256, (T, 216, 384, 3), dtype=np.uint8)
    tiles = rng.integers(0, 256, (K, 32, 256, 256, 3), dtype=np.uint8)
    cells = vl.sample_cells(rng, K)
    run_model(e1, e2, fu, low, tiles, cells, dev)
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        run_model(e1, e2, fu, low, tiles, cells, dev)
        sync(dev)
        ts.append(time.time() - t0)
    t = float(np.median(ts))
    return dict(frames=T, seconds_median=t, fps=T / t, realtime_x=T / t / 60.0)


def bench_nvdec(path, gpu):
    t0 = time.time()
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-hwaccel", "cuda", "-hwaccel_device", str(gpu),
                    "-hwaccel_output_format", "cuda", "-i", path, "-f", "null", "-"], check=True)
    t = time.time() - t0
    n = vl.probe(path)["frames"]
    return dict(clip=os.path.basename(path), frames=n, seconds=t, fps=n / t)


class Decoded(Dataset):
    def __init__(self, paths, gpu, threads):
        self.paths, self.gpu, self.threads = paths, gpu, threads

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        t0 = time.time()
        p = self.paths[i]
        pb = vl.probe(p)
        T = pb["frames"]
        tticks = np.arange(0, T, F6.TILE_EVERY)
        cells = vl.sample_cells(np.random.default_rng(i), len(tticks))
        v = dict(path=p, w=pb["w"], h=pb["h"], codec="hevc")
        low, full = vl.decode_window(v, 0, T, list(tticks), threads=self.threads, hw_device=self.gpu)
        view = vl.build_view(low, 0, full, np.arange(T), np.arange(T), cells, tticks, True)
        return view["low"], view["tiles"], cells, T, time.time() - t0


def bench_e2e(e1, e2, fu, dev, paths, gpu, workers, threads):
    out = {}
    # single stream: decode -> model, one clip at a time
    ds = Decoded(paths, gpu, threads)
    frames, dec_t, tot_t = 0, 0.0, 0.0
    for i in range(len(paths)):
        t0 = time.time()
        low, tiles, cells, T, td = ds[i]
        run_model(e1, e2, fu, low, tiles, cells, dev)
        sync(dev)
        tot_t += time.time() - t0
        dec_t += td
        frames += T
    out["stream"] = dict(clips=len(paths), frames=frames, fps=frames / tot_t, decode_share=dec_t / tot_t,
                         realtime_x=frames / tot_t / 60.0, decode_threads=threads)
    # throughput: decode in parallel processes, one GPU
    dl = DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=2)
    frames, t0 = 0, None
    for k, (low, tiles, cells, T, _) in enumerate(dl):
        if k == 0:                                   # exclude worker start-up
            t0 = time.time()
            continue
        run_model(e1, e2, fu, low, tiles, cells, dev)
        frames += int(T)
    sync(dev)
    t = time.time() - t0
    out["parallel"] = dict(clips=len(paths) - 1, frames=frames, fps=frames / t, realtime_x=frames / t / 60.0,
                           decode_workers=workers, decode_threads=threads)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda", a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    e1, e2 = F6.load_encoders(S1, S2, dev)
    fu = S3.Fusion().to(dev).eval()
    paths = clip_list("study")
    paths = [paths[i] for i in np.random.default_rng(0).choice(len(paths), a.clips, replace=False)]
    res = dict(gpu=torch.cuda.get_device_name(dev), cpu_cores=os.cpu_count(),
               note="pipeline: NVDEC decode, CPU downscale/crop, GPU encoders+fusion; fusion training ran on another GPU of the node")
    res["model"] = bench_model(e1, e2, fu, dev)
    print(json.dumps(res["model"]), flush=True)
    res["nvdec"] = [bench_nvdec(p, a.gpu) for p in paths[:3]]
    print(json.dumps(res["nvdec"]), flush=True)
    res["e2e"] = bench_e2e(e1, e2, fu, dev, paths, a.gpu, a.workers, a.threads)
    print(json.dumps(res["e2e"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/speed.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
