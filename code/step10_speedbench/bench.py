"""FrameDrop-VQA model: end-to-end speed on 4K60 video (same measurements as step7_speed.py).

  model   GPU compute only: Stage 1 tiles + Stage 2 motion + fusion on a synthetic
          10 s 4K60 clip (600 frames, 40 tile frames x 32 tiles), host->device copies
          included, bf16.
  nvdec   NVDEC decode ceiling per clip: ffmpeg -hwaccel cuda, frames stay on the GPU.
  e2e     end to end on real 4K60 clips: decode (NVDEC if available, else CPU),
          384x216 downscale + 4K tile crops on the CPU, encoders + fusion on the GPU.
          stream: one clip at a time; parallel: decode in --workers processes, one GPU.

Weights don't change timing: without --weights the model uses random weights of the
final shape. With --weights DIR, DIR must hold s1.pt, s2.pt, fusion.pt.

    python bench.py --gpu 0 --clips-dir clips --workers 8 --threads 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_video_loader as vl  # noqa: E402
import step4_model as M1  # noqa: E402
import step5_model as M2  # noqa: E402
import step6_features as F6  # noqa: E402
import step6_model as S3  # noqa: E402


def sync(dev):
    torch.cuda.synchronize(dev)


def load_model(weights, dev):
    e1, e2, fu = M1.TileMAE("s"), M2.VideoMAE(), S3.Fusion()
    if weights:
        for m, f in ((e1, "s1.pt"), (e2, "s2.pt"), (fu, "fusion.pt")):
            m.load_state_dict(torch.load(os.path.join(weights, f), map_location="cpu")["model"])
    return e1.to(dev).eval(), e2.to(dev).eval(), fu.to(dev).eval()


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
    return dict(frames=T, seconds_median=t, fps=T / t, realtime_x=T / t / 60.0,
                peak_mem_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30)


def codec(path):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                           "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()


def bench_nvdec(path, gpu):
    t0 = time.time()
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-hwaccel", "cuda", "-hwaccel_device", str(gpu),
                        "-hwaccel_output_format", "cuda", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    t = time.time() - t0
    n = vl.probe(path)["frames"]
    return dict(clip=os.path.basename(path), codec=codec(path), frames=n, seconds=t, fps=n / t,
                ok=r.returncode == 0 and not r.stderr.strip(), stderr=r.stderr.strip()[-200:])


class Decoded(Dataset):
    def __init__(self, paths, gpu, threads, hw_codecs):
        self.paths, self.gpu, self.threads, self.hw = paths, gpu, threads, hw_codecs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        t0 = time.time()
        p = self.paths[i]
        pb = vl.probe(p)
        T = pb["frames"]
        tticks = np.arange(0, T, F6.TILE_EVERY)
        cells = vl.sample_cells(np.random.default_rng(i), len(tticks))
        c = codec(p)
        v = dict(path=p, w=pb["w"], h=pb["h"], codec=c)
        low, full = vl.decode_window(v, 0, T, list(tticks), threads=self.threads,
                                     hw_device=self.gpu if c in self.hw else None)
        view = vl.build_view(low, 0, full, np.arange(T), np.arange(T), cells, tticks, True)
        return view["low"], view["tiles"], cells, T, time.time() - t0


def bench_e2e(e1, e2, fu, dev, paths, gpu, workers, threads, hw):
    out = {}
    ds = Decoded(paths, gpu, threads, hw)
    frames, dec_t, tot_t = 0, 0.0, 0.0
    ds[0]                                            # warm-up (file cache, cuDNN)
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
    rep = paths * max(1, -(-3 * workers // len(paths)))           # enough clips to keep the workers busy
    dl = DataLoader(Decoded(rep, gpu, threads, hw), batch_size=None, num_workers=workers, prefetch_factor=2)
    frames, t0 = 0, None
    for k, (low, tiles, cells, T, _) in enumerate(dl):
        if k == 0:                                   # exclude worker start-up
            t0 = time.time()
            continue
        run_model(e1, e2, fu, low, tiles, cells, dev)
        frames += int(T)
    sync(dev)
    t = time.time() - t0
    out["parallel"] = dict(clips=len(rep) - 1, frames=frames, fps=frames / t, realtime_x=frames / t / 60.0,
                           decode_workers=workers, decode_threads=threads)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--clips-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-nvdec", action="store_true", help="CPU decode only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda", a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    e1, e2, fu = load_model(a.weights, dev)
    paths = sorted(glob.glob(os.path.join(a.clips_dir, "*.mp4")))
    res = dict(gpu=torch.cuda.get_device_name(dev), cpu=platform.processor() or platform.machine(),
               cpu_cores=os.cpu_count(), torch=torch.__version__,
               ffmpeg=subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0],
               weights=a.weights or "random (timing is weight-independent)")
    res["model"] = bench_model(e1, e2, fu, dev)
    print(json.dumps(res["model"]), flush=True)
    hw = set()
    if not a.no_nvdec:
        res["nvdec"] = [bench_nvdec(p, a.gpu) for p in paths]
        print(json.dumps(res["nvdec"]), flush=True)
        hw = {d["codec"] for d in res["nvdec"] if d["ok"]}
    res["nvdec_codecs_used"] = sorted(hw)
    res["e2e"] = bench_e2e(e1, e2, fu, dev, paths, a.gpu, a.workers, a.threads, hw)
    print(json.dumps(res["e2e"]), flush=True)
    out = a.out or f"speed_{res['gpu'].replace(' ', '_')}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
