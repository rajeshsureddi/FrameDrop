"""Step 7: evaluation of the frozen final model (study = test, zero-shot).

  extract  frozen Stage 1/2 tokens for every clip of a dataset (pixels only,
           display order), one npz per clip
             study     FrameDrop-VQA study clips (4K60, as displayed)
             konvid1k  public MOS, as is (native resolution and frame rate:
                       native frames are the display ticks)
  score    Stage 3 fusion -> Q, V per clip (+ pooled features for Protocol A)
  metrics  Protocol B: Q vs MOS (SROCC, KROCC, PLCC, RMSE after 4-parameter
           logistic; also unmapped), V vs mean visibility (SROCC, AUC for
           noticed = mean >= 0.5), per-subset results.
           Protocol A: ridge / SVR on frozen pooled features, grouped folds
           (median of 20 random 80/20 splits grouped by source content).
The study's ground-truth file is read only here, after the model is frozen.

    python step7_eval.py extract --dataset study --gpu 0 --part 0 --parts 3
    python step7_eval.py score --dataset study --fusion runs/fusion/best.pt
    python step7_eval.py metrics --dataset study
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
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

BASE = "/scratch/09997/rs64952/FrameDrop-VQA"
ROOT = f"{BASE}/vqa_data"
OUT = f"{ROOT}/step7"
STUDY = f"{BASE}/mos_datasets/framedrop_vqa_subjective_study"
CELL_SEED = 2026                                  # same tile cells for every clip of a given length
DATASETS = dict(
    study=dict(videos=f"{STUDY}/mp4_for_study_crf10", labels=f"{STUDY}/ground_truth/mos_visibility_per_video.csv"),
    # Step 18: the same 4K study files, downscaled x2 (1920x1080 bicubic) inside the decode (no re-encode), native path
    study_half=dict(videos=os.environ.get("STEP18_VIDEOS", f"{STUDY}/mp4_for_study_crf10"),   # env: node-local copy labels=f"{STUDY}/ground_truth/mos_visibility_per_video.csv",
                    downscale=2),
    study_quarter=dict(videos=os.environ.get("STEP18_VIDEOS", f"{STUDY}/mp4_for_study_crf10"),   # x4: 960x540
                       labels=f"{STUDY}/ground_truth/mos_visibility_per_video.csv", downscale=4),
    konvid1k=dict(videos=f"{BASE}/mos_datasets/KonViD",
                  labels=f"{ROOT}/eval/labels/konvid1k/labels_FAST-VQA-and-FasterVQA.txt"),
    # LIVE-VQC (585 clips, MOS 0-100; labels from the FAST-VQA repo; videos from the official LIVE release). Eval only.
    livevqc=dict(videos=f"{BASE}/mos_datasets/LIVE-VQC", labels=f"{ROOT}/eval/labels/livevqc/labels_FAST-VQA-and-FasterVQA.txt"),
    # YouTube-UGC (10-21): CRF-10 H.264 versions from gs://ugc-dataset, 977 clips with "MOS full" (1-5). Step 14 only.
    ytugc=dict(videos=f"{BASE}/mos_datasets/YouTube-UGC", labels=f"{ROOT}/eval/labels/ytugc/labels.txt"),
    # LSVQ (labels from the FAST-VQA/DOVER repos; videos from the HF mirror teowu/LSVQ-videos).
    # lsvq_train is used for the step 8 fine-tune only; test / test_1080p are evaluation-only.
    lsvq_train=dict(videos=f"{BASE}/mos_datasets/LSVQ_PatchVQ/videos", labels=f"{ROOT}/eval/labels/lsvq/train_labels.txt"),
    lsvq_test=dict(videos=f"{BASE}/mos_datasets/LSVQ_PatchVQ/videos", labels=f"{ROOT}/eval/labels/lsvq/labels_test.txt"),
    lsvq_1080p=dict(videos=f"{BASE}/mos_datasets/LSVQ_PatchVQ/videos", labels=f"{ROOT}/eval/labels/lsvq/labels_1080p.txt"),
    # Step 10: 10% of LSVQ train held out (seed 0) for the 10b selection; never trained on by v2_90 / step10 candidates
    lsvq_val10=dict(videos=f"{BASE}/mos_datasets/LSVQ_PatchVQ/videos", labels=f"{ROOT}/eval/labels/lsvq/val10_seed0.txt"),
)

for _d in ("konvid1k", "livevqc", "ytugc"):            # Step 19: public sets at half resolution (x2 downscale in decode)
    DATASETS[f"{_d}_half"] = dict(DATASETS[_d], downscale=2)


def clip_name(dataset, rel):
    """Token / score key of a clip. LSVQ ids repeat across batches, so keep the batch."""
    return rel.strip().replace("/", "__") if dataset.startswith("lsvq") else os.path.basename(rel.strip())


TOK_ROOT = None                                     # --tok-root: tokens of other encoders (e.g. step 9 v3)


def token_dir(dataset):
    """Frozen-encoder tokens are model-independent (Stage 1/2 are frozen in v1 and v2)."""
    if TOK_ROOT:
        return f"{TOK_ROOT}/{dataset}/tokens"
    return f"{ROOT}/step8_lsvq/{dataset}/tokens" if dataset.startswith("lsvq") else f"{OUT}/{dataset}/tokens"


def clip_list(name):
    if name in ("study", "study_half", "study_quarter"):
        d = DATASETS[name]["videos"]
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mp4") and not f.endswith(".part.mp4"))
    with open(DATASETS[name]["labels"]) as f:
        return [os.path.join(DATASETS[name]["videos"], r[0].strip()) for r in csv.reader(f)]


def decodable_frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return 0


class ClipDS(Dataset):
    def __init__(self, paths, dataset):
        self.paths, self.dataset = paths, dataset

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        rng = np.random.default_rng(CELL_SEED)       # tile cells: fixed seed, never derived from the file name
        try:
            pb = vl.probe(p)
            if (self.dataset == "study" or (pb["w"], pb["h"]) == (3840, 2160)) and not DATASETS[self.dataset].get("downscale"):
                T = pb["frames"]
                tticks = np.arange(0, T, F6.TILE_EVERY)
                cells = vl.sample_cells(rng, len(tticks))
                v = dict(path=p, w=pb["w"], h=pb["h"])
                low, full = vl.decode_window(v, 0, T, list(tticks), threads=2)
                view = vl.build_view(low, 0, full, np.arange(T), np.arange(T), cells, tticks, True)
                return i, dict(low=view["low"], tiles=view["tiles"], cells=cells, fps=pb["fps"], wh=(pb["w"], pb["h"])), ""
            # public MOS as is: native frames are the display ticks, tiles every 15th native frame
            kw = dict(tiles=True, max_tiles=32, tile_every_s=F6.TILE_EVERY / pb["fps"],
                      seed=int(rng.integers(2 ** 31)), threads=2, downscale=DATASETS[self.dataset].get("downscale", 1))
            try:
                o = vl.mos_native(dict(path=p), **kw)
            except RuntimeError:
                # container frame count can exceed the decodable frames (some LSVQ clips): count and retry
                n = decodable_frames(p)
                if not n or n >= pb["frames"]:
                    raise
                o = vl.mos_native(dict(path=p), n=n, **kw)
            xy = o["tile_xy"]
            cells = np.stack([np.minimum(8, xy[..., 1] // 238), np.minimum(14, xy[..., 0] // 256)], -1)
            return i, dict(low=o["low"], tiles=o["tiles"], cells=cells, fps=pb["fps"], wh=(pb["w"], pb["h"])), ""
        except Exception as e:
            return i, None, str(e)[-400:]


def extract(a):
    dev = torch.device("cuda", a.gpu)
    e1, e2 = F6.load_encoders(a.s1, a.s2, dev)
    paths = clip_list(a.dataset)[a.part::a.parts]
    vdir = DATASETS[a.dataset]["videos"]
    od = token_dir(a.dataset)
    os.makedirs(od, exist_ok=True)
    name_of = lambda p: clip_name(a.dataset, os.path.relpath(p, vdir))  # noqa: E731
    todo = [p for p in paths if not os.path.exists(f"{od}/{name_of(p)}.npz")
            and not os.path.exists(f"{od}/{name_of(p)}.err")]
    missing = [p for p in todo if not os.path.exists(p)]      # not unpacked yet: picked up by a later run
    todo = [p for p in todo if os.path.exists(p)]
    print(f"{len(todo)}/{len(paths)} clips to extract ({len(missing)} videos missing)", flush=True)
    dl = DataLoader(ClipDS(todo, a.dataset), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    t0 = time.time()
    for k, (i, s, err) in enumerate(dl):
        name = name_of(todo[i])
        if s is None:
            with open(f"{od}/{name}.err", "w") as f:
                f.write(err)
            print("ERR", name, err[-200:], flush=True)
            continue
        low, tiles = s["low"], s["tiles"]
        T = (low.shape[0] // 2) * 2
        try:
            if T < F6.M2.T_WIN:
                raise ValueError(f"too short: {low.shape[0]} frames < {F6.M2.T_WIN} (one motion chunk)")
            sp, mo = F6.encode_view(e1, e2, low[:T], tiles, dev)
        except Exception as e:                        # one bad clip must not stop the run
            with open(f"{od}/{name}.err", "w") as f:
                f.write(f"{type(e).__name__}: {e}"[-400:])
            print("ERR", name, str(e)[-200:], flush=True)
            continue
        tmp = f"{od}/.{name}.tmp.npz"                  # atomic: a killed job never leaves a partial npz
        np.savez(tmp, spatial=sp, motion=mo, cells=np.asarray(s["cells"]).astype(np.int8),
                 fps=float(s["fps"]), wh=np.asarray(s["wh"]), frames=int(low.shape[0]))
        os.replace(tmp, f"{od}/{name}.npz")
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(todo)} {(k + 1) / (time.time() - t0):.2f} clips/s", flush=True)
    print("FINISHED extract", flush=True)


@torch.no_grad()
def score(a):
    dev = torch.device("cuda", a.gpu)
    sd = torch.load(a.fusion, map_location="cpu")["model"]
    m = S3.Fusion(n_q=S3.n_q_of(sd)).to(dev).eval()                # n_q queries per head (Step 20); 1 = original
    m.load_state_dict(sd)
    od = a.out or f"{OUT}/{a.dataset}"
    os.makedirs(od, exist_ok=True)
    rows, feats = [], {}
    for f in sorted(glob.glob(f"{token_dir(a.dataset)}/*.npz")):
        z = np.load(f)
        K = z["spatial"].shape[0]
        steps = z["motion"].shape[0]
        K = min(K, -(-steps * 2 // F6.TILE_EVERY))
        sp = torch.from_numpy(z["spatial"][:K]).float()[None].to(dev)
        ce = torch.from_numpy(z["cells"][:K].astype(np.int64))[None].to(dev)
        mo = torch.from_numpy(z["motion"]).float()[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Q, V, q_w, v_w = m(sp, ce, mo)
        rows.append(dict(clip=os.path.basename(f)[:-4], Q=float(Q), V=float(V), windows=int(q_w.shape[1]),
                         frames=int(z["frames"]), fps=float(z["fps"])))
        feats[rows[-1]["clip"]] = m.clip_pool[0].cpu().numpy()
    with open(f"{od}/scores.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(f"{od}/pooled_features.npz", clips=np.array(list(feats)), x=np.stack(list(feats.values())))
    print(f"scored {len(rows)} clips -> {od}/scores.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "score"])
    ap.add_argument("--dataset", default="study", choices=list(DATASETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--s1", default=f"{ROOT}/step4/runs/s_mae/final_avg4.pt")
    ap.add_argument("--s2", default=f"{ROOT}/step5/runs/v_mae/final_avg4.pt")
    ap.add_argument("--fusion", default=f"{ROOT}/step6/runs/fusion/best.pt")
    ap.add_argument("--out", default=None, help="score: output dir (default step7/<dataset>)")
    ap.add_argument("--tok-root", default=None, help="token root for encoders other than v1 (default: v1 dirs)")
    a = ap.parse_args()
    global TOK_ROOT
    TOK_ROOT = a.tok_root
    {"extract": extract, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
