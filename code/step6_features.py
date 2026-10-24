"""Step 6 (part 1): frozen Stage 1 / Stage 2 tokens for Stage 3.

Both encoders are frozen in Stage 3, so their tokens are computed once here
(pure speed-up: same pixels, same encoders, same outputs). A clip is
T = 300 displayed frames (2 windows of 240, hop 60) or 240 (1 window).

Per clip:
  spatial  (T/15, 32, 384) fp16   Stage 1 features (no mask) of 32 native tiles
                                    at every 15th displayed frame
  cells    (T/15, 32, 2)  int8    (row, col) on the 9 x 15 grid
  motion   (T/2, 12, 384) fp16    Stage 2 tokens (no mask, 3x4 regions) per
                                    2-frame step, from 32-frame chunks covering
                                    the clip (last chunk overlaps)

Modes
  pairs   training/valid pairs from VideoDataset (drops / rate / severity in
          equal proportions, p_none 0.1), written as shards of pairs
  clips   a list of video files (evaluation), each clip as is (native fps
          and resolution for public MOS; study clips as displayed), all frames

    python step6_features.py pairs --split train --n 1000 --shard 17 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_video_loader as vl  # noqa: E402
import step4_model as M1  # noqa: E402
import step5_model as M2  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
OUT = f"{ROOT}/step6"
TILE_EVERY = 15


def load_encoders(s1, s2, dev):
    sd = torch.load(s1, map_location="cpu")["model"]
    e1 = M1.TileMAE.from_state_dict(sd).to(dev).eval()
    e1.load_state_dict(sd)
    e2 = M2.VideoMAE().to(dev).eval()
    e2.load_state_dict(torch.load(s2, map_location="cpu")["model"])
    return e1, e2


def chunk_starts(T, t=M2.T_WIN):
    s = list(range(0, T - t + 1, t))
    if s[-1] + t < T:
        s.append(T - t)
    return s


@torch.no_grad()
def encode_view(e1, e2, low, tiles, dev, tile_bs=512):
    """low (T, 216, 384, 3) uint8, tiles (K, 32, 256, 256, 3) uint8 ->
    spatial (K, 32, 384) fp16, motion (T/2, 12, 384) fp16."""
    T = low.shape[0]
    K = tiles.shape[0]
    tiles, low = torch.as_tensor(tiles), torch.as_tensor(low)        # DataLoader may hand back tensors
    t = tiles.reshape(-1, 256, 256, 3).to(dev, non_blocking=True).permute(0, 3, 1, 2)
    sp = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(t), tile_bs):
            sp.append(e1.features(t[i:i + tile_bs]).half())
    spatial = torch.cat(sp).view(K, -1, sp[0].shape[-1])
    x = low.to(dev, non_blocking=True)
    starts = chunk_starts(T)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = e2.features(torch.stack([x[s:s + M2.T_WIN] for s in starts])).half()   # (C, 16, 12, d)
    motion = torch.zeros(T // 2, 12, f.shape[-1], dtype=torch.float16, device=dev)
    for c, s in enumerate(starts):
        motion[s // 2:s // 2 + M2.GT] = f[c]                  # later chunks overwrite the overlap
    return spatial.cpu().numpy(), motion.cpu().numpy()


class PairStream(IterableDataset):
    def __init__(self, split, seed, T):
        self.ds = vl.VideoDataset(f"{ROOT}/step2/encodes_manifest.csv", splits=(split,), mode="pair", T=T,
                                  tiles=True, p_none=0.1, pair_types=("drops", "rate", "severity"),
                                  plans_json=f"{ROOT}/step3/capture_plans.json", seed=seed,
                                  step1_manifest=f"{ROOT}/step1/manifest.csv", lowres_dir=f"{ROOT}/step3/lowres")
        self.T = T

    def __iter__(self):
        for s in self.ds:
            s["T"] = self.T
            yield s


def run_pairs(a):
    dev = torch.device("cuda", a.gpu)
    e1, e2 = load_encoders(a.s1, a.s2, dev)
    os.makedirs(f"{a.out}/{a.split}", exist_ok=True)
    dst = f"{a.out}/{a.split}/pairs_{a.shard:05d}.npz"
    if os.path.exists(dst):
        print("exists", dst)
        return
    dl = DataLoader(PairStream(a.split, 1000 + a.shard, a.T), batch_size=None, num_workers=a.workers,
                    prefetch_factor=2)
    want = {"drops": a.n // 3 + (a.n % 3 > 0), "rate": a.n // 3 + (a.n % 3 > 1), "severity": a.n // 3}
    S, Mo, C, meta = [], [], [], []
    t0 = time.time()
    for s in dl:
        if want.get(s["ptype"], 0) <= 0:
            continue
        want[s["ptype"]] -= 1
        sb, mb = encode_view(e1, e2, s["better"]["low"], s["better"]["tiles"], dev)
        sw, mw = encode_view(e1, e2, s["worse"]["low"], s["worse"]["tiles"], dev)
        S.append(np.stack([sb, sw]))
        Mo.append(np.stack([mb, mw]))
        C.append(np.asarray(s["cells"]).astype(np.int8))
        m0, m1 = s["meta"]
        meta.append(dict(ptype=s["ptype"], source_id=s["source_id"], mean_abs_diff=float(s["mean_abs_diff"]),
                         kind=m0["kind"], encode=(m0["encode"], m1["encode"]), rank=(m0["rank"], m1["rank"]),
                         severity=(m0["severity"], m1["severity"]), plan_id=m0["plan_id"], start=m0["start"],
                         family=m0["drop_family"], drops=(np.asarray(m0["drop_mask"]).astype(int).tolist(),
                                                              np.asarray(m1["drop_mask"]).astype(int).tolist())))
        if sum(want.values()) == 0:
            break
    tmp = dst[:-4] + ".tmp.npz"
    np.savez(tmp, spatial=np.stack(S), motion=np.stack(Mo), cells=np.stack(C), meta=json.dumps(meta),
             T=a.T, enc=json.dumps(dict(s1=a.s1, s2=a.s2)))
    os.replace(tmp, dst)
    print(f"wrote {dst}: {len(S)} pairs in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=600, help="pairs per shard (equal thirds per type)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--s1", default=f"{ROOT}/step4/runs/s_mae/final_avg4.pt")
    ap.add_argument("--s2", default=f"{ROOT}/step5/runs/v_mae/final_avg4.pt")
    ap.add_argument("--out", default=OUT, help="token root (train/ and valid/ go under it)")
    a = ap.parse_args()
    run_pairs(a)


if __name__ == "__main__":
    main()
