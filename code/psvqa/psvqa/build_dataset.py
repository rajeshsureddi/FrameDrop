"""Turn cached encodes into training samples (no human labels needed).

manifest.csv columns:
  source_id : clip identity (all encodes of one pristine clip share it)
  cache     : .npz from `extract --mode full`
  rank      : integer, higher = higher bitrate / better (pristine = highest)
  mos       : optional; if present, the row is a labeled public-dataset clip,
              used without synthetic drops for absolute anchoring
  dataset   : optional; MOS is z-scored per dataset

For every synthetic source we draw P base drop patterns shared by all its
encodes (-> bitrate-order pairs) and one strictly-worse superset of each
(-> drop-order pairs).

  python -m psvqa.build_dataset --manifest manifest.csv --predictor predictor.pt \
         --out train_samples.pkl --patterns 12
"""
import argparse
import csv
import pickle
from collections import defaultdict

import numpy as np

from .common import dup_flags, sample_mask, thumb_mad
from .drops import drops_to_index, sample_pattern, superset
from .temporal import per_frame_features, segment_stats, gate_inputs


def make_sample(cache, idx, runner, seg):
    z = runner.normalize(cache["z"][idx]) if runner else cache["z"][idx].astype(np.float32)
    th = cache["thumbs"][idx]
    F, dup = per_frame_features(z, th, runner)
    return dict(idx=idx.astype(np.int32),
                valid=sample_mask(dup),
                gate=gate_inputs(F, dup).astype(np.float16),
                tstats=segment_stats(F, dup, seg), seg=seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictor", default=None,
                    help="omit for the constant-velocity ablation")
    ap.add_argument("--out", default="train_samples.pkl")
    ap.add_argument("--patterns", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runner = None
    if a.predictor:
        from .predictor import PredictorRunner
        runner = PredictorRunner(a.predictor)

    rows = list(csv.DictReader(open(a.manifest)))
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source_id"]].append(r)

    rng = np.random.default_rng(a.seed)
    samples = []
    for n, (src, encs) in enumerate(by_src.items()):
        labeled = [r for r in encs if r.get("mos") not in (None, "")]
        for r in labeled:
            c = np.load(r["cache"])
            T = len(c["thumbs"])
            s = make_sample(c, np.arange(T), runner, int(round(float(c["fps"]))))
            s.update(cache=r["cache"], src=src, rank=int(r.get("rank") or 0),
                     pid=-1, parent=-1, visE=0.0, mos=float(r["mos"]),
                     dataset=r.get("dataset", "public"))
            samples.append(s)
        synth = [r for r in encs if r not in labeled]
        if not synth:
            continue
        T = min(len(np.load(r["cache"])["thumbs"]) for r in synth)
        pats = []
        for p in range(a.patterns):
            base, fam = sample_pattern(T, rng)
            pats.append((2 * p, -1, base, fam))
            pats.append((2 * p + 1, 2 * p, superset(base, rng), fam + "+sup"))
        for r in synth:
            c = np.load(r["cache"])
            c = {k: c[k][:T] if k in ("z", "thumbs", "spat") else c[k] for k in c.files}
            src_mad = thumb_mad(c["thumbs"]) * 255.0      # game-time motion
            seg = int(round(float(c["fps"])))
            for pid, parent, drop, fam in pats:
                idx = drops_to_index(drop)
                s = make_sample(c, idx, runner, seg)
                dup = idx[1:] == idx[:-1]
                visE = float(src_mad[1:][dup].sum())      # missed motion (proxy)
                s.update(cache=r["cache"], src=src, rank=int(r["rank"]), pid=pid,
                         parent=parent, visE=visE, mos=float("nan"),
                         family=fam, dataset="synthetic")
                samples.append(s)
        print(f"[{n + 1}/{len(by_src)}] {src}: {len(samples)} samples so far")

    with open(a.out, "wb") as f:
        pickle.dump(samples, f)
    print(f"saved {len(samples)} samples -> {a.out}")


if __name__ == "__main__":
    main()
