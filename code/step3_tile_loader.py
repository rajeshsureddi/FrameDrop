"""Step 3b: tile loader for Stage 1 (Tile-MAE).

Reads the per-source WebDataset tars written by step3_tiles.py. Every sample
the loader yields is a PAIR of 256x256 tiles cut from the same tile group
(same source, frame and position):
    x      uint8 (2, 256, 256, 3) RGB
    order  +1 -> x[0] is the better one (L_rank), 0 -> no order (L_rec only)
    ptype  "rate" | "capture_rate" | "severity" | "otf_severity" | "none"
Both tiles of a pair always get the same flip, colour jitter and on-the-fly
plan (except the one severity being ranked), so they stay identical except
along one axis. Stage 1 applies the same MAE mask to both.

On-the-fly distortion (p = 0.5): ops from video_distortions.py rendered on the
384x384 crop at its absolute 4K offset and source frame index (crop-consistent),
using the 64 px margin as context; plans whose margin() exceeds 64 px are
resampled. Clean-source ("orig") tiles may get any op; encoded tiles only the
post-decode ops. Temporal ops that need neighbouring frames (shutter_blur,
ghosting) cannot be rendered on a single tile and are excluded.

Batch diversity: workers interleave many source tars; DiverseBatcher (main
process) builds batches with at most `max_per_source` samples per source.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tarfile
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_common as c3  # noqa: E402
import step3_pairs as pr  # noqa: E402
import video_distortions as vd  # noqa: E402

POST_DECODE_OPS = ("rescale", "aliasing", "pixelation", "posterize", "chroma_bleed",
                   "chroma_quantize", "ghosting")
TEMPORAL = {n for n, s in vd.REGISTRY.items() if s.temporal}
OPS_ANY = tuple(n for n in vd.REGISTRY if n not in TEMPORAL)
OPS_ENCODED = tuple(n for n in POST_DECODE_OPS if n not in TEMPORAL)
PAIR_TYPES = ("rate", "capture_rate", "severity", "otf_severity", "none")

# one loader worker per core: OpenCV's default pool (one thread per core, in
# every worker) exhausted the per-user thread limit on 128-core nodes
cv2.setNumThreads(1)


# ------------------------------------------------------------------ reading
def iter_groups(tar_path):
    """Yield (meta, {"v00": png bytes, ...}) per tile group, streaming."""
    cur, meta, pngs = None, None, {}
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            key, ext = m.name.split(".", 1)
            if key != cur:
                if cur is not None:
                    yield meta, pngs
                cur, meta, pngs = key, None, {}
            data = tf.extractfile(m).read()
            if ext == "json":
                meta = json.loads(data)
            else:
                pngs[ext[:-4]] = data
    if cur is not None:
        yield meta, pngs


def decode_png(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]


def list_shards(tiles_dir, splits=("train",)):
    out = []
    for f in sorted(os.listdir(tiles_dir)):
        if f.endswith(".json") and not f.startswith("."):
            j = c3.load_json(os.path.join(tiles_dir, f))
            if j["split"] in splits:
                out.append(os.path.join(tiles_dir, j["tar"]))
    return out


# ------------------------------------------------------------------ pairs
def candidate_pairs(versions, plans):
    """All valid ordered pairs inside one tile group, by type: (better_v, worse_v)."""
    out = defaultdict(list)
    clean = [v for v in versions if v["kind"] == "clean"]
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            o = pr.rate_order(clean[i], clean[j])
            if o:
                out["rate"].append((o[0]["v"], o[1]["v"]))
    cap = [v for v in versions if v["kind"] == "capture"]
    for i in range(len(cap)):
        for j in range(i + 1, len(cap)):
            a, b = cap[i], cap[j]
            if a["plan_id"] != b["plan_id"]:
                continue
            if a["severity"] == b["severity"] and a["rank"] != b["rank"]:
                o = pr.rate_order(a, b)
                if o:
                    out["capture_rate"].append((o[0]["v"], o[1]["v"]))
            elif a["rank"] == b["rank"] and a["severity"] != b["severity"]:
                ops = plans.get(a["plan_id"], {}).get("ops", [])
                s = pr.severity_order(ops, a["severity"], b["severity"])
                if s:
                    out["severity"].append((a["v"], b["v"]) if s > 0 else (b["v"], a["v"]))
    return out


# ------------------------------------------------------------------ distortion
def sample_tile_plan(rng, allow, max_margin=c3.TILE_MARGIN, tries=30):
    r = np.random.default_rng(int(rng.integers(2**63)))
    for _ in range(tries):
        p = vd.sample_plan(r, c3.FPS, 5.0, n_ops=(1, 3), allow=allow)
        if p.margin(c3.H4K) <= max_margin:
            return p
    return vd.sample_plan(r, c3.FPS, 5.0, n_ops=(1, 1), allow=("posterize", "chroma_quantize"))


def render_tile(t, plan, frame, x0, y0):
    """t uint8 (384,384,3) crop at absolute (x0, y0) -> uint8."""
    y = vd.render(t[None], [frame], plan, full_size=(c3.W4K, c3.H4K), offset=(x0, y0))[0]
    return np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)


def jitter(t, flip, jit):
    """Horizontal flip + mild saturation/contrast/brightness, one fused affine:
    y = b*(c*(g + s*(x-g)) + (1-c)*m), g = luma, m = mean of the 256 core."""
    if flip:
        t = t[:, ::-1]
    t = np.ascontiguousarray(t)
    if jit is None:
        return t
    b, c, s = jit
    x = t.astype(np.float32)
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    m = float(center(x).mean())
    y = cv2.addWeighted(x, b * c * s, cv2.merge([g, g, g]), b * c * (1 - s), b * (1 - c) * m)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def center(t):
    m = c3.TILE_MARGIN
    return t[m:m + c3.TILE_CORE, m:m + c3.TILE_CORE]


# ------------------------------------------------------------------ dataset
class TileDataset(IterableDataset):
    def __init__(self, tiles_dir, splits=("train",), plans_json=None, p_ops=0.5,
                 items_per_group=4, open_per_worker=8, type_weights=None, seed=0,
                 rank=0, world=1, epoch=0, jitter_amp=0.05, shards=None, p_orig=0.5):
        self.tiles_dir, self.splits = tiles_dir, splits
        self.shards = shards if shards is not None else list_shards(tiles_dir, splits)
        plans_json = plans_json or os.path.join(os.path.dirname(tiles_dir.rstrip("/")), "capture_plans.json")
        self.plans = c3.load_json(plans_json)
        self.p_ops, self.ipg, self.K = p_ops, items_per_group, open_per_worker
        self.tw = type_weights or {k: 1.0 for k in PAIR_TYPES}
        self.seed, self.rank, self.world, self.epoch = seed, rank, world, epoch
        self.jamp, self.p_orig = jitter_amp, p_orig
        self.stats = Counter()

    def set_epoch(self, e):
        self.epoch = e

    def _my_shards(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        sh = list(self.shards)
        random.Random(self.seed * 1000003 + self.epoch).shuffle(sh)
        k, n = self.rank * nw + wid, self.world * nw
        return sh[k::n], np.random.default_rng([self.seed, self.epoch, self.rank, wid])

    def __iter__(self):
        shards, rng = self._my_shards()
        pending = list(shards)
        streams = []
        while pending or streams:
            while pending and len(streams) < self.K:
                streams.append(iter_groups(pending.pop(0)))
            i = int(rng.integers(len(streams)))
            try:
                meta, pngs = next(streams[i])
            except StopIteration:
                streams.pop(i)
                continue
            yield from self._items(meta, pngs, rng)

    # one group -> several pair samples
    def _items(self, meta, pngs, rng):
        vers = {v["v"]: v for v in meta["versions"]}
        cands = candidate_pairs(meta["versions"], self.plans)
        cache = {}

        def tile(v):
            if v not in cache:
                cache[v] = decode_png(pngs[v])
            return cache[v]

        x0, y0 = meta["x"] - meta["margin"], meta["y"] - meta["margin"]
        for _ in range(self.ipg):
            avail = [t for t in PAIR_TYPES if t in ("none", "otf_severity") or cands.get(t)]
            w = np.array([self.tw.get(t, 0.0) for t in avail], float)
            if w.sum() == 0:
                continue
            ptype = avail[int(rng.choice(len(avail), p=w / w.sum()))]
            if ptype in cands:
                better, worse = cands[ptype][int(rng.integers(len(cands[ptype])))]
                pair, order = [better, worse], 1
            elif ptype == "otf_severity":
                vs = list(vers)
                v = vs[0] if (vers[vs[0]]["kind"] == "orig" and rng.random() < self.p_orig) \
                    else vs[int(rng.integers(len(vs)))]
                pair, order = [v, v], 1
            else:
                vs = list(vers)
                a, b = rng.choice(len(vs), 2, replace=len(vs) < 2)
                pair, order = [vs[a], vs[b]], 0
            flip = rng.random() < 0.5
            ja = self.jamp
            jit = tuple(1 + rng.uniform(-ja, ja, 3)) if ja > 0 else None
            ts = [jitter(tile(v), flip, jit) for v in pair]
            any_orig = all(vers[v]["kind"] == "orig" for v in pair)
            allow = OPS_ANY if any_orig else OPS_ENCODED
            ops_used = ""
            if ptype == "otf_severity":
                ok = False
                for _try in range(10):
                    plan = sample_tile_plan(rng, allow)
                    s_hi = float(rng.uniform(0.3, 1.0))
                    s_lo = float(rng.uniform(0.0, max(0.0, s_hi - 0.25)))
                    ops = [(o.name, o.severity, o.P) for o in plan.ops]
                    if pr.severity_order(ops, s_lo, s_hi) == 1:
                        ok = True
                        break
                if not ok:
                    self.stats["otf_invalid"] += 1
                    continue
                ts = [render_tile(ts[0], plan.with_severity(s_lo), meta["frame"], x0, y0),
                      render_tile(ts[1], plan.with_severity(s_hi), meta["frame"], x0, y0)]
                ops_used = "+".join(o.name for o in plan.ops)
            elif rng.random() < self.p_ops:
                plan = sample_tile_plan(rng, allow)
                ts = [render_tile(t, plan, meta["frame"], x0, y0) for t in ts]
                ops_used = "+".join(o.name for o in plan.ops)
            ts = [center(t) for t in ts]
            if order:
                indist, d = pr.indistinguishable(ts[0], ts[1])
                if indist:
                    self.stats[f"indist_{ptype}"] += 1
                    order, ptype = 0, "none"
            self.stats[ptype] += 1
            yield dict(x=np.stack(ts), order=order, ptype=ptype, source_id=meta["source_id"],
                       frame=meta["frame"], xy=(meta["x"], meta["y"]),
                       versions=(vers[pair[0]]["encode_id"], vers[pair[1]]["encode_id"]),
                       encoders=(vers[pair[0]].get("encoder"), vers[pair[1]].get("encoder")),
                       plan_id=vers[pair[0]].get("plan_id") or vers[pair[1]].get("plan_id"),
                       split=meta.get("split"), ops=ops_used)


# ------------------------------------------------------------------ batching
class DiverseBatcher:
    """Pulls samples from a DataLoader(batch_size=None) and emits batches in
    which no source appears more than max_per_source times."""

    def __init__(self, loader, batch_size, max_per_source=2, pool_factor=4, seed=0):
        self.loader, self.B, self.mps = loader, batch_size, max_per_source
        self.pool_n = pool_factor * batch_size
        self.rng = random.Random(seed)

    def __iter__(self):
        pool, it = [], iter(self.loader)
        done = False
        while True:
            while not done and len(pool) < self.pool_n:
                try:
                    pool.append(next(it))
                except StopIteration:
                    done = True
            if not pool:
                return
            self.rng.shuffle(pool)
            cnt, batch, rest = Counter(), [], []
            for s in pool:
                if len(batch) < self.B and cnt[s["source_id"]] < self.mps:
                    batch.append(s)
                    cnt[s["source_id"]] += 1
                else:
                    rest.append(s)
            pool = rest
            if len(batch) < self.B and not done:
                pool.extend(batch)               # not diverse enough yet: pull more
                if len(pool) >= 16 * self.B:
                    raise RuntimeError("pool cannot fill a diverse batch: too few sources open")
                self.pool_n = len(pool) + self.B
                continue
            if batch:
                yield collate(batch)


def collate(batch):
    return dict(x=torch.from_numpy(np.stack([b["x"] for b in batch])).permute(0, 1, 4, 2, 3),
                order=torch.tensor([b["order"] for b in batch]),
                ptype=[b["ptype"] for b in batch], source_id=[b["source_id"] for b in batch],
                versions=[b["versions"] for b in batch], ops=[b["ops"] for b in batch],
                encoders=[b["encoders"] for b in batch], plan_id=[b["plan_id"] for b in batch],
                frame=[b["frame"] for b in batch], xy=[b["xy"] for b in batch])
