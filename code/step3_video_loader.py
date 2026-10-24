"""Step 3c: video loader for Stages 2-3.

A sample is a window of T displayed frames (default 240 = 4 s at 60 Hz) from a
Step 2 encode (clean ladder, capture or 2f post-drop) or from a public-MOS
clip. CPU workers decode on the fly and return, from ONE decode per video:
  low    uint8 (T, 216, 384, 3)  every displayed frame, area-downscaled
  tiles  uint8 (16, 32, 256, 256, 3)  native 4K tiles of every 15th displayed
         frame, 32 random cells of the 9x15 grid (row stride 238, col 256)
  cells  int (16, 32, 2)  (row, col) of each tile; tile_ticks (16,) display ticks
Nothing else about the video reaches the model (pixels-only rule); the
metadata returned next to it is for labels/logging.

Public-MOS clips (training L_mos and evaluation) are used AS IS (user decision
2026-10-15, replaces the plan's display-referenced conversion for them): native
resolution and frame rate, no upscale, no 4K padding, no frame repetition, no
drops/ops. The low-res stream is area-downscaled to fit 384x216 (never
upscaled; centred padding); tiles are native 256x256 crops on a grid fitted to
the clip's own size (tile_xy = pixel top-left). Step 2 encodes are 3840x2160
at 60 fps.

Drops: a drop pattern over the T ticks is applied as an index remap
(psvqa drops.py); none with probability p_none.

Pairs (mode="pair"), always the same window, same tile cells, and:
  drops     same encode, P vs superset(P)      (P may be none)
  rate      same P, two clean-ladder rungs (or two rungs of one capture plan)
  severity  same P, severities 0.5 vs 1.0 of one capture plan, same rung
validity rules in step3_pairs.py; indistinguishable pairs (mean |A-B| < 0.5 on
the low-res frames) are resampled.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "psvqa"))
import step3_common as c3  # noqa: E402
import step3_pairs as pr  # noqa: E402
from psvqa import drops as dr  # noqa: E402

LOW_W, LOW_H = 384, 216
T_WIN = 240
TILE_EVERY = 15
GRID_ROWS, GRID_COLS = 9, 15
ROW_STRIDE, COL_STRIDE = 238, 256
TILES_PER_STEP = 32
NVDEC_CODECS = {"h264", "hevc", "vp9"}
CODEC_OF = {"libx264": "h264", "libx265": "hevc", "libvpx-vp9": "vp9", "libsvtav1": "av1"}


# ------------------------------------------------------------------ catalogue
def load_encodes(manifest, splits=("train",), min_frames=T_WIN, step1_manifest=None):
    """Step 2 encodes grouped per source; with step1_manifest also the clean
    source spans themselves (kind "orig", decoded from the source file)."""
    by_src = defaultdict(list)
    spans = {}
    with open(manifest) as f:
        for r in csv.DictReader(f):
            if r["split"] not in splits or r["ok"] != "True" or int(r["frames"]) < min_frames:
                continue
            frames = int(r["frames"])
            start = int(r["span_end"]) - frames if r["kind"] == "capture" else int(r["span_start"])
            spans[r["source_id"]] = (int(r["span_start"]), int(r["span_end"]))
            by_src[r["source_id"]].append(dict(
                source_id=r["source_id"], kind=r["kind"], path=r["path"], json_path=r["json_path"],
                rank=int(r["rank"]) if r["rank"] else None, plan_id=r["plan_id"] or None,
                severity=float(r["severity"]) if r["severity"] else 0.0,
                size_bytes=int(r["size_bytes"]), crf=float(r["crf"]) if r["crf"] else None,
                encoder=r["encoder"], codec=CODEC_OF.get(r["encoder"]), frames=frames, start=start, fps=60.0, w=c3.W4K, h=c3.H4K))
    if step1_manifest:
        from step2_common import source_id
        with open(step1_manifest) as f:
            for r in csv.DictReader(f):
                sid = source_id(r["path"]) if r["keep"] == "True" else None
                if sid in spans and spans[sid][1] - spans[sid][0] >= min_frames:
                    a, b = spans[sid]
                    by_src[sid].append(dict(
                        source_id=sid, kind="orig", path=r["path"], json_path=None, rank=None,
                        plan_id=None, severity=0.0, size_bytes=None, crf=None, encoder=None,
                        codec=r["codec"], frames=b - a, start=a, file_offset=a, fps=60.0,
                        w=c3.W4K, h=c3.H4K))
    return dict(by_src)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets"
                          ":stream_side_data=rotation", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    w, h = int(s["width"]), int(s["height"])
    rot = next((int(d["rotation"]) for d in s.get("side_data_list", []) if "rotation" in d), 0)
    if abs(rot) % 180 == 90:                         # ffmpeg autorotates: frames come out as displayed
        w, h = h, w
    return dict(w=w, h=h, fps=fps, frames=int(s["nb_read_packets"]))


# ------------------------------------------------------------------ decoding
def display_map(n_src, fps):
    """display tick -> source frame (closest in time); 1:1 for ~60 fps."""
    if abs(fps - 60.0) < 0.5:
        return np.arange(n_src)
    n_disp = int(np.floor((n_src - 1) * 60.0 / fps)) + 1
    m = np.floor(np.arange(n_disp) * fps / 60.0 + 0.5).astype(int)
    return np.minimum(m, n_src - 1)


def _read_exact(fd, n_frames, fsz, out):
    with os.fdopen(fd, "rb", buffering=0) as f:
        for k in range(n_frames):
            buf = bytearray(fsz)
            mv, got = memoryview(buf), 0
            while got < fsz:
                m = f.readinto(mv[got:])
                if not m:
                    break
                got += m
            if got < fsz:
                break
            out.append(bytes(buf))
        while f.read(1 << 20):             # drain
            pass


def decode_window(v, lo, hi, full_frames, threads=2, hw_device=None):
    """Decode source/encode frames [lo, hi) of video v.
    Returns low (hi-lo, 216, 384, 3) RGB uint8 and {frame: yuv420p bytes at 4K}
    for the requested full-resolution frames (absolute indices of v)."""
    n = hi - lo
    lo, hi = lo + v.get("file_offset", 0), hi + v.get("file_offset", 0)
    pre = f"trim=start_frame={lo}:end_frame={hi},setpts=N/({c3.FPS}*TB)"
    if v.get("lowres_cache"):                 # lossless 384x216 yuv444p cache (step3_lowres.py)
        assert not full_frames
        fc = f"[0:v]{pre},format=yuv444p[lo]"
        r_lo, w_lo = os.pipe()
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", v["path"], "-filter_complex", fc,
               "-map", "[lo]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{w_lo}"]
        return _run_decode(cmd, [w_lo], [(r_lo, n, LOW_W * LOW_H * 3)], v, lo, hi, n, [])
    if (v["w"], v["h"]) != (c3.W4K, c3.H4K):
        pre += (f",scale={c3.W4K}:{c3.H4K}:flags=lanczos:force_original_aspect_ratio=decrease"
                f":out_color_matrix=bt709:out_range=tv,pad={c3.W4K}:{c3.H4K}:(ow-iw)/2:(oh-ih)/2")
    full = sorted(set(int(f) - lo for f in full_frames))
    r_lo, w_lo = os.pipe()
    fds = [w_lo]
    if full:
        r_hi, w_hi = os.pipe()
        fds.append(w_hi)
        sel = "+".join(f"eq(n\\,{k})" for k in full)
        fc = (f"[0:v]{pre},format=yuv420p,split=2[a][b];"
              f"[a]scale={LOW_W}:{LOW_H}:flags=area,format=yuv444p[lo];"
              f"[b]select={sel},format=yuv420p[hi]")
        outs = ["-map", "[lo]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{w_lo}",
                "-map", "[hi]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{w_hi}"]
    else:
        fc = f"[0:v]{pre},format=yuv420p,scale={LOW_W}:{LOW_H}:flags=area,format=yuv444p[lo]"
        outs = ["-map", "[lo]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{w_lo}"]
    hw = []
    if hw_device is not None and v.get("codec") in NVDEC_CODECS:     # bit-exact (checked), AV1 not on A100
        hw = ["-hwaccel", "cuda", "-hwaccel_device", str(hw_device)]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", str(threads), *hw, "-i", v["path"],
           "-filter_complex", fc, "-filter_complex_threads", "1", *outs]
    reads = [(r_lo, n, LOW_W * LOW_H * 3)]
    if full:
        reads.append((r_hi, len(full), c3.W4K * c3.H4K * 3 // 2))
    return _run_decode(cmd, fds, reads, v, lo, hi, n, full)


def _run_decode(cmd, fds, reads, v, lo, hi, n, full):
    p = subprocess.Popen(cmd, pass_fds=fds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for fd in fds:
        os.close(fd)
    bufs = [[] for _ in reads]
    th = [threading.Thread(target=_read_exact, args=(fd, k, sz, b)) for (fd, k, sz), b in zip(reads, bufs)]
    for t in th:
        t.start()
    err = p.stderr.read().decode(errors="replace")
    rc = p.wait()
    for t in th:
        t.join()
    low_b = bufs[0]
    hi_b = bufs[1] if len(bufs) > 1 else []
    if len(low_b) != n or len(hi_b) != len(full):
        raise RuntimeError(f"decode {v['path']} [{lo},{hi}): low {len(low_b)}/{n} "
                           f"full {len(hi_b)}/{len(full)} rc={rc} {err[-400:]}")
    a = np.frombuffer(b"".join(low_b), np.uint8).reshape(n, 3, LOW_H, LOW_W)
    low = c3.yuv444_to_rgb(a[:, 0], a[:, 1], a[:, 2])
    off = v.get("file_offset", 0)
    return low, {lo - off + k: b for k, b in zip(full, hi_b)}


# ------------------------------------------------------------------ windows
def sample_drops(T, rng, p_none):
    if rng.random() < p_none:
        return np.zeros(T, bool), "none"
    while True:
        d, fam = dr.sample_pattern(T, rng)
        if d.any():
            return d, fam


def sample_cells(rng, n_steps):
    cells = np.zeros((n_steps, TILES_PER_STEP, 2), np.int64)
    for k in range(n_steps):
        c = rng.choice(GRID_ROWS * GRID_COLS, TILES_PER_STEP, replace=False)
        cells[k, :, 0], cells[k, :, 1] = c // GRID_COLS, c % GRID_COLS
    return cells


def build_view(low, lo, full, src_of_tick, idx, cells, tile_ticks, want_tiles):
    """low: decoded low-res frames for source frames [lo, ...); src_of_tick:
    window tick -> absolute source frame (before drops); idx: drop remap."""
    shown = src_of_tick[idx]                       # displayed tick -> source frame
    out = dict(low=low[shown - lo])
    if want_tiles:
        tiles = np.zeros((len(tile_ticks), TILES_PER_STEP, 256, 256, 3), np.uint8)
        for k, d in enumerate(tile_ticks):
            Y, U, V = c3.yuv_planes(full[int(shown[d])], c3.W4K, c3.H4K)
            for j, (r, c) in enumerate(cells[k]):
                tiles[k, j] = c3.yuv_crop_to_rgb(Y, U, V, int(c) * COL_STRIDE, int(r) * ROW_STRIDE, 256, 256)
        out["tiles"] = tiles
    return out


class VideoDataset(IterableDataset):
    """mode: "single" | "pair".  pair_types: subset of ("drops", "rate", "severity").
    tiles=False skips the 4K tiles (Stage 2 only needs the low-res frames)."""

    def __init__(self, manifest, splits=("train",), mode="single", T=T_WIN, tiles=True,
                 p_none=0.1, pair_types=("drops", "rate", "severity"), kinds=("orig", "clean", "capture", "postdrop"),
                 plans_json=None, seed=0, rank=0, world=1, threads=2, mos_items=None, max_retries=8,
                 hwaccel_gpus=None, step1_manifest=None, lowres_dir=None):
        self.by_src = load_encodes(manifest, splits, T, step1_manifest) if manifest else {}
        self.lowres_dir = lowres_dir                 # use the lossless 384x216 cache when tiles=False
        for s in list(self.by_src):
            self.by_src[s] = [v for v in self.by_src[s] if v["kind"] in kinds]
            if not self.by_src[s]:
                del self.by_src[s]
        self.sources = sorted(self.by_src)
        self.mos_items = mos_items or []            # [{path, mos, dataset}] public MOS clips
        self.mode, self.T, self.tiles, self.p_none = mode, T, tiles, p_none
        self.pair_types = pair_types
        self.plans = c3.load_json(plans_json) if plans_json else {}
        self.seed, self.rank, self.world, self.threads = seed, rank, world, threads
        self.max_retries = max_retries
        self.hw_gpus = hwaccel_gpus                # e.g. [0, 1, 2]: NVDEC for h264/hevc/vp9
        self.hw_device = None
        self.stats = defaultdict(int)

    def _rng(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        if self.hw_gpus:
            self.hw_device = self.hw_gpus[wid % len(self.hw_gpus)]
        return np.random.default_rng([self.seed, self.rank, wid, os.getpid()])

    def __iter__(self):
        rng = self._rng()
        while True:
            try:
                if self.mos_items:
                    yield self.mos_sample(self.mos_items[int(rng.integers(len(self.mos_items)))], rng)
                    continue
                sid = self.sources[int(rng.integers(len(self.sources)))]   # sources evenly
                s = self.pair_sample(sid, rng) if self.mode == "pair" else self.single_sample(sid, rng)
                if s is not None:
                    yield s
            except RuntimeError as e:
                self.stats["decode_error"] += 1
                print(f"[VideoDataset] {e}", file=sys.stderr)

    # -- helpers
    def _window(self, vs, rng):
        a0 = max(v["start"] for v in vs)
        a1 = min(v["start"] + v["frames"] for v in vs) - self.T
        if a1 < a0:
            return None
        return int(rng.integers(a0, a1 + 1))

    def _baked(self, v, a):
        """Drops already baked into a 2f post-drop encode, over window ticks [a, a+T)."""
        m = np.zeros(self.T, bool)
        if v["kind"] == "postdrop":
            if "baked" not in v:
                j = c3.load_json(v["json_path"])
                v["baked"] = np.asarray(j["drop_mask"], np.int64)   # encode-frame indices
            k = v["baked"] - (a - v["start"])
            m[k[(k >= 0) & (k < self.T)]] = True
        return m

    def _decode(self, v, a, shown_sets, cells_ticks):
        """Decode the window [a, a+T) of v; full-res frames for the displayed
        source frames at the tile ticks of every view."""
        full = set()
        if self.tiles:
            for shown in shown_sets:
                full.update(int(shown[d]) for d in cells_ticks)
        if not self.tiles:
            return self._low_only(v, a), {}
        low, fb = decode_window(v, a - v["start"], a + self.T - v["start"],
                                [f - v["start"] for f in full], self.threads, self.hw_device)
        return low, {f + v["start"]: b for f, b in fb.items()}

    def _low_only(self, v, a):
        """Low-res frames of window [a, a+T) of v, from the cache if present."""
        if self.lowres_dir:
            eid = "orig" if v["kind"] == "orig" else os.path.basename(v["path"])[:-4]
            cp = os.path.join(self.lowres_dir, v["source_id"], eid + ".mkv")
            if os.path.exists(cp):
                self.stats["lowres_cache_hit"] += 1
                c = dict(path=cp, lowres_cache=True)
                return decode_window(c, a - v["start"], a + self.T - v["start"], [])[0]
            self.stats["lowres_cache_miss"] += 1
        return decode_window(v, a - v["start"], a + self.T - v["start"], [], self.threads,
                             self.hw_device)[0]

    def _views(self, v, a, idxs, cells, tticks):
        src_of_tick = np.arange(a, a + self.T)
        shown_sets = [src_of_tick[i] for i in idxs]
        low, full = self._decode(v, a, shown_sets, tticks)
        return [build_view(low, a, full, src_of_tick, i, cells, tticks, self.tiles) for i in idxs]

    def _meta(self, v, a, P, fam):
        return dict(source_id=v["source_id"], kind=v["kind"], encode=os.path.basename(v["path"]),
                    rank=v["rank"], plan_id=v["plan_id"], severity=v["severity"], start=a,
                    drop_family=fam, n_dropped=int(P.sum()), drop_mask=P)

    # -- samples
    def single_sample(self, sid, rng):
        vs = self.by_src[sid]
        v = vs[int(rng.integers(len(vs)))]
        a = self._window([v], rng)
        if a is None:
            return None
        P, fam = sample_drops(self.T, rng, self.p_none)
        tticks = np.arange(0, self.T, TILE_EVERY)
        cells = sample_cells(rng, len(tticks))
        (view,) = self._views(v, a, [dr.drops_to_index(P)], cells, tticks)
        self.stats[f"single_{v['kind']}"] += 1
        return dict(view, cells=cells, tile_ticks=tticks, meta=self._meta(v, a, P, fam), source_id=sid)

    def pair_sample(self, sid, rng):
        vs = self.by_src[sid]
        tticks = np.arange(0, self.T, TILE_EVERY)
        for _ in range(self.max_retries):
            ptype = self.pair_types[int(rng.integers(len(self.pair_types)))]
            P, fam = sample_drops(self.T, rng, self.p_none)
            cells = sample_cells(rng, len(tticks))
            if ptype == "drops":
                v = vs[int(rng.integers(len(vs)))]
                a = self._window([v], rng)
                if a is None:
                    continue
                baked = self._baked(v, a)          # 2f: count only drops not already baked in
                # decode the low-res window once and search (P, superset) on it
                low0 = self._low_only(v, a)
                found = None
                for _k in range(20):
                    if _k:
                        P, fam = sample_drops(self.T, rng, self.p_none)
                    q = dr.superset(P, rng)
                    q[0] = False
                    if not pr.drops_superset_ok(P | baked, q | baked):
                        self.stats["drops_superset_too_small"] += 1
                        continue
                    iP, iQ = dr.drops_to_index(P), dr.drops_to_index(q)
                    indist, d = pr.indistinguishable(low0[iP], low0[iQ])
                    if indist:
                        self.stats["indist_drops_candidate"] += 1
                        continue
                    found = (q, iP, iQ)
                    break
                if found is None:
                    self.stats["drops_window_static"] += 1
                    continue
                Q, iP, iQ = found
                if self.tiles:
                    vb, vw = self._views(v, a, [iP, iQ], cells, tticks)
                else:
                    vb, vw = dict(low=low0[iP]), dict(low=low0[iQ])
                metas = (self._meta(v, a, P, fam), self._meta(v, a, Q, fam + "+superset"))
            else:
                cands = []
                if ptype == "rate":
                    clean = [v for v in vs if v["kind"] == "clean"]
                    cands = [(x, y) for i, x in enumerate(clean) for y in clean[i + 1:]]
                    capg = defaultdict(list)
                    for v in vs:
                        if v["kind"] == "capture":
                            capg[(v["plan_id"], v["severity"])].append(v)
                    for g in capg.values():
                        cands += [(x, y) for i, x in enumerate(g) for y in g[i + 1:]]
                    cands = [pr.rate_order(x, y) for x, y in cands]
                    cands = [c for c in cands if c]
                else:
                    capg = defaultdict(list)
                    for v in vs:
                        if v["kind"] == "capture":
                            capg[(v["plan_id"], v["rank"])].append(v)
                    for g in capg.values():
                        for i, x in enumerate(g):
                            for y in g[i + 1:]:
                                ops = self.plans.get(x["plan_id"], {}).get("ops", [])
                                s = pr.severity_order(ops, x["severity"], y["severity"])
                                if s:
                                    cands.append((x, y) if s > 0 else (y, x))
                if not cands:
                    self.stats[f"{ptype}_no_candidates"] += 1
                    continue
                vbest, vworst = cands[int(rng.integers(len(cands)))]
                a = self._window([vbest, vworst], rng)
                if a is None:
                    continue
                idx = dr.drops_to_index(P)
                (vb,) = self._views(vbest, a, [idx], cells, tticks)
                (vw,) = self._views(vworst, a, [idx], cells, tticks)
                metas = (self._meta(vbest, a, P, fam), self._meta(vworst, a, P, fam))
            indist, d = pr.indistinguishable(vb["low"], vw["low"])
            if indist:
                self.stats[f"indist_{ptype}"] += 1
                continue
            self.stats[f"pair_{ptype}"] += 1
            return dict(better=vb, worse=vw, ptype=ptype, mean_abs_diff=d, cells=cells,
                        tile_ticks=tticks, meta=metas, source_id=sid)
        self.stats["pair_gave_up"] += 1
        return None

    def mos_sample(self, item, rng):
        """Random window of T native frames (whole clip if shorter), as is."""
        if "probe" not in item:
            item["probe"] = probe(item["path"])
        nf = item["probe"]["frames"]
        n = min(self.T, nf)
        a = int(rng.integers(0, nf - n + 1))
        out = mos_native(item, a, n, tiles=self.tiles, seed=int(rng.integers(2 ** 31)), threads=self.threads)
        self.stats["mos"] += 1
        return out


# ------------------------------------------------------------------ public-MOS clips (as is)
def _even(x):
    return int(x) // 2 * 2


def native_grid(w, h, t=256):
    """Top-left (x, y) of tiles covering a w x h frame, evenly spaced, even coords."""
    def axis(n):
        if n <= t:
            return [0]
        k = int(np.ceil((n - t) / t)) + 1
        return sorted({_even(round(v)) for v in np.linspace(0, n - t, k)})
    return [(x, y) for y in axis(h) for x in axis(w)]


def mos_native(item, start=0, n=None, tiles=True, max_tiles=TILES_PER_STEP, tile_every_s=TILE_EVERY / 60.0,
               seed=0, threads=2, downscale=1):
    """Public-MOS clip for evaluation, as is: native frames [start, start+n) (n=None:
    to the end), no display conversion. Returns
      low    (n, 216, 384, 3)  every native frame, area-downscaled to fit (never
             upscaled), centred on black
      tiles  (K, M, 256, 256, 3) native-resolution tiles of the frames at
             tile_frames (one every tile_every_s seconds); M = all grid cells, or
             max_tiles of them (fixed seed); tiles past a small frame's edge are black
      tile_xy (K, M, 2) top-left pixel (x, y) in the native frame
    downscale > 1 (Step 18 multi-scale): the decoded frame is first bicubic-downscaled by that factor inside the same
    ffmpeg graph (no re-encode); everything above then refers to the downscaled frame.
    """
    if "probe" not in item:
        item["probe"] = probe(item["path"])
    pb = item["probe"]
    w, h, fps = pb["w"], pb["h"], pb["fps"]
    ds = ""
    if downscale != 1:
        w, h = _even(round(w / downscale)), _even(round(h / downscale))
        ds = f"scale={w}:{h}:flags=bicubic,"
    n = pb["frames"] - start if n is None else n
    if n <= 0 or start + n > pb["frames"]:
        raise RuntimeError(f"{item['path']}: window [{start},{start + n}) outside {pb['frames']} frames")
    s = min(1.0, LOW_W / w, LOW_H / h)
    lw, lh = max(2, _even(round(w * s))), max(2, _even(round(h * s)))
    we, he = w + w % 2, h + h % 2                    # yuv420p needs even dims: pad 1 px if odd
    pre = (f"trim=start_frame={start}:end_frame={start + n},setpts=N/({c3.FPS}*TB),{ds}"
           f"scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,pad={we}:{he}:0:0")
    low_f = (f"scale={lw}:{lh}:flags=area,format=yuv444p,"
             f"pad={LOW_W}:{LOW_H}:{_even((LOW_W - lw) // 2)}:{_even((LOW_H - lh) // 2)}")
    tf = sorted({min(n - 1, int(round(k * tile_every_s * fps))) for k in range(int(n / (tile_every_s * fps)) + 1)}) \
        if tiles else []
    r_lo, w_lo = os.pipe()
    fds, reads = [w_lo], [(r_lo, n, LOW_W * LOW_H * 3)]
    if tf:
        r_hi, w_hi = os.pipe()
        fds.append(w_hi)
        reads.append((r_hi, len(tf), we * he * 3 // 2))
        sel = "+".join(f"eq(n\\,{k})" for k in tf)
        fc = f"[0:v]{pre},split=2[a][b];[a]{low_f}[lo];[b]select={sel},format=yuv420p[hi]"
        outs = ["-map", "[lo]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{w_lo}",
                "-map", "[hi]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{w_hi}"]
    else:
        fc = f"[0:v]{pre},{low_f}[lo]"
        outs = ["-map", "[lo]", "-fps_mode", "passthrough", "-f", "rawvideo", f"pipe:{w_lo}"]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", str(threads), "-i", item["path"],
           "-filter_complex", fc, "-filter_complex_threads", "1", *outs]
    v = dict(path=item["path"])
    low, full = _run_decode(cmd, fds, reads, v, start, start + n, n, tf)
    out = dict(low=low, tile_frames=np.array(tf, np.int64), native_wh=(w, h), fps=fps, start=start,
               meta=dict(path=item["path"], dataset=item.get("dataset"), mos=item.get("mos"),
                         src_fps=fps, src_wh=(w, h), start=start, frames=n, as_is=True),
               source_id=item["path"])
    if tf:
        grid = native_grid(w, h)
        r = np.random.default_rng(seed)
        m = len(grid) if max_tiles is None else min(max_tiles, len(grid))
        xy = np.zeros((len(tf), m, 2), np.int64)
        t = np.zeros((len(tf), m, 256, 256, 3), np.uint8)
        for k, f in enumerate(tf):
            Y, U, V = c3.yuv_planes(full[start + f], we, he)
            pick = np.sort(r.choice(len(grid), m, replace=False)) if m < len(grid) else np.arange(m)
            for j, g in enumerate(pick):
                x, y = grid[g]
                cw, ch = min(256, we - x), min(256, he - y)
                t[k, j, :ch, :cw] = c3.yuv_crop_to_rgb(Y, U, V, x, y, cw, ch)
                xy[k, j] = (x, y)
        out.update(tiles=t, tile_xy=xy)
    return out


class SubWindows(IterableDataset):
    """Stage 2 helper: cut each decoded window (tiles=False) into consecutive
    sub-windows of t displayed frames (32 for Video-MAE), so one 4K decode
    serves several training windows. Drop pairs are re-validated inside each
    sub-window (superset rule + indistinguishable filter); single windows keep
    their slice of the drop pattern."""

    def __init__(self, base: VideoDataset, t=32):
        assert not base.tiles
        self.base, self.t = base, t

    def __iter__(self):
        for s in self.base:
            T = s["better"]["low"].shape[0] if "better" in s else s["low"].shape[0]
            for k in range(T // self.t):
                sl = slice(k * self.t, (k + 1) * self.t)
                if "better" not in s:
                    m = dict(s["meta"], drop_mask=s["meta"]["drop_mask"][sl], sub=k)
                    m["n_dropped"] = int(m["drop_mask"].sum())
                    yield dict(low=s["low"][sl], meta=m, source_id=s["source_id"])
                    continue
                b, w = s["better"]["low"][sl], s["worse"]["low"][sl]
                order = 1
                p = s["meta"][0]["drop_mask"][sl]
                q = s["meta"][1]["drop_mask"][sl]
                if s["ptype"] == "drops" and not pr.drops_superset_ok(p, q):
                    order = 0                      # no extra drops here: unordered
                if order and pr.indistinguishable(b, w)[0]:
                    order = 0
                yield dict(better=b, worse=w, order=order, ptype=s["ptype"], sub=k,
                           source_id=s["source_id"], masks=(p, q), family=s["meta"][0]["drop_family"],
                           kinds=(s["meta"][0]["kind"], s["meta"][1]["kind"]))
