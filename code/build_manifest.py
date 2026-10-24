"""Step 1: audit 4K source videos and build a source-grouped manifest.

For every video:
  * ffprobe: codec, resolution, fps (nominal vs average -> VFR), bit depth,
    transfer function (HDR), bitrate -> bits per pixel (source-quality proxy)
  * full decode at 128x72 luma: decode errors, repeated-frame fraction,
    longest repeat run, and REPEAT CADENCE (24/30 fps material upconverted to
    60 fps by frame repetition; it would teach the model that repeats are normal)
  * perceptual hashes of 3 frames for near-duplicate grouping

Then: group related clips (same shoot, cross-dataset copies), drop any group
that matches the excluded study sources, and split train/valid/audit BY GROUP.

Usage:
  python build_manifest.py --data LMM-10K=/data/lmm10k --data Inter4K=/data/inter4k \
      --exclude /data/study_sources --out manifest_dir --workers 16

Re-runs are resumable: per-video results are cached in <out>/probe_cache/.
Uses only ffmpeg/ffprobe + numpy.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".y4m"}
TW, TH = 128, 72
DUP_MEAN = 0.25          # mean |diff| (gray levels) below this ...
DUP_MAX = 4              # ... and max |diff| at most this -> repeated frame
HW, HH = 960, 540        # repeat confirmation + crop detection resolution
HI_DUP_MEAN = 0.25       # a suspected repeat is confirmed at 960x540 when mean |diff| < this
HI_DUP_MAX = 2           # ... and max |diff| <= this (slow sub-pixel motion is not a hold)
STATIC_MOTION = 0.3      # mean 128x72 |diff| per frame below this = near-static
CROP_LIMIT = 24          # 8-bit luma at or below this across the whole clip = black bar
CROP_COVER = 0.5         # bar edge: the first active row/col must be > CROP_LIMIT over at least
                         # this fraction of its width in at least half the frames
HDR_TRC = {"smpte2084", "arib-std-b67"}
SCAN_VERSION = 5         # bump when scan() output changes; part of the cache key
DENSE_EVERY = 10         # dataset clips: dhash every N frames (study-exclusion matching)
FLAT_STD = 5.0           # thumbnails with luma std below this give unreliable fingerprints


# ----------------------------------------------------------------- probing ---
def _frac(s):
    try:
        f = Fraction(s)
        return float(f) if f.denominator else 0.0
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,"
           "color_transfer,color_primaries,bit_rate,nb_frames,duration:"
           "stream_tags=rotate:stream_side_data=rotation:format=duration,bit_rate",
           "-of", "json", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"probe_error": r.stderr.strip()[:200]}
    j = json.loads(r.stdout or "{}")
    st = (j.get("streams") or [{}])[0]
    fm = j.get("format", {})
    w, h = int(st.get("width", 0)), int(st.get("height", 0))
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float((st.get("tags") or {}).get("rotate", 0) or 0))
    if abs(rot) in (90, 270):
        w, h = h, w
    dur = float(st.get("duration") or fm.get("duration") or 0)
    br = float(st.get("bit_rate") or fm.get("bit_rate") or 0)
    r_fps, a_fps = _frac(st.get("r_frame_rate")), _frac(st.get("avg_frame_rate"))
    fps = a_fps or r_fps
    pix = st.get("pix_fmt", "")
    return dict(codec=st.get("codec_name", ""), width=w, height=h,
                fps=round(fps, 3), fps_nominal=round(r_fps, 3),
                vfr=bool(r_fps and a_fps and abs(r_fps - a_fps) / r_fps > 0.01),
                pix_fmt=pix, bit_depth=12 if "12" in pix else 10 if "10" in pix else 8,
                transfer=st.get("color_transfer", ""), primaries=st.get("color_primaries", ""),
                duration=round(dur, 3), bitrate=br,
                bpp=round(br / (w * h * fps), 5) if w and h and fps and br else 0.0)


# --------------------------------------------------------- frame scanning ---
def dhash(img):
    # block-average to 8 rows x 9 cols, compare horizontal neighbours
    rows = np.array_split(img.astype(np.float32), 8, axis=0)
    g = np.array([[c.mean() for c in np.array_split(r, 9, axis=1)] for r in rows])
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def tiny(img):
    """128x72 thumbnail -> 32x18 by 4x4 block mean (exclusion fingerprint)."""
    return img.astype(np.float32).reshape(TH // 4, 4, TW // 4, 4).mean((1, 3))


def highpass(t):
    """Remove the 3x3 local mean so correlation compares detail, not global
    layout (a bright-sky-over-dark-ground layout alone correlates ~0.9)."""
    p = np.pad(t, 1, mode="symmetric")
    blur = sum(p[dy:dy + t.shape[0], dx:dx + t.shape[1]] for dy in range(3) for dx in range(3)) / 9.0
    return t - blur


def cadence(dup):
    """Detect periodic repetition typical of 24/25/30 -> 50/60 upconversion."""
    d = dup[1:].astype(np.float32)
    if len(d) < 30 or d.mean() < 0.15:
        return 0, 0.0
    d = d - d.mean()
    denom = (d * d).sum() + 1e-9
    best_lag, best = 0, 0.0
    for lag in range(2, 6):
        ac = float((d[:-lag] * d[lag:]).sum() / denom)
        if ac > best:
            best_lag, best = lag, ac
    return best_lag, best


def _read_all(fobj, out):
    out.append(fobj.read())


def _hi_stats(path_fifo, res):
    """Stream 960x540 luma frames: per-transition mean/max |diff|, the
    elementwise max frame and per-frame row/col coverage (crop detection)."""
    fsz = HW * HH
    prev, mx = None, None
    hm, hx, rows, cols = [], [], [], []
    with open(path_fifo, "rb") as f:
        while True:
            b = f.read(fsz)
            if len(b) < fsz:
                break
            fr = np.frombuffer(b, np.uint8).reshape(HH, HW)
            lit = fr > CROP_LIMIT
            rows.append(lit.mean(1).astype(np.float32))
            cols.append(lit.mean(0).astype(np.float32))
            mx = fr.copy() if mx is None else np.maximum(mx, fr)
            if prev is not None:
                d = np.abs(fr.astype(np.int16) - prev.astype(np.int16))
                hm.append(float(d.mean()))
                hx.append(int(d.max()))
            prev = fr
    res.update(hi_mean=np.array([0.0] + hm), hi_max=np.array([0] + hx), max_frame=mx,
               rows=np.array(rows), cols=np.array(cols))


def detect_active(max_frame, rows, cols, w, h):
    """Letterbox/pillarbox detection: bars are border rows/cols that stay at or
    below CROP_LIMIT in every frame. The picture edge next to a bar must span
    the frame (the rows/cols within 3 px of it are lit over >= CROP_COVER of
    their length in >= half the frames), which rejects dark content such as
    a moon or a small object on black. Returns [x, y, w, h] in source pixels."""
    if max_frame is None:
        return [0, 0, w, h]
    rmax, cmax = max_frame.max(1), max_frame.max(0)

    def edges(m, cover):
        n = len(m)
        a = 0
        while a < n and m[a] <= CROP_LIMIT:
            a += 1
        b = n
        while b > a and m[b - 1] <= CROP_LIMIT:
            b -= 1
        if b - a < n // 4:                              # nearly all black: no crop
            return 0, n
        spans_frame = lambda lo, hi: (cover[:, lo:hi].max(1) >= CROP_COVER).mean() >= 0.5
        ok_a = a == 0 or spans_frame(a, min(b, a + 3))
        ok_b = b == n or spans_frame(max(a, b - 3), b)
        return (a if ok_a else 0), (b if ok_b else n)

    y0, y1 = edges(rmax, rows)
    x0, x1 = edges(cmax, cols)
    sx, sy = w / HW, h / HH
    X0, X1 = int(np.ceil(x0 * sx / 2) * 2), int(np.floor(x1 * sx / 2) * 2)
    Y0, Y1 = int(np.ceil(y0 * sy / 2) * 2), int(np.floor(y1 * sy / 2) * 2)
    return [X0, Y0, X1 - X0, Y1 - Y0]


def find_spans(motion, cut, conf, min_len):
    """Split a clip at scene cuts and at confirmed repeat runs in moving
    content; keep spans of >= min_len frames. A confirmed run t..t+k-1 (each
    frame equal to its predecessor) removes those frames: the span before
    ends at t (exclusive), the next starts at t+k. Runs in near-static
    context (median motion of the surrounding non-repeat frames below
    STATIC_MOTION) are content, not holds, and do not split.
    Returns [[start, end), ...], the boundary list and the held frame count."""
    n = len(motion)
    bad = np.zeros(n, bool)          # frame t cannot be in a span
    brk = np.zeros(n, bool)          # a span cannot contain both t-1 and t
    brk[cut] = True
    held = 0
    t = 1
    while t < n:
        if conf[t]:
            k = t
            while k < n and conf[k]:
                k += 1
            lo, hi = max(1, t - 30), min(n, k + 30)
            ctx = np.r_[motion[lo:t], motion[k:hi]]
            ctx = ctx[ctx > 0]
            if len(ctx) and float(np.median(ctx)) >= STATIC_MOTION:
                bad[t:k] = True
                if k < n:
                    brk[k] = True
                held += k - t
            t = k
        else:
            t += 1
    spans, start = [], None
    for t in range(n):
        if bad[t]:
            if start is not None:
                spans.append([start, t])
            start = None
            continue
        if start is None:
            start = t
        elif brk[t]:
            spans.append([start, t])
            start = t
    if start is not None:
        spans.append([start, n])
    return [sp for sp in spans if sp[1] - sp[0] >= min_len], held


def scan(path, max_seconds, min_span=240, dense_every=DENSE_EVERY, threads=0, w=0, h=0):
    tmp = tempfile.mkdtemp(prefix="bm_")
    fifo = os.path.join(tmp, "hi.raw")
    os.mkfifo(fifo)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
    if threads:
        cmd += ["-threads", str(threads)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    # passthrough: count the frames actually stored; the default CFR mode for
    # rawvideo output may duplicate/drop frames on irregular timestamps.
    # One decode feeds the 128x72 analysis stream (stdout) and the 960x540
    # confirmation/crop stream (fifo); the 128x72 chain is unchanged.
    cmd += ["-i", path, "-an", "-sn", "-fps_mode", "passthrough", "-filter_complex",
            f"[0:v]split=2[a][b];[a]scale={TW}:{TH}:flags=area,format=gray[lo];"
            f"[b]scale={HW}:{HH}:flags=area,format=gray[hi]",
            "-map", "[lo]", "-f", "rawvideo", "pipe:1",
            "-map", "[hi]", "-fps_mode", "passthrough", "-f", "rawvideo", fifo]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lo_buf, err_buf, hi = [], [], {}
    th = [threading.Thread(target=_read_all, args=(p.stdout, lo_buf)),
          threading.Thread(target=_read_all, args=(p.stderr, err_buf)),
          threading.Thread(target=_hi_stats, args=(fifo, hi))]
    for x in th:
        x.start()
    p.wait()
    for x in th:
        x.join()
    shutil.rmtree(tmp, ignore_errors=True)
    buf = np.frombuffer(lo_buf[0], np.uint8)
    n = len(buf) // (TW * TH)
    out = dict(decode_errors=len(err_buf[0].decode(errors="ignore").strip().splitlines()),
               frames_scanned=n)
    if n < 2 or len(hi.get("hi_mean", [])) != n:
        if n >= 2:
            out["decode_errors"] += 1          # lo/hi frame counts disagree
        out.update(dup_frac=0.0, longest_repeat=0, cadence_lag=0, cadence_strength=0.0,
                   unique_fps_ratio=1.0, mean_motion=0.0, hashes=[], n_cuts=0,
                   seg_start=-1, seg_motion=0.0, seg_frames=0, spans=[], span_motion=[],
                   span_static=[], usable_frames=0, held_frames=0, dup_frac_lo=0.0,
                   longest_repeat_lo=0, active_xywh=[0, 0, w, h],
                   dense_hashes=[], dense_std=[], dense_tiny=[])
        return out
    fr = buf[: n * TW * TH].reshape(n, TH, TW)
    sus = np.zeros(n, bool)                  # suspected repeat (128x72 rule)
    motion = np.zeros(n, np.float32)
    for t in range(1, n):
        d = np.abs(fr[t].astype(np.int16) - fr[t - 1].astype(np.int16))
        motion[t] = d.mean()
        sus[t] = d.mean() < DUP_MEAN and d.max() <= DUP_MAX
    conf = sus & (hi["hi_mean"] < HI_DUP_MEAN) & (hi["hi_max"] <= HI_DUP_MAX)
    conf[0] = False

    def longest_run(x):
        run = best = 0
        for v in x[1:]:
            run = run + 1 if v else 0
            best = max(best, run)
        return best
    # cadence stays on the 128x72 rule: upconverted material that was re-encoded
    # afterwards has near-identical (not bit-exact) repeats the 960x540 rule rejects
    lag, strength = cadence(sus)
    med = float(np.median(motion[1:])) if n > 2 else 0.0
    cut = motion > max(25.0, 8.0 * med + 1.0)          # scene cut / hard transition
    cut[0] = False
    spans, held = find_spans(motion, cut, conf, min_span)
    smot = [round(float(motion[a + 1:b].mean()), 3) for a, b in spans]
    out["n_cuts"] = int(cut.sum())
    out.update(spans=spans, span_motion=smot, span_static=[m < STATIC_MOTION for m in smot],
               usable_frames=int(sum(b - a for a, b in spans)), held_frames=int(held))
    if spans:                                # primary span = longest (earliest on ties)
        k = max(range(len(spans)), key=lambda i: (spans[i][1] - spans[i][0], -i))
        out.update(seg_start=spans[k][0], seg_frames=spans[k][1] - spans[k][0], seg_motion=smot[k])
    else:
        out.update(seg_start=-1, seg_frames=0, seg_motion=0.0)
    out["active_xywh"] = detect_active(hi["max_frame"], hi["rows"], hi["cols"], w, h)
    dense = range(0, n, max(1, dense_every))
    out["dense_hashes"] = [format(dhash(fr[t]), "016x") for t in dense]
    out["dense_std"] = [round(float(fr[t].std()), 2) for t in dense]
    out["dense_tiny"] = [np.clip(np.round(tiny(fr[t])), 0, 255).astype(np.uint8).tobytes().hex()
                         for t in dense]
    keys = [fr[int(n * q)] for q in (0.1, 0.5, 0.9)]
    out.update(dup_frac=round(float(conf[1:].mean()), 4), longest_repeat=int(longest_run(conf)),
               dup_frac_lo=round(float(sus[1:].mean()), 4), longest_repeat_lo=int(longest_run(sus)),
               cadence_lag=int(lag), cadence_strength=round(strength, 3),
               unique_fps_ratio=round(1.0 - float(sus[1:].mean()), 4),
               mean_motion=round(float(motion[1:].mean()), 3),
               hashes=[format(dhash(k), "016x") for k in keys])
    return out


def analyse(args):
    dataset, path, cache_dir, max_seconds, min_span, threads = args
    dense_every = 1 if dataset == "__EXCLUDE__" else DENSE_EVERY
    key = hashlib.sha1(json.dumps([os.path.abspath(path), min_span, max_seconds,
                                   dense_every, SCAN_VERSION]).encode()).hexdigest()
    cpath = os.path.join(cache_dir, key + ".json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f)
    rec = dict(dataset=dataset, path=os.path.abspath(path),
               size_mb=round(os.path.getsize(path) / 2**20, 2))
    rec.update(probe(path))
    if "probe_error" not in rec:
        rec.update(scan(path, max_seconds, min_span, dense_every, threads,
                        rec["width"], rec["height"]))
    with open(cpath, "w") as f:
        json.dump(rec, f)
    return rec


# ------------------------------------------------------------ decisions ---
def flags_for(r, a):
    f = []
    if "probe_error" in r:
        return ["probe_error"]
    if r.get("frames_scanned", 0) < 2 or r.get("decode_errors", 0) > 0:
        f.append("decode_error")
    if r["height"] < a.min_height or r["width"] < a.min_width:
        f.append("not_4k" if r["height"] < r["width"] else "portrait_or_small")
    if r["height"] > r["width"]:
        f.append("portrait")
    if r["fps"] < a.min_fps:
        f.append("low_fps")
    if r["vfr"]:
        f.append("vfr")
    if r["transfer"] in HDR_TRC:
        f.append("hdr")
    if r["duration"] < a.min_duration:
        f.append("too_short")
    if 0 < r["bpp"] < a.min_bpp:
        f.append("low_bpp")
    if (r.get("unique_fps_ratio", 1) < 0.8 and r.get("cadence_strength", 0) > 0.3):
        f.append("upconverted_cadence")
    elif r.get("held_frames", 0) > 0:
        f.append("native_freeze")          # soft: held frames are cut out of the spans
    if r.get("span_static") and all(r["span_static"]):
        f.append("static")                 # soft: near-static clips are kept, tagged
    ax = r.get("active_xywh") or [0, 0, r["width"], r["height"]]
    if ax[2] < r["width"] or ax[3] < r["height"]:
        f.append("letterbox")              # soft: use active_xywh downstream
    if not r.get("spans"):
        f.append("no_usable_span")
    return f


HARD = {"no_usable_span", "probe_error", "decode_error", "not_4k", "portrait_or_small", "portrait",
        "low_fps", "vfr", "hdr", "too_short", "upconverted_cadence", "low_bpp"}


# ------------------------------------------------------------ grouping ---
def _key_feats(r):
    """High-passed, normalised 32x18 thumbnails at the 3 key-frame positions
    (nearest dense frame: <= DENSE_EVERY/2 frames away), or None if flat."""
    step = 1 if r["dataset"] == "__EXCLUDE__" else DENSE_EVERY
    n, tiny_, sd = r.get("frames_scanned", 0), r.get("dense_tiny", []), r.get("dense_std", [])
    out = []
    for q in (0.1, 0.5, 0.9):
        k = min(len(tiny_) - 1, int(round(int(n * q) / step))) if tiny_ else -1
        if k < 0 or sd[k] < FLAT_STD:
            out.append(None)
            continue
        v = highpass(np.frombuffer(bytes.fromhex(tiny_[k]), np.uint8).astype(np.float32)
                     .reshape(TH // 4, TW // 4)).ravel()
        v -= v.mean()
        out.append(v / (np.linalg.norm(v) + 1e-6))
    return out


def group_ids(records, thresh, verify_ncc=0.0):
    """Union-find over videos whose key-frame hashes are within `thresh` bits.
    With verify_ncc > 0 a hash match only links two videos if the matched key
    frames' high-passed thumbnails also correlate >= verify_ncc: 64-bit dHashes
    of generic layouts (e.g. a bright centre band, f0f0f0f0...) collide across
    unrelated clips and union-find chains them into huge false groups."""
    owners, hs, keyk = [], [], []
    for i, r in enumerate(records):
        for k, h in enumerate(r.get("hashes", [])):
            owners.append(i)
            hs.append(int(h, 16))
            keyk.append(k)
    feats = [_key_feats(r) for r in records] if verify_ncc > 0 else None
    K = np.array(keyk)
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if hs:
        H = np.array(hs, dtype=np.uint64)
        O = np.array(owners)
        bytes_ = H.view(np.uint8).reshape(-1, 8)
        lut = np.array([bin(i).count("1") for i in range(256)], np.uint8)
        chunk = 2048
        for s in range(0, len(H), chunk):
            x = bytes_[s:s + chunk, None, :] ^ bytes_[None, :, :]
            dist = lut[x].sum(-1, dtype=np.int32)
            ii, jj = np.nonzero(dist <= thresh)
            for i, j, ki, kj in zip(O[ii + s], O[jj], K[ii + s], K[jj]):
                if i != j and feats is not None:
                    fa, fb = feats[i][ki], feats[j][kj]
                    if fa is None or fb is None or float(fa @ fb) < verify_ncc:
                        continue
                if i != j:
                    a, b = find(int(i)), find(int(j))
                    if a != b:
                        parent[a] = b
    return [find(i) for i in range(len(records))]


def exclusion_matches(records, ncc_thresh, min_frames):
    """Dense study-exclusion check. Excluded clips are fingerprinted on every
    frame, dataset clips every DENSE_EVERY frames, as 32x18 luma thumbnails
    (near-flat ones skipped). A dataset frame matches a study frame when the
    normalised correlation of their high-passed thumbnails is >= ncc_thresh. A copy or excerpt keeps
    a constant frame offset (study frame - dataset frame) while chance matches
    do not, so a clip match needs >= min_frames dataset frames matching at a
    consistent offset (5-frame bins merged with neighbours).
    (64-bit dHashes and plain low-res correlation were too unspecific here:
    unrelated static scenes with a similar layout matched study frames.) Catches excerpts that the 3 key-frame
    hashes used for grouping miss (different start time or length).
    Returns {record index: {excluded record index: offset-consistent frames}}."""
    def dense(i):
        r = records[i]
        step = 1 if r["dataset"] == "__EXCLUDE__" else DENSE_EVERY
        out = [(highpass(np.frombuffer(bytes.fromhex(t), np.uint8).astype(np.float32)
                         .reshape(TH // 4, TW // 4)).ravel(), k * step)
               for k, (t, sd) in enumerate(zip(r.get("dense_tiny", []), r.get("dense_std", [])))
               if sd >= FLAT_STD]
        if not out:
            return None, None
        z = np.stack([v for v, _ in out])
        z -= z.mean(1, keepdims=True)
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-6
        return z, np.array([t for _, t in out])
    ez, eo, ef = [], [], []
    for i, r in enumerate(records):
        if r["dataset"] == "__EXCLUDE__":
            z, t = dense(i)
            if z is not None:
                ez.append(z)
                eo += [i] * len(z)
                ef.append(t)
    out = {}
    if not ez:
        return out
    E, EO, EF = np.concatenate(ez), np.array(eo), np.concatenate(ef)
    for i, r in enumerate(records):
        if r["dataset"] == "__EXCLUDE__":
            continue
        D, DF = dense(i)
        if D is None:
            continue
        hit = (D @ E.T) >= ncc_thresh        # (dataset frames, study frames)
        if not hit.any():
            continue
        votes = defaultdict(set)              # (excluded clip, offset bin) -> dataset frames
        for f, j in zip(*np.nonzero(hit)):
            b = int(np.floor((EF[j] - DF[f]) / 5.0))
            for bb in (b - 1, b, b + 1):
                votes[(int(EO[j]), bb)].add(int(f))
        cnt = Counter()
        for (o, _), fs in votes.items():
            cnt[o] = max(cnt[o], len(fs))
        cnt = {o: c for o, c in cnt.items() if c >= min_frames}
        if cnt:
            out[i] = cnt
    return out


def assign_splits(groups, fractions, seed):
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    cut1, cut2 = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    split = {g: ("train" if k < cut1 else "valid" if k < cut2 else "audit")
             for k, g in enumerate(uniq)}
    return [split[g] for g in groups]


# ------------------------------------------------------------------ main ---
def list_videos(root):
    out = []
    for d, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXT:
                out.append(os.path.join(d, fn))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="NAME=/path (repeatable)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="folder of study/test source videos to keep out of training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="decode at most this many seconds per video (0 = all)")
    ap.add_argument("--min-width", type=int, default=3840)
    ap.add_argument("--min-height", type=int, default=2160)
    ap.add_argument("--min-fps", type=float, default=50.0)
    ap.add_argument("--min-duration", type=float, default=3.0)
    ap.add_argument("--min-bpp", type=float, default=0.03,
                    help="hard flag: sources below this bits/pixel (cutoff set at Step 1 review)")
    ap.add_argument("--hash-thresh", type=int, default=6)
    ap.add_argument("--exclude-ncc", type=float, default=0.6,
                    help="dense exclusion: high-passed 32x18 thumbnail correlation for a frame match")
    ap.add_argument("--group-verify-ncc", type=float, default=0.6,
                    help="key-hash links must also pass this high-passed thumbnail correlation "
                         "(0 = original hash-only grouping)")
    ap.add_argument("--exclude-min-frames", type=int, default=3,
                    help="dense exclusion: matched (non-flat) frames needed to call a study overlap")
    ap.add_argument("--decode-threads", type=int, default=0,
                    help="ffmpeg -threads per scan (0 = ffmpeg default)")
    ap.add_argument("--split", default="0.85,0.05,0.10")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-span-frames", type=int, default=240,
                    help="keep every span between cuts / confirmed holds of at least this many frames")
    ap.add_argument("--encodes-per-source", type=int, default=15,
                    help="for the storage estimate: 5 ladder + 8 capture + 2 post-drop re-encodes")
    ap.add_argument("--mbps", type=float, default=15.0,
                    help="for the storage estimate: average bitrate of stored encodes")
    a = ap.parse_args()

    cache = os.path.join(a.out, f"probe_cache_span{a.min_span_frames}")
    os.makedirs(cache, exist_ok=True)
    jobs = []
    for spec in a.data:
        name, root = spec.split("=", 1)
        jobs += [(name, p, cache, a.max_seconds, a.min_span_frames, a.decode_threads)
                 for p in list_videos(root)]
    for root in a.exclude:
        jobs += [("__EXCLUDE__", p, cache, a.max_seconds, a.min_span_frames, a.decode_threads)
                 for p in list_videos(root)]
    print(f"{len(jobs)} videos to analyse with {a.workers} workers", flush=True)

    records = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(analyse, jobs, chunksize=4), 1):
            records.append(r)
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
    records.sort(key=lambda r: r["path"])

    groups = group_ids(records, a.hash_thresh, a.group_verify_ncc)
    excluded_groups = {g for g, r in zip(groups, records) if r["dataset"] == "__EXCLUDE__"}
    key_matched = {i for i, (g, r) in enumerate(zip(groups, records))
                   if r["dataset"] != "__EXCLUDE__" and g in excluded_groups}
    dense_matched = exclusion_matches(records, a.exclude_ncc, a.exclude_min_frames)
    excluded_groups |= {groups[i] for i in dense_matched}
    excl_name = {i: os.path.basename(r["path"]) for i, r in enumerate(records)
                 if r["dataset"] == "__EXCLUDE__"}
    fr = [float(x) for x in a.split.split(",")]

    rows = []
    for i, (r, g) in enumerate(zip(records, groups)):
        if r["dataset"] == "__EXCLUDE__":
            continue
        f = flags_for(r, a)
        if g in excluded_groups:
            f.append("matches_excluded_source")
        hard = [x for x in f if x in HARD or x == "matches_excluded_source"]
        how = []
        if i in key_matched:
            how.append("keyhash")
        if i in dense_matched:
            how += [f"dense:{excl_name[o]}:{c}" for o, c in sorted(dense_matched[i].items())]
        if g in excluded_groups and not how:
            how.append("group")
        rows.append(dict(r, flags="|".join(f), keep=not hard, group_id=f"g{g}",
                         hashes="|".join(r.get("hashes", [])), excl_match="|".join(how),
                         spans=json.dumps(r.get("spans", [])),
                         span_motion=json.dumps(r.get("span_motion", [])),
                         span_static=json.dumps(r.get("span_static", [])),
                         active_xywh=json.dumps(r.get("active_xywh", []))))
    keep_groups = [row["group_id"] for row in rows if row["keep"]]
    split_of = dict(zip(keep_groups, assign_splits(keep_groups, fr, a.seed)))
    for row in rows:
        row["split"] = split_of.get(row["group_id"], "") if row["keep"] else ""
        st = row.get("seg_start", -1)
        row["seg_start_s"] = round(st / row["fps"], 3) if st is not None and st >= 0 and row["fps"] else ""

    cols = ["dataset", "path", "keep", "split", "group_id", "flags", "codec", "width",
            "height", "fps", "fps_nominal", "vfr", "pix_fmt", "bit_depth", "transfer",
            "primaries", "duration", "bitrate", "bpp", "size_mb", "frames_scanned",
            "decode_errors", "dup_frac", "longest_repeat", "cadence_lag",
            "cadence_strength", "unique_fps_ratio", "mean_motion", "n_cuts",
            "seg_start", "seg_start_s", "seg_motion", "hashes", "probe_error",
            "seg_frames", "excl_match", "spans", "usable_frames", "span_motion",
            "span_static", "held_frames", "active_xywh", "dup_frac_lo", "longest_repeat_lo"]
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- report
    lines = []
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds]
        kept = [r for r in rs if r["keep"]]
        fl = Counter(x for r in rs for x in r["flags"].split("|") if x)
        bpp = np.array([r["bpp"] for r in rs if r.get("bpp")])
        lines.append(f"\n== {ds}: {len(rs)} videos, kept {len(kept)} "
                     f"({sum(r['duration'] for r in kept) / 3600:.2f} h of clips, "
                     f"{sum(r['usable_frames'] / r['fps'] for r in kept) / 3600:.2f} h usable in "
                     f"{sum(len(r['spans']) and len(json.loads(r['spans'])) for r in kept)} spans)")
        lines.append("   flags: " + ", ".join(f"{k}={v}" for k, v in fl.most_common()))
        lines.append("   fps: " + ", ".join(f"{k}={v}" for k, v in
                                           Counter(round(r['fps']) for r in rs).most_common(6)))
        lines.append("   codecs: " + ", ".join(f"{k}={v}" for k, v in
                                              Counter(r['codec'] for r in rs).most_common()))
        if len(bpp):
            q = np.percentile(bpp, [5, 25, 50, 75, 95])
            lines.append("   bpp p5/p25/p50/p75/p95: " + " / ".join(f"{x:.4f}" for x in q))
    kept = [r for r in rows if r["keep"]]
    gsize = Counter(r["group_id"] for r in kept)
    lines.append(f"\ngroups: {len(gsize)} (largest {max(gsize.values()) if gsize else 0} clips); "
                 f"multi-clip groups: {sum(1 for v in gsize.values() if v > 1)}")
    cross = defaultdict(set)
    for r in kept:
        cross[r["group_id"]].add(r["dataset"])
    lines.append(f"groups spanning >1 dataset: {sum(1 for v in cross.values() if len(v) > 1)}")
    lines.append("split (clips): " + ", ".join(f"{k}={v}" for k, v in
                                             Counter(r['split'] for r in kept).items()))
    lines.append(f"excluded-source matches dropped: "
                 f"{sum('matches_excluded_source' in r['flags'] for r in rows)} "
                 f"(excluded study clips scanned: {len(excl_name)}; "
                 f"key-hash group matches: {len(key_matched)}, dense matches: {len(dense_matched)}, "
                 f"same-group only: {sum(r['excl_match'] == 'group' for r in rows)})")
    for r in rows:
        if r["excl_match"]:
            lines.append(f"   {r['dataset']}/{os.path.basename(r['path'])} {r['group_id']} "
                         f"[{r['excl_match']}] {'flags=' + r['flags']}")
    gsplit = defaultdict(set)
    for r in kept:
        gsplit[r["group_id"]].add(r["split"])
    lines.append(f"kept clips: {len(kept)}; kept groups appearing in >1 split: "
                 f"{sum(1 for v in gsplit.values() if len(v) > 1)}")
    sp = [(b - a_, m, st) for r in kept for (a_, b), m, st in
          zip(json.loads(r["spans"]), json.loads(r["span_motion"]), json.loads(r["span_static"]))]
    if sp:
        L = np.array([x[0] for x in sp])
        mot = np.array([x[1] for x in sp])
        lines.append(f"usable spans: {len(sp)} (clips with >1 span: "
                     f"{sum(len(json.loads(r['spans'])) > 1 for r in kept)}); span frames "
                     f"p10/p50/p90/max: " + " / ".join(f"{v:.0f}" for v in np.percentile(L, [10, 50, 90, 100])))
        lines.append("span motion p10/p50/p90 (gray levels/frame): " +
                     " / ".join(f"{v:.2f}" for v in np.percentile(mot, [10, 50, 90])) +
                     f"; static spans: {sum(x[2] for x in sp)}")
    lines.append(f"kept clips tagged static: {sum('static' in r['flags'].split('|') for r in kept)}; "
                 f"letterbox: {sum('letterbox' in r['flags'].split('|') for r in kept)}; "
                 f"with held frames cut out: {sum(r['held_frames'] > 0 for r in kept)}")
    usable_s = sum(r["usable_frames"] / r["fps"] for r in kept)
    tb = usable_s * a.encodes_per_source * a.mbps / 8 / 1e6
    lines.append(f"storage estimate: {usable_s / 3600:.2f} h usable x {a.encodes_per_source} encodes "
                 f"@ {a.mbps:g} Mbps = {tb:.2f} TB")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    sys.exit(main())
