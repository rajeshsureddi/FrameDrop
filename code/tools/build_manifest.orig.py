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
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".y4m"}
TW, TH = 128, 72
DUP_MEAN = 0.25          # mean |diff| (gray levels) below this ...
DUP_MAX = 4              # ... and max |diff| at most this -> repeated frame
HDR_TRC = {"smpte2084", "arib-std-b67"}


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


def scan(path, max_seconds):
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", path, "-an", "-sn",
            "-vf", f"scale={TW}:{TH}:flags=area,format=gray",
            "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True)
    buf = np.frombuffer(r.stdout, np.uint8)
    n = len(buf) // (TW * TH)
    out = dict(decode_errors=len(r.stderr.decode(errors="ignore").strip().splitlines()),
               frames_scanned=n)
    if n < 2:
        out.update(dup_frac=0.0, longest_repeat=0, cadence_lag=0, cadence_strength=0.0,
                   unique_fps_ratio=1.0, mean_motion=0.0, hashes=[])
        return out
    fr = buf[: n * TW * TH].reshape(n, TH, TW)
    dup = np.zeros(n, bool)
    motion = np.zeros(n, np.float32)
    for t in range(1, n):
        d = np.abs(fr[t].astype(np.int16) - fr[t - 1].astype(np.int16))
        motion[t] = d.mean()
        dup[t] = d.mean() < DUP_MEAN and d.max() <= DUP_MAX
    run = longest = 0
    for x in dup[1:]:
        run = run + 1 if x else 0
        longest = max(longest, run)
    lag, strength = cadence(dup)
    med = float(np.median(motion[1:])) if n > 2 else 0.0
    cut = motion > max(25.0, 8.0 * med + 1.0)          # scene cut / hard transition
    cut[0] = False
    out["n_cuts"] = int(cut.sum())
    out.update(pick_segment(motion, dup, cut, SEGMENT_FRAMES.get("n", 300)))
    keys = [fr[int(n * q)] for q in (0.1, 0.5, 0.9)]
    out.update(dup_frac=round(float(dup[1:].mean()), 4), longest_repeat=int(longest),
               cadence_lag=int(lag), cadence_strength=round(strength, 3),
               unique_fps_ratio=round(1.0 - float(dup[1:].mean()), 4),
               mean_motion=round(float(motion[1:].mean()), 3),
               hashes=[format(dhash(k), "016x") for k in keys])
    return out


SEGMENT_FRAMES = {}


def pick_segment(motion, dup, cut, L):
    """Best L-frame window: no cut, no repeat runs > 2 frames, not static;
    among valid windows prefer the one with the clip's typical motion."""
    n = len(motion)
    if n < L:
        return dict(seg_start=-1, seg_motion=0.0)
    bad = cut.copy()
    run = 0
    for t in range(1, n):
        run = run + 1 if dup[t] else 0
        if run > 2:
            bad[t] = True
    cs_bad = np.r_[0, np.cumsum(bad)]
    cs_m = np.r_[0, np.cumsum(motion)]
    starts = np.arange(0, n - L + 1, max(1, L // 10))
    valid = [(st, (cs_m[st + L] - cs_m[st + 1]) / (L - 1)) for st in starts
             if cs_bad[st + L] - cs_bad[st + 1] == 0]
    valid = [(st, m) for st, m in valid if m > 0.3]   # skip near-static windows
    if not valid:
        return dict(seg_start=-1, seg_motion=0.0)
    target = np.median([m for _, m in valid])
    st, m = min(valid, key=lambda v: abs(v[1] - target))
    return dict(seg_start=int(st), seg_motion=round(float(m), 3))


def _init_worker(nframes):
    SEGMENT_FRAMES["n"] = nframes


def analyse(args):
    dataset, path, cache_dir, max_seconds = args
    key = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()
    cpath = os.path.join(cache_dir, key + ".json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f)
    rec = dict(dataset=dataset, path=os.path.abspath(path),
               size_mb=round(os.path.getsize(path) / 2**20, 2))
    rec.update(probe(path))
    if "probe_error" not in rec:
        rec.update(scan(path, max_seconds))
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
    elif r.get("longest_repeat", 0) > a.max_repeat_run:
        f.append("native_freeze")
    if r.get("dup_frac", 0) > 0.9:
        f.append("static")
    if r.get("seg_start", 0) < 0:
        f.append("no_clean_segment")
    return f


HARD = {"no_clean_segment", "probe_error", "decode_error", "not_4k", "portrait_or_small", "portrait",
        "low_fps", "vfr", "hdr", "too_short", "upconverted_cadence", "native_freeze"}


# ------------------------------------------------------------ grouping ---
def group_ids(records, thresh):
    """Union-find over videos whose key-frame hashes are within `thresh` bits."""
    owners, hs = [], []
    for i, r in enumerate(records):
        for h in r.get("hashes", []):
            owners.append(i)
            hs.append(int(h, 16))
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
            for i, j in zip(O[ii + s], O[jj]):
                if i != j:
                    a, b = find(int(i)), find(int(j))
                    if a != b:
                        parent[a] = b
    return [find(i) for i in range(len(records))]


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
    ap.add_argument("--min-bpp", type=float, default=0.02,
                    help="flag (soft) sources below this bits/pixel; inspect the histogram first")
    ap.add_argument("--max-repeat-run", type=int, default=2)
    ap.add_argument("--hash-thresh", type=int, default=6)
    ap.add_argument("--split", default="0.85,0.05,0.10")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--segment", type=float, default=5.0,
                    help="seconds of each source used downstream (trimmed when encoding)")
    ap.add_argument("--encodes-per-source", type=int, default=11,
                    help="for the storage estimate: clean ladder + capture-plan variants")
    ap.add_argument("--mbps", type=float, default=15.0,
                    help="for the storage estimate: average bitrate of stored encodes")
    a = ap.parse_args()

    cache = os.path.join(a.out, f"probe_cache_seg{a.segment:g}")
    os.makedirs(cache, exist_ok=True)
    nseg = int(round(a.segment * 60))
    jobs = []
    for spec in a.data:
        name, root = spec.split("=", 1)
        jobs += [(name, p, cache, a.max_seconds)
                 for p in list_videos(root)]
    for root in a.exclude:
        jobs += [("__EXCLUDE__", p, cache, a.max_seconds)
                 for p in list_videos(root)]
    print(f"{len(jobs)} videos to analyse with {a.workers} workers", flush=True)

    records = []
    with Pool(a.workers, initializer=_init_worker, initargs=(nseg,)) as pool:
        for k, r in enumerate(pool.imap_unordered(analyse, jobs, chunksize=4), 1):
            records.append(r)
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
    records.sort(key=lambda r: r["path"])

    groups = group_ids(records, a.hash_thresh)
    excluded_groups = {g for g, r in zip(groups, records) if r["dataset"] == "__EXCLUDE__"}
    fr = [float(x) for x in a.split.split(",")]

    rows = []
    for r, g in zip(records, groups):
        if r["dataset"] == "__EXCLUDE__":
            continue
        f = flags_for(r, a)
        if g in excluded_groups:
            f.append("matches_excluded_source")
        hard = [x for x in f if x in HARD or x == "matches_excluded_source"]
        rows.append(dict(r, flags="|".join(f), keep=not hard, group_id=f"g{g}",
                         hashes="|".join(r.get("hashes", []))))
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
            "seg_start", "seg_start_s", "seg_motion", "hashes", "probe_error"]
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
                     f"({sum(r['duration'] for r in kept) / 3600:.1f} h)")
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
                 f"{sum('matches_excluded_source' in r['flags'] for r in rows)}")
    mot = np.array([r["seg_motion"] for r in kept]) if kept else np.zeros(1)
    lines.append("segment motion p10/p50/p90 (gray levels/frame): " +
                 " / ".join(f"{v:.2f}" for v in np.percentile(mot, [10, 50, 90])))
    n_src = len(kept)
    tb = n_src * a.encodes_per_source * a.mbps * a.segment / 8 / 1e6
    lines.append(f"storage estimate: {n_src} sources x {a.encodes_per_source} encodes x "
                 f"{a.segment:g}s @ {a.mbps:g} Mbps = {tb:.2f} TB")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    sys.exit(main())
