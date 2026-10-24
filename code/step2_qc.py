"""Step 2e: quality control, encodes manifest and review thumbnails.

  * encodes_manifest.csv: one row per encode (clean / capture / postdrop) from its JSON
  * per-encode checks: decodes without errors, frame count == its span length,
    source resolution (3840x2160), 60 fps, codec == encoder, keyframe count
    consistent with the GOP, bitrate vs target (bitrate modes)
  * PSNR (ffmpeg) and VMAF (libvmaf CLI, vmaf_4k_v0.6.1) of every clean-ladder
    encode of a random 2% of sources against the re-decoded reference span;
    quality must rise with rank for each source (violations listed)
  * realized encoder x mode x preset distribution, storage used, throughput
  * step2_review/: 20 random capture variants, reference | severity 0.5 | 1.0
    (same plan, same rung), at mid-span

python step2_qc.py --manifest .../step1/manifest.csv --out .../step2 --workers 16 [--frac 0.02]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

import step2_common as sc
from step2_ladder import kept_sources

VMAF = os.environ.get("VMAF_BIN", "/work/09997/rs64952/ls6/FrameDrop-VQA/tools_ext/vmaf/bin/vmaf")
VMAF_LIB = os.path.join(os.path.dirname(os.path.dirname(VMAF)), "lib64")
CODEC = {"libx264": "h264", "libx265": "hevc", "libsvtav1": "av1", "libvpx-vp9": "vp9"}
COLS = ["source_id", "group_id", "split", "kind", "plan_id", "severity", "encoder", "gop", "target_kbps",
        "actual_kbps", "rank", "path", "json_path", "frames", "decode_ok",
        # extra (not in the brief's list): settings and span details
        "dataset", "mode", "preset", "tune", "bframes", "rate_kind", "rate_value", "span_index",
        "span_start", "span_end", "expected_frames", "width", "height", "fps", "codec", "keyframes",
        "size_bytes", "crf", "drop_family", "n_dropped", "ok", "check_fail"]


def load_jsons(out, pattern="*.json"):
    for j in sorted(glob.glob(os.path.join(out, "encodes", "*", pattern))):
        if os.path.basename(j).startswith("tmp_") or j.endswith(".qc.json"):
            continue
        yield j, json.load(open(j))


def checks(R):
    p, S = R.get("probe") or {}, R["settings"]
    fails = []
    if not p.get("decode_ok"):
        fails.append("decode")
    if p.get("frames") != R["span_frames"]:
        fails.append(f"frames {p.get('frames')}!={R['span_frames']}")
    if (p.get("width"), p.get("height")) != (3840, 2160):
        fails.append(f"size {p.get('width')}x{p.get('height')}")
    if abs(p.get("fps", 0) - 60) > 0.01:
        fails.append(f"fps {p.get('fps')}")
    if p.get("codec") != CODEC[S["encoder"]]:
        fails.append(f"codec {p.get('codec')}")
    if S["gop"] != "default" and p.get("frames"):
        need = -(-p["frames"] // int(S["gop"]))        # at least one keyframe per GOP
        if p.get("keyframes", 0) < need:
            fails.append(f"keyframes {p.get('keyframes')}<{need} for gop {S['gop']}")
    return fails


def row_of(j, R):
    S, p = R["settings"], R.get("probe") or {}
    f = checks(R)
    return dict(source_id=R["source_id"], group_id=R["group_id"], split=R["split"], kind=R["kind"],
                plan_id=R.get("plan_id", ""), severity=R.get("severity", ""), encoder=S["encoder"],
                gop=S["gop"], target_kbps=R["rate_value"] if R["rate_kind"] == "kbps" else "",
                actual_kbps=p.get("actual_kbps"), rank=R.get("rank", ""), path=R["path"], json_path=j,
                frames=p.get("frames"), decode_ok=p.get("decode_ok"), dataset=R["dataset"], mode=S["mode"],
                preset=S["preset"], tune=S["tune"] or "", bframes=S["bframes"], rate_kind=R["rate_kind"],
                rate_value=R["rate_value"], span_index=R["span_index"], span_start=R["span"][0],
                span_end=R["span"][1], expected_frames=R["span_frames"], width=p.get("width"),
                height=p.get("height"), fps=p.get("fps"), codec=p.get("codec"), keyframes=p.get("keyframes"),
                size_bytes=p.get("size_bytes"), crf=R.get("crf", ""), drop_family=R.get("drop_family", ""),
                n_dropped=R.get("n_dropped", ""), ok=R.get("ok") and not f, check_fail="|".join(f))


# ------------------------------------------------------------- PSNR / VMAF
def _ref_cmd(src, a, b, dst_fmt, dst):
    return ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-fps_mode", "passthrough", "-vf",
            f"trim=start_frame={a}:end_frame={b},setpts=N/(60*TB),format=yuv420p",
            "-f", dst_fmt, "-pix_fmt", "yuv420p", dst]


def quality(job):
    """PSNR (Y, average) and VMAF (4K model) of one encode vs its re-decoded span.
    Cached next to the encode (<encode>.qc.json) so a --quality-only run can go ahead early."""
    src, a, b, enc = job
    cpath = enc[:-4] + ".qc.json"
    if os.path.exists(cpath):
        c = json.load(open(cpath))
        if c.get("psnr") is not None and c.get("vmaf") is not None:
            return enc, c["psnr"], c["vmaf"]
    tmp = tempfile.mkdtemp(prefix="qc_")
    rf, df = os.path.join(tmp, "r.y4m"), os.path.join(tmp, "d.y4m")
    os.mkfifo(rf)
    os.mkfifo(df)
    pr = subprocess.Popen(_ref_cmd(src, a, b, "yuv4mpegpipe", rf), stderr=subprocess.DEVNULL)
    pd = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", enc, "-f", "yuv4mpegpipe",
                           "-pix_fmt", "yuv420p", df], stderr=subprocess.DEVNULL)
    vj = os.path.join(tmp, "v.json")
    env = dict(os.environ, LD_LIBRARY_PATH=VMAF_LIB)
    v = subprocess.run([VMAF, "-r", rf, "-d", df, "--model", "version=vmaf_4k_v0.6.1", "--threads", "4",
                        "--json", "-o", vj, "-q"], capture_output=True, text=True, env=env)
    pr.wait()
    pd.wait()
    vm = json.load(open(vj))["pooled_metrics"]["vmaf"]["mean"] if v.returncode == 0 else None
    ps = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", enc, "-i", src, "-filter_complex",
                         f"[1:v]trim=start_frame={a}:end_frame={b},setpts=N/(60*TB),format=yuv420p[r];"
                         "[0:v]format=yuv420p[d];[d][r]psnr=stats_file=-", "-f", "null", "-"],
                        capture_output=True, text=True)
    vals = [float(t.split(":")[1]) for line in ps.stdout.splitlines() for t in line.split()
            if t.startswith("psnr_y:") and t.split(":")[1] not in ("inf",)]
    psnr = float(np.mean(vals)) if vals else None
    for f in (rf, df, vj):
        if os.path.exists(f):
            os.remove(f)
    os.rmdir(tmp)
    with open(cpath + ".tmp", "w") as f:
        json.dump(dict(psnr=psnr, vmaf=vm, ref=src, span=[a, b]), f)
    os.replace(cpath + ".tmp", cpath)
    return enc, psnr, vm


# ------------------------------------------------------------- thumbnails
def frame_at(path, n, trim=None):
    vf = (f"trim=start_frame={trim[0]}:end_frame={trim[1]},setpts=N/(60*TB)," if trim else "") + \
         f"select=eq(n\\,{n}),scale=640:360,format=bgr24"
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vf", vf, "-frames:v", "1",
                        "-f", "rawvideo", "-"], capture_output=True)
    b = np.frombuffer(r.stdout, np.uint8)
    return b.reshape(360, 640, 3).copy() if len(b) == 640 * 360 * 3 else np.zeros((360, 640, 3), np.uint8)


def review(rows, out, n=20, seed=0):
    d = os.path.join(out, "step2_review")
    os.makedirs(d, exist_ok=True)
    cap = defaultdict(dict)
    for r in rows:
        if r["kind"] == "capture" and r["ok"]:
            cap[(r["plan_id"], r["rank"])][float(r["severity"])] = r
    keys = sorted(k for k, v in cap.items() if len(v) == 2)
    rng = np.random.default_rng(seed)
    pick = [keys[i] for i in sorted(rng.choice(len(keys), min(n, len(keys)), replace=False))]
    lines = []
    for k in pick:
        a, b = cap[k][0.5], cap[k][1.0]
        R = json.load(open(a["json_path"]))
        ra, rb = R["render_span"]
        mid = (rb - ra) // 2
        ref = frame_at(R["source_path"], mid, (ra, rb))
        tiles = [ref, frame_at(a["path"], mid), frame_at(b["path"], mid)]
        names = [o["op"] for o in R["labels"]["ops"]]
        for t, lab in zip(tiles, ["reference", "severity 0.5", "severity 1.0"]):
            cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        img = np.concatenate([tiles[0], np.full((360, 4, 3), 255, np.uint8), tiles[1],
                              np.full((360, 4, 3), 255, np.uint8), tiles[2]], 1)
        cv2.putText(img, f"{k[0]} r{k[1]} {a['encoder']} {a['mode']} | {', '.join(names)}"[:110],
                    (8, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        fn = f"{k[0]}_r{k[1]}.jpg"
        cv2.imwrite(os.path.join(d, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lines.append(f"{fn}: ops {names}, terms {R['labels']['terms']}")
    open(os.path.join(d, "index.txt"), "w").write("\n".join(lines) + "\n")
    return len(pick)


def quality_jobs(a, kept, rows):
    """The random a.frac of kept sources and their verified clean-ladder encodes."""
    rng = np.random.default_rng(a.seed)
    sids = sorted(kept)
    sample = sorted(rng.choice(sids, max(1, int(round(a.frac * len(sids)))), replace=False))
    ss = set(sample)
    jobs, meta = [], {}
    for r in rows:
        if r["kind"] == "clean" and r["ok"] and r["source_id"] in ss:
            jobs.append((kept[r["source_id"]]["path"], int(r["span_start"]), int(r["span_end"]), r["path"]))
            meta[r["path"]] = r
    return jobs, meta, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--quality-only", action="store_true",
                    help="only compute (and cache) PSNR/VMAF of the sampled clean encodes; no report")
    ap.add_argument("--throughput", default="", help="text lines to include (from the job logs)")
    a = ap.parse_args()
    kept = {r["source_id"]: r for r in kept_sources(a.manifest)}
    if a.quality_only:
        rows = [row_of(j, R) for j, R in load_jsons(a.out, "s*_clean_r*.json")]
        jobs = quality_jobs(a, kept, rows)[0]
        with Pool(a.workers) as pool:
            for i, (enc, ps, vm) in enumerate(pool.imap_unordered(quality, jobs), 1):
                if i % 50 == 0 or ps is None or vm is None:
                    print(f"  {i}/{len(jobs)} {os.path.basename(enc)} psnr={ps} vmaf={vm}", flush=True)
        print(f"quality-only done: {len(jobs)} clean encodes", flush=True)
        return 0
    rows = [row_of(j, R) for j, R in load_jsons(a.out)]
    with open(os.path.join(a.out, "encodes_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    L = [f"encodes: {len(rows)}  " + ", ".join(f"{k}={v}" for k, v in Counter(r["kind"] for r in rows).items())]
    # coverage: every kept source has a 5-rung clean ladder for every span
    clean = defaultdict(set)
    for r in rows:
        if r["kind"] == "clean" and r["ok"]:
            clean[(r["source_id"], r["span_index"])].add(r["rank"])
    want = [(sid, k) for sid, r in kept.items() for k in range(len(r["spans_list"]))]
    miss = [x for x in want if clean.get(x) != {0, 1, 2, 3, 4}]
    L.append(f"kept sources {len(kept)}, spans {len(want)}; spans without a complete verified 5-rung "
             f"ladder: {len(miss)}" + (f" (first: {miss[:5]})" if miss else ""))
    cap_src = {r["source_id"] for r in rows if r["kind"] == "capture" and r["ok"]}
    pd_src = Counter(r["source_id"] for r in rows if r["kind"] == "postdrop" and r["ok"])
    L.append(f"sources with capture variants: {len(cap_src)}; with 2 post-drop variants: "
             f"{sum(v == 2 for v in pd_src.values())}")
    # split/group inheritance
    bad_inh = [r for r in rows if r["source_id"] in kept and (r["split"], r["group_id"]) !=
               (kept[r["source_id"]]["split"], kept[r["source_id"]]["group_id"])]
    L.append(f"encodes whose split/group differ from the Step 1 manifest: {len(bad_inh)}")
    # per-encode checks
    fails = [r for r in rows if not r["ok"]]
    L.append(f"\nper-encode checks (decode, frames == span, 3840x2160, 60 fps, codec, keyframes vs GOP): "
             f"{len(rows) - len(fails)} pass, {len(fails)} fail")
    for k, v in Counter(x for r in fails for x in (r["check_fail"] or "json not ok").split("|")).most_common(10):
        L.append(f"   {k}: {v}")
    br = [r for r in rows if r["rate_kind"] == "kbps" and r["actual_kbps"]]
    for mode in ("abr", "capped", "cbr"):
        x = np.array([float(r["actual_kbps"]) / float(r["target_kbps"]) for r in br if r["mode"] == mode])
        if len(x):
            L.append(f"   actual/target bitrate, {mode:6s}: p5/p50/p95 = " +
                     " / ".join(f"{v:.2f}" for v in np.percentile(x, [5, 50, 95])) +
                     f"  (n={len(x)}, outside [0.5, 2]: {int(((x < 0.5) | (x > 2)).sum())})")
    # realized distribution (per source)
    seen = {}
    for r in rows:
        seen.setdefault(r["source_id"], r)
    L.append(f"\nrealized settings over {len(seen)} sources:")
    for key in ("encoder", "mode", "tune", "bframes", "gop"):
        L.append(f"   {key:8s}: " + ", ".join(f"{k or 'none'}={v}" for k, v in
                                             sorted(Counter(str(r[key]) for r in seen.values()).items())))
    L.append("   encoder x mode x preset:")
    for (e, m, p), v in sorted(Counter((r["encoder"], r["mode"], str(r["preset"])) for r in seen.values()).items()):
        L.append(f"      {e:11s} {m:6s} {p:10s} {v}")
    tot = sum(int(r["size_bytes"] or 0) for r in rows)
    by = Counter()
    for r in rows:
        by[r["kind"]] += int(r["size_bytes"] or 0)
    L.append(f"\nstorage used: {tot / 1e12:.3f} TB (" + ", ".join(f"{k} {v / 1e12:.3f} TB" for k, v in by.items()) + ")")
    if a.throughput:
        L.append("throughput:\n" + open(a.throughput).read().strip())
    # PSNR / VMAF monotonicity
    if not a.skip_quality:
        jobs, meta, sample = quality_jobs(a, kept, rows)
        with Pool(a.workers) as pool:
            q = pool.map(quality, jobs)
        per = defaultdict(list)
        for enc, ps, vm in q:
            r = meta[enc]
            per[(r["source_id"], r["span_index"])].append((int(r["rank"]), ps, vm, r["encoder"], r["mode"]))
        viol = []
        with open(os.path.join(a.out, "qc_quality.csv"), "w") as f:
            f.write("source_id,span_index,rank,encoder,mode,psnr_y,vmaf\n")
            for (sid, k), v in sorted(per.items()):
                v.sort()
                for rank, ps, vm, e, m in v:
                    f.write(f"{sid},{k},{rank},{e},{m},{ps},{vm}\n")
                for name, col in (("PSNR", 1), ("VMAF", 2)):
                    s = [x[col] for x in v]
                    if None in s:
                        viol.append(f"{sid} s{k} {name}: missing")
                    elif any(b2 < a2 for a2, b2 in zip(s, s[1:])):
                        viol.append(f"{sid} s{k} {name} not monotone in rank ({v[0][3]}, {v[0][4]}): "
                                    + " ".join(f"{x:.2f}" for x in s))
        L.append(f"\nPSNR/VMAF on {len(sample)} sources ({len(per)} spans, {len(jobs)} clean encodes, "
                 f"VMAF model vmaf_4k_v0.6.1, libvmaf {os.path.basename(os.path.dirname(os.path.dirname(VMAF)))}): "
                 f"{len(viol)} monotonicity violations")
        L += ["   " + v for v in viol]
    n = review(rows, a.out)
    L.append(f"\nstep2_review: {n} capture variants (reference | severity 0.5 | severity 1.0)")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.out, "qc_report.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
