"""Synthetic checks for build_manifest.py (Step 1, span-based audit).

Builds small clips with ffmpeg and runs build_manifest.py on them:
  clean      7 s continuous motion                    -> kept, one span over the whole clip
  cut_mid    7 s, hard cut at 3.5 s                   -> both pieces < 240 frames: no_usable_span
  cut_early  7 s, hard cut at 0.5 s                   -> kept, one span starting at the cut
  two_spans  9 s, hard cut at 4.5 s                   -> kept, two spans of 270 frames
  hold       9 s motion, frames 240-249 repeat frame 239 bit-exactly (lossless)
                                                      -> native_freeze (soft), held frames cut out,
                                                         spans [0,240) and [250,480)
  slow       a still rotated slowly (sub-pixel motion)        -> no held frames, one span, not upconverted
  still      a still image                            -> kept, tagged static, one span
  upconv     30 fps content repeated to 60 fps        -> upconverted_cadence (hard)
  letterbox  motion inside black top/bottom bars      -> letterbox, active_xywh = picture area
  darkobj    small bright object moving on black      -> not cropped
  dupe_copy  re-encode of an "excluded" clip          -> matches_excluded_source
  excerpt    5 s excerpt of a 10 s excluded clip      -> matches_excluded_source (dense)
Then re-runs and checks the manifest is byte-identical (cache reuse).

python test_build_manifest.py
"""
import csv
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 320, 180


def ff(*args, inp=None):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, input=inp)


def enc(src_args, out, extra_vf=None, rate=60):
    vf = [f"fps={rate}"] + ([extra_vf] if extra_vf else [])
    ff(*src_args, "-vf", ",".join(vf), "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
       "-pix_fmt", "yuv420p", out)


def mandel(t, start=0.0, neg=False, rate=60, end_pts=400, size=(W, H)):
    # continuously zooming fractal: smooth global motion
    src = (f"mandelbrot=s={size[0]}x{size[1]}:rate={rate}:start_scale=3:end_scale=0.3:"
           f"end_pts={end_pts}:maxiter=256")
    if neg:   # scene B: inverted fractal, very different luma from scene A
        src += ",negate"
    return ["-f", "lavfi", "-i", src, "-ss", str(start), "-t", str(t)]


def concat(a, b, out, tmp):
    lst = os.path.join(tmp, os.path.basename(out) + ".txt")
    with open(lst, "w") as f:
        f.write(f"file '{a}'\nfile '{b}'\n")
    ff("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "12",
       "-preset", "veryfast", "-pix_fmt", "yuv420p", out)


def decode_yuv(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p",
                          "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, W * H * 3 // 2)


def main():
    tmp = tempfile.mkdtemp(prefix="bm_test_")
    data, excl, parts, out = (os.path.join(tmp, d) for d in ("data", "excl", "parts", "out"))
    for d in (data, excl, parts):
        os.makedirs(d)
    p = lambda d, n: os.path.join(d, n + ".mp4")
    enc(mandel(7), p(data, "clean"))
    enc(mandel(3.5), p(parts, "a35"))
    enc(mandel(3.5, neg=True), p(parts, "b35"))
    concat(p(parts, "a35"), p(parts, "b35"), p(data, "cut_mid"), parts)
    enc(mandel(0.5, start=10), p(parts, "a05"))
    enc(mandel(6.5, neg=True), p(parts, "b65"))
    concat(p(parts, "a05"), p(parts, "b65"), p(data, "cut_early"), parts)
    enc(mandel(4.5, start=5), p(parts, "a45"))
    enc(mandel(4.5, start=5, neg=True), p(parts, "b45"))
    concat(p(parts, "a45"), p(parts, "b45"), p(data, "two_spans"), parts)
    # hold: decode 8 s of motion, repeat frame 239 over 240..249 (game time advances,
    # so frame 250 is a catch-up jump), encode losslessly so the repeats stay exact
    enc(mandel(9, start=2), p(parts, "hold_src"))
    fr = decode_yuv(p(parts, "hold_src")).copy()
    fr[240:250] = fr[239]
    ff("-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-r", "60", "-i", "-",
       "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p",
       p(data, "hold"), inp=fr.tobytes())
    ff(*mandel(0.1, start=6), "-frames:v", "1", os.path.join(parts, "still.png"))
    # slow: the still rotated by ~0.03 px/frame at the border (bilinear: sub-pixel motion);
    # 128x72 suspects repeats, 960x540 must reject them
    ff("-loop", "1", "-framerate", "60", "-i", os.path.join(parts, "still.png"), "-t", "7", "-vf",
       "rotate=a=t*0.01:c=black@0:ow=iw:oh=ih", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
       "-pix_fmt", "yuv420p", p(data, "slow"))
    ff("-loop", "1", "-framerate", "60", "-i", os.path.join(parts, "still.png"), "-t", "6", "-c:v", "libx264",
       "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", p(data, "still"))
    enc(mandel(7, rate=30), p(data, "upconv"), rate=60)
    enc(mandel(7, start=4, size=(W, 136)), p(data, "letterbox"), extra_vf=f"pad={W}:{H}:0:22:black")
    ff("-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=60", "-f", "lavfi",
       "-i", "color=c=white:s=24x24:r=60", "-filter_complex",
       "[0:v][1:v]overlay=x='60+t*30':y='70+10*sin(t*3)':shortest=1", "-t", "6",
       "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", p(data, "darkobj"))
    # excluded "study" clip (moving test pattern), its re-encoded copy and an excerpt
    enc(["-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:rate=60", "-t", "10"], p(excl, "study_scene"))
    ff("-i", p(excl, "study_scene"), "-c:v", "libx264", "-b:v", "1500k", "-pix_fmt", "yuv420p",
       p(data, "dupe_copy"))
    ff("-ss", "3", "-i", p(excl, "study_scene"), "-t", "5", "-c:v", "libx264", "-crf", "20",
       "-pix_fmt", "yuv420p", p(data, "excerpt"))

    cmd = [sys.executable, os.path.join(HERE, "build_manifest.py"), "--data", f"SYN={data}",
           "--exclude", excl, "--out", out, "--workers", "4", "--min-width", str(W),
           "--min-height", str(H), "--min-bpp", "0"]
    run = lambda: subprocess.run(cmd, check=True, capture_output=True, text=True)
    r1 = run()
    shutil.copy(os.path.join(out, "manifest.csv"), os.path.join(tmp, "manifest_run1.csv"))
    rows = {os.path.splitext(os.path.basename(r["path"]))[0]: r
            for r in csv.DictReader(open(os.path.join(out, "manifest.csv")))}

    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    fl = lambda n: rows[n]["flags"].split("|") if rows[n]["flags"] else []
    sp = lambda n: json.loads(rows[n]["spans"])
    kept = lambda n: rows[n]["keep"] == "True"
    desc = lambda n: (f"flags={rows[n]['flags']}, spans={rows[n]['spans']}, cuts={rows[n]['n_cuts']}, "
                      f"held={rows[n]['held_frames']}")
    check(len(rows) == 12, f"12 dataset clips in manifest, excluded clip absent ({sorted(rows)})")
    n = int(rows["clean"]["frames_scanned"])
    check(kept("clean") and sp("clean") == [[0, n]] and rows["clean"]["n_cuts"] == "0",
          f"clean: kept, one span over all {n} frames ({desc('clean')})")
    check(int(rows["cut_mid"]["n_cuts"]) >= 1 and "no_usable_span" in fl("cut_mid") and not kept("cut_mid"),
          f"cut_mid: cut detected, both pieces < 240 frames ({desc('cut_mid')})")
    s = sp("cut_early")
    check(kept("cut_early") and len(s) == 1 and 28 <= s[0][0] <= 32,
          f"cut_early: one span starting at the cut ({desc('cut_early')})")
    s = sp("two_spans")
    check(kept("two_spans") and len(s) == 2 and all(b - a >= 240 for a, b in s),
          f"two_spans: two spans >= 240 frames ({desc('two_spans')})")
    s = sp("hold")
    check(kept("hold") and "native_freeze" in fl("hold") and rows["hold"]["held_frames"] == "10"
          and s == [[0, 240], [250, int(rows["hold"]["frames_scanned"])]],
          f"hold: held frames cut out, spans [0,240) + [250,end) ({desc('hold')})")
    check(kept("slow") and rows["slow"]["held_frames"] == "0" and len(sp("slow")) == 1
          and "upconverted_cadence" not in fl("slow") and float(rows["slow"]["dup_frac_lo"]) > 0.05
          and float(rows["slow"]["dup_frac"]) < 0.01,
          f"slow: sub-pixel motion not treated as holds ({desc('slow')}, "
          f"dup_frac_lo={rows['slow']['dup_frac_lo']}, dup_frac={rows['slow']['dup_frac']})")
    check(kept("still") and "static" in fl("still") and len(sp("still")) == 1,
          f"still: kept and tagged static ({desc('still')})")
    u = rows["upconv"]
    check("upconverted_cadence" in fl("upconv") and not kept("upconv"),
          f"upconv flagged (flags={u['flags']}, dup_frac_lo={u['dup_frac_lo']}, lag={u['cadence_lag']})")
    ax = json.loads(rows["letterbox"]["active_xywh"])
    check("letterbox" in fl("letterbox") and kept("letterbox") and ax[0] == 0 and ax[2] == W
          and abs(ax[1] - 22) <= 2 and abs(ax[3] - 136) <= 4,
          f"letterbox: active_xywh ~ [0, 22, {W}, 136] (got {ax})")
    ax = json.loads(rows["darkobj"]["active_xywh"])
    check(ax == [0, 0, W, H] and "letterbox" not in fl("darkobj"),
          f"darkobj: dark content is not cropped (got {ax})")
    check("matches_excluded_source" in fl("dupe_copy") and not kept("dupe_copy"),
          f"dupe_copy excluded ({rows['dupe_copy']['excl_match']})")
    check("matches_excluded_source" in fl("excerpt") and "dense:" in rows["excerpt"]["excl_match"],
          f"excerpt excluded by dense matching ({rows['excerpt']['excl_match']})")
    others = [k for k in rows if k not in ("dupe_copy", "excerpt") and "matches_excluded_source" in fl(k)]
    check(not others, f"no false exclusion of unrelated clips {others}")
    run()
    check(filecmp.cmp(os.path.join(out, "manifest.csv"), os.path.join(tmp, "manifest_run1.csv"),
                      shallow=False), "re-run with cache gives identical manifest.csv")
    print(r1.stdout.strip().splitlines()[-1] if r1.stdout.strip() else "")
    if fails:
        print(f"{len(fails)} FAILED (artifacts in {tmp})")
        return 1
    shutil.rmtree(tmp)
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
