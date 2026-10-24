"""Step 2 shared code: per-source encoder settings, ffmpeg arguments, spans.

Encoding settings are generic and sampled ONCE per source (rng keyed by the
source id), then reused for that source's whole clean ladder, its capture
variants (2c) and its post-drop re-encodes (2f), so ranking pairs differ only
in the rate parameter / severity / drops (AGENT_BRIEF 2b):

  encoder   libx264 | libx265 | libsvtav1 | libvpx-vp9                (equal)
  mode      crf | abr | capped | cbr                                  (equal)
  preset    uniform over x264/x265 ultrafast..slow, svt-av1 4..12, vp9 good cpu-used 0..5
  tune      none 60%, else uniform over the encoder's tunes
  bframes   default 60%, none 20%, high 20%
  gop       uniform over {30, 60, 120, 240, 600, default}
  ladder    5 rungs, rank 0 = worst: bitrate modes log-uniform 1.5-60 Mbps (4K);
            crf: x264/x265 [18, 51], svt-av1/vp9 [18, 60], rungs >= 4 apart

Every output is 8-bit yuv420p tagged BT.709, CFR 60 fps (59.94 fps sources are
retimed, not resampled: every source frame is kept once), frame-accurate
span trim. Nothing here is copied from the study or from psvqa/encode.py.
"""
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np

ENCODERS = ["libx264", "libx265", "libsvtav1", "libvpx-vp9"]
MODES = ["crf", "abr", "capped", "cbr"]
X26X_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]
PRESETS = {"libx264": X26X_PRESETS, "libx265": X26X_PRESETS,
           "libsvtav1": list(range(4, 13)), "libvpx-vp9": list(range(0, 6))}
TUNES = {"libx264": ["film", "animation", "grain", "stillimage", "psnr", "ssim", "fastdecode", "zerolatency"],
         "libx265": ["psnr", "ssim", "grain", "fastdecode", "zerolatency", "animation"],
         "libsvtav1": ["tune0_vq"],       # default tune is 1 (PSNR); tune 2 (SSIM) is CRF-only and
                                          # flagged "testing and debugging" by SVT-AV1 3.1
         "libvpx-vp9": ["content_screen", "content_film", "ssim"]}
GOPS = [30, 60, 120, 240, 600, "default"]
CRF_RANGE = {"libx264": (18, 51), "libx265": (18, 51), "libsvtav1": (18, 60), "libvpx-vp9": (18, 60)}
KBPS_RANGE_4K = (1500, 60000)
KBPS_RANGE_1080 = (500, 20000)
FPS_OUT = 60
X265_LOOKAHEAD = {"ultrafast": 5, "superfast": 10, "veryfast": 15, "faster": 15, "fast": 15,
                  "medium": 20, "slow": 25}      # x265 3.5 preset defaults
SVT_CAPPED_CRF = 23     # SVT-AV1 3.1 caps bitrate only on top of CRF (see realize())
SETTINGS_VERSION = 2


def source_id(path):
    """Stable id from dataset-relative identity (dataset dir name + file stem)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    ds = os.path.basename(os.path.dirname(os.path.realpath(path)))
    return f"{ds}__{stem}".replace(" ", "_")


def _rng(sid, salt, seed):
    h = hashlib.sha256(f"{seed}|{salt}|{sid}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def sample_settings(sid, height, seed=2026):
    """Per-source encoder settings (deterministic in sid and seed)."""
    r = _rng(sid, "settings", seed)
    enc = ENCODERS[int(r.integers(len(ENCODERS)))]
    mode = MODES[int(r.integers(len(MODES)))]
    preset = PRESETS[enc][int(r.integers(len(PRESETS[enc])))]
    tune = None if r.random() < 0.6 else TUNES[enc][int(r.integers(len(TUNES[enc])))]
    u = r.random()
    bf = "default" if u < 0.6 else "none" if u < 0.8 else "high"
    bf_n = int(r.integers(3, 9)) if bf == "high" else None        # x264/x265: -bf 3..8
    gop = GOPS[int(r.integers(len(GOPS)))]
    cbr_buf = float(r.uniform(0.5, 2.0))                            # CBR: bufsize = k x target
    if mode == "crf":
        lo, hi = CRF_RANGE[enc]
        while True:                                                 # 5 rungs, >= 4 apart
            v = np.sort(r.integers(lo, hi + 1, 5))
            if np.all(np.diff(v) >= 4):
                break
        rungs = [int(x) for x in v[::-1]]                           # rank 0 = highest CRF = worst
        rate = dict(kind="crf", rungs=rungs)
    else:
        lo, hi = KBPS_RANGE_4K if height > 1500 else KBPS_RANGE_1080
        k = np.sort(np.exp(r.uniform(np.log(lo), np.log(hi), 5)))
        rate = dict(kind="kbps", rungs=[int(round(x)) for x in k])  # rank 0 = lowest bitrate
    S = dict(settings_version=SETTINGS_VERSION, seed=seed, encoder=enc, mode=mode,
             preset=preset, tune=tune, bframes=bf, bframes_n=bf_n, gop=gop,
             cbr_bufsize_factor=round(cbr_buf, 4), rate=rate)
    return realize(S)


def realize(S):
    """Resolve combinations the libraries reject, to the nearest valid one, and
    record each change in S["adjustments"] (the sampled values stay in
    S["sampled"]). Found by encoding every encoder x mode x B-frames x tune x
    GOP combination:
      x265 3.5   tune zerolatency sets lookahead 0, which must exceed bframes
                 -> zerolatency implies no B-frames
      SVT-AV1 3.1  CBR only in low-delay -> pred-struct 1 (no B-frames)
                   VBR (abr) not in low-delay -> default (random-access) structure
                   max bitrate only on top of CRF -> capped = CRF SVT_CAPPED_CRF
                   with -maxrate = the rung bitrate"""
    S = dict(S, sampled=dict(bframes=S["bframes"], bframes_n=S["bframes_n"]), adjustments=[])
    enc, mode = S["encoder"], S["mode"]
    if enc == "libx265" and S["tune"] == "zerolatency" and S["bframes"] == "high":
        S.update(bframes="none", bframes_n=None)
        S["adjustments"].append("x265 zerolatency: bframes high -> none")
    elif enc == "libx265" and S["bframes"] == "high":
        # x265 needs rc-lookahead > bframes; ultrafast sets 5, superfast 10
        need = S["bframes_n"] + 1
        if need > X265_LOOKAHEAD[S["preset"]]:
            S["x265_rc_lookahead"] = need
            S["adjustments"].append(f"x265 {S['preset']}: rc-lookahead {X265_LOOKAHEAD[S['preset']]} -> {need} "
                                    f"(must exceed bframes {S['bframes_n']})")
    if enc == "libsvtav1" and mode == "cbr" and S["bframes"] != "none":
        S.update(bframes="none", bframes_n=None)
        S["adjustments"].append("svt-av1 cbr: low-delay structure (bframes none)")
    if enc == "libsvtav1" and mode == "abr" and S["bframes"] == "none":
        S.update(bframes="default")
        S["adjustments"].append("svt-av1 vbr: bframes none -> default")
    if enc == "libsvtav1" and mode == "capped":
        S["adjustments"].append(f"svt-av1 capped: crf {SVT_CAPPED_CRF} + maxrate = rung")
    return S


def encoder_args(S, rung_value, threads=0):
    """ffmpeg output arguments for one encode with settings S at one ladder rung
    (a CRF value or a target kbps). Returns (args, description dict)."""
    enc, mode = S["encoder"], S["mode"]
    a = ["-c:v", enc]
    svt, x265p = [], []
    # ---- preset / speed
    if enc in ("libx264", "libx265"):
        a += ["-preset", S["preset"]]
    elif enc == "libsvtav1":
        a += ["-preset", str(S["preset"])]
    else:
        a += ["-deadline", "good", "-cpu-used", str(S["preset"]), "-row-mt", "1"]
    # ---- tune
    t = S["tune"]
    if t:
        if enc in ("libx264", "libx265"):
            a += ["-tune", t]
        elif enc == "libsvtav1":
            svt.append("tune=" + ("0" if t == "tune0_vq" else "2"))
        else:
            a += {"content_screen": ["-tune-content", "screen"], "content_film": ["-tune-content", "film"],
                  "ssim": ["-tune", "ssim"]}[t]
    # ---- B-frames / reference structure
    bf = S["bframes"]
    if bf != "default":
        if enc == "libx264":
            a += ["-bf", "0" if bf == "none" else str(S["bframes_n"])]
        elif enc == "libx265":
            x265p.append(f"bframes={0 if bf == 'none' else S['bframes_n']}")
            if S.get("x265_rc_lookahead"):
                x265p.append(f"rc-lookahead={S['x265_rc_lookahead']}")
        elif enc == "libsvtav1":
            svt.append("pred-struct=1" if bf == "none" else "hierarchical-levels=5")
        else:   # vp9 has no B-frames: alt-ref frames are the analogue
            a += (["-auto-alt-ref", "0", "-lag-in-frames", "0"] if bf == "none"
                  else ["-auto-alt-ref", "6", "-lag-in-frames", "25", "-arnr-maxframes", "7"])
    # ---- GOP
    if S["gop"] != "default":
        g = int(S["gop"])
        a += ["-g", str(g)]
        if enc == "libx264":
            a += ["-keyint_min", str(g)]
        if enc == "libsvtav1":
            svt.append(f"keyint={g}")
    # ---- rate control
    v = int(rung_value)
    if mode == "crf":
        a += ["-crf", str(v)] + (["-b:v", "0"] if enc == "libvpx-vp9" else [])
    else:
        if enc == "libsvtav1" and mode == "capped":
            a += ["-crf", str(SVT_CAPPED_CRF), "-maxrate", f"{v}k"]
        else:
            a += ["-b:v", f"{v}k"]
        if mode == "capped" and enc != "libsvtav1":
            a += ["-maxrate", f"{int(round(1.5 * v))}k", "-bufsize", f"{2 * v}k"]
        elif mode == "cbr":
            buf = int(round(S["cbr_bufsize_factor"] * v))
            if enc == "libsvtav1":
                svt.append("rc=2")
                a += ["-bufsize", f"{buf}k"]
            else:
                a += ["-minrate", f"{v}k", "-maxrate", f"{v}k", "-bufsize", f"{buf}k"]
                if enc == "libx264":
                    a += ["-x264-params", "nal-hrd=cbr"]
    if threads:
        if enc == "libx265":
            x265p.append(f"pools={threads}")
        elif enc == "libsvtav1":
            svt.append(f"lp={threads}")
        else:
            a += ["-threads", str(threads)]
    if svt:
        a += ["-svtav1-params", ":".join(svt)]
    if x265p:
        a += ["-x265-params", ":".join(x265p)]
    a += ["-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
          "-color_primaries", "bt709", "-color_trc", "bt709", "-an", "-movflags", "+faststart"]
    return a


def span_filter(start, end, in_label="0:v", w=None, h=None):
    """Frame-accurate trim of source frames [start, end), retimed to CFR 60 fps
    (each source frame kept exactly once)."""
    f = (f"[{in_label}]trim=start_frame={start}:end_frame={end},"
         f"setpts=N/({FPS_OUT}*TB)")
    return f


def ladder_command(src, span, S, outs, threads=0):
    """One ffmpeg process: decode the source once, trim the span, split to the
    5 ladder encodes. outs: list of 5 output paths in rank order."""
    a, b = span
    n = len(outs)
    fc = span_filter(a, b) + f",format=yuv420p,split={n}" + "".join(f"[o{i}]" for i in range(n))
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-filter_complex", fc]
    for i, (path, rung) in enumerate(zip(outs, S["rate"]["rungs"])):
        cmd += ["-map", f"[o{i}]", "-r", str(FPS_OUT), "-fps_mode", "cfr"] + encoder_args(S, rung, threads) + [path]
    return cmd


def probe_encode(path):
    """Decoded frame count, size, fps, codec, keyframe positions, bitrate, decode errors."""
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,pix_fmt,"
         "color_primaries,color_transfer,color_space:format=duration,bit_rate,size",
         "-of", "json", path], capture_output=True, text=True).stdout or "{}")
    st = (j.get("streams") or [{}])[0]
    fm = j.get("format", {})
    kf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                         "-show_entries", "frame=pts", "-of", "csv=p=0", path],
                        capture_output=True, text=True).stdout.split()
    dec = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-f", "null", "-"],
                         capture_output=True, text=True)
    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    return dict(codec=st.get("codec_name"), width=int(st.get("width", 0)), height=int(st.get("height", 0)),
                fps=round(float(num) / float(den), 3) if float(den) else 0.0,
                frames=int(st.get("nb_read_frames") or 0), pix_fmt=st.get("pix_fmt"),
                color=[st.get("color_primaries"), st.get("color_transfer"), st.get("color_space")],
                duration=float(fm.get("duration") or 0), size_bytes=int(fm.get("size") or 0),
                actual_kbps=round(float(fm.get("bit_rate") or 0) / 1000, 1),
                keyframes=len(kf), decode_ok=dec.returncode == 0 and not dec.stderr.strip(),
                decode_stderr=dec.stderr.strip()[:300])


# ------------------------------------------------------------------ fan-out
def needs_fanout(S, n_outputs):
    """SVT-AV1 3.1.2 crashes (general protection fault in libSvtAv1Enc) when two
    or more encoder instances run in one ffmpeg process at high CRF (e.g. 59
    and 50; one instance per process is fine). For SVT-AV1 with several outputs,
    decode once and fan the raw frames out to one encoder process per output."""
    return S["encoder"] == "libsvtav1" and n_outputs > 1


def fanout_encode(decode_cmd, w, h, S, rungs, outs, threads=0):
    """One decode process (raw yuv420p on stdout) -> one encoder process per
    output, frames written to every encoder over 1 MB pipes. Returns
    (returncode, stderr, {"decode": cmd, "encode": [cmds]}); returncode is the
    first non-zero of all processes."""
    import fcntl
    import queue
    import threading
    fsz = w * h * 3 // 2
    rd, wd = os.pipe()
    try:
        fcntl.fcntl(wd, 1031, 1 << 20)
    except OSError:
        pass
    dec = subprocess.Popen(decode_cmd, stdout=wd, stderr=subprocess.PIPE)
    os.close(wd)
    enc_cmds = [["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                 "-s", f"{w}x{h}", "-r", str(FPS_OUT), "-i", "pipe:0"] + encoder_args(S, v, threads) + [o]
                for v, o in zip(rungs, outs)]
    encs = []
    for c in enc_cmds:
        e = subprocess.Popen(c, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            fcntl.fcntl(e.stdin.fileno(), 1031, 1 << 20)
        except OSError:
            pass
        encs.append(e)
    qs = [queue.Queue(maxsize=8) for _ in encs]
    errs = [None] * len(encs)

    def writer(i):
        e = encs[i]
        try:
            while True:
                b = qs[i].get()
                if b is None:
                    break
                e.stdin.write(b)
        except BrokenPipeError:
            while qs[i].get() is not None:      # keep draining so the reader never blocks
                pass
        finally:
            try:
                e.stdin.close()
            except BrokenPipeError:
                pass

    def drain(i):
        errs[i] = encs[i].stderr.read().decode(errors="ignore")

    ths = [threading.Thread(target=writer, args=(i,)) for i in range(len(encs))] + \
          [threading.Thread(target=drain, args=(i,)) for i in range(len(encs))]
    for t in ths:
        t.start()
    f = os.fdopen(rd, "rb", buffering=0)
    while True:
        buf = bytearray(fsz)
        mv, k = memoryview(buf), 0
        while k < fsz:
            m = f.readinto(mv[k:])
            if not m:
                break
            k += m
        if k < fsz:
            break
        b = bytes(buf)
        for q in qs:
            q.put(b)
    f.close()
    for q in qs:
        q.put(None)
    for t in ths:
        t.join()
    rcs = [e.wait() for e in encs]
    dec_err = dec.stderr.read().decode(errors="ignore")
    rcs.append(dec.wait())
    rc = next((x for x in rcs if x != 0), 0)
    return rc, (dec_err + "".join(e or "" for e in errs)), dict(decode=decode_cmd, encode=enc_cmds)


def span_decode_cmd(src, span):
    a, b = span
    return ["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-fps_mode", "passthrough", "-vf",
            f"trim=start_frame={a}:end_frame={b},setpts=N/({FPS_OUT}*TB),format=yuv420p",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]


# ------------------------------------------------------------------- locks
LOCK_STALE_S = 3 * 3600      # a job killed at its walltime leaves its lock; reclaim after this


class ItemLock:
    """Atomic per-item lock (O_CREAT|O_EXCL) so several jobs can work on the
    same shard without encoding the same item twice. Usage:
        with ItemLock(path) as got:
            if got: ...work..."""

    def __init__(self, path):
        self.path, self.got = path, False

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.uname().nodename} {os.getpid()} {time.time():.0f}\n".encode())
                os.close(fd)
                self.got = True
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
                        os.remove(self.path)          # stale: owner was killed
                        continue
                except FileNotFoundError:
                    continue
                break
        return self.got

    def __exit__(self, *exc):
        if self.got:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
