"""Step 2c: capture-distortion -> compression variants for every kept source.

Two stages, so that lossy encoding runs on CPU nodes and only rendering on the GPU:

  render (GPU node): per source, 2 plans = sample_plan(rng, fps=60,
      duration=span seconds, n_ops=(1, 3)) from a source-keyed rng. The primary
      span is decoded once per plan (RGB, BT.709 matrix forced) and rendered at
      plan.with_severity(0.5) and (1.0) with the TRUE source frame indices, in
      chunks carrying plan.margin_t() frames of history. Each rendered stream
      goes straight into a lossless x264 (-qp 0) yuv420p intermediate: no
      uncompressed frame is written to disk. Decoding outputs yuv420p and the
      BT.709 limited-range YUV <-> RGB conversion runs on the GPU
      (COLOR_PIPELINE below; versioned in every JSON). Decode, render and the
      two intermediate writers run in separate threads over 1 MB pipes (a
      default 64 KB pipe moves only ~0.4 GB/s of 4K frames).
  encode (CPU node): each intermediate is decoded once and encoded at the
      source's high and mid ladder rungs with the source's exact settings
      (step2_common.encoder_args); the intermediate is deleted once both
      encodes verify.

Temporal history: frames before a span start belong to another shot (cut) or
to a hold, so history is taken inside the span: the rendered span starts
margin_t frames after the span start (identical to the brief when margin_t = 0).

Each final encode's JSON holds the full plan (ops, severities, parameters,
seeds, envelopes, supports, bipolar flags), describe() labels without the
per-frame arrays, render span, chunking, settings, rung and commands, so the
frames can be re-rendered (video_distortions_torch.render is bit-exact on the
same GPU type) and re-encoded.

python step2_capture.py render --manifest M --out OUT [--shard k/n] [--limit N]
python step2_capture.py encode --manifest M --out OUT --workers 16 --threads 4 [--shard k/n]
"""
import argparse
import dataclasses
import fcntl
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
from multiprocessing import Pool

import numpy as np

import step2_common as sc
from step2_ladder import in_shard, kept_sources, primary_span, tool_versions

SEVERITIES = (0.5, 1.0)
N_PLANS = 2
CHUNK = 48          # rendered frames per GPU batch (plus margin_t history)
TEMPORAL_FRAMES = 40  # plans with temporal ops: chunk + history <= this (4K float copies fit in 40 GB)


def chunk_for(plan):
    import video_distortions as vd
    if any(vd.REGISTRY[o.name].temporal for o in plan.ops):
        return max(8, TEMPORAL_FRAMES - plan.margin_t())
    return CHUNK
COLOR_PIPELINE = ("v1: decode yuv420p limited range; chroma 2x upsample bilinear (align_corners=False); "
                  "BT.709 YCbCr->RGB float [0,1] clamped; render; RGB->YCbCr BT.709; chroma 2x2 mean; "
                  "round to 8-bit limited range (Y 16+219y, C 128+224c), clamp 0..255")
F_SETPIPE_SZ = 1031
KR, KB = 0.2126, 0.0722


def _big_pipe(fd):
    try:
        fcntl.fcntl(fd, F_SETPIPE_SZ, 1 << 20)
    except OSError:
        pass


def yuv_to_rgb(Y, U, V):
    """uint8 planes on the GPU (T,h,w), (T,h/2,w/2) -> float RGB (T,h,w,3)."""
    import torch
    import torch.nn.functional as F
    y = (Y.float() - 16) / 219
    c = torch.stack([U, V], 1).float()
    c = (F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False) - 128) / 224
    cb, cr = c[:, 0], c[:, 1]
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / (1 - KR - KB)
    return torch.stack([r, g, b], -1).clamp(0, 1)


def rgb_to_yuv_bytes(X):
    """float RGB (T,h,w,3) on the GPU -> yuv420p bytes (T frames)."""
    import torch
    import torch.nn.functional as F
    r, g, b = X[..., 0], X[..., 1], X[..., 2]
    y = KR * r + (1 - KR - KB) * g + KB * b
    cb, cr = (b - y) / (2 * (1 - KB)), (r - y) / (2 * (1 - KR))
    c = F.avg_pool2d(torch.stack([cb, cr], 1), 2)
    Y8 = torch.round(16 + 219 * y).clamp(0, 255).to(torch.uint8)
    C8 = torch.round(128 + 224 * c).clamp(0, 255).to(torch.uint8)
    T = X.shape[0]
    out = torch.cat([Y8.reshape(T, -1), C8[:, 0].reshape(T, -1), C8[:, 1].reshape(T, -1)], 1)
    return out.cpu().numpy()


# ----------------------------------------------------------- plan <-> json
def _py(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.str_,)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return [_py(x) for x in o]
    if isinstance(o, list):
        return [_py(x) for x in o]
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    return o


def plan_to_json(plan):
    return dict(fps=float(plan.fps), seed=int(plan.seed), ops=[
        dict(name=o.name, severity=float(o.severity), P=_py(o.P), seed=int(o.seed),
             envelope=_py(dataclasses.asdict(o.envelope)), support=_py(dataclasses.asdict(o.support)),
             bipolar=bool(o.bipolar)) for o in plan.ops])


def plan_from_json(j):
    import video_distortions as vd
    ops = []
    for o in j["ops"]:
        P = dict(o["P"])
        if "ghosts" in P:
            P["ghosts"] = [tuple(g) for g in P["ghosts"]]
        ops.append(vd.Op(o["name"], o["severity"], P, o["seed"], vd.Envelope(**o["envelope"]),
                         vd.Support(**o["support"]), o["bipolar"]))
    return vd.Plan(ops, j["fps"], j["seed"])


def labels(plan, frame_idx):
    import video_distortions as vd
    d = vd.describe(plan, frame_idx)
    for op in d["ops"]:
        op.pop("strength", None)           # per-frame arrays are re-derivable from the plan
    return _py(d)


def capture_choices(r, S):
    """Deterministic per-source plans and rungs (same for both severities)."""
    import video_distortions as vd
    rng = sc._rng(r["source_id"], "capture", S["seed"])
    a, b = r["spans_list"][primary_span(r)]
    plans = [vd.sample_plan(rng, fps=60.0, duration=(b - a) / 60.0, n_ops=(1, 3)) for _ in range(N_PLANS)]
    rungs = dict(high=int(rng.choice([3, 4])), mid=2)
    return plans, rungs


# ------------------------------------------------------------------ render
def _decode_cmd(src, a, b):
    # passthrough: the default CFR output duplicates/drops frames on sources with
    # irregular timestamps (e.g. average 60.109 or 58.824 fps)
    return ["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-fps_mode", "passthrough", "-vf",
            f"trim=start_frame={a}:end_frame={b},setpts=N/({sc.FPS_OUT}*TB)",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]


def _lossless_cmd(w, h, dst):
    return ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{w}x{h}", "-r", str(sc.FPS_OUT), "-i", "pipe:0",
            "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-threads", "12",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", dst]


def render_source(r, out, device):
    d = os.path.join(out, "encodes", r["source_id"])
    os.makedirs(d, exist_ok=True)
    with sc.ItemLock(os.path.join(d, ".lock_render")) as got:
        if not got:                       # another render job has this source
            return []
        return _render_source(r, out, device)


def _render_source(r, out, device):
    import torch
    import video_distortions_torch as vt
    S = sc.sample_settings(r["source_id"], int(r["height"]))
    plans, rungs = capture_choices(r, S)
    k = primary_span(r)
    a, b = r["spans_list"][k]
    w, h = int(r["width"]), int(r["height"])
    d = os.path.join(out, "encodes", r["source_id"])
    os.makedirs(d, exist_ok=True)
    done = []
    for p, plan in enumerate(plans):
        mt = plan.margin_t()
        chunk = chunk_for(plan)
        ra = a + mt                                   # rendered span [ra, b), history [a, ra)
        tags = [f"s{k}_cap_p{p}_sev{int(round(s * 100)):03d}" for s in SEVERITIES]
        inter = [os.path.join(d, f"tmp_{t}.mkv") for t in tags]
        ij = [x[:-4] + ".json" for x in inter]
        finals = [os.path.join(d, f"{t}_r{rungs[n]}.json") for t in tags for n in ("high", "mid")]
        if all(os.path.exists(f) and json.load(open(f)).get("ok") for f in finals) or \
                all(os.path.exists(f) and json.load(open(f)).get("ok") for f in ij):
            continue
        t0 = time.time()
        fsz = w * h * 3 // 2
        rd, wd = os.pipe()
        _big_pipe(wd)
        dec = subprocess.Popen(_decode_cmd(r["path"], a, b), stdout=wd, stderr=subprocess.PIPE)
        os.close(wd)
        encs = []
        for x in inter:
            e = subprocess.Popen(_lossless_cmd(w, h, x), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _big_pipe(e.stdin.fileno())
            encs.append(e)
        frames_q = queue.Queue(maxsize=2 * chunk)

        def reader():
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
                frames_q.put(np.frombuffer(buf, np.uint8))
            frames_q.put(None)
            f.close()

        out_qs = [queue.Queue(maxsize=2 * CHUNK) for _ in encs]

        def writer(e, q):
            while True:
                item = q.get()
                if item is None:
                    break
                e.stdin.write(item.tobytes())
            e.stdin.close()

        ths = [threading.Thread(target=reader)] + \
              [threading.Thread(target=writer, args=(e, q)) for e, q in zip(encs, out_qs)]
        for t_ in ths:
            t_.start()
        sev_plans = [plan.with_severity(s) for s in SEVERITIES]
        hist = None                                   # last mt frames as float RGB on the GPU
        t_abs, n_out, eof = a, 0, False
        while not eof:
            buf = []
            while len(buf) < chunk:
                item = frames_q.get()
                if item is None:
                    eof = True
                    break
                buf.append(item)
            if not buf:
                break
            raw = torch.as_tensor(np.stack(buf), device=device)
            Yp = raw[:, :w * h].reshape(-1, h, w)
            Up = raw[:, w * h:w * h * 5 // 4].reshape(-1, h // 2, w // 2)
            Vp = raw[:, w * h * 5 // 4:].reshape(-1, h // 2, w // 2)
            new = yuv_to_rgb(Yp, Up, Vp)
            X = torch.cat([hist, new]) if hist is not None else new
            first = t_abs - (len(X) - len(new))       # absolute source index of X[0]
            idx = np.arange(first, first + len(X))
            # output only NEW frames of this chunk (history frames were output by the
            # previous chunk) that lie in the rendered span
            keep = torch.as_tensor((idx >= ra) & (idx >= t_abs), device=device)
            if bool(keep.any()):
                for q, sp in zip(out_qs, sev_plans):
                    Y = vt.render(X, idx, sp)[keep]
                    for fr in rgb_to_yuv_bytes(Y):
                        q.put(fr)
                n_out += int(keep.sum())
            t_abs += len(new)
            hist = X[-mt:] if mt else None
        for q in out_qs:
            q.put(None)
        for t_ in ths:
            t_.join()
        errs = [e.stderr.read().decode(errors="ignore").strip() for e in encs]
        for e in encs:
            e.wait()
        dec.wait()
        dt = time.time() - t0
        for s, sp, x, j, err, e in zip(SEVERITIES, sev_plans, inter, ij, errs, encs):
            rec = dict(stage="rendered", source_id=r["source_id"], span_index=k, span=[a, b],
                       render_span=[ra, b], render_frames=b - ra, history_frames=mt, chunk=chunk,
                       plan_index=p, plan_id=f"{r['source_id']}__p{p}", severity=s,
                       plan=plan_to_json(plan), plan_at_severity=plan_to_json(sp),
                       labels=labels(sp, np.arange(ra, b)), rungs=rungs, intermediate=x,
                       color_pipeline=COLOR_PIPELINE,
                       frames_written=n_out, render_device=str(torch.cuda.get_device_name(0)
                                                               if torch.cuda.is_available() else "cpu"),
                       torch=torch.__version__, cmd_decode=_decode_cmd(r["path"], a, b),
                       cmd_lossless=_lossless_cmd(w, h, x), wall_s_render_both=round(dt, 1),
                       ok=bool(e.returncode == 0 and dec.returncode == 0 and n_out == b - ra),
                       stderr=err[-300:])
            with open(j + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(j + ".tmp", j)
            done.append(rec)
    return done


def pending_tb(out):
    """Total size of rendered intermediates not yet encoded (throttles rendering)."""
    tot = 0
    with os.scandir(os.path.join(out, "encodes")) as it:
        for d in it:
            if d.is_dir():
                with os.scandir(d.path) as it2:
                    tot += sum(f.stat().st_size for f in it2 if f.name.startswith("tmp_") and f.name.endswith(".mkv"))
    return tot / 1e12


def cmd_render(a):
    import torch
    rows = [r for r in kept_sources(a.manifest) if in_shard(r["source_id"], a.shard)]
    if a.limit:
        rows = rows[:a.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"render shard {a.shard}: {len(rows)} sources on {dev}", flush=True)
    t0, frames = time.time(), 0
    for i, r in enumerate(rows, 1):
        if i % 10 == 1 and a.max_pending_tb != float("inf"):   # the scan is slow on 20 TB of scratch
            while pending_tb(a.out) > a.max_pending_tb:
                print(f"  pending intermediates > {a.max_pending_tb} TB: waiting for the encode stage", flush=True)
                time.sleep(300)
        for rec in render_source(r, a.out, dev):
            frames += rec["render_frames"]
        if i % 10 == 0 or i == len(rows):
            hrs = (time.time() - t0) / 3600
            print(f"  {i}/{len(rows)} sources, {frames} rendered frames per severity, "
                  f"{i / max(hrs, 1e-9):.0f} sources/h", flush=True)
    if not a.limit:
        k, n = a.shard
        open(os.path.join(a.out, "logs", f"render_done_shard{k}of{n}"), "w").write(time.strftime("%c") + "\n")


# ------------------------------------------------------------------ encode
def encode_intermediate(job):
    ij = job[0]
    with sc.ItemLock(ij[:-5] + ".lock") as got:
        if not got:                       # another job is encoding this intermediate
            return []
        return _encode_intermediate(job)


def _encode_intermediate(job):
    ij, r, threads, versions = job
    R = json.load(open(ij))
    if not R.get("ok"):
        return [dict(ok=False, error="render not ok", intermediate=ij)]
    S = sc.sample_settings(r["source_id"], int(r["height"]))
    base = os.path.basename(ij)[4:-5]                 # s{k}_cap_p{p}_sev{nnn}
    d = os.path.dirname(ij)
    ranks = [R["rungs"]["high"], R["rungs"]["mid"]]
    outs = [os.path.join(d, f"{base}_r{rk}.mp4") for rk in ranks]
    js = [o[:-4] + ".json" for o in outs]
    if all(os.path.exists(j) and json.load(open(j)).get("ok") for j in js):
        return [json.load(open(j)) for j in js]
    fc = "[0:v]format=yuv420p,split=2[o0][o1]"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", R["intermediate"], "-filter_complex", fc]
    for i, (o, rk) in enumerate(zip(outs, ranks)):
        cmd += ["-map", f"[o{i}]", "-r", str(sc.FPS_OUT), "-fps_mode", "cfr"] + \
               sc.encoder_args(S, S["rate"]["rungs"][rk], threads) + [o]
    t0 = time.time()
    if sc.needs_fanout(S, len(outs)):
        w, h = int(r["width"]), int(r["height"])
        dcmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", R["intermediate"], "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
        rc, err, cmd = sc.fanout_encode(dcmd, w, h, S, [S["rate"]["rungs"][rk] for rk in ranks], outs, threads)
        p = subprocess.CompletedProcess(cmd, rc, "", err)
    else:
        p = subprocess.run(cmd, capture_output=True, text=True)
    res = []
    for o, j, rk in zip(outs, js, ranks):
        pr = sc.probe_encode(o) if os.path.exists(o) else {}
        rec = dict(kind="capture", source_id=r["source_id"], dataset=r["dataset"], source_path=r["path"],
                   group_id=r["group_id"], split=r["split"], span_index=R["span_index"], span=R["span"],
                   render_span=R["render_span"], span_frames=R["render_frames"],
                   history_frames=R["history_frames"], render_chunk=R["chunk"],
                   active_xywh=json.loads(r["active_xywh"]), plan_id=R["plan_id"],
                   plan_index=R["plan_index"], severity=R["severity"], plan=R["plan"],
                   plan_at_severity=R["plan_at_severity"], labels=R["labels"], rank=rk,
                   rung_role="high" if rk == R["rungs"]["high"] else "mid",
                   rate_kind=S["rate"]["kind"], rate_value=S["rate"]["rungs"][rk], settings=S, path=o,
                   cmd_render_decode=R["cmd_decode"], cmd_lossless=R["cmd_lossless"], cmd_encode=cmd,
                   render_device=R["render_device"], torch=R["torch"], color_pipeline=R["color_pipeline"],
                   ffmpeg_rc=p.returncode,
                   ffmpeg_stderr=p.stderr.strip()[-500:], wall_s_encode_pair=round(time.time() - t0, 1),
                   versions=versions, probe=pr)
        rec["ok"] = bool(p.returncode == 0 and pr.get("decode_ok") and pr.get("frames") == R["render_frames"]
                         and abs(pr.get("fps", 0) - sc.FPS_OUT) < 0.01)
        with open(j + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(j + ".tmp", j)
        res.append(rec)
    if all(x["ok"] for x in res):
        os.remove(R["intermediate"])
        R["intermediate_deleted"] = True
        with open(ij, "w") as f:
            json.dump(R, f)
    return res


def cmd_encode(a):
    """Encode every rendered intermediate of this shard. With --loop, keep
    polling until all render shards (--render-shards N) have written their
    done-marker and no intermediate is left."""
    by_id = {r["source_id"]: r for r in kept_sources(a.manifest) if in_shard(r["source_id"], a.shard)}
    versions = tool_versions()
    t0, done, bad, failed = time.time(), 0, 0, set()
    with Pool(a.workers) as pool:
        while True:
            jobs = []
            for sid, r in sorted(by_id.items()):
                for ij in sorted(glob.glob(os.path.join(a.out, "encodes", sid, "tmp_*_cap_*.json"))):
                    R = json.load(open(ij))
                    if R.get("ok") and not R.get("intermediate_deleted") and ij not in failed:
                        jobs.append((ij, r, a.threads, versions))
            print(f"encode shard {a.shard}: {len(jobs)} rendered intermediates pending", flush=True)
            for job, res in zip(jobs, pool.imap(encode_intermediate, jobs)):
                done += 1
                if res and not all(x.get("ok") for x in res):
                    bad += sum(not x.get("ok") for x in res)
                    failed.add(job[0])
                if done % 50 == 0:
                    print(f"  {done} intermediates, {bad} bad encodes, "
                          f"{done / max((time.time() - t0) / 3600, 1e-9):.0f}/h", flush=True)
            if not a.loop:
                break
            markers = glob.glob(os.path.join(a.out, "logs", f"render_done_shard*of{a.render_shards}"))
            if not jobs and len(markers) >= a.render_shards:
                break
            if not jobs:
                time.sleep(300)
    print(f"encode shard {a.shard} finished: {done} intermediates, {bad} bad encodes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "encode"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pending-tb", type=float, default=float("inf"),
                    help="render: pause while un-encoded intermediates exceed this size")
    ap.add_argument("--loop", action="store_true",
                    help="encode: keep polling for new intermediates until the render shards finish")
    ap.add_argument("--render-shards", type=int, default=9, help="encode --loop: number of render shards")
    a = ap.parse_args()
    a.shard = tuple(int(x) for x in a.shard.split("/"))
    (cmd_render if a.stage == "render" else cmd_encode)(a)


if __name__ == "__main__":
    sys.exit(main())
