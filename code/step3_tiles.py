"""Step 3a: tile store for Stage 1 (Tile-MAE).

For every kept source: the original span ("orig"), its 5 clean-ladder encodes
and its 8 capture encodes are decoded in lockstep. Every 6th source frame
(absolute index span[0] + 6k) gets 16 random tile positions on the absolute
4K grid; the same positions are cut from every version, so ladder and
severity pairs share content. Each tile is the 384x384 RGB uint8 crop (256
core + 64 px margin per side), stored losslessly as PNG.

Output (WebDataset layout, one tar per source):
    <out>/tiles/<sid>.tar       samples "<sid>__f<frame>_t<j>" with
        .json      group metadata (source, frame, x, y, versions[])
        .vNN.png   one 384x384 RGB tile per version
    <out>/tiles/<sid>.json      per-source index (written last = done marker)

x, y in the metadata are the CORE's top-left on the 3840x2160 grid; the crop
starts at (x-64, y-64). Positions are even (yuv420 chroma alignment), keep the
whole 384 crop inside the frame and the core inside the source's active area.

Resumable: per-source lock files, skip sources whose index exists.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_common as c3  # noqa: E402
from step2_common import ItemLock, source_id  # noqa: E402

SEED = 2026
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]
STORE_VERSION = 1


def _rng(sid, salt):
    h = hashlib.sha256(f"{SEED}|{sid}|{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def tile_positions(sid, frames, active_xywh):
    """{frame: [(x, y) core top-left] * 16}, deterministic per source."""
    r = _rng(sid, "tiles")
    ax, ay, aw, ah = active_xywh
    m, core = c3.TILE_MARGIN, c3.TILE_CORE
    x_lo, x_hi = max(m, ax), min(c3.W4K - core - m, ax + aw - core)
    y_lo, y_hi = max(m, ay), min(c3.H4K - core - m, ay + ah - core)
    if x_hi < x_lo:
        x_lo, x_hi = m, c3.W4K - core - m
    if y_hi < y_lo:
        y_lo, y_hi = m, c3.H4K - core - m
    out = {}
    for f in frames:
        xs = r.integers(-(-x_lo // 2), x_hi // 2 + 1, c3.TILES_PER_FRAME) * 2
        ys = r.integers(-(-y_lo // 2), y_hi // 2 + 1, c3.TILES_PER_FRAME) * 2
        out[f] = [(int(x), int(y)) for x, y in zip(xs, ys)]
    return out


def source_versions(sid, enc_dir):
    """Versions of one source: orig + clean + capture, from the Step 2 JSONs."""
    d = os.path.join(enc_dir, sid)
    js = sorted(f for f in os.listdir(d) if f.startswith("s") and f.endswith(".json")
                and not f.endswith(".qc.json") and ("_clean_" in f or "_cap_" in f))
    vers = []
    for f in js:
        j = c3.load_json(os.path.join(d, f))
        if not j.get("ok"):
            raise RuntimeError(f"encode not ok: {f}")
        pr = j["probe"]
        start = j.get("render_span", j["span"])[0]
        vers.append(dict(
            encode_id=f[:-5], kind=j["kind"], rank=j["rank"], plan_id=j.get("plan_id"),
            severity=j.get("severity", 0.0), encoder=j["settings"]["encoder"],
            mode=j["settings"]["mode"], rate_kind=j["rate_kind"], rate_value=j["rate_value"],
            size_bytes=pr["size_bytes"], actual_kbps=pr["actual_kbps"], frames=pr["frames"],
            start=start, path=os.path.join(d, f[:-5] + ".mp4")))
    j0 = c3.load_json(os.path.join(d, js[0]))
    vers.sort(key=lambda v: ({"clean": 1, "capture": 2}[v["kind"]], v["encode_id"]))
    orig = dict(encode_id="orig", kind="orig", rank=None, plan_id=None, severity=0.0,
                encoder=None, mode=None, rate_kind=None, rate_value=None, size_bytes=None,
                actual_kbps=None, frames=j0["span"][1] - j0["span"][0], start=j0["span"][0],
                path=j0["source_path"])
    return j0, [orig] + vers


def _select_cmd(v, span, threads):
    s0 = span[0]
    off = (v["start"] - s0) % c3.STORE_STEP
    sel = f"select=eq(mod(n+{off}\\,{c3.STORE_STEP})\\,0)"
    if v["kind"] == "orig":
        vf = f"trim=start_frame={span[0]}:end_frame={span[1]},setpts=N/({c3.FPS}*TB),{sel},format=yuv420p"
    else:
        vf = f"{sel},format=yuv420p"
    return ["ffmpeg", "-v", "error", "-nostdin", "-threads", str(threads), "-i", v["path"],
            "-fps_mode", "passthrough", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]


def _add(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    tf.addfile(ti, io.BytesIO(data))


def process_source(sid, enc_dir, out_dir, threads=1, local_tmp="/tmp"):
    j0, vers = source_versions(sid, enc_dir)
    span = j0["span"]
    frames = list(range(span[0], span[1], c3.STORE_STEP))
    pos = tile_positions(sid, frames, j0.get("active_xywh") or [0, 0, c3.W4K, c3.H4K])
    # which stored frames each version has, in order
    for v in vers:
        v["sel"] = [f for f in frames if v["start"] <= f < v["start"] + v["frames"]]
    tmp = os.path.join(out_dir, f"{sid}.tar.tmp")
    meta_v = [{k: v[k] for k in ("encode_id", "kind", "rank", "plan_id", "severity", "encoder",
                                 "mode", "rate_kind", "rate_value", "size_bytes", "actual_kbps")}
              for v in vers]
    n_groups = n_tiles = 0
    hw = c3.TILE_FULL
    # One decoder at a time (a 4K decoder holds ~300 MB of reference frames);
    # PNG tiles are spilled to node-local disk, then assembled into groups.
    spill_dir = tempfile.mkdtemp(prefix=f"tiles_{sid}_", dir=local_tmp)
    try:
        where = {}                                   # (i, f, t) -> (offset, length) in spill file i
        for i, v in enumerate(vers):
            got = 0
            with open(os.path.join(spill_dir, f"v{i:02d}.bin"), "wb") as sp:
                off = 0
                for b in c3.iter_frames(_select_cmd(v, span, threads)):
                    if got >= len(v["sel"]):
                        got += 1
                        continue
                    f = v["sel"][got]
                    got += 1
                    Y, U, V = c3.yuv_planes(b, c3.W4K, c3.H4K)
                    for t, (x, y) in enumerate(pos[f]):
                        tile = c3.yuv_crop_to_rgb(Y, U, V, x - c3.TILE_MARGIN, y - c3.TILE_MARGIN, hw, hw)
                        ok, png = cv2.imencode(".png", tile[..., ::-1], PNG_PARAMS)   # cv2 is BGR
                        assert ok
                        sp.write(png)
                        where[(i, f, t)] = (off, len(png))
                        off += len(png)
            if got != len(v["sel"]):
                raise RuntimeError(f"{v['encode_id']}: decoded {got} selected frames, want "
                                   f"{len(v['sel'])} {c3.iter_frames.last_err}")
        spills = [open(os.path.join(spill_dir, f"v{i:02d}.bin"), "rb") for i in range(len(vers))]
        with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tf:
            for f in frames:
                for t, (x, y) in enumerate(pos[f]):
                    key = f"{sid}__f{f:04d}_t{t:02d}"
                    vlist = []
                    for i, v in enumerate(vers):
                        if (i, f, t) not in where:
                            continue
                        o, n = where[(i, f, t)]
                        spills[i].seek(o)
                        _add(tf, f"{key}.v{i:02d}.png", spills[i].read(n))
                        vlist.append(dict(meta_v[i], v=f"v{i:02d}", enc_frame=f - v["start"]))
                        n_tiles += 1
                    meta = dict(source_id=sid, split=j0["split"], group_id=j0["group_id"],
                                dataset=j0["dataset"], frame=f, x=x, y=y, margin=c3.TILE_MARGIN,
                                versions=vlist)
                    _add(tf, f"{key}.json", json.dumps(meta).encode())
                    n_groups += 1
        for h in spills:
            h.close()
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)
    tar = os.path.join(out_dir, f"{sid}.tar")
    os.replace(tmp, tar)
    idx = dict(source_id=sid, split=j0["split"], group_id=j0["group_id"], dataset=j0["dataset"],
               span=span, frames=frames, n_groups=n_groups, n_tiles=n_tiles,
               tar=os.path.basename(tar), tar_bytes=os.path.getsize(tar), store_version=STORE_VERSION,
               versions=[dict(meta_v[i], v=f"v{i:02d}", start=vers[i]["start"],
                              n_frames=len(vers[i]["sel"])) for i in range(len(vers))])
    with open(os.path.join(out_dir, f"{sid}.json.tmp"), "w") as fo:
        json.dump(idx, fo)
    os.replace(os.path.join(out_dir, f"{sid}.json.tmp"), os.path.join(out_dir, f"{sid}.json"))
    return n_groups, n_tiles


def work(job):
    sid, enc_dir, out_dir, threads = job
    idx = os.path.join(out_dir, f"{sid}.json")
    if os.path.exists(idx):
        return sid, "skip", 0, 0.0
    with ItemLock(os.path.join(out_dir, f".lock_{sid}")) as got:
        if not got or os.path.exists(idx):
            return sid, "locked", 0, 0.0
        t0 = time.time()
        try:
            _, nt = process_source(sid, enc_dir, out_dir, threads)
            return sid, "ok", nt, time.time() - t0
        except Exception:
            with open(os.path.join(out_dir, f"{sid}.err"), "w") as fe:
                fe.write(traceback.format_exc())
            try:
                os.remove(os.path.join(out_dir, f"{sid}.tar.tmp"))
            except FileNotFoundError:
                pass
            return sid, "ERR", 0, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encodes", default="/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/encodes")
    ap.add_argument("--out", default="/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step3/tiles")
    ap.add_argument("--manifest", default="/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step1/manifest.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2, help="ffmpeg threads per decoder")
    ap.add_argument("--only", nargs="*", help="source ids")
    ap.add_argument("--order-seed", type=int, default=0, help="per-node source order")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only:
        sids = a.only
    else:
        import csv
        sids = []
        with open(a.manifest) as f:
            for r in csv.DictReader(f):
                if r["keep"] == "True":
                    sids.append(source_id(r["path"]))
        sids = [s for s in sids if os.path.isdir(os.path.join(a.encodes, s))]
        np.random.default_rng(a.order_seed).shuffle(sids)
    print(f"{len(sids)} sources, {a.workers} workers", flush=True)
    jobs = [(s, a.encodes, a.out, a.threads) for s in sids]
    t0, n_ok, n_err, tiles = time.time(), 0, 0, 0
    with Pool(a.workers) as pool:
        for k, (sid, st, nt, dt) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            if st == "ok":
                n_ok += 1
                tiles += nt
            elif st == "ERR":
                n_err += 1
            if st in ("ok", "ERR"):
                el = time.time() - t0
                print(f"[{time.strftime('%m-%d %H:%M:%S')}] {k + 1}/{len(sids)} {sid} {st} "
                      f"{dt:.0f}s | done here {n_ok} err {n_err} tiles {tiles} "
                      f"rate {n_ok / el * 3600:.0f} src/h", flush=True)
    print("FINISHED", n_ok, "ok", n_err, "err", flush=True)


if __name__ == "__main__":
    main()
