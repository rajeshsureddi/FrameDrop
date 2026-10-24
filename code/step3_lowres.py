"""Step 3c support: lossless low-res cache for Stage 2.

For every Step 2 encode (clean, capture, post-drop) and every source span
("orig"), store all frames area-downscaled to 384x216 exactly as the video
loader produces them on the fly (yuv420p -> swscale area -> yuv444p), as
lossless H.264 (qp 0, yuv444p). Decoding the cache returns byte-identical
yuv444p frames (checked per item via md5), so Stage 2 sees the same pixels
without decoding 4K.

    <out>/<sid>/<encode_id>.mkv  +  <encode_id>.json  (done marker: frames, md5)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_common as c3  # noqa: E402
from step2_common import ItemLock, source_id  # noqa: E402

LOW = "scale=384:216:flags=area,format=yuv444p"


def items(enc_manifest, step1_manifest):
    out, spans = [], {}
    with open(enc_manifest) as f:
        for r in csv.DictReader(f):
            if r["ok"] != "True":
                continue
            out.append(dict(sid=r["source_id"], eid=os.path.basename(r["path"])[:-4], path=r["path"],
                            trim=None, frames=int(r["frames"])))
            spans[r["source_id"]] = (int(r["span_start"]), int(r["span_end"]))
    with open(step1_manifest) as f:
        for r in csv.DictReader(f):
            if r["keep"] != "True":
                continue
            sid = source_id(r["path"])
            if sid in spans:
                a, b = spans[sid]
                out.append(dict(sid=sid, eid="orig", path=r["path"], trim=(a, b), frames=b - a))
    return out


def build(it, out_dir, threads):
    d = os.path.join(out_dir, it["sid"])
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, it["eid"] + ".mkv")
    pre = "format=yuv420p"
    if it["trim"]:
        pre = f"trim=start_frame={it['trim'][0]}:end_frame={it['trim'][1]},setpts=N/({c3.FPS}*TB),format=yuv420p"
    # one decode: raw low-res to us (for the md5) and lossless encode to disk
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", str(threads), "-i", it["path"],
           "-filter_complex", f"[0:v]{pre},{LOW},split=2[a][b]", "-fps_mode", "passthrough",
           "-map", "[a]", "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p",
           "-threads", "2", dst + ".tmp.mkv",
           "-map", "[b]", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-400:])
    fsz = 384 * 216 * 3
    n = len(p.stdout) // fsz
    if n != it["frames"] or len(p.stdout) % fsz:
        raise RuntimeError(f"{it['eid']}: {n} frames, want {it['frames']}")
    m_src = hashlib.md5(p.stdout).hexdigest()
    q = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", dst + ".tmp.mkv",
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv444p", "pipe:1"],
                       capture_output=True)
    m_cache = hashlib.md5(q.stdout).hexdigest()
    if m_cache != m_src:
        raise RuntimeError(f"{it['eid']}: cache not lossless ({m_cache} != {m_src})")
    os.replace(dst + ".tmp.mkv", dst)
    with open(dst[:-4] + ".json", "w") as f:
        json.dump(dict(source_id=it["sid"], encode_id=it["eid"], src=it["path"], trim=it["trim"],
                       frames=n, md5_yuv444=m_src, bytes=os.path.getsize(dst), filter=LOW), f)


def work(job):
    it, out_dir, threads = job
    done = os.path.join(out_dir, it["sid"], it["eid"] + ".json")
    if os.path.exists(done):
        return "skip", it["eid"]
    os.makedirs(os.path.join(out_dir, it["sid"]), exist_ok=True)
    with ItemLock(done[:-5] + ".lock") as got:
        if not got or os.path.exists(done):
            return "locked", it["eid"]
        try:
            build(it, out_dir, threads)
            return "ok", it["eid"]
        except Exception:
            with open(done[:-5] + ".err", "w") as f:
                f.write(traceback.format_exc())
            return "ERR", f"{it['sid']}/{it['eid']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-manifest", default="/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/encodes_manifest.csv")
    ap.add_argument("--manifest", default="/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step1/manifest.csv")
    ap.add_argument("--out", default="/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step3/lowres")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    its = items(a.enc_manifest, a.manifest)
    np.random.default_rng(a.order_seed).shuffle(its)
    if a.limit:
        its = its[:a.limit]
    print(f"{len(its)} items, {a.workers} workers", flush=True)
    t0, n_ok, n_err = time.time(), 0, 0
    with Pool(a.workers) as pool:
        for k, (st, eid) in enumerate(pool.imap_unordered(work, [(i, a.out, a.threads) for i in its], chunksize=4)):
            n_ok += st == "ok"
            n_err += st == "ERR"
            if st == "ERR":
                print("ERR", eid, flush=True)
            if (k + 1) % 500 == 0:
                print(f"[{time.strftime('%m-%d %H:%M:%S')}] {k + 1}/{len(its)} ok {n_ok} err {n_err} "
                      f"rate {n_ok / (time.time() - t0) * 3600:.0f}/h", flush=True)
    print("FINISHED", n_ok, "ok", n_err, "err", flush=True)


if __name__ == "__main__":
    main()
