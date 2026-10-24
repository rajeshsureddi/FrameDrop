"""Step 2f: post-drop re-encodes (2 variants for every kept source).

Real files are often re-encoded after frames were dropped, so held frames
become near-identical rather than exact. Per variant: pick a clean-ladder
encode of the source's primary (longest) usable span, sample a drop pattern from
psvqa/drops.py (broad family, uniform placement), stream the decoded frames
through the display index map, and re-encode with the source's encoder
settings (same encoder/preset/tune/B-frames/GOP) in CRF mode at a CRF sampled
uniformly per variant: [18, 51] x264/x265, [18, 60] svt-av1/vp9.
The JSON stores the drop mask, the exact display index map, the CRF and the
command, so the clip can be regenerated from the clean encode.

python step2_postdrop.py --manifest .../step1/manifest.csv --out .../step2 --workers 32 --threads 4 [--shard k/n]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

import step2_common as sc
from step2_ladder import in_shard, kept_sources, primary_span, tool_versions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "psvqa"))
from psvqa import drops  # noqa: E402

N_VARIANTS = 2


def plan_variants(r, S):
    """Deterministic per-source choice of rungs, drop patterns and CRFs."""
    rng = sc._rng(r["source_id"], "postdrop", S["seed"])
    a, b = r["spans_list"][primary_span(r)]
    T = b - a
    lo, hi = sc.CRF_RANGE[S["encoder"]]
    out = []
    for v in range(N_VARIANTS):
        rank = int(rng.integers(0, 5))
        mask, fam = drops.sample_pattern(T, rng)
        while not mask.any():                    # a post-drop clip must contain drops
            mask, fam = drops.sample_pattern(T, rng)
        crf = int(rng.integers(lo, hi + 1))
        out.append(dict(variant=v, rank=rank, family=fam, drop_mask=mask, crf=crf))
    return out


def _valid_output(enc_rc, pr, T):
    return (enc_rc == 0 and bool(pr.get("decode_ok")) and pr.get("frames") == T
            and abs(pr.get("fps", 0) - sc.FPS_OUT) < 0.01)


def run_variant(r, S, V, d, threads, versions):
    k = primary_span(r)
    src = os.path.join(d, f"s{k}_clean_r{V['rank']}.mp4")
    dst = os.path.join(d, f"s{k}_postdrop_v{V['variant']}_crf{V['crf']}.mp4")
    jpath = dst[:-4] + ".json"
    if os.path.exists(jpath):
        old = json.load(open(jpath))
        if old.get("ok"):
            return old
        if "dec_complete" not in old and _valid_output(old.get("ffmpeg_rc"), old.get("probe", {}), old["span_frames"]):
            # pre-fix record: failed only on the decoder's broken-pipe exit after a tail drop
            old.update(ok=True, revalidated="decoder rc ignored (pipe closed early after tail drop)")
            with open(jpath + ".tmp", "w") as f:
                json.dump(old, f)
            os.replace(jpath + ".tmp", jpath)
            return old
    a, b = r["spans_list"][k]
    T = b - a
    idx = drops.drops_to_index(V["drop_mask"])
    w, h = int(r["width"]), int(r["height"])
    fsz = w * h * 3 // 2
    S2 = dict(S, mode="crf")
    enc_cmd = (["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{w}x{h}", "-r", str(sc.FPS_OUT), "-i", "-"]
               + sc.encoder_args(S2, V["crf"], threads) + [dst])
    dec_cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-fps_mode", "passthrough", "-f", "rawvideo",
               "-pix_fmt", "yuv420p", "-"]
    t0 = time.time()
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cur, cur_i, short = None, -1, False
    try:
        for t in range(T):                        # idx is non-decreasing: stream, O(1) memory
            while cur_i < idx[t]:
                cur = dec.stdout.read(fsz)
                cur_i += 1
                short |= len(cur) != fsz
            enc.stdin.write(cur)
    finally:
        enc.stdin.close()
        dec.stdout.close()
    enc_err = enc.stderr.read().decode(errors="ignore")
    enc.wait()
    dec.wait()
    pr = sc.probe_encode(dst) if os.path.exists(dst) else {}
    rec = dict(kind="postdrop", source_id=r["source_id"], dataset=r["dataset"], source_path=r["path"],
               group_id=r["group_id"], split=r["split"], span_index=k, span=[a, b], span_frames=T,
               active_xywh=json.loads(r["active_xywh"]), variant=V["variant"], base_encode=src,
               base_rank=V["rank"], drop_family=V["family"],
               drop_mask=np.flatnonzero(V["drop_mask"]).tolist(), display_index=idx.tolist(),
               n_dropped=int(V["drop_mask"].sum()), crf=V["crf"], rate_kind="crf", rate_value=V["crf"],
               settings=S2, source_settings=S, path=dst, cmd_decode=dec_cmd, cmd_encode=enc_cmd,
               ffmpeg_rc=enc.returncode, dec_rc=dec.returncode, dec_complete=not short, ffmpeg_stderr=enc_err.strip()[-500:],
               wall_s=round(time.time() - t0, 1), versions=versions, probe=pr)
    # the decoder is not checked by exit code: when the last frames are dropped we stop reading and
    # close its pipe, so it exits non-zero; a real decode failure shows up as a short frame read
    rec["ok"] = bool(not short and _valid_output(enc.returncode, pr, T))
    with open(jpath + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(jpath + ".tmp", jpath)
    return rec


def do_source(job):
    r, out, threads, versions = job
    d = os.path.join(out, "encodes", r["source_id"])
    os.makedirs(d, exist_ok=True)
    with sc.ItemLock(os.path.join(d, ".lock_postdrop")) as got:
        if not got:
            return r["source_id"], []
        return _do_source(r, out, threads, versions, d)


def _do_source(r, out, threads, versions, d):
    S = sc.sample_settings(r["source_id"], int(r["height"]))
    res = []
    for V in plan_variants(r, S):
        if not os.path.exists(os.path.join(d, f"s{primary_span(r)}_clean_r{V['rank']}.mp4")):
            res.append(dict(ok=False, source_id=r["source_id"], error="clean ladder encode missing"))
            continue
        res.append(run_variant(r, S, V, d, threads, versions))
    return r["source_id"], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--skip-shard", default="", help="leave out sources of this shard k/n")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    shard = tuple(int(x) for x in a.shard.split("/"))
    rows = [r for r in kept_sources(a.manifest) if in_shard(r["source_id"], shard)]
    if a.skip_shard:                       # sources owned by another (e.g. projection) run
        skip = tuple(int(x) for x in a.skip_shard.split("/"))
        rows = [r for r in rows if not in_shard(r["source_id"], skip)]
    if a.limit:
        rows = rows[:a.limit]
    versions = tool_versions()
    print(f"shard {a.shard}: {len(rows)} sources x {N_VARIANTS} post-drop variants", flush=True)
    t0, done, bad = time.time(), 0, 0
    todo = rows
    with Pool(a.workers) as pool:
        for attempt in range(31):          # revisit sources another job had locked
            skipped = []
            for r, (sid, res) in zip(todo, pool.imap(do_source, [(r, a.out, a.threads, versions) for r in todo])):
                if not res:
                    skipped.append(r)
                    continue
                done += 1
                bad += sum(not x.get("ok") for x in res)
                if done % 50 == 0:
                    h = (time.time() - t0) / 3600
                    print(f"  {done}/{len(rows)} sources, {bad} bad, {done / max(h, 1e-9):.0f} sources/h", flush=True)
            if not skipped:
                break
            print(f"  {len(skipped)} sources locked by another job; retrying in 10 min", flush=True)
            todo = skipped
            time.sleep(600)
    print(f"finished: {done} sources processed here, {bad} bad", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
