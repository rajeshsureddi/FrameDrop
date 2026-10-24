"""Step 2b: clean compression ladders for every usable span of every kept source.

For each span: ONE ffmpeg process decodes the source once, trims the span
frame-accurately and feeds the 5 ladder encodes (per-source settings from
step2_common.sample_settings). Each encode gets a JSON with everything needed
to regenerate it (source, span, settings, rung, exact command, tool versions)
plus its probe results. Resumable: an encode whose JSON says ok is skipped.

python step2_ladder.py --manifest .../step1/manifest.csv --out .../step2 \
    --workers 16 --threads 4 [--shard 0/4] [--limit 200]
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import step2_common as sc


def tool_versions():
    v = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    return dict(ffmpeg=v, step2_common_settings_version=sc.SETTINGS_VERSION)


def kept_sources(manifest):
    rows = [r for r in csv.DictReader(open(manifest)) if r["keep"] == "True"]
    for r in rows:
        r["source_id"] = sc.source_id(r["path"])
        r["spans_list"] = json.loads(r["spans"])
    ids = [r["source_id"] for r in rows]
    assert len(ids) == len(set(ids)), "source_id collision"
    return sorted(rows, key=lambda r: r["source_id"])


def primary_span(r):
    """Index of the longest usable span (earliest on ties): used by 2c and 2f."""
    sp = r["spans_list"]
    return max(range(len(sp)), key=lambda i: (sp[i][1] - sp[i][0], -i))


def in_shard(sid, shard):
    k, n = shard
    return int(hashlib.sha1(sid.encode()).hexdigest(), 16) % n == k


def encode_dir(out, sid):
    return os.path.join(out, "encodes", sid)


def do_source(job):
    r, out, threads, versions = job
    d = encode_dir(out, r["source_id"])
    os.makedirs(d, exist_ok=True)
    with sc.ItemLock(os.path.join(d, ".lock_ladder")) as got:
        if not got:                       # another job is encoding this source right now
            return r["source_id"], []
        return _do_source(r, out, threads, versions, d)


def _do_source(r, out, threads, versions, d):
    sid = r["source_id"]
    S = sc.sample_settings(sid, int(r["height"]))
    results = []
    for k, (a, b) in enumerate(r["spans_list"]):
        outs = [os.path.join(d, f"s{k}_clean_r{rank}.mp4") for rank in range(5)]
        jsons = [p[:-4] + ".json" for p in outs]
        if all(os.path.exists(j) and json.load(open(j)).get("ok") for j in jsons):
            results += [json.load(open(j)) for j in jsons]
            continue
        t0 = time.time()
        if sc.needs_fanout(S, len(outs)):
            rc, err, cmd = sc.fanout_encode(sc.span_decode_cmd(r["path"], (a, b)), int(r["width"]),
                                            int(r["height"]), S, S["rate"]["rungs"], outs, threads)
            p = subprocess.CompletedProcess(cmd, rc, "", err)
        else:
            cmd = sc.ladder_command(r["path"], (a, b), S, outs, threads)
            p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        for rank, (o, j) in enumerate(zip(outs, jsons)):
            pr = sc.probe_encode(o) if os.path.exists(o) else {}
            rec = dict(
                kind="clean", source_id=sid, dataset=r["dataset"], source_path=r["path"],
                group_id=r["group_id"], split=r["split"], span_index=k, span=[a, b], span_frames=b - a,
                source_fps=float(r["fps"]), active_xywh=json.loads(r["active_xywh"]),
                span_static=json.loads(r["span_static"])[k], rank=rank,
                rate_kind=S["rate"]["kind"], rate_value=S["rate"]["rungs"][rank], settings=S,
                path=o, cmd=cmd, ffmpeg_rc=p.returncode, ffmpeg_stderr=p.stderr.strip()[-500:],
                wall_s_ladder=round(dt, 1), versions=versions, probe=pr)
            rec["ok"] = bool(p.returncode == 0 and pr.get("decode_ok") and pr.get("frames") == b - a
                             and pr.get("width") == int(r["width"]) and pr.get("height") == int(r["height"])
                             and abs(pr.get("fps", 0) - sc.FPS_OUT) < 0.01)
            with open(j + ".tmp", "w") as f:
                json.dump(rec, f, indent=1)
            os.replace(j + ".tmp", j)
            results.append(rec)
    return sid, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4, help="threads per encoder (5 encoders per worker)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--skip-shard", default="", help="leave out sources of this shard k/n")
    ap.add_argument("--limit", type=int, default=0, help="first N sources of this shard (projection run)")
    a = ap.parse_args()
    shard = tuple(int(x) for x in a.shard.split("/"))
    rows = [r for r in kept_sources(a.manifest) if in_shard(r["source_id"], shard)]
    if a.skip_shard:                       # sources owned by another (e.g. projection) run
        skip = tuple(int(x) for x in a.skip_shard.split("/"))
        rows = [r for r in rows if not in_shard(r["source_id"], skip)]
    if a.limit:
        rows = rows[:a.limit]
    versions = tool_versions()
    print(f"shard {a.shard}: {len(rows)} sources, {sum(len(r['spans_list']) for r in rows)} spans, "
          f"{a.workers} workers x 5 encoders x {a.threads} threads", flush=True)
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
                if done % 20 == 0:
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
