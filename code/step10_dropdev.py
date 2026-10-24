"""Step 10a: DropDev, a drop-ordering development benchmark (no study data).

Built only from `audit`-split sources (never used in any training stage or
checkpoint selection). For each source, the existing Step 2 clean-ladder
encodes at rungs 4 / 2 / 0 (top / mid / bottom; the source's own Step 2
encoder settings) are the bases. On each base, 8 pairs (better, worse):

  burst_length   one burst of L1 vs L2 frames at the same start, L1 < L2,
                 L in {1, 2, 3, 5, 8, 15, 25, 40, 60}
  burst_count    k vs k + 2 non-overlapping bursts of the same length
  density        Bernoulli p1 vs p2 (nested: worse drops a superset), p2 >= p1 + 0.1
  none_vs_some   no drops vs one burst of >= 3 frames

one pair per type as an index remap of the base decode ("raw"), and one pair per
type re-encoded as in 2f ("reenc": both members re-encoded with the source's
encoder at the same CRF, sampled uniformly per base from the 2f range). So 50%
of the pairs are re-encoded and the two members of a pair differ only in drops.

Label-side measurements (never model inputs): mean |A - B| of the displayed
384x216 luma over the ticks whose displayed frame differs between A and B
(indistinguishable if < 0.5 gray levels; the all-frames mean is kept too), and motion from the
clean source span's frame differences: mean |S_t - S_t-1| over the frames the
worse member additionally drops (pair motion), and over the whole span.

    python step10_dropdev.py plan
    python step10_dropdev.py encode --workers 32 --threads 4 [--shard k/n]
    python step10_dropdev.py measure --workers 64 [--shard k/n]
    python step10_dropdev.py collect
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "psvqa"))
import step2_common as sc  # noqa: E402
from psvqa import drops  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
VARIANT = os.environ.get("DROPDEV", "base")    # "hard": DropDev-hard (10-19), see HARD below
OUT = f"{ROOT}/step10_dropdev" + ("_hard" if VARIANT == "hard" else "")
MANIFEST = f"{ROOT}/step2/encodes_manifest.csv"
RUNGS = (4, 2, 0)
TYPES = ("burst_length", "burst_count", "density", "none_vs_some")
LENGTHS = (1, 2, 3, 5, 8, 15, 25, 40, 60)
EDGE = 8                       # no drops in the first / last EDGE frames
LOW_W, LOW_H = 384, 216
SEED = 2026


def in_shard(key, shard):
    k, n = shard
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % n == k


def mask_from(T, idx):
    m = np.zeros(T, bool)
    m[list(idx)] = True
    m[0] = False
    return m


# ------------------------------------------------------------------ patterns
def p_burst_length(T, rng):
    i, j = sorted(rng.choice(len(LENGTHS), 2, replace=False))
    L1, L2 = LENGTHS[i], LENGTHS[j]
    s = int(rng.integers(EDGE, T - EDGE - L2))
    return (mask_from(T, range(s, s + L1)), mask_from(T, range(s, s + L2)),
            dict(L1=L1, L2=L2, start=s))


def _place_bursts(T, n, L, rng, gap=4):
    for _ in range(1000):
        st = np.sort(rng.integers(EDGE, T - EDGE - L, n))
        if n == 1 or np.all(np.diff(st) >= L + gap):
            return [int(x) for x in st]
    return None


def p_burst_count(T, rng):
    while True:
        L = int(rng.choice([1, 2, 3, 5, 8]))
        k = int(rng.integers(1, 5))
        st = _place_bursts(T, k + 2, L, rng)
        if st is not None:
            break
    keep = sorted(rng.choice(k + 2, k, replace=False))
    b = [st[q] for q in keep]
    better = mask_from(T, [s + d for s in b for d in range(L)])
    worse = mask_from(T, [s + d for s in st for d in range(L)])
    return better, worse, dict(L=L, k1=k, k2=k + 2, starts=st, better_starts=b)


def p_density(T, rng):
    while True:
        p1 = float(rng.uniform(0.03, 0.6))
        p2 = float(rng.uniform(p1 + 0.1, 0.8))
        u = rng.random(T)
        inner = np.zeros(T, bool)
        inner[EDGE:T - EDGE] = True
        better, worse = (u < p1) & inner, (u < p2) & inner
        better[0] = worse[0] = False
        if better.any() and (worse & ~better).sum() >= 3:
            return better, worse, dict(p1=round(p1, 4), p2=round(p2, 4))


def p_none_vs_some(T, rng):
    L = int(round(np.exp(rng.uniform(np.log(3), np.log(60)))))
    s = int(rng.integers(EDGE, T - EDGE - L))
    return np.zeros(T, bool), mask_from(T, range(s, s + L)), dict(L=L, start=s)


GEN = dict(burst_length=p_burst_length, burst_count=p_burst_count, density=p_density, none_vs_some=p_none_vs_some)

# DropDev-hard (10-19, after the base gate saturated at Q ~0.96 for v1/v2/v3): same audit bases, rungs, pair
# types, raw/reenc modes, measurements, metric and 10a/10b rules; only the drop differences are smaller.
# Chosen from pair properties only (no model was scored on any hard pair before this was fixed):
#   burst_length   L1 < L2 <= 6 (base: up to 60)
#   burst_count    L in {1, 2}, k vs k + 1 bursts (base: L <= 8, k vs k + 2)
#   density        nested, p2 = p1 + U(0.03, 0.08), p1 in [0.02, 0.4] (base: gap >= 0.1)
#   none_vs_some   one burst of 1-3 frames (base: 3-60)
#   reenc          CRF from the upper half of the source encoder's 2f range (stronger masking)
HARD_LENGTHS = (1, 2, 3, 4, 5, 6)


def h_burst_length(T, rng):
    i, j = sorted(rng.choice(len(HARD_LENGTHS), 2, replace=False))
    L1, L2 = HARD_LENGTHS[i], HARD_LENGTHS[j]
    s = int(rng.integers(EDGE, T - EDGE - L2))
    return (mask_from(T, range(s, s + L1)), mask_from(T, range(s, s + L2)),
            dict(L1=L1, L2=L2, start=s))


def h_burst_count(T, rng):
    while True:
        L = int(rng.choice([1, 2]))
        k = int(rng.integers(1, 5))
        st = _place_bursts(T, k + 1, L, rng)
        if st is not None:
            break
    keep = sorted(rng.choice(k + 1, k, replace=False))
    b = [st[q] for q in keep]
    better = mask_from(T, [s + d for s in b for d in range(L)])
    worse = mask_from(T, [s + d for s in st for d in range(L)])
    return better, worse, dict(L=L, k1=k, k2=k + 1, starts=st, better_starts=b)


def h_density(T, rng):
    while True:
        p1 = float(rng.uniform(0.02, 0.4))
        p2 = p1 + float(rng.uniform(0.03, 0.08))
        u = rng.random(T)
        inner = np.zeros(T, bool)
        inner[EDGE:T - EDGE] = True
        better, worse = (u < p1) & inner, (u < p2) & inner
        better[0] = worse[0] = False
        if better.any() and (worse & ~better).any():
            return better, worse, dict(p1=round(p1, 4), p2=round(p2, 4))


def h_none_vs_some(T, rng):
    L = int(rng.integers(1, 4))
    s = int(rng.integers(EDGE, T - EDGE - L))
    return np.zeros(T, bool), mask_from(T, range(s, s + L)), dict(L=L, start=s)


if VARIANT == "hard":
    GEN = dict(burst_length=h_burst_length, burst_count=h_burst_count, density=h_density, none_vs_some=h_none_vs_some)
    SEED = 2027


def clip_id(base_id, mask, crf):
    h = hashlib.sha1(np.packbits(mask).tobytes()).hexdigest()[:12] if mask.any() else "none"
    return f"{base_id}__{'raw' if crf is None else f'crf{crf}'}__{h}"


# ------------------------------------------------------------------ plan
def plan(a):
    rows = [r for r in csv.DictReader(open(MANIFEST))
            if r["split"] == "audit" and r["kind"] == "clean" and r["ok"] == "True" and int(r["rank"]) in RUNGS]
    os.makedirs(f"{OUT}/clips", exist_ok=True)
    clips, pairs = {}, []
    for r in sorted(rows, key=lambda r: (r["source_id"], -int(r["rank"]))):
        j = json.load(open(r["json_path"]))
        assert j["split"] == "audit"
        base_id = f"{r['source_id']}__r{r['rank']}"
        T = int(r["frames"])
        rng = sc._rng(base_id, "dropdev", SEED)
        lo, hi = sc.CRF_RANGE[r["encoder"]]
        if VARIANT == "hard":
            lo = (lo + hi + 1) // 2
        crf = int(rng.integers(lo, hi + 1))
        for typ in TYPES:
            for mode in ("raw", "reenc"):
                better, worse, info = GEN[typ](T, rng)
                assert not (better & ~worse).any() and (worse & ~better).any()   # strict superset
                c = None if mode == "raw" else crf
                ids = []
                for m in (better, worse):
                    cid = clip_id(base_id, m, c)
                    clips.setdefault(cid, dict(clip=cid, base_id=base_id, source_id=r["source_id"],
                                               group_id=r["group_id"], rank=int(r["rank"]), base=r["path"],
                                               base_json=r["json_path"], encoder=r["encoder"], T=T,
                                               w=int(r["width"]), h=int(r["height"]),
                                               crf=c, drop=np.flatnonzero(m).tolist()))
                    ids.append(cid)
                pairs.append(dict(pair=f"{base_id}__{typ}__{mode}", type=typ, mode=mode, base_id=base_id,
                                  source_id=r["source_id"], group_id=r["group_id"], rank=int(r["rank"]),
                                  encoder=r["encoder"], crf=c, better=ids[0], worse=ids[1],
                                  n_better=int(better.sum()), n_worse=int(worse.sum()),
                                  n_extra=int((worse & ~better).sum()), span_static=j.get("span_static"),
                                  source_path=j["source_path"], span=j["span"], info=info))
    json.dump(dict(variant=VARIANT, seed=SEED, rungs=RUNGS, types=TYPES, n_bases=len(rows), clips=list(clips.values()), pairs=pairs),
              open(f"{OUT}/plan.json", "w"))
    n_re = sum(c["crf"] is not None for c in clips.values())
    print(f"{len(rows)} bases from {len({r['source_id'] for r in rows})} audit sources: "
          f"{len(pairs)} pairs, {len(clips)} clips ({n_re} to re-encode)", flush=True)


def load_plan():
    return json.load(open(f"{OUT}/plan.json"))


# ------------------------------------------------------------------ encode (2f style)
def reencode(job):
    c, threads = job
    try:
        return _reencode(c, threads)
    except Exception as e:                    # noqa: BLE001  (one failed encode must not stop the pool)
        dst = f"{OUT}/clips/{c['clip']}.json"
        with open(dst + ".tmp", "w") as f:
            json.dump(dict(c, ok=False, error=f"{type(e).__name__}: {str(e)[-300:]}"), f)
        os.replace(dst + ".tmp", dst)
        return c["clip"], False, 0.0


def _reencode(c, threads):
    dst = f"{OUT}/clips/{c['clip']}.mp4"
    jpath = dst[:-4] + ".json"
    if os.path.exists(jpath) and json.load(open(jpath)).get("ok"):
        return c["clip"], True, 0.0
    with sc.ItemLock(dst + ".lock") as got:
        if not got:
            return c["clip"], None, 0.0
        bj = json.load(open(c["base_json"]))
        S = bj["settings"] if isinstance(bj["settings"], dict) else json.loads(bj["settings"])
        S2 = dict(S, mode="crf")
        T = c["T"]
        idx = drops.drops_to_index(mask_from(T, c["drop"]))
        w, h = c["w"], c["h"]
        fsz = w * h * 3 // 2
        enc_cmd = (["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s", f"{w}x{h}", "-r", str(sc.FPS_OUT), "-i", "-"]
                   + sc.encoder_args(S2, c["crf"], threads) + [dst])
        dec_cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", c["base"], "-fps_mode", "passthrough",
                   "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
        t0 = time.time()
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        cur, cur_i, short = None, -1, False
        try:
            for t in range(T):
                while cur_i < idx[t]:
                    cur = dec.stdout.read(fsz)
                    cur_i += 1
                    short |= len(cur) != fsz
                enc.stdin.write(cur)
        except BrokenPipeError:
            short = True                          # encoder died; recorded as not ok, retried on rerun
        finally:
            for f_ in (enc.stdin, dec.stdout):
                try:
                    f_.close()
                except BrokenPipeError:
                    pass
            if short:
                dec.kill()
        err = enc.stderr.read().decode(errors="ignore")
        enc.wait()
        dec.wait()
        pr = sc.probe_encode(dst) if os.path.exists(dst) else {}
        ok = bool(not short and enc.returncode == 0 and pr.get("decode_ok") and pr.get("frames") == T)
        rec = dict(c, path=dst, display_index=idx.tolist(), settings=S2, cmd_encode=enc_cmd, cmd_decode=dec_cmd,
                   ffmpeg_rc=enc.returncode, ffmpeg_stderr=err.strip()[-500:], wall_s=round(time.time() - t0, 1),
                   probe=pr, ok=ok)
        with open(jpath + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(jpath + ".tmp", jpath)
        return c["clip"], ok, rec["wall_s"]


def encode(a):
    P = load_plan()
    shard = tuple(int(x) for x in a.shard.split("/"))
    todo = [c for c in P["clips"] if c["crf"] is not None and in_shard(c["clip"], shard)]
    todo.sort(key=lambda c: c["encoder"] != "libvpx-vp9")         # slowest first
    print(f"shard {a.shard}: {len(todo)} re-encodes", flush=True)
    t0, done, bad = time.time(), 0, 0
    for attempt in range(20):
        locked = []
        with Pool(a.workers) as pool:
            for k, (cid, ok, ws) in enumerate(pool.imap_unordered(reencode, [(c, a.threads) for c in todo])):
                if ok is None:
                    locked.append(cid)
                    continue
                done += 1
                bad += not ok
                if done % 100 == 0:
                    print(f"  {done}/{len(todo)} done, {bad} bad, {done / (time.time() - t0) * 3600:.0f}/h", flush=True)
        if not locked:
            break
        print(f"  {len(locked)} locked by another job; retry in 10 min", flush=True)
        todo = [c for c in todo if c["clip"] in set(locked)]
        time.sleep(600)
    print(f"FINISHED encode: {done} done, {bad} bad", flush=True)


# ------------------------------------------------------------------ measure (label side)
def decode_gray(path, span=None):
    vf = (f"trim=start_frame={span[0]}:end_frame={span[1]},setpts=N/(60*TB)," if span else "")
    vf += f"scale={LOW_W}:{LOW_H}:flags=area,format=gray"
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", path, "-vf", vf,
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, LOW_H, LOW_W)


def measure_base(job):
    base_id, pairs, clips = job
    dst = f"{OUT}/measure/{base_id}.json"
    if os.path.exists(dst):
        return base_id, "skip"
    try:
        c0 = clips[pairs[0]["better"]]
        T = c0["T"]
        base = decode_gray(c0["base"]).astype(np.int16)
        src = decode_gray(pairs[0]["source_path"], pairs[0]["span"]).astype(np.int16)
        if len(base) != T or len(src) != T:
            raise RuntimeError(f"frames base {len(base)} src {len(src)} != {T}")
        sdiff = np.zeros(T)
        sdiff[1:] = np.abs(np.diff(src, axis=0)).mean(axis=(1, 2))

        cache = {}

        def shown(cid):
            if cid not in cache:
                c = clips[cid]
                if c["crf"] is None:
                    cache[cid] = base[drops.drops_to_index(mask_from(T, c["drop"]))]
                else:
                    cj = json.load(open(f"{OUT}/clips/{cid}.json"))
                    if not cj.get("ok"):
                        raise RuntimeError(f"re-encode not ok: {cid}")
                    cache[cid] = decode_gray(cj["path"]).astype(np.int16)
            return cache[cid]

        res = []
        for p in pairs:
            A, B = shown(p["better"]), shown(p["worse"])
            extra = mask_from(T, clips[p["worse"]]["drop"]) & ~mask_from(T, clips[p["better"]]["drop"])
            ib = drops.drops_to_index(mask_from(T, clips[p["better"]]["drop"]))
            iw = drops.drops_to_index(mask_from(T, clips[p["worse"]]["drop"]))
            ch = ib != iw                                      # ticks whose displayed frame differs
            d = np.abs(A - B).mean(axis=(1, 2))
            res.append(dict(pair=p["pair"], pix_diff=float(d[ch].mean()), pix_diff_all=float(d.mean()),
                            n_changed=int(ch.sum()),
                            motion_pair=float(sdiff[extra].mean()), motion_clip=float(sdiff[1:].mean())))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(res, open(dst + ".tmp", "w"))
        os.replace(dst + ".tmp", dst)
        return base_id, "ok"
    except Exception as e:                                   # noqa: BLE001
        return base_id, f"ERR {type(e).__name__}: {str(e)[-300:]}"


def measure(a):
    P = load_plan()
    clips = {c["clip"]: c for c in P["clips"]}
    by = {}
    for p in P["pairs"]:
        by.setdefault(p["base_id"], []).append(p)
    shard = tuple(int(x) for x in a.shard.split("/"))
    jobs = [(b, ps, {k: clips[k] for p in ps for k in (p["better"], p["worse"])})
            for b, ps in by.items() if in_shard(b, shard)]
    print(f"shard {a.shard}: {len(jobs)} bases to measure", flush=True)
    n = 0
    with Pool(a.workers) as pool:
        for b, st in pool.imap_unordered(measure_base, jobs):
            n += 1
            if st.startswith("ERR"):
                print(b, st, flush=True)
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print("FINISHED measure", flush=True)


def collect(a):
    """pairs.csv: plan + measurements + motion level (tertiles of pair motion over all pairs)."""
    P = load_plan()
    m = {}
    for p in P["pairs"]:
        f = f"{OUT}/measure/{p['base_id']}.json"
        if p["base_id"] not in m and os.path.exists(f):
            m.update({x["pair"]: x for x in json.load(open(f))})
    rows = [dict({k: v for k, v in p.items() if k not in ("info", "span", "source_path")},
                 info=json.dumps(p["info"]), **m[p["pair"]]) for p in P["pairs"] if p["pair"] in m]
    mo = np.array([r["motion_pair"] for r in rows])
    t1, t2 = np.quantile(mo, [1 / 3, 2 / 3])
    for r in rows:
        r["motion_level"] = "low" if r["motion_pair"] < t1 else "mid" if r["motion_pair"] < t2 else "high"
        r["indistinguishable"] = r["pix_diff"] < 0.5
        r["static"] = bool(r["span_static"]) or r["motion_clip"] < 0.5
    with open(f"{OUT}/pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)}/{len(P['pairs'])} pairs measured; motion tertiles {t1:.2f} / {t2:.2f}")
    print("per type:", dict(Counter(r["type"] for r in rows)))
    print("indistinguishable per type:", dict(Counter(r["type"] for r in rows if r["indistinguishable"])))
    print("static:", sum(r["static"] for r in rows))
    json.dump(dict(motion_tertiles=[float(t1), float(t2)], n_pairs=len(rows)), open(f"{OUT}/pairs_meta.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "encode", "measure", "collect"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    dict(plan=plan, encode=encode, measure=measure, collect=collect)[a.cmd](a)


if __name__ == "__main__":
    main()
