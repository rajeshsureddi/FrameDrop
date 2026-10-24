"""Step 10a/10b: score models on DropDev (no study data).

  extract  tokens of every DropDev clip for one encoder pair (s1, s2). A base
           encode is decoded once; its raw clips are display index remaps of
           that decode, re-encoded clips are decoded from their own files. All
           clips share the same tile positions (fixed seed, depends only on length), tiles
           every 15th displayed frame, as in step7_eval.
  score    fusion -> Q, V per clip (same windowing as step7_eval.score)
  metrics  per pair: Q correct if Q(better) > Q(worse), V correct if
           V(worse) > V(better) (ties count 0.5).

Fixed definition (10b, set before any candidate is trained): "overall" =
unweighted mean of the four per-type accuracies over the distinguishable pairs
(display pixel difference >= 0.5 gray levels). Indistinguishable and static
pairs are reported separately and do not enter "overall".

    python step10_eval.py extract --name v1enc --s1 ... --s2 ... --gpu 0 --part 0 --parts 3
    python step10_eval.py score --name v2 --tok v1enc --fusion ...
    python step10_eval.py metrics --name v2
    python step10_eval.py average --name v23avg --of v2 v3     (10c-5: mean of Q and of V)
    python step10_eval.py compare --of v2 v3

Gate (fixed 10-18, before any DropDev number was seen): DropDev reproduces the
study's v2 > v3 gap if v2's overall Q-ordering exceeds v3's and the 95% paired
bootstrap interval of the difference (2,000 resamples of audit sources) excludes 0.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "psvqa"))
import step3_video_loader as vl  # noqa: E402
import step6_features as F6  # noqa: E402
import step6_model as S3  # noqa: E402
import step2_common as sc  # noqa: E402
from psvqa import drops  # noqa: E402
from step10_dropdev import OUT, load_plan, mask_from, in_shard  # noqa: E402

TYPES = ("burst_length", "burst_count", "density", "none_vs_some")


class BaseClips(IterableDataset):
    def __init__(self, bases, done):
        self.bases, self.done = bases, done

    def __iter__(self):
        wi = get_worker_info()
        mine = self.bases[wi.id::wi.num_workers] if wi else self.bases
        for base_id, clips in mine:
            clips = [c for c in clips if c["clip"] not in self.done]
            if not clips:
                continue
            T = clips[0]["T"]
            tticks = np.arange(0, T, F6.TILE_EVERY)
            rng = np.random.default_rng(CELL_SEED)   # fixed seed, never derived from a name/id
            cells = vl.sample_cells(rng, len(tticks))
            raw = [c for c in clips if c["crf"] is None]
            try:
                if raw:
                    idxs = {c["clip"]: drops.drops_to_index(mask_from(T, c["drop"])) for c in raw}
                    need = sorted({int(i[t]) for i in idxs.values() for t in tticks})
                    v = dict(path=raw[0]["base"], w=raw[0]["w"], h=raw[0]["h"])
                    low, full = vl.decode_window(v, 0, T, need, threads=2)
                    for c in raw:
                        view = vl.build_view(low, 0, full, np.arange(T), idxs[c["clip"]], cells, tticks, True)
                        yield c["clip"], view["low"], view["tiles"], cells, ""
                    del low, full
                for c in clips:
                    if c["crf"] is None:
                        continue
                    p = f"{OUT}/clips/{c['clip']}.mp4"
                    v = dict(path=p, w=c["w"], h=c["h"])
                    low, full = vl.decode_window(v, 0, T, list(tticks), threads=2)
                    view = vl.build_view(low, 0, full, np.arange(T), np.arange(T), cells, tticks, True)
                    yield c["clip"], view["low"], view["tiles"], cells, ""
            except Exception as e:                       # noqa: BLE001
                for c in clips:
                    yield c["clip"], None, None, None, f"{type(e).__name__}: {str(e)[-300:]}"


CELL_SEED = 2026                                     # = step7_eval.CELL_SEED


def tok_dir(name):
    return f"{OUT}/tokens/{name}"


def extract(a):
    dev = torch.device("cuda", a.gpu)
    e1, e2 = F6.load_encoders(a.s1, a.s2, dev)
    od = tok_dir(a.name)
    os.makedirs(od, exist_ok=True)
    json.dump(dict(s1=a.s1, s2=a.s2), open(f"{od}/../{a.name}.encoders.json", "w"))
    P = load_plan()
    by = defaultdict(list)
    for c in P["clips"]:
        by[c["base_id"]].append(c)
    bases = sorted((b, cs) for b, cs in by.items() if in_shard(b, (a.part, a.parts))
                   and (a.only_base is None or b == a.only_base))
    done = {os.path.basename(f)[:-4] for f in glob.glob(f"{od}/*.npz")}
    if a.skip_missing:                               # early start: only clips whose re-encode is finished
        def ready(c):
            try:
                return c["crf"] is None or json.load(open(f"{OUT}/clips/{c['clip']}.json"))["ok"]
            except (OSError, ValueError, KeyError):
                return False
        bases = [(b, [c for c in cs if ready(c)]) for b, cs in bases]
    n_todo = sum(c["clip"] not in done for _, cs in bases for c in cs)
    print(f"part {a.part}/{a.parts}: {len(bases)} bases, {n_todo} clips to extract", flush=True)
    dl = DataLoader(BaseClips(bases, done), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    t0, k = time.time(), 0
    for cid, low, tiles, cells, err in dl:
        if low is None:
            open(f"{od}/{cid}.err", "w").write(err)
            print("ERR", cid, err[-200:], flush=True)
            continue
        low = np.asarray(low)
        T = (low.shape[0] // 2) * 2
        sp, mo = F6.encode_view(e1, e2, low[:T], tiles, dev)
        tmp = f"{od}/.{cid}.tmp.npz"
        np.savez(tmp, spatial=sp, motion=mo, cells=np.asarray(cells).astype(np.int8), frames=int(low.shape[0]))
        os.replace(tmp, f"{od}/{cid}.npz")
        k += 1
        if k % 200 == 0:
            print(f"{k}/{n_todo} {k / (time.time() - t0):.2f} clips/s", flush=True)
    print("FINISHED extract", flush=True)


class TokenFiles(torch.utils.data.Dataset):
    """token .npz -> (path, spatial[:K], cells[:K] int64, motion), K = tile steps covered by the motion tokens."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as z:
            sp, ce, mo = z["spatial"], z["cells"], z["motion"]
        K = min(sp.shape[0], -(-mo.shape[0] * 2 // F6.TILE_EVERY))
        return self.files[i], sp[:K], ce[:K].astype(np.int64), mo


@torch.no_grad()
def score(a):
    dev = torch.device("cuda", a.gpu)
    sd = torch.load(a.fusion, map_location="cpu")["model"]
    m = S3.Fusion(n_q=S3.n_q_of(sd)).to(dev).eval()                # n_q queries per head (Step 20); 1 = original
    m.load_state_dict(sd)
    rows = []
    files = sorted(glob.glob(f"{tok_dir(a.tok)}/*.npz"))
    dl = DataLoader(TokenFiles(files), batch_size=None, num_workers=a.workers, prefetch_factor=8)
    for f, sp, ce, mo in dl:                         # each file read once, in parallel workers, in sorted order
        sp, ce, mo = (torch.as_tensor(x)[None].to(dev) for x in (sp, ce, mo))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Q, V, _, _ = m(sp.float(), ce, mo.float())
        rows.append(dict(clip=os.path.basename(f)[:-4], Q=float(Q), V=float(V)))
    od = f"{OUT}/eval/{a.name}"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/scores.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(tok=a.tok, fusion=a.fusion), open(f"{od}/model.json", "w"))
    print(f"scored {len(rows)} clips -> {od}/scores.csv", flush=True)


def load_scores(name):
    return {r["clip"]: (float(r["Q"]), float(r["V"])) for r in csv.DictReader(open(f"{OUT}/eval/{name}/scores.csv"))}


def average(a):
    S = [load_scores(n) for n in a.of]
    keys = set.intersection(*(set(s) for s in S))
    od = f"{OUT}/eval/{a.name}"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/scores.csv", "w") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "Q", "V"])
        for k in sorted(keys):
            w.writerow([k, np.mean([s[k][0] for s in S]), np.mean([s[k][1] for s in S])])
    json.dump(dict(average_of=a.of), open(f"{od}/model.json", "w"))


def _acc(x):
    return dict(n=len(x), acc=float(np.mean(x)) if len(x) else None)


def metrics(a):
    S = load_scores(a.name)
    pairs = list(csv.DictReader(open(f"{OUT}/pairs.csv")))
    rec, miss = [], 0
    for p in pairs:
        if p["better"] not in S or p["worse"] not in S:
            miss += 1
            continue
        (qb, vb), (qw, vw) = S[p["better"]], S[p["worse"]]
        rec.append(dict(p, q=1.0 if qb > qw else 0.5 if qb == qw else 0.0,
                        v=1.0 if vw > vb else 0.5 if vw == vb else 0.0,
                        dist=p["indistinguishable"] != "True", static=p["static"] == "True"))

    def table(sel):
        out = {}
        for key in ("q", "v"):
            out[key] = {t: _acc([r[key] for r in sel if r["type"] == t]) for t in TYPES}
            out[key]["all"] = _acc([r[key] for r in sel])
        return out

    main = [r for r in rec if r["dist"] and not r["static"]]
    res = dict(model=a.name, pairs_scored=len(rec), pairs_missing=miss, n_main=len(main))
    res["main"] = table(main)
    for key in ("q", "v"):
        accs = [res["main"][key][t]["acc"] for t in TYPES]
        res[f"overall_{key.upper()}"] = float(np.mean(accs)) if None not in accs else None
    res["by_motion"] = {lv: table([r for r in main if r["motion_level"] == lv]) for lv in ("low", "mid", "high")}
    res["by_mode"] = {md: table([r for r in main if r["mode"] == md]) for md in ("raw", "reenc")}
    res["by_rank"] = {rk: table([r for r in main if r["rank"] == rk]) for rk in ("4", "2", "0")}
    res["static"] = table([r for r in rec if r["static"]])
    res["indistinguishable"] = table([r for r in rec if not r["dist"]])
    bl = defaultdict(list)                           # burst-length detail: by the longer burst L2
    for r in main:
        if r["type"] == "burst_length":
            bl[json.loads(r["info"])["L2"]].append(r)
    res["burst_length_by_L2"] = {str(k): dict(q=_acc([r["q"] for r in v]), v=_acc([r["v"] for r in v]))
                                 for k, v in sorted(bl.items())}
    od = f"{OUT}/eval/{a.name}"
    json.dump(res, open(f"{od}/dropdev_metrics.json", "w"), indent=1)
    print(f"{a.name}: overall Q {res['overall_Q']:.4f}  V {res['overall_V']:.4f}  (main n={len(main)}, missing {miss})")
    for key in ("q", "v"):
        print("  ", key, {t: round(res["main"][key][t]["acc"], 4) for t in TYPES})


def per_pair(name):
    S = load_scores(name)
    out = {}
    for p in csv.DictReader(open(f"{OUT}/pairs.csv")):
        if p["indistinguishable"] == "True" or p["static"] == "True" or p["better"] not in S or p["worse"] not in S:
            continue
        (qb, vb), (qw, vw) = S[p["better"]], S[p["worse"]]
        out[p["pair"]] = (p["source_id"], p["type"], float(qb > qw) + 0.5 * (qb == qw), float(vw > vb) + 0.5 * (vw == vb))
    return out


def compare(a):
    """paired bootstrap over sources of overall (mean of per-type) ordering: of[0] - of[1]"""
    A, B = per_pair(a.of[0]), per_pair(a.of[1])
    keys = sorted(set(A) & set(B))
    src = sorted({A[k][0] for k in keys})
    sidx = {s: i for i, s in enumerate(src)}
    ti = {t: i for i, t in enumerate(TYPES)}
    # per source x type sums and counts, for both models and both heads
    sums = np.zeros((2, 2, len(src), len(TYPES)))
    cnt = np.zeros((len(src), len(TYPES)))
    for k in keys:
        i, j = sidx[A[k][0]], ti[A[k][1]]
        cnt[i, j] += 1
        for m, D in enumerate((A, B)):
            sums[m, 0, i, j] += D[k][2]
            sums[m, 1, i, j] += D[k][3]

    def overall(w):
        c = (w[:, None] * cnt).sum(0)
        s = (w[None, None, :, None] * sums).sum(2)            # (model, head, type)
        return (s / c).mean(-1)                                # (model, head)

    rng = np.random.default_rng(0)
    base = overall(np.ones(len(src)))
    boots = np.array([overall(np.bincount(rng.integers(0, len(src), len(src)), minlength=len(src)).astype(float))
                      for _ in range(2000)])
    res = dict(models=a.of, n_pairs=len(keys), n_sources=len(src))
    for h, name in enumerate(("Q", "V")):
        d = boots[:, 0, h] - boots[:, 1, h]
        res[name] = dict(a=float(base[0, h]), b=float(base[1, h]), diff=float(base[0, h] - base[1, h]),
                         ci95=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))])
    res["gap_reproduced_Q"] = bool(res["Q"]["diff"] > 0 and res["Q"]["ci95"][0] > 0)
    os.makedirs(f"{OUT}/eval", exist_ok=True)
    json.dump(res, open(f"{OUT}/eval/compare_{a.of[0]}_vs_{a.of[1]}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "score", "metrics", "average", "compare"])
    ap.add_argument("--name")
    ap.add_argument("--tok", default=None)
    ap.add_argument("--s1")
    ap.add_argument("--s2")
    ap.add_argument("--fusion")
    ap.add_argument("--of", nargs="+")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only-base", default=None, help="smoke test: one base id")
    ap.add_argument("--skip-missing", action="store_true", help="extract: skip re-encodes not finished yet")
    a = ap.parse_args()
    dict(extract=extract, score=score, metrics=metrics, average=average, compare=compare)[a.cmd](a)


if __name__ == "__main__":
    main()
