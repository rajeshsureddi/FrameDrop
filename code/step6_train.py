"""Step 6 (part 2): train Stage 3 (fusion + Q/V heads) on frozen-encoder tokens.

Losses (METHOD_PLAN Step 6, with the user's override: public MOS is
evaluation-only, so there is no L_mos and checkpoints are selected on
valid-split ranking accuracy only):
  L_rank  on Q, margin 0.1, three pair types in equal proportions
          (more drops, lower bitrate, higher capture severity)
  L_drop  on V, margin 0.1: V(superset(P)) > V(P), including P = none
  L_inv   on V, Huber(V_a - V_b) for rate / severity pairs (same drops), weight 0.1
AdamW lr 3e-4, cosine, 5% warmup, batch 32 clips (16 pairs), 30k steps.
Batches: equal thirds per pair type, <= 2 pairs per source.

    python step6_train.py --name fusion
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as S3  # noqa: E402
from step4_train import lr_at, param_groups  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
OUT = f"{ROOT}/step6"
MARGIN = 0.1
TYPES = ("drops", "rate", "severity")


def load_shard(p):
    z = np.load(p)
    return dict(spatial=z["spatial"], motion=z["motion"], cells=z["cells"].astype(np.int64),
                meta=json.loads(str(z["meta"])))


class ShardPool:
    """Keeps `k` shards in memory, rotates one shard every `rotate` batches."""

    def __init__(self, files, k, seed):
        self.files, self.k, self.rng = list(files), k, random.Random(seed)
        self.rng.shuffle(self.files)
        self.next = 0
        self.loaded = [self._load() for _ in range(min(k, len(self.files)))]

    def _load(self):
        p = self.files[self.next % len(self.files)]
        self.next += 1
        if self.next % len(self.files) == 0:
            self.rng.shuffle(self.files)
        return load_shard(p)

    def rotate(self):
        if len(self.files) > self.k:
            self.loaded[self.rng.randrange(len(self.loaded))] = self._load()

    def batch(self, n_pairs):
        """equal thirds per type, <= 2 pairs per source."""
        idx = [(si, i) for si, sh in enumerate(self.loaded) for i in range(len(sh["meta"]))]
        self.rng.shuffle(idx)
        want = {t: n_pairs // 3 + (k < n_pairs % 3) for k, t in enumerate(TYPES)}
        cnt, out = Counter(), []
        for si, i in idx:
            m = self.loaded[si]["meta"][i]
            if want[m["ptype"]] > 0 and cnt[m["source_id"]] < 2:
                want[m["ptype"]] -= 1
                cnt[m["source_id"]] += 1
                out.append((si, i))
            if len(out) == n_pairs:
                break
        return self.collate(out)

    def collate(self, sel):
        g = lambda k: np.stack([self.loaded[si][k][i] for si, i in sel])  # noqa: E731
        return dict(spatial=torch.from_numpy(g("spatial")), motion=torch.from_numpy(g("motion")),
                    cells=torch.from_numpy(g("cells")), ptype=[self.loaded[si]["meta"][i]["ptype"] for si, i in sel])


def forward_pairs(model, b, dev):
    """b: spatial (P, 2, K, 32, 384) ... -> Q, V (P, 2)"""
    P = b["spatial"].shape[0]
    sp = b["spatial"].to(dev, non_blocking=True).flatten(0, 1).float()
    mo = b["motion"].to(dev, non_blocking=True).flatten(0, 1).float()
    ce = b["cells"].to(dev, non_blocking=True)[:, None].expand(-1, 2, -1, -1, -1).flatten(0, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Q, V, _, _ = model(sp, ce, mo)
    return Q.float().view(P, 2), V.float().view(P, 2)


def losses(Q, V, ptype, dev, w_inv=0.1):
    t = np.array(ptype)
    dr = torch.from_numpy(t == "drops").to(dev)
    rs = ~dr
    l_rank = F.relu(MARGIN - (Q[:, 0] - Q[:, 1])).mean()
    l_drop = F.relu(MARGIN - (V[dr, 1] - V[dr, 0])).mean() if dr.any() else Q.sum() * 0
    l_inv = F.huber_loss(V[rs, 0], V[rs, 1]) if rs.any() else Q.sum() * 0
    return l_rank + l_drop + w_inv * l_inv, dict(l_rank=float(l_rank), l_drop=float(l_drop), l_inv=float(l_inv))


@torch.no_grad()
def evaluate(model, val, dev, bs=32):
    model.eval()
    Qs, Vs, types = [], [], []
    for sh in val:
        n = len(sh["meta"])
        for i in range(0, n, bs):
            b = dict(spatial=torch.from_numpy(sh["spatial"][i:i + bs]), motion=torch.from_numpy(sh["motion"][i:i + bs]),
                     cells=torch.from_numpy(sh["cells"][i:i + bs]))
            Q, V = forward_pairs(model, b, dev)
            Qs.append(Q.cpu())
            Vs.append(V.cpu())
        types += [m["ptype"] for m in sh["meta"]]
    model.train()
    Q, V, t = torch.cat(Qs).numpy(), torch.cat(Vs).numpy(), np.array(types)
    r = {}
    for k in TYPES:
        m = t == k
        if m.any():
            r[f"q_acc_{k}"] = float((Q[m, 0] > Q[m, 1]).mean())
    dm = t == "drops"
    r["v_acc_drops"] = float((V[dm, 1] > V[dm, 0]).mean()) if dm.any() else None
    r["v_absdiff_rate_sev"] = float(np.abs(V[~dm, 0] - V[~dm, 1]).mean()) if (~dm).any() else None
    r["q_acc_mean"] = float(np.mean([r[f"q_acc_{k}"] for k in TYPES if f"q_acc_{k}" in r]))
    r["select"] = (r["q_acc_mean"] + (r["v_acc_drops"] or 0)) / 2         # checkpoint selection metric
    r["n"] = int(len(t))
    return r


def train(a):
    dev = torch.device("cuda", a.gpu)
    torch.manual_seed(a.seed)
    run_dir = f"{a.root}/runs/{a.name}"
    os.makedirs(run_dir, exist_ok=True)
    files = sorted(f for f in glob.glob(f"{a.root}/train/pairs_*.npz") if ".tmp" not in f)
    val = [load_shard(p) for p in sorted(glob.glob(f"{a.root}/valid/pairs_0*.npz")) if ".tmp" not in p]
    print(f"{len(files)} train shards, {sum(len(v['meta']) for v in val)} valid pairs", flush=True)
    pool = ShardPool(files, a.pool, a.seed)
    model = S3.Fusion(drop_path=a.drop_path).to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.999))
    warm = int(0.05 * a.steps)
    log = open(f"{run_dir}/train_log.jsonl", "a")
    json.dump(dict(args=vars(a), train_shards=len(files), params=sum(p.numel() for p in model.parameters())),
              open(f"{run_dir}/config.json", "w"), indent=1)
    best, acc, n, t0 = -1, defaultdict(float), 0, time.time()
    for step in range(1, a.steps + 1):
        b = pool.batch(a.pairs)
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, a.steps, a.lr, warm)
        Q, V = forward_pairs(model, b, dev)
        loss, parts = losses(Q, V, b["ptype"], dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in parts.items():
            acc[k] += v
        acc["q_acc"] += float((Q[:, 0] > Q[:, 1]).float().mean())
        n += 1
        if step % a.rotate == 0:
            pool.rotate()
        if step % a.log_every == 0:
            rec = dict(step=step, lr=opt.param_groups[0]["lr"], **{k: v / n for k, v in acc.items()},
                       a=float(torch.sigmoid(model.a_logit)), steps_per_s=n / (time.time() - t0),
                       time=time.strftime("%m-%d %H:%M:%S"))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            acc, n, t0 = defaultdict(float), 0, time.time()
        if step % a.val_every == 0 or step == a.steps:
            r = evaluate(model, val, dev)
            r.update(step=step, kind="val")
            if r["select"] > best:
                best = r["select"]
                torch.save(dict(model=model.state_dict(), step=step, val=r), f"{run_dir}/best.pt")
                r["best"] = True
            torch.save(dict(model=model.state_dict(), step=step), f"{run_dir}/last.pt")
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(json.dumps(r), flush=True)
    print("FINISHED best", best, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fusion")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--pairs", type=int, default=16, help="16 pairs = 32 clips")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--rotate", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=OUT, help="token root with train/ valid/ (runs/ are written under it)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
