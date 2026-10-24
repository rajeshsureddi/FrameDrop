"""Train the PSVQA head.

Losses
  rank_drop : same encode, pattern vs its superset      -> Q(base) > Q(superset)
  rank_rate : same pattern, higher vs lower bitrate      -> Q(hi) > Q(lo)
  vis_rank  : superset that removes clearly more motion -> V(superset) > V(base)
  vis_ord   : ordinal (CORAL) loss on a physical proxy level (weak, low weight)
  mos       : optional public-MOS clips (no drops): MSE on per-dataset z-scores
              + (1 - PLCC); anchors the absolute scale across content

  python -m psvqa.train_head --samples train_samples.pkl --out head.pt
"""
import argparse
import pickle
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as Fn

from .head import PSVQAHead, clip_tensors


class SpatCache:
    def __init__(self, cap=64):
        self.d, self.cap = OrderedDict(), cap

    def get(self, path):
        if path in self.d:
            self.d.move_to_end(path)
            return self.d[path]
        a = np.load(path)["spat"]
        self.d[path] = a
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return a


def fit_stats(samples):
    g = np.concatenate([s["gate"].astype(np.float32) for s in samples[::7]])
    t = np.concatenate([s["tstats"] for s in samples])
    return dict(g_mu=g.mean(0), g_sd=g.std(0) + 1e-4, t_mu=t.mean(0), t_sd=t.std(0) + 1e-4)


def vis_levels(samples):
    E = np.array([s["visE"] for s in samples if s["dataset"] == "synthetic"])
    pos = E[E > 1.0]
    q = np.percentile(pos, [33, 67]) if len(pos) else np.array([1.0, 2.0])
    for s in samples:
        e = s["visE"]
        s["vis"] = 0 if e <= 1.0 else 1 + int(e > q[0]) + int(e > q[1])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="head.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--w_vis_ord", type=float, default=0.2)
    ap.add_argument("--w_mos", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    samples = pickle.load(open(a.samples, "rb"))
    stats = fit_stats(samples)
    vis_q = vis_levels(samples)
    synth = [s for s in samples if s["dataset"] == "synthetic"]
    labeled = [s for s in samples if s["dataset"] != "synthetic"]
    for ds in {s["dataset"] for s in labeled}:
        m = np.array([s["mos"] for s in labeled if s["dataset"] == ds])
        for s in labeled:
            if s["dataset"] == ds:
                s["mos_z"] = (s["mos"] - m.mean()) / (m.std() + 1e-6)

    # index pairs
    key = {(s["cache"], s["pid"]): s for s in synth}
    drop_pairs = [(key[(s["cache"], s["parent"])], s) for s in synth
                  if s["parent"] >= 0 and (s["cache"], s["parent"]) in key]
    by_pat = defaultdict(list)
    for s in synth:
        by_pat[(s["src"], s["pid"])].append(s)
    rate_pairs = [(hi, lo) for grp in by_pat.values() for hi in grp for lo in grp
                  if hi["rank"] > lo["rank"]]
    print(f"{len(drop_pairs)} drop pairs, {len(rate_pairs)} bitrate pairs, "
          f"{len(labeled)} labeled clips, vis thresholds {vis_q}")

    dev = a.device
    net = PSVQAHead().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    cache = SpatCache()
    rng = np.random.default_rng(0)

    def run(s):
        sp, g, ts = clip_tensors(s, cache.get(s["cache"]), stats, dev)
        return net(sp, g, ts, s["seg"])

    for step in range(a.steps):
        loss, logs = 0.0, {}
        l_rank, l_vr, l_vo = [], [], []
        for _ in range(a.pairs):
            b, w = drop_pairs[rng.integers(len(drop_pairs))]
            (Qb, Vb, _), (Qw, Vw, _) = run(b), run(w)
            l_rank.append(Fn.relu(a.margin - (Qb - Qw)))
            if w["visE"] > 1.5 * b["visE"] + 1.0:
                l_vr.append(Fn.relu(a.margin - (Vw - Vb)))
            for s, V in ((b, Vb), (w, Vw)):
                tgt = torch.tensor([s["vis"] > k for k in range(3)], dtype=torch.float32, device=dev)
                l_vo.append(Fn.binary_cross_entropy_with_logits(net.vis_logits(V), tgt))
            if rate_pairs:
                hi, lo = rate_pairs[rng.integers(len(rate_pairs))]
                l_rank.append(Fn.relu(a.margin - (run(hi)[0] - run(lo)[0])))
        loss = torch.stack(l_rank).mean() + a.w_vis_ord * torch.stack(l_vo).mean()
        if l_vr:
            loss = loss + torch.stack(l_vr).mean()
        if labeled and a.w_mos > 0:
            bl = [labeled[i] for i in rng.choice(len(labeled), min(8, len(labeled)), replace=False)]
            Q = torch.stack([run(s)[0] for s in bl])
            y = torch.tensor([s["mos_z"] for s in bl], dtype=torch.float32, device=dev)
            qc, yc = Q - Q.mean(), y - y.mean()
            plcc = (qc * yc).sum() / (qc.norm() * yc.norm() + 1e-6)
            loss = loss + a.w_mos * (Fn.mse_loss(Q, y) + (1 - plcc))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f}")

    torch.save(dict(state=net.state_dict(), stats=stats, vis_q=vis_q), a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
