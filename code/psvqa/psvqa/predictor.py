"""Causal latent-motion predictor, trained ONLY on clean (drop-free) video.

It learns what normal gameplay/general motion looks like in the frozen dense
feature space: given past per-frame latent motions m_<t, predict m_t. It never
sees a frame drop, so it cannot overfit to any drop protocol.

Usage:
  python -m psvqa.predictor --caches clean_caches/*.npz --out predictor.pt
"""
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn

from .common import dup_flags


class MotionPredictor(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.inp = nn.Linear(dim, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, m):                       # m: (B,L,D) -> preds (B,L,D)
        B, L, _ = m.shape
        h = m.new_zeros(B, self.cell.hidden_size)
        preds = []
        for t in range(L):
            h = self.cell(torch.relu(self.inp(m[:, t])), h)
            preds.append(self.out(h))
        return torch.stack(preds, 1)


class PredictorRunner:
    """Step-wise wrapper matching temporal.per_frame_features' interface."""

    def __init__(self, ckpt, device="cpu"):
        c = torch.load(ckpt, map_location=device, weights_only=False)
        self.mu, self.sd = c["mu"], c["sd"]
        self.net = MotionPredictor(len(self.mu), c["hidden"]).to(device).eval()
        self.net.load_state_dict(c["state"])
        self.device = device

    def normalize(self, z):
        return ((z.astype(np.float32) - self.mu) / self.sd).astype(np.float32)

    def init(self):
        return torch.zeros(1, self.net.cell.hidden_size, device=self.device)

    @torch.no_grad()
    def step(self, h, m):
        x = torch.from_numpy(m).to(self.device)[None]
        h = self.net.cell(torch.relu(self.net.inp(x)), h)
        return self.net.out(h)[0].cpu().numpy(), h


def unique_motion(z, thumbs):
    """Per-step latent motion over unique frames (genuine engine repeats in
    clean footage are skipped exactly as in the temporal branch)."""
    dup, _ = dup_flags(thumbs)
    keep = np.where(~dup)[0]
    zz = z[keep]
    gaps = np.diff(keep).astype(np.float32)[:, None]
    return (zz[1:] - zz[:-1]) / gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--out", default="predictor.pt")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    files = [f for p in a.caches for f in glob.glob(p)]

    zs = [np.load(f)["z"].astype(np.float32) for f in files]
    allz = np.concatenate(zs)
    mu, sd = allz.mean(0), allz.std(0) + 1e-4
    seqs = []
    for f, z in zip(files, zs):
        m = unique_motion((z - mu) / sd, np.load(f)["thumbs"])
        if len(m) > a.win + 1:
            seqs.append(m.astype(np.float32))
    print(f"{len(seqs)} clean sequences, dim={len(mu)}")

    net = MotionPredictor(len(mu), a.hidden).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = np.random.default_rng(0)
    for step in range(a.steps):
        batch = []
        for _ in range(a.bs):
            s = seqs[rng.integers(len(seqs))]
            i = rng.integers(0, len(s) - a.win - 1)
            batch.append(s[i:i + a.win + 1])
        x = torch.from_numpy(np.stack(batch)).to(a.device)
        pred = net(x[:, :-1])
        loss = ((pred - x[:, 1:]) ** 2).mean()
        # reference: constant-velocity error, to check the GRU actually helps
        cv = ((x[:, 1:-1] - x[:, 2:]) ** 2).mean()
        zero = (x[:, 1:] ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            print(f"step {step} mse {loss.item():.4f} const-vel {cv.item():.4f} zero {zero.item():.4f}")
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, hidden=a.hidden), a.out)


if __name__ == "__main__":
    main()
