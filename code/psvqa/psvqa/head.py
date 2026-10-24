"""PSVQA head: fuses spatial quality with perceptual-surprise temporal stats.

Spatial pooling is over DISPLAYED frames, so a frozen image contributes once
per tick it stays on screen, and a learned gate (freeze length, last real
motion) lets the model up-weight it when temporal masking disappears.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .common import ffill_index


class PSVQAHead(nn.Module):
    def __init__(self, spat_dim=4096, tdim=12, h=128):
        super().__init__()
        self.sproj = nn.Sequential(nn.LayerNorm(spat_dim), nn.Linear(spat_dim, h), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        self.tproj = nn.Sequential(nn.Linear(tdim, 64), nn.GELU())
        self.gru = nn.GRU(h + 64, 64, batch_first=True)
        self.q = nn.Linear(64, 1)
        self.vis = nn.Sequential(nn.Linear(tdim, 32), nn.GELU(), nn.Linear(32, 1))
        self.vis_b = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))
        self.lam = nn.Parameter(torch.tensor(0.0))

    def thresholds(self):
        b = self.vis_b
        return torch.cat([b[:1], b[:1] + torch.cumsum(Fn.softplus(b[1:]), 0)])

    def forward(self, spat, gate, tstats, seg):
        """One clip. spat (T,4096) displayed & forward-filled, gate (T,2),
        tstats (S,12) standardized. Returns Q (scalar), vis score, per-seg q."""
        T, S = spat.shape[0], tstats.shape[0]
        w = Fn.softplus(self.gate(gate)).squeeze(-1) + 1e-3        # (T,)
        f = self.sproj(spat)                                        # (T,h)
        seg_id = torch.clamp(torch.arange(T, device=spat.device) // seg, max=S - 1)
        num = torch.zeros(S, f.shape[1], device=f.device).index_add_(0, seg_id, f * w[:, None])
        den = torch.zeros(S, device=f.device).index_add_(0, seg_id, w)
        spat_seg = num / den[:, None]
        x = torch.cat([spat_seg, self.tproj(tstats)], -1)[None]
        q = self.q(self.gru(x)[0])[0, :, 0]                          # (S,)
        a = torch.sigmoid(self.lam)                                  # mean vs worst-part
        Q = a * q.mean() + (1 - a) * (torch.softmax(-q / 0.5, 0) * q).sum()
        V = self.vis(tstats)[:, 0].max()
        return Q, V, q

    def vis_logits(self, V):
        return V - self.thresholds()                                 # P(level > k)


def displayed_spatial(spat_src, idx, valid):
    """Spatial feature seen at each display tick (latest sampled frame)."""
    return spat_src[idx[ffill_index(valid)]]


def clip_tensors(sample, spat_src, stats, device):
    sp = displayed_spatial(spat_src, sample["idx"], sample["valid"]).astype(np.float32)
    g = (sample["gate"].astype(np.float32) - stats["g_mu"]) / stats["g_sd"]
    ts = (sample["tstats"] - stats["t_mu"]) / stats["t_sd"]
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
    return to(sp), to(g), to(ts)
