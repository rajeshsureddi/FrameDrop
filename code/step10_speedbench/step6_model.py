"""Step 6: Stage 3 fusion transformer and heads (both encoders frozen).

Window = 240 displayed frames (4 s); hop 60 (1 s).
  spatial tokens  16 tile frames x 32 tiles        512 x 384
  motion tokens   120 steps x 12 regions           1440 x 384
  project each to 256 + type emb + display-order pos + grid pos   1952 x 256
  fusion transformer depth 4, dim 256, 4 heads
  attention pooling, 2 learned queries -> Q head, V head (MLP 256->64->1)
Clip: Q = (1-a) * mean(q_w) + a * softmin_tau(q_w);  V = softmax_tau(v_w)
(a, tau learned; softmin/softmax_tau = weighted mean with weights softmax(-+x/tau)).
The only positional information: display order of each token's frame inside the
window, the tile's grid cell, the motion token's region (rule 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from step4_model import Block

WIN, HOP, TILE_EVERY, STEP = 240, 60, 15, 2
N_TF, N_STEPS = WIN // TILE_EVERY, WIN // STEP          # 16, 120
GRID = 9 * 15


class AttnPool(nn.Module):
    def __init__(self, dim, heads, n_q=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_q, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.q.expand(x.shape[0], -1, -1)
        return self.norm(self.attn(q, x, x, need_weights=False)[0])


def head(dim):
    return nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))


class Fusion(nn.Module):
    def __init__(self, in_dim=384, dim=256, depth=4, heads=4, drop_path=0.1):
        super().__init__()
        self.proj_s, self.proj_m = nn.Linear(in_dim, dim), nn.Linear(in_dim, dim)
        self.type_emb = nn.Parameter(torch.zeros(2, dim))
        self.frame_pos = nn.Embedding(WIN, dim)              # display order inside the window
        self.cell_pos = nn.Embedding(GRID, dim)              # tile grid cell (9 x 15)
        self.region_pos = nn.Embedding(12, dim)              # motion region (3 x 4)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, d) for d in dpr])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pool = AttnPool(dim, heads)
        self.q_head, self.v_head = head(dim), head(dim)
        self.a_logit = nn.Parameter(torch.zeros(()))         # a = sigmoid -> 0.5 at start
        self.tau_q = nn.Parameter(torch.zeros(()))           # tau = softplus(.) + 0.05
        self.tau_v = nn.Parameter(torch.zeros(()))
        for e in (self.frame_pos, self.cell_pos, self.region_pos):
            nn.init.normal_(e.weight, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)

    def window(self, spatial, cells, motion):
        """spatial (B, 16, 32, 384), cells (B, 16, 32, 2) long, motion (B, 120, 12, 384)
        -> q_w, v_w (B,)."""
        B, dev = spatial.shape[0], spatial.device
        tf = torch.arange(spatial.shape[1], device=dev) * TILE_EVERY    # shorter for clips < 240 frames
        s = self.proj_s(spatial) + self.type_emb[0] + self.frame_pos(tf)[None, :, None] \
            + self.cell_pos(cells[..., 0] * 15 + cells[..., 1])
        st = torch.arange(motion.shape[1], device=dev) * STEP
        m = self.proj_m(motion) + self.type_emb[1] + self.frame_pos(st)[None, :, None] \
            + self.region_pos(torch.arange(12, device=dev))[None, None]
        x = torch.cat([s.flatten(1, 2), m.flatten(1, 2)], 1)
        for b in self.blocks:
            x = b(x)
        p = self.pool(self.norm(x))
        self.last_pool = p.detach()                          # (B, 2, dim): Protocol A features
        return self.q_head(p[:, 0]).squeeze(-1), self.v_head(p[:, 1]).squeeze(-1)

    def clip(self, q_w, v_w, n_w):
        """q_w, v_w (C, W) window outputs; n_w (C,) number of valid windows."""
        valid = torch.arange(q_w.shape[1], device=q_w.device)[None] < n_w[:, None]
        tq = F.softplus(self.tau_q) + 0.05
        tv = F.softplus(self.tau_v) + 0.05
        neg = torch.finfo(q_w.dtype).min
        mean_q = (q_w * valid).sum(1) / valid.sum(1)
        wq = torch.softmax(torch.where(valid, -q_w / tq, neg), 1)
        a = torch.sigmoid(self.a_logit)
        Q = (1 - a) * mean_q + a * (wq * q_w).sum(1)
        wv = torch.softmax(torch.where(valid, v_w / tv, neg), 1)
        return Q, (wv * v_w).sum(1)

    def forward(self, spatial, cells, motion, n_w=None):
        """Clip-level. spatial (C, K, 32, 384) with K = T/15 tile frames, cells
        (C, K, 32, 2), motion (C, T/2, 12, 384). Windows start every HOP frames."""
        C, K = spatial.shape[:2]
        T = motion.shape[1] * STEP
        W = max(1, (T - WIN) // HOP + 1)             # a clip shorter than 4 s is one (short) window
        qs, vs, pools = [], [], []
        for w in range(W):
            a_tf, a_st = w * HOP // TILE_EVERY, w * HOP // STEP
            q, v = self.window(spatial[:, a_tf:a_tf + N_TF], cells[:, a_tf:a_tf + N_TF],
                               motion[:, a_st:a_st + N_STEPS])
            qs.append(q)
            vs.append(v)
            pools.append(self.last_pool.float())
        q_w, v_w = torch.stack(qs, 1), torch.stack(vs, 1)
        n_w = torch.full((C,), W, device=spatial.device) if n_w is None else n_w
        Q, V = self.clip(q_w.float(), v_w.float(), n_w)
        self.clip_pool = torch.stack(pools, 1).mean(1).flatten(1)  # (C, 2*dim) mean over windows
        return Q, V, q_w, v_w
