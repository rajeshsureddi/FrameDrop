"""Step 5: Stage 2 motion encoder (Video-MAE on low-res displayed frames).

  input window         B x 3 x 32 x 216 x 384 (32 consecutive displayed frames)
  tubelets 2x24x24     16 steps x (9 x 16) = 2304 tokens x 3456 values
  embed                linear 3456 -> 384 + spatial pos + display-order pos
  mask                 50% of time steps fully, then 50% of the remaining tokens
                       -> 576 visible
  encoder ViT-S        depth 12, 6 heads, joint space-time attention
  decoder              dim 192, depth 4, 6 heads
  reconstruction head  2304 x (3456 + 6)
  score head           mean pool -> MLP 384 -> 128 -> 1
  inference output     no mask; 9x16 grid pooled to 3x4 regions: 16 x 12 x 384

Initialization (plan): public VideoMAE ViT-S (MCG-NJU/videomae-small-finetuned-ssv2,
CC-BY-NC-4.0; only ViT-S VideoMAE weights on the HF hub, MAE-pretrained then
fine-tuned on SSv2). Its 2x16x16 patch embedding is resized to 2x24x24 with the
FlexiViT pseudo-inverse resize; position embeddings are fixed sin-cos, rebuilt
for the 16 x 9 x 16 grid. The checkpoint was trained with 6 heads (official
VideoMAE vit_small); the HF config's 16 heads is a conversion artefact.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from step4_model import Block, MEAN, STD

T_WIN, H, W = 32, 216, 384
TP, SP = 2, 24                                 # tubelet: 2 frames x 24 x 24
GT, GH, GW = T_WIN // TP, H // SP, W // SP     # 16 x 9 x 16


def sincos_1d(dim, pos):
    om = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    o = np.einsum("m,d->md", pos.astype(np.float64), om)
    return np.concatenate([np.sin(o), np.cos(o)], 1)


def sincos_3d(dim, gt, gh, gw):
    """(gt*gh*gw, dim): display-order (t) + spatial (h, w) sin-cos, t-major."""
    dt = dim // 4 // 2 * 2
    ds = (dim - dt) // 2 // 2 * 2
    dw = dim - dt - ds
    t, h, w = np.meshgrid(np.arange(gt), np.arange(gh), np.arange(gw), indexing="ij")
    e = np.concatenate([sincos_1d(dt, t.ravel()), sincos_1d(ds, h.ravel()), sincos_1d(dw, w.ravel())], 1)
    return torch.from_numpy(e).float()


class VideoMAE(nn.Module):
    def __init__(self, dim=384, depth=12, heads=6, dec_dim=192, dec_depth=4, dec_heads=6, drop_path=0.1,
                 step_mask=0.5, token_mask=0.5):
        super().__init__()
        self.dim, self.step_mask, self.token_mask = dim, step_mask, token_mask
        self.n = GT * GH * GW
        self.patch_embed = nn.Conv3d(3, dim, (TP, SP, SP), (TP, SP, SP))
        self.register_buffer("pos", sincos_3d(dim, GT, GH, GW)[None], persistent=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, d) for d in dpr])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.decoder_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_3d(dec_dim, GT, GH, GW)[None], persistent=False)
        self.decoder_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(dec_dim, TP * SP * SP * 3 + 6)
        self.score_head = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1, 1), persistent=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.mask_token, std=0.02)

    def normalize(self, x_uint8):
        """(B, T, H, W, 3) uint8 -> (B, 3, T, H, W) normalized float."""
        return (x_uint8.permute(0, 4, 1, 2, 3).float().div_(255.0) - self.mean) / self.std

    def tubeletify(self, x):
        """(B, 3, 32, 216, 384) -> (B, 2304, 3456): values ordered (t, y, x, c)."""
        B = x.shape[0]
        x = x.reshape(B, 3, GT, TP, GH, SP, GW, SP)
        return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, self.n, TP * SP * SP * 3)

    def targets(self, xn):
        t = self.tubeletify(xn)
        m, v = t.mean(-1, keepdim=True), t.var(-1, keepdim=True)
        pix = (t - m) / (v + 1e-6).sqrt()
        tc = t.view(*t.shape[:2], -1, 3)
        return torch.cat([pix, tc.mean(2), tc.std(2)], -1)

    def masking_ids(self, B, device, generator=None):
        """50% of time steps fully masked, then 50% of the remaining tokens.
        Returns ids_keep (B, 576), ids_restore (B, 2304), mask (B, 2304; 1 = masked),
        step_masked (B, 16)."""
        S = GH * GW
        n_steps_m = int(round(GT * self.step_mask))
        st = torch.rand(B, GT, device=device, generator=generator).argsort(1)
        step_masked = torch.zeros(B, GT, device=device, dtype=torch.bool)
        step_masked.scatter_(1, st[:, :n_steps_m], True)
        noise = torch.rand(B, GT, S, device=device, generator=generator)
        # rank tokens inside the unmasked steps; keep the lowest (1 - token_mask) fraction
        n_keep = int(round((GT - n_steps_m) * S * (1 - self.token_mask)))
        noise = noise + step_masked[..., None].float() * 2.0          # masked steps sort last
        ids_shuffle = noise.view(B, -1).argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        mask = torch.ones(B, self.n, device=device)
        mask[:, :n_keep] = 0
        return ids_shuffle[:, :n_keep], ids_restore, torch.gather(mask, 1, ids_restore), step_masked

    def encode(self, xn, ids_keep=None):
        x = self.patch_embed(xn).flatten(2).transpose(1, 2) + self.pos
        if ids_keep is not None:
            x = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, x.shape[-1]))
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def score(self, lat):
        return self.score_head(lat.mean(1)).squeeze(-1)

    def decode(self, lat, ids_restore):
        x = self.decoder_embed(lat)
        B = x.shape[0]
        y = torch.cat([x, self.mask_token.expand(B, self.n - x.shape[1], -1)], 1)
        y = torch.gather(y, 1, ids_restore[..., None].expand(-1, -1, y.shape[-1])) + self.dec_pos
        for b in self.decoder_blocks:
            y = b(y)
        return self.decoder_pred(self.decoder_norm(y))

    def forward(self, xn, ids_keep, ids_restore):
        lat = self.encode(xn, ids_keep)
        return lat, self.decode(lat, ids_restore), self.score(lat)

    @torch.no_grad()
    def features(self, x_uint8):
        """Inference output (B, 16, 12, dim): no mask, 9x16 grid pooled to 3x4 regions."""
        lat = self.encode(self.normalize(x_uint8))
        g = lat.view(lat.shape[0], GT, GH, GW, -1).permute(0, 1, 4, 2, 3).flatten(0, 1)   # (B*16, d, 9, 16)
        r = F.adaptive_avg_pool2d(g, (3, 4))
        return r.view(lat.shape[0], GT, -1, 12).permute(0, 1, 3, 2)


def pi_resize_patch(w, new_hw):
    """FlexiViT pseudo-inverse resize of a conv kernel's spatial dims.
    w: (out, in, t, h, w) -> (out, in, t, H, W)."""
    o, i, t, h, ww = w.shape
    Hn, Wn = new_hw
    basis = torch.eye(h * ww, dtype=torch.float64).view(h * ww, 1, h, ww)
    up = F.interpolate(basis, size=(Hn, Wn), mode="bilinear", align_corners=False)   # (h*w, 1, H, W)
    Rm = up.view(h * ww, Hn * Wn).T                                                   # (H*W, h*w)
    pinv = torch.linalg.pinv(Rm.T)                                                    # (H*W, h*w)
    flat = w.double().reshape(o * i * t, h * ww).T                                    # (h*w, N)
    return (pinv @ flat).T.reshape(o, i, t, Hn, Wn).to(w.dtype)


def load_videomae_hf(model: VideoMAE, path):
    sd = torch.load(path, map_location="cpu")
    out = {}
    pw = sd["videomae.embeddings.patch_embeddings.projection.weight"]
    out["patch_embed.weight"] = pi_resize_patch(pw, (SP, SP))
    out["patch_embed.bias"] = sd["videomae.embeddings.patch_embeddings.projection.bias"]
    i = 0
    while f"videomae.encoder.layer.{i}.attention.attention.query.weight" in sd:
        p = f"videomae.encoder.layer.{i}."
        q, k, v = (sd[p + f"attention.attention.{n}.weight"] for n in ("query", "key", "value"))
        out[f"blocks.{i}.attn.qkv.weight"] = torch.cat([q, k, v], 0)
        qb, vb = sd[p + "attention.attention.q_bias"], sd[p + "attention.attention.v_bias"]
        out[f"blocks.{i}.attn.qkv.bias"] = torch.cat([qb, torch.zeros_like(qb), vb])
        out[f"blocks.{i}.attn.proj.weight"] = sd[p + "attention.output.dense.weight"]
        out[f"blocks.{i}.attn.proj.bias"] = sd[p + "attention.output.dense.bias"]
        out[f"blocks.{i}.mlp.fc1.weight"] = sd[p + "intermediate.dense.weight"]
        out[f"blocks.{i}.mlp.fc1.bias"] = sd[p + "intermediate.dense.bias"]
        out[f"blocks.{i}.mlp.fc2.weight"] = sd[p + "output.dense.weight"]
        out[f"blocks.{i}.mlp.fc2.bias"] = sd[p + "output.dense.bias"]
        for a, b in (("norm1", "layernorm_before"), ("norm2", "layernorm_after")):
            out[f"blocks.{i}.{a}.weight"] = sd[p + f"{b}.weight"]
            out[f"blocks.{i}.{a}.bias"] = sd[p + f"{b}.bias"]
        i += 1
    if "fc_norm.weight" in sd:                         # mean-pooling checkpoint: its fc_norm is the final norm
        out["norm.weight"], out["norm.bias"] = sd["fc_norm.weight"], sd["fc_norm.bias"]
    missing, unexpected = model.load_state_dict(out, strict=False)
    enc_missing = [k for k in missing if k.split(".")[0] in ("patch_embed", "blocks", "norm")]
    if enc_missing or unexpected or i != len(model.blocks):
        raise RuntimeError(f"videomae load: layers {i}, missing {enc_missing[:5]}, unexpected {unexpected[:5]}")
    return dict(loaded=len(out), layers=i, patch_resize="2x16x16 -> 2x24x24 (pinv)", pos="fixed sincos (rebuilt)")
