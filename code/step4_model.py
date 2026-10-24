"""Step 4: Stage 1 spatial encoder (Tile-MAE).

  input tile           B x 3 x 256 x 256
  patch tokens         B x 256 x 768 -> linear -> B x 256 x 384
  visible (75% mask)   B x 64 x 384
  encoder ViT-S/16     depth 12, 6 heads (timm key names, so public MAE
                       ViT-S/16 checkpoints load directly)
  decoder              dim 256, depth 4, 8 heads, mask tokens
  reconstruction head  B x 256 x (768 + 6)
  score head           mean pool -> MLP 384 -> 128 -> 1
  inference output     no mask, mean of the 256 patch tokens: 1 x 384 per tile

L_rec target per patch: the patch pixels normalized by their own mean/std
(768) plus the per-channel patch mean and std (6, in the normalized input
space), so brightness and contrast changes are not normalized away.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VARIANTS = dict(s=dict(dim=384, depth=12, heads=6), b=dict(dim=768, depth=12, heads=12),
                l=dict(dim=1024, depth=24, heads=16))


def sincos_2d(dim, n):
    """MAE fixed 2D sin-cos position embedding, (n*n, dim)."""
    g = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64))
    g = np.stack(g, 0).reshape(2, -1)                      # (w, h)
    om = 1.0 / 10000 ** (np.arange(dim // 4, dtype=np.float64) / (dim / 4.0))

    def one(p):
        o = np.einsum("m,d->md", p, om)
        return np.concatenate([np.sin(o), np.cos(o)], 1)
    return torch.from_numpy(np.concatenate([one(g[0]), one(g[1])], 1)).float()


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, heads, drop_path=0.0, layerscale=False):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), Mlp(dim, 4 * dim)
        self.ls1, self.ls2 = (LayerScale(dim), LayerScale(dim)) if layerscale else (nn.Identity(), nn.Identity())
        self.dp = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp(self.ls1(self.attn(self.norm1(x))))
        return x + self.dp(self.ls2(self.mlp(self.norm2(x))))


class TileMAE(nn.Module):
    def __init__(self, variant="s", img=256, patch=16, dec_dim=256, dec_depth=4, dec_heads=8,
                 drop_path=0.1, mask_ratio=0.75, layerscale=False):
        super().__init__()
        cfg = VARIANTS[variant]
        dim, depth, heads = cfg["dim"], cfg["depth"], cfg["heads"]
        self.img, self.p, self.g, self.dim, self.mask_ratio = img, patch, img // patch, dim, mask_ratio
        self.n = self.g * self.g
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.cat([torch.zeros(1, dim), sincos_2d(dim, self.g)])[None])
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, d, layerscale) for d in dpr])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # decoder
        self.decoder_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("decoder_pos_embed",
                             torch.cat([torch.zeros(1, dec_dim), sincos_2d(dec_dim, self.g)])[None],
                             persistent=False)
        self.decoder_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(dec_dim, patch * patch * 3 + 6)
        self.score_head = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self._init()

    def _init(self):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @classmethod
    def from_state_dict(cls, sd, **kw):
        """Build with the geometry stored in a state dict (patch size, grid, layer scale), e.g. the
        DINOv2 variant (patch 14 on a 252 crop); plain ViT-S/16 checkpoints give the default model."""
        dim = sd["patch_embed.proj.weight"].shape[0]
        patch = sd["patch_embed.proj.weight"].shape[-1]
        g = math.isqrt(sd["pos_embed"].shape[1] - 1)
        variant = next(k for k, v in VARIANTS.items() if v["dim"] == dim)
        return cls(variant, img=g * patch, patch=patch, layerscale="blocks.0.ls1.gamma" in sd, **kw)

    # -- input
    def normalize(self, x_uint8):
        """(B, 3, H, W) uint8 RGB -> normalized float; tiles larger than the model input
        (256 for the patch-14 / 252 model) are centre-cropped."""
        H, W = x_uint8.shape[-2:]
        if H != self.img or W != self.img:
            y, x = (H - self.img) // 2, (W - self.img) // 2
            x_uint8 = x_uint8[..., y:y + self.img, x:x + self.img]
        return (x_uint8.float().div_(255.0) - self.mean) / self.std

    def patchify(self, x):
        B, p, g = x.shape[0], self.p, self.g
        return x.reshape(B, 3, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * 3)

    def targets(self, xn):
        """xn normalized input -> (B, N, p*p*3 + 6)."""
        t = self.patchify(xn)                                        # (B, N, p*p*3), channel last
        m, v = t.mean(-1, keepdim=True), t.var(-1, keepdim=True)
        pix = (t - m) / (v + 1e-6).sqrt()
        tc = t.view(*t.shape[:2], -1, 3)
        stats = torch.cat([tc.mean(2), tc.std(2)], -1)              # (B, N, 6)
        return torch.cat([pix, stats], -1)

    # -- masking (one mask per pair: both tiles get the same mask)
    def random_masking_ids(self, B, device, generator=None):
        keep = int(round(self.n * (1 - self.mask_ratio)))
        noise = torch.rand(B, self.n, device=device, generator=generator)
        ids_shuffle = noise.argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        mask = torch.ones(B, self.n, device=device)
        mask[:, :keep] = 0
        return ids_shuffle[:, :keep], ids_restore, torch.gather(mask, 1, ids_restore)

    # -- encoder
    def encode(self, xn, ids_keep=None):
        x = self.patch_embed.proj(xn).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:]
        if ids_keep is not None:
            x = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, x.shape[-1]))
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], 1)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def pooled(self, lat):
        return lat[:, 1:].mean(1)

    def score(self, lat):
        return self.score_head(self.pooled(lat)).squeeze(-1)

    def decode(self, lat, ids_restore):
        x = self.decoder_embed(lat)
        B, n_vis = x.shape[0], x.shape[1] - 1
        mt = self.mask_token.expand(B, self.n - n_vis, -1)
        y = torch.cat([x[:, 1:], mt], 1)
        y = torch.gather(y, 1, ids_restore[..., None].expand(-1, -1, y.shape[-1]))
        x = torch.cat([x[:, :1], y], 1) + self.decoder_pos_embed
        for b in self.decoder_blocks:
            x = b(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:]

    def forward(self, xn, ids_keep, ids_restore):
        """Training forward (masked): latents, reconstruction, score."""
        lat = self.encode(xn, ids_keep)
        return lat, self.decode(lat, ids_restore), self.score(lat)

    @torch.no_grad()
    def features(self, x_uint8):
        """Inference output: (B, dim) per tile, no mask, mean of the patch tokens."""
        return self.pooled(self.encode(self.normalize(x_uint8)))


def load_timm_vit(model: TileMAE, path):
    """Load a timm-format ViT encoder state dict (e.g. MAE ViT-S/16 @224),
    interpolating the 14x14 position embedding to the 16x16 grid."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd)
    pe = sd["pos_embed"]
    n_old = int(math.isqrt(pe.shape[1] - 1))
    if n_old != model.g:
        grid = pe[:, 1:].reshape(1, n_old, n_old, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=(model.g, model.g), mode="bicubic", align_corners=False)
        sd["pos_embed"] = torch.cat([pe[:, :1], grid.permute(0, 2, 3, 1).reshape(1, model.g * model.g, -1)], 1)
    enc = {k: v for k, v in sd.items() if k.split(".")[0] in ("patch_embed", "cls_token", "pos_embed", "blocks", "norm")}
    missing, unexpected = model.load_state_dict(enc, strict=False)
    enc_missing = [k for k in missing if k.split(".")[0] in ("patch_embed", "cls_token", "pos_embed", "blocks", "norm")]
    if enc_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {enc_missing[:5]} unexpected {list(unexpected)[:5]}")
    return dict(loaded=len(enc), skipped=sorted(set(sd) - set(enc)), pos_interp=(n_old, model.g))


def load_dinov2_hf(model: TileMAE, repo="facebook/dinov2-small"):
    """Load a HF-format DINOv2 encoder (facebook/dinov2-*: patch 14, layer scale, 37x37 position
    table for 518 px) into a TileMAE built with patch=14, layerscale=True; the position table is
    interpolated to the model grid (18x18 for a 252 input). DINOv2's mask_token is not used."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    d = snapshot_download(repo, allow_patterns=["*.safetensors", "config.json"])
    sd = load_file(f"{d}/model.safetensors")
    pe = sd["embeddings.position_embeddings"]
    n_old = int(math.isqrt(pe.shape[1] - 1))
    grid = pe[:, 1:].reshape(1, n_old, n_old, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(model.g, model.g), mode="bicubic", align_corners=False)
    out = {"cls_token": sd["embeddings.cls_token"],
           "pos_embed": torch.cat([pe[:, :1], grid.permute(0, 2, 3, 1).reshape(1, model.g * model.g, -1)], 1),
           "patch_embed.proj.weight": sd["embeddings.patch_embeddings.projection.weight"],
           "patch_embed.proj.bias": sd["embeddings.patch_embeddings.projection.bias"],
           "norm.weight": sd["layernorm.weight"], "norm.bias": sd["layernorm.bias"]}
    i = 0
    while f"encoder.layer.{i}.norm1.weight" in sd:
        p, q = f"encoder.layer.{i}.", f"blocks.{i}."
        a = p + "attention.attention."
        out[q + "attn.qkv.weight"] = torch.cat([sd[a + n + ".weight"] for n in ("query", "key", "value")])
        out[q + "attn.qkv.bias"] = torch.cat([sd[a + n + ".bias"] for n in ("query", "key", "value")])
        for src, dst in (("attention.output.dense", "attn.proj"), ("norm1", "norm1"), ("norm2", "norm2"),
                         ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            out[q + dst + ".weight"], out[q + dst + ".bias"] = sd[p + src + ".weight"], sd[p + src + ".bias"]
        out[q + "ls1.gamma"], out[q + "ls2.gamma"] = sd[p + "layer_scale1.lambda1"], sd[p + "layer_scale2.lambda1"]
        i += 1
    missing, unexpected = model.load_state_dict(out, strict=False)
    enc_missing = [k for k in missing if k.split(".")[0] in ("patch_embed", "cls_token", "pos_embed", "blocks", "norm")]
    if enc_missing or unexpected:
        raise RuntimeError(f"dinov2 load: missing {enc_missing[:5]} unexpected {list(unexpected)[:5]}")
    return dict(loaded=len(out), layers=i, pos_interp=(n_old, model.g),
                skipped=sorted(k for k in sd if k.startswith("embeddings.mask_token")))
