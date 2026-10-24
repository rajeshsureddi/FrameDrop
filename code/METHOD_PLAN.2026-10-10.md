# Method plan: pixels-only real-time 4K VQA (supersedes earlier plans)

This is the full plan from data to evaluation. Steps 1–2 are in `AGENT_BRIEF_steps1-2.md` and are unchanged, except for the addendum (2f) below. Do each step in order and stop for human review where marked.

## What the model is

A no-reference model that watches a 4K60 video and outputs, for every 4 s window (1 s hop) and for the whole clip:
- **Q**: overall quality (higher is better)
- **V**: frame-drop visibility (higher is more visible)

It has three learned parts, trained in order, each frozen before the next:

| Stage | Module | Sees | Purpose |
|---|---|---|---|
| 1 | Spatial encoder (Tile-MAE) | native 256×256 tiles of a few frames per second | compression and capture artifacts at full 4K detail |
| 2 | Motion encoder (Video-MAE) | every displayed frame at 384×216 | motion, freezes, judder, flicker |
| 3 | Fusion transformer + heads | tokens from stages 1 and 2 | combine them into Q and V |

## Hard rules (in addition to those in the Step 1–2 brief)

1. **Pixels only.** The model's only input is decoded pixels in display order. Never feed timestamps/PTS, duplicate-frame flags, dwell counts, frame-difference statistics, codec or bitstream data, or the distortion parameters. The only positional information allowed is each frame's index within the window (display order) and each tile's position in the frame.
2. **Labels are pair order only.** "Version A is worse than version B" holds when A has a superset of B's drops, a lower bitrate in the same ladder, or a higher capture severity of the same plan. The only absolute labels are MOS from *public* datasets (Step 6). Never use the injected parameters as regression targets.
3. **The study is test-only.** No study video, score or visibility rating is used for training, tuning, early stopping or checkpoint selection. Select checkpoints on the validation split of the training data (ranking accuracy) and on public-MOS validation folds.
4. **Display-referenced input.** Every video, training or public-MOS, is first converted to what a 3840×2160 60 Hz display shows: spatially upscaled with Lanczos if smaller, and lower frame rates shown by frame repetition (30 fps → each frame twice; 24 fps → 3:2 pattern; 50 fps → closest-in-time frame). Never interpolate frames.
5. **Batch diversity.** Every batch draws its samples from many different sources (at most 2 samples per source per batch). Log the mean pairwise cosine similarity of frozen DINOv2 features within batches.

---

## Ground-truth order for L_rank (all stages)

A pair (better, worse) is valid only when the two versions are **identical except along one axis**, and the worse one is degraded further along it. The model never sees the axis or its value; only the order is used, in a margin loss `max(0, m − (f(better) − f(worse)))` where f is the score.

| Pair type | Better | Worse | Everything else identical |
|---|---|---|---|
| Drops | drop pattern P (or none) | superset(P): every drop in P plus more | source, encode, window, tiles |
| Bitrate | ladder rank r | rank r' < r | source, encoder, GOP, segment, drop pattern, tiles |
| Capture | `plan.with_severity(s)` | `plan.with_severity(s')`, s' > s | source, plan parameters/seeds, encoder, bitrate, drops, tiles |

Never rank across axes (drops vs bitrate, or two different capture plans): there is no ground truth for that. Only `L_mos` on public data ties the axes together.

**Validity filters** ("more degraded" is not always "visibly worse"):
1. **Minimum gap:** bitrate ratio ≥ 1.5×; severity gap ≥ 0.25; drops: the superset adds at least 3 dropped frames or one burst of ≥ 2.
2. **Indistinguishable pairs are skipped.** If the two versions' displayed pixels differ by less than a threshold (mean |A − B| on the 384×216 frames < 0.5 gray levels, e.g. extra drops in a static scene or two bitrates both at saturation), drop the pair. This difference is computed only for the label; it is never a model input.
3. **Ops that can look better at low strength** (`halo`, `saturation` with k > 0, `low_contrast`, `white_balance`, `hue_shift`): rank them only when the worse severity is ≥ 0.6 and the gap is ≥ 0.5.
4. **Margin:** m = 0.1. Log the fraction of pairs violated per type. A type that stays above 30% violated late in training is usually a bad pair definition, not a model failure.
5. **Same view:** both members of a pair use the same tile positions, same window, same frames and the same MAE mask.

---

## Step 2 addendum: 2f. Post-hold re-encoded subset (offline)

Real files are often re-encoded after frames were dropped, so held frames become *near*-identical rather than exact. On **every** kept source (2 variants each): take a clean-ladder encode, apply a random drop pattern (from `psvqa/drops.py`, the broad family), and re-encode with a CRF **sampled uniformly per variant**: libx264/libx265 in [18, 51] (their maximum), libsvtav1 in [18, 60] (its scale reaches 63). Use the source's encoder family. Store it with a JSON of the exact display index map and the CRF. These clips appear in Stage 2 and 3 training alongside on-the-fly drops, and also form bitrate-style ranking pairs (same drops, lower CRF is better) when the CRF gap is ≥ 6. No storage cap.

---

## Step 3: Tile store and data loaders

**3a. Tile store (for Stage 1).** For every clean source segment and every ladder or capture encode, decode sequentially. Every 6th frame (10 per second), cut 16 tiles at random positions on the absolute 4K grid. The **same positions** are used across all encodes of one source, so ladder pairs share content. Store each tile as a 384×384 uint8 crop (256 core + 64 px margin per side) in WebDataset shards, with keys: source_id, encode_id, rank, plan_id, severity, frame index, x, y. At about 0.44 MB per tile, expect several TB; there is no cap, but sample sources evenly so no source dominates.

**3b. Tile loader (Stage 1).**
- With probability 0.5, apply on-the-fly ops from `video_distortions.py`, using the stored margin as context (ops whose `margin()` exceeds 64 px are skipped):
  - on **clean-source** tiles: any op
  - on **encoded** tiles: only post-decode ops (`rescale`, `aliasing`, `pixelation`, `posterize`, `chroma_bleed`, `chroma_quantize`, `ghosting`)
- Center-crop to 256×256. Augment with horizontal flip and mild color jitter applied **before** distortion. No random-resized crops, no blur, no JPEG.
- Ranking pairs: two tiles with the same source, position and frame that differ only in ladder rank, or only in `plan.with_severity(s)`.

**3c. Video loader (Stages 2–3).**
- CPU workers decode an encode (any length ≥ 240 frames; windows are sampled anywhere inside it, never across spans) or a public-MOS clip on the fly, apply the display-referenced conversion, then apply a random drop pattern as an index remap (none with probability 0.1). From the same decode, produce:
  - (i) every displayed frame downscaled to 384×216 (area filter)
  - (ii) for every 15th displayed frame, 32 native 256×256 tiles at random positions on a 9×15 grid. Grid rows use stride 238 px so 9 rows cover 2160 px; columns use stride 256 px.
- Ranking pairs: the same encode with drop pattern P and with superset(P) (`drops.superset`); the same P on two ladder ranks; the same P on two severities of one capture plan.
- Lonestar6 note: decoding is CPU-bound. Size the dataloader workers to the node's cores and check that GPU utilization stays above 70%.

**STOP for review:** loader throughput, sample visualizations (tiles, low-res strips with drops), batch-diversity numbers.

---

## Step 4: Stage 1, spatial encoder (Tile-MAE)

| Tensor | Shape |
|---|---|
| input tile | B × 3 × 256 × 256 |
| patch tokens (16×16 patches) | B × 256 × 768 → linear → B × 256 × 384 |
| visible after 75% random mask | B × 64 × 384 |
| encoder ViT-S/16 (depth 12, 6 heads, ~22 M params) | B × 64 × 384 |
| decoder (dim 256, depth 4, 8 heads; mask tokens added) | B × 256 × 256 |
| reconstruction head | B × 256 × (768 + 6) |
| score head (mean pool → MLP 384→128→1) | B × 1 |
| **inference output** (no mask, mean of 256 tokens) | **1 × 384 per tile** |

**Losses**
- `L_rec`: MSE on the masked patches only. Targets are the per-patch normalized pixels of the **distorted** tile (768) plus the per-channel patch mean and std (6), so brightness and contrast distortions aren't normalized away.
- `L_rank`: max(0, 0.1 − (s_better − s_worse)) on pairs from 3b. Both tiles of a pair use the **same mask**. Weight 0.1, switched on after 10% of steps.

**Initialization:** start from a public self-supervised **MAE ViT-S/16** checkpoint (pixel-reconstruction pretraining matches this objective), then continue MAE on distorted native 4K tiles. Record the checkpoint's source and its pretraining data. Run the from-scratch variant as an ablation. Pretrained semantic encoders (DINO-style) are a second ablation; they tend to discard low-level detail.

**Model size:** ViT-S is the deployed encoder because of the real-time budget. Train a ViT-B variant with the same recipe for the size ablation. Optionally train a ViT-L teacher offline and distill its tile tokens into ViT-S (feature-matching loss on unmasked tiles) if the size sweep shows ViT-L clearly ahead.

**Training:** AdamW (β 0.9, 0.95), lr 1.5e-4 × batch/256, cosine schedule, 5% warmup, weight decay 0.05, drop path 0.1, batch 512, about 200 k steps (adjust to data size; fewer when initialized from a checkpoint). bf16. Average the last 4 checkpoints.

**Checks before moving on:**
- On held-out sources, reconstruction error rises with severity.
- Pair-ranking accuracy is above 90% on validation pairs (report it per op family and per codec).
- Features don't collapse: the per-dimension std of pooled tile features stays above 0.1.

---

## Step 5: Stage 2, motion encoder (Video-MAE on low-res pixels)

| Tensor | Shape |
|---|---|
| input window: 32 consecutive displayed frames, 384×216 | B × 3 × 32 × 216 × 384 |
| tubelets 2 × 24 × 24 | 16 steps × (9 × 16) = 2304 tokens × 3456 values |
| embed (linear 3456 → 384) + spatial pos + display-order pos | B × 2304 × 384 |
| mask: 50% of time steps fully, then 50% of remaining tokens | B × 576 × 384 visible |
| encoder ViT-S (depth 12, 6 heads, joint space-time attention) | B × 576 × 384 |
| decoder (dim 192, depth 4, 6 heads) | B × 2304 × 192 |
| reconstruction head | B × 2304 × (3456 + 6) |
| score head (mean pool → MLP → 1) | B × 1 |
| **inference output** (no mask; pool 9×16 grid to 3×4 regions) | **16 steps × 12 × 384 per window** |

**Losses**
- `L_vmae`: MSE on masked tubelets, using normalized pixels plus per-channel mean and std, as in Stage 1. Whole-step masking forces the model to predict frames from their neighbours. A held frame is easy to predict and the catch-up frame after it is not, so the model learns normal motion with no drop labels.
- `L_rank`: margin 0.1 on (P, superset(P)) pairs of the same window, with the same mask. Weight 0.1, switched on after 10% of steps.

**Data:** clean and encoded spans, the 2f subset, and on-the-fly drops in 50% of windows (none in the rest, so normal motion dominates). No time reversal.

**Initialization:** start from a public **VideoMAE ViT-S** checkpoint (Kinetics/SSv2 pretraining). Its tubelets are 2×16×16, so resize the patch-embedding weights to 2×24×24 (FlexiViT-style pseudo-inverse resize) and interpolate the position embeddings to the 16×9 grid. Ablation: from scratch.

**Training:** same optimizer recipe as Stage 1, batch 64 windows, about 150 k steps.

**Checks:**
- On validation, reconstruction error of masked steps is higher at catch-up frames than at normal frames.
- Ranking accuracy on drop pairs, reported by drop family and by burst length (1, 2–4, 5–15, 16–60 frames).
- Static scenes: the score difference for drops in near-static content should be small. Report it; don't force it.

---

## Step 6: Stage 3, fusion transformer and heads

Both encoders are frozen. A **window** is 4 s (240 displayed frames); hop 1 s at inference.

| Tensor | Shape |
|---|---|
| spatial tokens: 16 sampled frames (every 15th) × 32 tiles | 512 × 384 |
| motion tokens: 120 steps × 12 regions | 1440 × 384 |
| project each to 256 + type emb + display-order pos + grid pos | 1952 × 256 |
| fusion transformer (depth 4, dim 256, 4 heads) | 1952 × 256 |
| attention pooling, 2 learned queries | 2 × 256 |
| Q head (MLP 256→64→1), V head (MLP 256→64→1) | q_w, v_w per window |
| clip outputs | Q = (1−a)·mean(q_w) + a·softmin_τ(q_w); V = softmax_τ(v_w) |

`a` and τ are learned. A clip is scored from all its windows (a 300-frame training clip gives 2 windows; longer spans give more).

**Losses**
- `L_rank` on Q: margin 0.1 over three pair types in equal proportions: more drops, lower bitrate, higher capture severity.
- `L_mos` on Q: public MOS datasets after display conversion. MSE on per-dataset z-scored MOS + (1 − PLCC) within each batch. Batches are drawn from one dataset at a time.
- `L_drop` on V: margin 0.1, V(superset(P)) > V(P), including P = no drops.
- `L_inv` on V: Huber(V_a − V_b) for pairs that differ **only** in bitrate or capture severity, with the same drops. This keeps V a drop-visibility score rather than a general quality score. Weight 0.1.

**Public MOS sets** (check licenses, keep a record):
- LIVE-YT-Gaming (gaming, 30/60 fps)
- AVT-VQDB-UHD-1 (4K, includes frame-rate reduction)
- LIVE-YT-HFR (frame rates 24–120 fps)
- KoNViD-1k (general UGC)

Hold out 20% of each for validation.

**Training:** AdamW lr 3e-4, cosine, batch 32 clips, about 30 k steps. Select the checkpoint on public-MOS validation SROCC plus ranking accuracy, never on the study.

---

## Step 7: Evaluation (study = test)

- **Protocol B (headline, zero-shot):** run the trained model on all 1,823 study clips, displayed-order pixels only.
  - Q vs MOS: SROCC, KROCC, PLCC and RMSE after the standard 4-parameter logistic fit, also reported unmapped.
  - V vs mean visibility rating (Not = 0 … Clearly = 3): SROCC, and AUC for noticed (mean ≥ 0.5) vs not. V is a score, not calibrated to 0–3, so only rank-based metrics are used.
  - Per-subset results: uncompressed+drops, consecutive 5/15/25, random 15/30/60%, and by bitrate region.
- **Protocol A (comparable to the baseline table):** the same grouped holdout folds as the baselines (median of 20), with only a ridge/SVR on frozen pooled features trained on the training folds.
- **Baselines:** the existing table (GAMIVAL, VSFA, ...), FAST-VQA/DOVER, P.1204.3 on the original encodes, and the PSVQA hand-crafted-feature model **as a baseline only**.
- **Speed:** end-to-end fps at 4K60 including NVDEC decode, on an A100 and on one consumer GPU.

---

## Step 8: Ablations

| Question | Comparison |
|---|---|
| Do encoders need ranking? | MAE only vs MAE + L_rank (stages 1 and 2) |
| Mean/std targets | normalized pixels only vs + mean/std |
| Native tiles | native 256 tiles vs whole frame resized to 384×216 |
| Every frame | motion encoder on every frame vs uniform 4 fps |
| Whole-step masking | random tube masking only vs + whole-step masking |
| Visibility head | with vs without L_inv |
| Display conversion | public MOS with vs without frame-repetition conversion |
| Post-hold re-encode | with vs without the 2f subset |
| Batch diversity | contiguous batches vs mixed-source |
| Initialization | from scratch vs MAE/VideoMAE checkpoint vs DINO-style checkpoint |
| Encoder size | ViT-S vs ViT-B vs ViT-L (accuracy vs fps curve); ViT-L→ViT-S distillation |
| Event labels (deliberately excluded) | + hold/onset/resume auxiliary head, reported as an ablation only |

## Real-time budget (per second of 4K60, rough)

| Part | Work | Cost |
|---|---|---|
| Spatial encoder | 4 frames × 32 tiles × ViT-S/16 @ 256 tokens | ~0.8 TMAC |
| Motion encoder | ~1.9 windows × 2304 tokens × ViT-S | ~0.2 TMAC |
| Fusion | 1 window × 1952 tokens × 4 layers @ 256 | < 0.05 TMAC |

Total is about 1 TMAC/s, which fits an A100 or a recent consumer GPU in bf16, plus NVDEC decode. **Measure it; don't infer it.** Reusing a tile's cached features when its pixels are bit-identical to the previous sampled frame is allowed as a pure speed-up, because it doesn't change any output.
