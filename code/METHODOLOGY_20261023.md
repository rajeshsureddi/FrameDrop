# FrameDrop-VQA method: what was built (as of 2026-10-23) and where to improve

A no-reference, pixels-only quality model for 4K60 video that outputs two scores: **Q** (overall quality) and **V** (frame-drop visibility).
Its training supervision is label-free synthetic pair order plus public MOS. The FrameDrop-VQA study (1,823 videos, 35 contents) is never trained on; it is evaluated only.

---

## 1. Data preparation

### 1.1 Source footage (Step 1)
| | Inter4K | LMM-10K | Total kept |
|---|---|---|---|
| Videos scanned | 1,000 | 9,907 | |
| Kept (clean 4K60 SDR) | 529 | 6,844 | **7,373 sources** (~10.3 h) |

**Filters:** HDR, scene cuts, upconverted cadence (30→60 repeats), native freezes, static clips, low bits-per-pixel and letterbox are all detected. Each kept source contributes one usable span of ≥ 240 frames (median 302).

**Study exclusion:** all 35 study clips are fingerprinted, using both key-hash and dense matching; there were 0 matches.

**Splits by source group:** train 6,267 / valid 365 / audit 741.
- The audit split is used only for the DropDev benchmarks.

### 1.2 Distorted versions (Step 2), about 110.6k real encodes
Each source gets one randomly sampled encoder setup, reused for all of its versions so that pairs differ along exactly one axis:
- **Encoder:** x264, x265, SVT-AV1 or VP9.
- **Rate mode:** CRF, ABR, capped or CBR.
- **Other settings:** preset, tune, B-frames and GOP length.

The versions:
- **Clean rate ladder:** 5 rungs per source (36,865 encodes). Bitrate rungs span 1.5–60 Mbps at 4K; CRF rungs are ≥ 4 apart.
- **Capture distortions:** 2 random plans per source, each with 1–3 ops from a 32-op distortion bank (blur, noise, banding, ghosting, shutter, colour and so on), at 2 severities (0.5 and 1.0). Each is encoded at 2 rates, giving 58,984 encodes.
- **Post-drop re-encodes (2f):** 2 per source (14,746). Drops are applied and the result is re-encoded at a random CRF, so held frames are near-identical rather than exact copies.
- **No compression is simulated.** All compression comes from real encoders.

### 1.3 Frame drops (applied on the fly)
Drops are index remaps on the display timeline, never baked into files (except 2f). The drop families are:
- **burst:** 1–5 runs of 1–60 frames each
- **Bernoulli:** p from 0.03 to 0.7
- **Gilbert–Elliott:** bursty loss
- **pacing:** every m-th frame, m from 2 to 6
- **mixtures:** two of the above (12% of patterns)
- **none:** 8% of patterns

The median dropped fraction is about 12%.

### 1.4 Stores and loaders (Step 3)
- **Tile store:** every 6th frame of every version, 16 tiles per frame at the *same* positions across versions. Tiles are 384×384 crops (a 256 core plus margin), stored lossless, about 12.8 TB.
- **Low-res cache:** every version at 384×216, lossless, 117,968 items.
- **Video loader (for Stages 2–3):**
  - every displayed frame at 384×216;
  - native 256×256 tiles every 15th displayed frame, 32 random cells of a 9×15 grid.

### 1.5 Supervision
**Ranking pairs.** Each pair differs in exactly one thing; the model only learns that one version is worse:

| Pair | Better | Worse | Validity filters |
|---|---|---|---|
| Drops | drop pattern P | superset of P | at least 3 more dropped frames, or one extra burst of ≥ 2 frames |
| Rate | larger actual file | smaller actual file | file-size ratio ≥ 1.5 and agrees with the nominal ladder order |
| Capture | severity s | severity s′ > s | gap ≥ 0.25 (≥ 0.5 for ops that can look *better* at low strength) |

- **Indistinguishable pairs are dropped.** If the two versions differ by less than 0.5 gray levels (mean |A − B|), the pair is not used. This difference is computed only to filter labels and is never a model input.
- **Pairs never compare across axes,** for example drops vs bitrate.

**Public MOS** (absolute labels, used from v2 on):

| Dataset | Use |
|---|---|
| LSVQ train, 28,056 clips | MOS training (v2, v3, v4) |
| YouTube-UGC split0 train, 684 clips | MOS training (v4 only) |
| LIVE-YT-Gaming train, 430 clips (study titles excluded) | MOS training (v4 only) |
| LSVQ test, LSVQ 1080p, KoNViD-1k, LIVE-VQC, YouTube-UGC | evaluation only, at native resolution and frame rate |

**Development benchmarks** (from the audit sources only):
- **DropDev:** 17,784 pairs of four types (burst length, burst count, random density, none vs some), half of them re-encoded.
- **DropDev-hard:** the same four types with subtler gaps (bursts of ≤ 6 frames, k vs k+1 bursts, density gaps of 0.03–0.08).

---

## 2. Model

Three stages. Each is trained, then frozen before the next; in v3 and v4 all three are fine-tuned end to end.

| Stage | Backbone | Initialisation | Input | Output |
|---|---|---|---|---|
| **1. Spatial encoder (Tile-MAE)** | ViT-S/16: 12 layers, dim 384, 6 heads, ~22M params | MedMAE ViT-S/16, MAE-pretrained on ImageNet-1k (`1aurent/vit_small_patch16_224.medmae_in1k_mae`) | native 256×256 tiles, 32 per sampled frame, every 15th frame | 384-d per tile (mean of 256 tokens) |
| **2. Motion encoder (Video-MAE)** | ViT-S: 12 layers, dim 384, joint space-time attention | VideoMAE-S, MAE then fine-tuned on SSv2 (`MCG-NJU/videomae-small-finetuned-ssv2`). Its 2×16×16 tubelets were resized to 2×24×24. | every displayed frame at 384×216, in 32-frame windows | 16 time steps × 12 regions × 384 per window |
| **3. Fusion** | transformer: depth 4, dim 256, 4 heads, 3.75M params | scratch | per 4 s window: 512 spatial + 1,440 motion tokens, with type, time and grid embeddings | 2 attention-pool queries → Q head and V head (MLP 256→64→1); clip features: **512-d pooled** |

**Clip score:**
- Q = (1−a)·mean(q_w) + a·softmin(q_w).
- V = softmax pooling over windows.
- a and the temperatures are learned.

**Stage 1 training:**
- **Loss:** MAE reconstruction with 75% masking. Targets are normalised pixels plus the per-patch mean and std, so brightness and contrast changes aren't normalised away. Rank loss on rate and severity tile pairs, weight 0.1.
- **Schedule:** 100k steps × 516 tiles on 3 A100s; the final model averages the last 4 checkpoints.

**Stage 2 training:**
- **Loss:** VideoMAE reconstruction. Masking covers 50% of time steps fully, then 50% of the remaining tokens, so held frames must be predicted from their neighbours. Rank loss on (P, superset(P)) drop pairs, weight 0.1.
- **Schedule:** 100k steps × 66 windows; the final model averages the last 4 checkpoints.

**Stage 3 losses:**
- L_rank on Q over drop, rate and severity pairs.
- L_drop on V: V(superset) > V(P).
- L_inv on V, weight 0.1: V must not change between versions that differ only in rate or severity.
- From v2 on, L_mos: MSE on z-scored MOS plus (1 − PLCC).

### 2.1 Model versions
| Version | What changed | Training data |
|---|---|---|
| v1 | 3 stages, pairs only, no MOS | synthetic pairs |
| v2 | fusion fine-tuned with MOS + pairs (encoders frozen) | + LSVQ train |
| v3 | **end-to-end**: all stages, layer-wise LR decay 0.75, 128-frame native windows; every 4th step is a pair batch | LSVQ train + pairs |
| d1 | Stage 1 re-initialised from **DINOv2-S** (patch 14), full chain | as v3 (train90) |
| **v4** | v3 + more MOS (sampled LSVQ .6 / YT-UGC .25 / Gaming .15 of batches), 6k steps | + YT-UGC and LIVE-YT-Gaming train |

### 2.2 Evaluation protocols
- **Zero-shot:** model output Q vs MOS, with a 4-parameter logistic fit for PLCC.
- **Protocol A (frozen features):** pooled 512-d features + ridge or SVR, on the baselines' 20 content-grouped splits. Step 18 added multi-scale features: full, half and quarter resolution, downscaled inside the decode.
- **Full fine-tune (Step 14):** 10 or 20 splits, best validation epoch, test once.

### 2.3 Where it stands
| | Ours | Best baseline |
|---|---|---|
| Study, zero-shot SROCC | **v4 .816** | UVQ1.5 .770 |
| Study, frozen features + SVR | v4 3-scale .845 (1 scale .827) | GAMIVAL .886 |
| Study, full fine-tune | v3 .923 | (GAMIVAL .886, not matched) |
| Study, V vs human visibility | AUC .65; accuracy ≈ chance on "noticed" | VLMs / DOVER / FasterVQA ≈ .50 |
| KoNViD / LIVE-VQC / YT-UGC, frozen + SVR | .837 / .776 / .792 | TLVQM .768 / **.803** / .659 |
| Same, full fine-tune | .848 / .821 / .829 | HFVQA .927 / .915 / .921 |
| DropDev / DropDev-hard, Q ordering | .964 / .851 (v4) | n/a |

---

## 3. Diagnosed weaknesses (evidence)

1. **The spatial encoder is the weakest link in the features.**
   - Its features score .025 SROCC below a plain DINOv2-S on the same tiles. Bigger DINOv2 models (B/L) don't help.
   - Two Stage 1 checks failed: reconstruction error doesn't rise with severity, and feature spread is low (median std .097).
   - Multi-scale input helps only where sources have spatial detail (study +.017, YT-UGC +.033). The single-scale tile view is under-using the frame.
2. **The motion branch is weak on subtle temporal events.**
   - Single-frame drops are ranked correctly only 89.6% of the time, and the catch-up reconstruction check failed.
   - DropDev-hard Q ordering is .85.
   - V is at chance on consecutive-5-frame and random-30% drops versus human visibility.
   - On LIVE-VQC (camera shake and motion) we lose to TLVQM's hand-crafted motion features.
   - Likely cause: motion is seen only at **384×216**, where small motion and single repeats in low-motion scenes are below the resolution.
3. **V has no perceptual signal.** It learns only "more drops = worse", with no data on *when people notice* (motion, content, duration). That is why it ranks severity well but humans' visibility poorly.
4. **Cross-content calibration and domain.** Within-content ordering is strong (median .90–.92), but the overall level shifts by content. Gaming MOS data is tiny: 430 clips helped in-domain but cost about .01 on unseen sets.
5. **Selection hygiene.** The study has now been used to choose between options in evals 11–18. The next model should be picked on development data alone, then evaluated on the study once.

---

## 4. Options to improve (ranked)

| # | Change | Targets weakness | Expected gain | Cost |
|---|---|---|---|---|
| **A** | **High-resolution temporal branch:** native-resolution "temporal tiles", i.e. 256×256 tubes over 32 consecutive 60 fps frames at several positions. Encoded by a VideoMAE-S (tubelet 1–2 frames), trained with the same MAE + drop-pair recipe and fused as a third token type. Pixels only, display order. | 2, 3, partly LIVE-VQC | largest on drops / visibility / DropDev-hard; some MOS | Stage 2b training ~1 day on 3 GPUs, then fusion + end-to-end |
| **B** | **Better spatial encoder:** (i) distil DINOv2-S into Tile-MAE (feature matching) while keeping MAE + rank losses; or (ii) a learned multi-scale tile stream (native + ½ + ¼ tiles as token types) inside the model, instead of concatenating frozen features | 1 | +.02–.03 frozen-feature SROCC (probe and Step 18 evidence) | ~1 day each |
| **C** | **Temporal perceptual MOS:** LIVE-YT-HFR (frame-rate MOS, 24–120 fps; partially downloaded, labels still to fetch) and AVT-VQDB-UHD-1 (4K, includes frame-rate reduction), added to L_mos | 3, 4 | V and Q sensitivity to *visible* temporal loss | data wait + a v4-style fine-tune |
| **D** | **Cross-content anchor pairs** (plan 9b, never run): clean top rung of source A vs a very low rung or heavy drops of source B | 4 | calibration (overall vs within-content gap) | cheap (synthetic) |
| **E** | **Gaming domain:** own 4K60 gameplay captures from titles not in the study, for Stage 1–2 continued pretraining; plus GamingVideoSET, KUGVD and CGVDS MOS | 4 | domain match (GAMIVAL's advantage) | needs data |

**Recommended order:** A first. It's the clearest gap, it's unique to our method (no baseline sees 60 fps at native resolution), and DropDev-hard plus LIVE-VQC give development-only selection signals. Then B (ii), then C/D as data arrives.
Selection for any new model: development sets only (LSVQ val10, DropDev / DropDev-hard, held-out public MOS splits), then one logged study evaluation.
