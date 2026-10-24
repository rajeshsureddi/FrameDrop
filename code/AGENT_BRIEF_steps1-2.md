# Agent brief: 4K VQA training data, Steps 1–2

You are preparing training data for a real-time no-reference video quality model for 4K60 content with compression and frame drops. Step 1 audits the source videos and builds the source manifest. Step 2 creates the offline compressed and distorted variants. **Stop after each step** and hand back the report for human review before continuing.

---

## Fill in before starting

| Item | Value |
|---|---|
| Inter4K root | `/path/to/inter4k` |
| LMM-10K root | `/path/to/lmm10k` |
| Study clips (exclusion only, frames only, never scores) | `/path/to/study_sources` |
| Own gameplay captures (optional) | `/path/to/gaming` |
| Output root | `/path/to/vqa_data` |
| GPU(s) available | e.g. 1× RTX 4090 / H100 |
| Storage | no cap: Lonestar6 `$SCRATCH` is unlimited (mind its purge policy; keep manifests, JSONs, code and reports in `$WORK`) |
| CPU cores | e.g. 32 |

## Provided code

| File | What it is |
|---|---|
| `build_manifest.py` | Step 1 audit and manifest (works; extend it, see 1.3) |
| `video_distortions.py` | numpy distortion bank, 32 ops, deterministic, crop-consistent |
| `test_video_distortions.py` | its tests (must keep passing) |
| `psvqa/psvqa/drops.py` | drop patterns (keyframe anchoring removed; drops placed uniformly) |
| `psvqa/psvqa/encode.py` | **do not use for Step 2**: its low-latency NVENC/zerolatency settings mirror the study; follow 2b instead |

## Hard rules (apply to every step)

1. **The study videos are test data.** Use the study source folder only to detect and exclude overlapping content. Never train on, encode, tune thresholds on, or look at MOS for study videos.
2. **SDR only, up to 4K.** Drop HDR (PQ/HLG) sources; never tone-map them.
3. **Splits are by source group** (same shoot or cross-dataset duplicates stay together). Never split by clip.
4. **Never modify or delete the downloaded datasets.** Write only under the output root.
5. **Everything must be resumable and deterministic:** per-item caches, fixed seeds, and a JSON record of every parameter used to create every file.
6. **Frame drops are NOT applied in Steps 1–2.** They are applied later, on the fly, as index remaps.
7. **Never fake codec artifacts** (no JPEG/DCT simulation). Compression comes only from real encoders.

---

## Step 1: Source audit and manifest

### Goal
A table of usable, clean 4K60 SDR sources, each with its usable frame span(s), grouped and split by source, with no overlap with the study.

### Tasks

**1.1 Environment.** Confirm `ffmpeg`/`ffprobe` (record versions), numpy, OpenCV. Run `python test_video_distortions.py`; it must print `ALL PASSED`.

**1.2 Smoke test.** Run `build_manifest.py` on ~50 clips from each dataset with `--workers` set to the core count. Check `report.txt` looks sane and estimate the full runtime.

**1.3 Extend `build_manifest.py`** (keep existing behaviour and columns):
- **Scene-cut detection:** flag frames whose mean luma difference (128×72 thumbnail) is far above the clip's median (e.g. > max(25, 8 × median + 1) gray levels). Store `n_cuts`.
- **Usable spans (revised after review):** there is no fixed segment length. Split each clip at scene cuts and at native repeat runs (confirmed at 960×540, see 1.6); every remaining span of **≥ 240 frames** is usable. Store `spans` (list of [start, end) frame ranges), `usable_frames`, `span_motion`. A clip is kept if it has ≥ 1 usable span. Do not discard near-static clips: tag them `static` and keep them (they teach the model that drops in static scenes are invisible).
- **Storage estimate** in the report: `usable seconds × encodes per source × average Mbps`.
- Include the span rules in the cache key so changing them re-scans.

**1.6 Revisions after review.** Repeats: confirm suspected repeats on 960×540 luma (mean |diff| < 0.25 and max ≤ 2 gray levels); slow footage with sub-pixel motion is not a repeat. Letterboxing: run crop detection, store the active area `active_xywh`, and use only the active area for tiles and the motion encoder. `--min-bpp 0.03`. Study exclusion/grouping: detail-correlation ≥ 0.6 on ≥ 3 frames, as implemented.
- Add a test on synthetic clips (one with a hard cut, one upconverted 30→60 fps, one copy of an "excluded" clip) proving each case is handled.

**1.4 Full run** over Inter4K and LMM-10K (plus gaming captures if provided), with `--exclude` pointing at the study sources.

**1.5 Spot-check the filters (needed for the human review).** For each of `upconverted_cadence`, `native_freeze`, `low_bpp` and `no_usable_span`, save one mid-clip thumbnail and the 30-frame duplicate pattern for 10 flagged and 10 kept clips into `step1_review/`. Do not change thresholds yourself; report what you see.

### Deliverables (in `<output root>/step1/`)
- `manifest.csv`: one row per clip with `keep`, `split`, `group_id`, `flags`, probe stats, span fields
- `report.txt`: per-dataset counts, flag counts, fps/codec breakdown, bpp percentiles, kept hours, group stats, split counts, excluded-source matches, span-motion percentiles, storage estimate
- `step1_review/`: spot-check images
- `STEP1_SUMMARY.md`: what ran, runtimes, tool versions, anomalies, open questions

### Acceptance checks
- [ ] Every video under both roots appears in the manifest exactly once.
- [ ] No kept clip is HDR, below 3840×2160, below 50 fps, VFR, or flagged with a hard flag.
- [ ] Every kept clip has at least one span of ≥ 240 frames inside its decoded range.
- [ ] No `group_id` appears in more than one split.
- [ ] Every clip matching an excluded study source is dropped, and the count is reported.
- [ ] Re-running the command reuses the cache and produces an identical `manifest.csv`.

**STOP. Hand back `STEP1_SUMMARY.md` and `report.txt`.** The human sets the `--min-bpp` cutoff and approves the filter thresholds before Step 2.

---

## Step 2: Offline encodes and capture-distortion variants

### Goal
For every usable span of every kept source: a **clean compression ladder**, plus a few **capture-distortion → compression** variants, all from real encoders with generic, randomly sampled settings (2b), each with a JSON record. Reference frames are never stored: they are re-decoded from the original file plus the span's frame range.

### 2a. GPU port of the distortion bank
- Port `video_distortions.py` to PyTorch (`video_distortions_torch.py`) with the **same API, registry, parameters and hash functions**, so a plan renders identically on either backend.
- Parity tests against the numpy version on random plans, frames and crops: max abs difference < 2e-3 per op and for 2–3-op combinations; identity at severity 0; crop consistency; seekability.
- Benchmark on the target GPU at 3840×2160: report ms/frame per op. Target: under ~20 ms for most ops. List any that miss.

### 2b. Clean compression ladders (all kept sources)
- **Input:** decode each usable span frame-accurately (all its frames; no fixed length) and verify the decoded frame count equals the span length.
- **Encoding settings are generic, not copied from the study.** Do not default to low-latency, CBR, no-B-frame or HEVC-heavy settings because the study used them; the study's encoder configuration must not shape the training distribution. Everything below is sampled **once per source** (ranking pairs need everything except the rate parameter to be identical) and recorded in every JSON:
  - **Encoder**, equal weights: `libx264`, `libx265`, `libsvtav1`, `libvpx-vp9`.
  - **Rate-control mode**, equal weights:
    - CRF / constant quality (x264/x265 `-crf`, svt-av1 `-crf`, vp9 `-crf -b:v 0`)
    - one-pass average bitrate (`-b:v`)
    - constrained/capped (`-b:v` with `-maxrate` 1.5× and `-bufsize` 2×)
    - CBR-like (`-b:v = -maxrate`, `-bufsize` 0.5–2× target)
  - **Preset/speed:** uniform over the encoder's full range (e.g. x264/x265 ultrafast…slow; svt-av1 preset 4–12; vp9 `-deadline good`, `-cpu-used` 0–5).
  - **Tune:** none (60%), or one of the encoder's tunes (e.g. film, grain, zerolatency/fastdecode), uniformly.
  - **B-frames / reference structure:** encoder default (60%), 0 (20%), or high (20%; e.g. x264/x265 `-bf 3–8`).
  - **GOP:** uniform from {30, 60, 120, 240, 600, encoder default}.
- **Ladder:** 5 rungs per source of the chosen rate parameter, sorted, rank 0 = worst:
  - bitrate modes: log-uniform over 1.5–60 Mbps for 4K (0.5–20 Mbps for 1080p)
  - CRF mode: uniform over [18, 51] for x264/x265, [18, 60] for svt-av1, [18, 60] for vp9, with rungs at least 4 CRF apart
- **Efficiency:** decode each span once and feed all ladder encodes from that single decode (one ffmpeg command with multiple outputs, or a tee). Run encoding on CPU nodes, not the GPU node; the GPU is only needed for 2a and 2c rendering.
- **Pixel format:** 8-bit `yuv420p`, BT.709 tagged.
- Do **not** reuse the study's protocol: no RD-curve knee/saturation picking, no "3 bitrates per clip", no low-latency/CBR/HEVC preference, no NVENC-specific settings. Report the realized distribution of encoder × mode × preset in `qc_report.txt`.

### 2c. Capture-distortion variants (all kept sources)
- For each selected source: sample 2 plans with `sample_plan(rng, fps=60, duration=5, n_ops=(1, 3))`.
- For each plan, render 2 severities (`plan.with_severity(0.5)` and `1.0`) and encode each at 2 bitrates drawn from that source's ladder (one high, one mid).
- **Pipeline:** decode the original span → render frames with the GPU bank (pass the true source frame indices) → pipe raw frames directly into the encoder. Never write uncompressed frames to disk.
- **Save** the plan as JSON next to the encode (op names, severities, parameters, seeds, envelopes, supports, bipolar flags), plus `describe()` labels without the per-frame arrays.
- **Temporal ops** (`shutter_blur`, `ghosting`) need `plan.margin_t()` frames of history: decode that many frames before the span start (when available inside the clip; otherwise start the span later by that many frames), and drop them before encoding.

### 2d. Scale
- No storage cap: run 2b, 2c and 2f on **all** kept sources. Report the storage actually used and throughput (sources/hour).
- Scratch is purged when files go unused: every encode must be regenerable from its JSON, and the manifests/JSONs live in `$WORK`.
- Use parallel workers sized to GPU encoder sessions and CPU cores. Report throughput (sources/hour).

### 2e. Quality control
- For a random 2% of sources, compute PSNR and VMAF (if `libvmaf` is available) of every clean-ladder encode against the re-decoded reference span. Check quality rises with bitrate for each source and report any violations.
- Verify every encode: decodes without errors, has exactly the span's frame count, is 3840×2160 at 60 fps, and its codec/GOP/bitrate match its JSON.
- For 20 random capture variants, save side-by-side thumbnails (reference / severity 0.5 / severity 1.0) for human review.

### Deliverables (in `<output root>/step2/`)
- `encodes/<source_id>/...mp4` and a `.json` for each
- `encodes_manifest.csv`, one row per encode: `source_id, group_id, split, kind (clean|capture), plan_id, severity, encoder, gop, target_kbps, actual_kbps, rank, path, json_path, frames, decode_ok`
- `video_distortions_torch.py` with parity tests and `benchmark.txt`
- `qc_report.txt`: VMAF/PSNR monotonicity, decode checks, storage used, throughput
- `step2_review/`: thumbnails
- `STEP2_SUMMARY.md`: what ran, runtimes, versions, deviations, open questions

### Acceptance checks
- [ ] Torch parity tests pass, and `test_video_distortions.py` still passes.
- [ ] Every kept source has a 5-step clean ladder, and every encode decodes to its span's frame count.
- [ ] The QC sample's quality is monotone in bitrate (any exception is listed).
- [ ] Every capture variant has a plan JSON from which the same frames can be re-rendered exactly.
- [ ] Each source's encodes inherit its split and group from the Step 1 manifest.
- [ ] Storage used is reported, and manifests/JSONs are mirrored in `$WORK`.

**STOP. Hand back `STEP2_SUMMARY.md` and `qc_report.txt`.** Step 3 (tile store and data loader) starts after review.
