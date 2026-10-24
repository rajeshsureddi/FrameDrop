# PSVQA: Predictive-Surprise Video Quality Assessment

A real-time no-reference VQA model for compression plus frame drops. It is trained without any data from the subjective study, which is used only as a zero-shot test set.

## Method in one paragraph

Every displayed frame is embedded at 256x144 by a small frozen CNN (the dense branch). A causal GRU, trained **only on clean video**, predicts the latent motion the viewer expects next. Temporal distortion is then measured as *perceptual surprise*, i.e. the mismatch between expected and shown motion in that feature space:

- **Freeze surprise:** motion that was expected but did not happen. This is zero for a freeze in a static scene.
- **Catch-up excess:** the size of the visible skip when playback resumes.
- **Prediction error and trajectory curvature:** these capture judder and irregular pacing.

A ResNet-50 runs on native-resolution 224x224 fragments of about 2 unique frames per second, plus every frame that gets held during a freeze. Its spatial features are pooled over *displayed* frames. A frozen image therefore counts once per tick it stays on screen, and a learned gate (driven by freeze length and last real motion) models the loss of temporal masking during freezes. A GRU over 1 s segments outputs the overall quality Q and an ordinal frame-drop visibility level from 0 to 3.

## Why it needs no test-set information

- **Predictor:** trained on clean video only (one-class), so it never sees a drop.
- **Head:** trained with ranking losses whose order is known without labels. For the same encode, adding drops makes it worse. For the same drop pattern, a lower bitrate makes it worse. Optionally, public MOS datasets (e.g. LIVE-YT-Gaming, KoNViD) anchor the absolute scale.
- **Drop generator:** broader than the study protocol. It samples bursts of 1–60 frames placed anywhere, Bernoulli loss, Gilbert–Elliott loss, pacing judder, and mixtures.
- **Source clips:** use training clips that do not overlap the study's 35 sources, ideally from different game titles.

## Pipeline

```bash
pip install -r requirements.txt

# 1. Compressed training versions of clean clips (+ manifest.csv)
python -m psvqa.encode clean_clips/*.mp4 --out encodes/ --n 5 \
       --encoders libx264 hevc_nvenc --cache_dir caches_train

# 2. Features for every training video (pristine + encodes), full mode.
#    Drops are applied later by re-indexing, so extract each encode once.
python -m psvqa.extract clean_clips/*.mp4 encodes/*.mp4 --out caches_train --mode full

# 3. Motion predictor on CLEAN caches only (pristine + encodes, no drops)
python -m psvqa.predictor --caches "caches_train/*.npz" --out predictor.pt

# 4. Training samples: encodes x drop patterns (+ supersets)
#    Optionally append public-MOS rows to manifest.csv (columns mos, dataset)
python -m psvqa.build_dataset --manifest encodes/manifest.csv \
       --predictor predictor.pt --out train_samples.pkl --patterns 12

# 5. Head
python -m psvqa.train_head --samples train_samples.pkl --out head.pt

# 6. Zero-shot test: study videos in POLICY mode (the real-time path)
python -m psvqa.extract study_videos/*.mp4 --out caches_test --mode policy
python -m psvqa.evaluate --test test.csv --head head.pt --predictor predictor.pt
```

`test.csv` needs the columns `cache` and `mos`, and optionally `visibility` and `group`:

- **visibility:** mean human rating with Not noticeable = 0, Barely = 1, Slightly = 2, Clearly = 3.
- **group:** subset label such as `uncompressed`, `consecutive-15`, `random-60`; per-group SROCC is printed when present.

## Ablations for the paper

| Variant | How |
|---|---|
| Constant-velocity predictor (no learning) | omit `--predictor` in steps 4 and 6 |
| No freeze gate (uniform pooling) | fix the gate output to a constant in `head.py` |
| Spatial only (VSFA-like) | zero the temporal stats |
| Temporal only | zero the spatial features |
| Spatial sampling rate | `SAMPLE_EVERY` in `common.py` |
| Stronger spatial encoder | swap `SpatialEncoder` for a FAST-VQA/DOVER backbone |

Report both zero-shot results and FPS next to the baseline table. The baselines in that table were trained in-dataset.

## Things to check on your data first

- **Repeat detection** (`common.is_dup_pair`) uses the mean and max difference on 128x72 luma thumbnails. On synthetic tests it is exact for decoded repeats, with no false positives on clean moving content. If the study videos were re-encoded for storage, however, recall falls: about 80% at CRF 28 and 65% at CRF 38 in my test. Run it on a few study clips with known patterns and tune `DUP_MEAN`/`DUP_MAX` if needed. This uses no MOS labels.
- **Frame rate:** the predictor assumes 60 fps. Public 30 fps datasets are fine for the MOS anchor but should not be used to train the predictor.
- **Visibility supervision is weak.** It uses a physical proxy (missed pixel motion summed over freezes, split at quantiles) plus ranking. The human visibility labels in the study are used only for testing.
- **Real-time throughput:** decoding with PyAV on the CPU is the bottleneck at 4K. For real time, use NVDEC and resize on the GPU. The rest of the pipeline is light: a MobileNet at 256x144 on every frame, a ResNet-50 on about 2–4 frames per second, and one GRU step per frame.
