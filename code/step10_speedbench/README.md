# FrameDrop-VQA speed benchmark (consumer GPU)

This measures the same three things as the A100 run (`step7_speed.py`), on any CUDA GPU.

1. **model**: GPU compute only, for a synthetic 10 s 4K60 clip. That is Stage 1 (40 tile frames × 32 tiles, ViT-S/16), Stage 2 (600 frames at 384×216, ViT-S), and fusion, in bf16. Host-to-device copies are included.
2. **nvdec**: the NVDEC decode ceiling for each clip in `clips/`. Frames stay on the GPU.
3. **e2e**: end to end on the real 4K60 clips in `clips/`. Decoding uses NVDEC where it works for the codec, otherwise the CPU. The 384×216 downscale and 4K tile crops run on the CPU; the encoders and fusion run on the GPU.
   - `stream`: one clip at a time, for single-stream fps.
   - `parallel`: decoding in `--workers` processes feeding one GPU, for throughput.

`clips/` holds 8 clips: 2 each of H.264, HEVC, VP9 and AV1. Each is 3840×2160, 60 fps, about 5 s. They are top-rung Step 2 encodes of LMM-10K sources from the audit split. They are not study clips.

## Setup
```
conda create -n fdvqa python=3.11 && conda activate fdvqa
pip install torch numpy      # a CUDA build matching your driver
# ffmpeg/ffprobe >= 6 on PATH, built with NVDEC (cuda/cuvid)
# check: ffmpeg -hwaccels   (should list "cuda")
```

## Run
```
python bench.py --gpu 0 --workers 8 --threads 4
# without NVDEC:
python bench.py --gpu 0 --no-nvdec
# with trained weights (directory with s1.pt, s2.pt, fusion.pt), optional:
python bench.py --gpu 0 --weights /path/to/weights
```
Set `--workers` to about the number of physical CPU cores divided by `--threads`.

The script writes `speed_<GPU name>.json`. Send that file back.

Timing does not depend on the weights. Without `--weights`, the model uses random weights of the final shape.

The model part needs about 3–4 GB of GPU memory.
