#!/bin/bash
# Step 18 (user 10-23: "different scales (full and half resolution) ours v3 or v4 only, that gets like 1024 features"):
# half-resolution copies of the 1823 study clips: 3840x2160 -> 1920x1080 bicubic, 60 fps, all frames, full range kept,
# H.264 CRF 10. Read from the node-local copy if present.  Usage: step18_half_res.sh <part> <parts> <jobs>
part=$1; parts=$2; J=${3:-24}
FF=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin/ffmpeg
SRC=/scratch/09997/rs64952/FrameDrop-VQA/mos_datasets/framedrop_vqa_subjective_study/mp4_for_study_crf10
LOC=/tmp/rs64952_step14/study
OUT=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step18_multiscale/study_1080p
ls $SRC | grep '\.mp4$' | sort | awk -v p=$part -v n=$parts 'NR % n == p' | \
xargs -P $J -I{} bash -c '
  o='$OUT'/{}; [ -s "$o" ] && exit 0
  i='$LOC'/{}; [ -s "$i" ] || i='$SRC'/{}
  '$FF' -nostdin -v error -threads 4 -i "$i" -vf scale=1920:1080:flags=bicubic -c:v libx264 -crf 10 -preset fast \
     -pix_fmt yuvj420p -color_range pc -an -threads 4 "$o.part.mp4" && mv "$o.part.mp4" "$o" || echo "FAIL {}"'
echo "DONE part $part $(date -Is)"
