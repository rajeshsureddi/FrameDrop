#!/bin/bash
# Step 19 (user 10-23: multi-scale frozen features + same regressors on the public datasets): tokens for one GPU.
# Usage: step19_extract.sh <part> <parts> <gpu> [workers]   (each task split into <parts>; this worker does <part>)
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
D=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; O=$D/step19_public
V3=$D/frozen_v3_20261018; V4=$D/step15_v4/runs/v4
for t in v3:ytugc v3:ytugc_half v3:konvid1k_half v3:livevqc_half v4:konvid1k_half v4:livevqc_half; do
  m=${t%%:*}; ds=${t##*:}; [ $m = v3 ] && R=$V3 || R=$V4
  echo "$(date -Is) $m $ds part $1/$2"
  python step7_eval.py extract --dataset $ds --part $1 --parts $2 --gpu $3 --workers ${4:-20} --s1 $R/s1.pt --s2 $R/s2.pt --tok-root $O/tokens_$m
done
echo "STEP19_EXTRACT_DONE $(date -Is)"
