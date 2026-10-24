#!/bin/bash
# on the gpu-h100 node: one render process per GPU, shards in the given order
# usage: render_h100.sh <gpu> <shard> [<shard> ...]
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step2; cd $R/method
g=$1; shift
for k in "$@"; do
  CUDA_VISIBLE_DEVICES=$g python step2_capture.py render --manifest $R/vqa_data/step1/manifest.csv --out $O \
     --shard $k/9 >> $O/logs/render_h100_gpu${g}_shard${k}of9.log 2>&1
done
