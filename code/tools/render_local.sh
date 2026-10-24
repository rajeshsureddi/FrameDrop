#!/bin/bash
# 2c render shards on this session's GPU (gpu-a100-small job), one after another
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step2; cd $R/method
for k in "$@"; do
  python step2_capture.py render --manifest $R/vqa_data/step1/manifest.csv --out $O --shard $k/9 > $O/logs/render_shard${k}of9.log 2>&1
done
