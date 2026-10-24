#!/bin/bash
# low-priority capture encoding on the gpu-h100 node's idle cores (renders keep priority)
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step2; cd $R/method
exec nice -n 19 python step2_capture.py encode --manifest $R/vqa_data/step1/manifest.csv --out $O \
   --shard 0/1 --workers ${1:-12} --threads ${2:-4} --loop --render-shards 9
