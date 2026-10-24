#!/bin/bash
# low-priority 2b ladders, then 2f post-drop, on half the gpu-h100 node's cores (all sources; locks)
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step2; cd $R/method
WK=${1:-10}; TH=${2:-2}
nice -n 19 python step2_ladder.py --manifest $R/vqa_data/step1/manifest.csv --out $O --shard 0/1 --workers $WK --threads $TH
echo "ladder pass done $(date -Is)"
nice -n 19 python step2_postdrop.py --manifest $R/vqa_data/step1/manifest.csv --out $O --shard 0/1 --workers $(( WK * 2 )) --threads $TH
echo "postdrop pass done $(date -Is)"
