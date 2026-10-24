#!/bin/bash
# usage: lowres_after.sh PID WORKERS SEED -> when PID exits, build the low-res cache on this node
while kill -0 $1 2>/dev/null; do sleep 60; done
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
cd /scratch/09997/rs64952/FrameDrop-VQA/method
echo "=== pid $1 exited $(date -Is); starting low-res cache"
exec nice -n 10 python step3_lowres.py --workers $2 --order-seed $3
