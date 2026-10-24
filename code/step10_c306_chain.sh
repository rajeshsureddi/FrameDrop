#!/bin/bash
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
cd /scratch/09997/rs64952/FrameDrop-VQA/method/step10_speedbench
CUDA_VISIBLE_DEVICES=0 python -I bench.py --gpu 0 --weights /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/frozen_v3_20261018 --workers 48 --threads 2 --out /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/speed/speed_A100_v3_w48.json > /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/logs/speed_A100_c306_w48.log 2>&1
cd /scratch/09997/rs64952/FrameDrop-VQA/method
nice -n 10 python step10_dropdev.py encode --workers 22 --threads 5 > /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/logs/encode_c306-002.log 2>&1
