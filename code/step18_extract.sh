#!/bin/bash
# Step 18: v4-encoder tokens for the half-resolution study clips.  Usage: step18_extract.sh <part> <parts> <gpu> [workers]
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 STEP18_VIDEOS=/tmp/rs64952_step14/study
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step15_v4/runs/v4
TR=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step18_multiscale/tokens_v4
python step7_eval.py extract --dataset ${DS:-study_half} --part $1 --parts $2 --gpu $3 --workers ${4:-20} --s1 $R/s1.pt --s2 $R/s2.pt --tok-root $TR
