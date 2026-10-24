#!/bin/bash
#SBATCH -J s14ft
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00
#SBATCH -A ASC22049
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step14_finetune/logs/%x_%j.out
# usage: sbatch step14.sbatch <init>   (3 GPUs, one queue worker each)
M=/scratch/09997/rs64952/FrameDrop-VQA/method
for g in 0 1 2; do $M/step14_queue.sh $g ${1:-v3} & done
wait
echo ALL_DONE $(date)
