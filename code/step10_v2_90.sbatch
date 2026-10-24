#!/bin/bash
#SBATCH -J v2_90
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 06:00:00
#SBATCH -A ASC22049
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_cand/logs/%x_%j.out
/scratch/09997/rs64952/FrameDrop-VQA/method/step10_cand_run.sh v2_90 0
