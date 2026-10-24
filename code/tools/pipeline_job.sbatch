#!/bin/bash
#SBATCH -J fdvqa_pipe
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A ASC22049
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/logs_batch/pipe_%j.out
cd /scratch/09997/rs64952/FrameDrop-VQA/method
exec bash tools/pipeline_resume.sh
