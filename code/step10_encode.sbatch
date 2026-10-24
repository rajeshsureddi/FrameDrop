#!/bin/bash
#SBATCH -J dd_enc
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 48:00:00
#SBATCH -A ASC22049
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/logs/%x_%j.out
# DropDev re-encodes (all jobs share shard 0/1; per-clip locks), then label-side measure.
source /work/09997/rs64952/ls6/miniconda3/etc/profile.d/conda.sh
conda activate /work/09997/rs64952/ls6/miniconda3/envs/vlm-r1
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export OMP_NUM_THREADS=1
python step10_dropdev.py encode --workers 24 --threads 5
python step10_dropdev.py measure --workers 48
