#!/bin/bash
#SBATCH -J dd_meas
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 12:00:00
#SBATCH -A ASC22049
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/logs/%x_%j.out
# After all encode jobs: measure what is left, then write pairs.csv.
source /work/09997/rs64952/ls6/miniconda3/etc/profile.d/conda.sh
conda activate /work/09997/rs64952/ls6/miniconda3/envs/vlm-r1
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export OMP_NUM_THREADS=1
python step10_dropdev.py encode --workers 24 --threads 5
python step10_dropdev.py measure --workers 64
python step10_dropdev.py collect
