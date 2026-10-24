#!/bin/bash
#SBATCH -J step2_cpu
#SBATCH -p normal
#SBATCH -A ASC22049
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 12:00:00
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/logs/%x.%j.out
# usage: sbatch step2_cpu.sbatch <shard k/n> [workers] [threads]
set -e
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; M=$R/method; O=$R/vqa_data/step2; cd $M
SH=${1:-0/1}; WK=${2:-16}; TH=${3:-4}; SKIP=${4:+--skip-shard $4}
echo "node $(hostname) shard $SH workers $WK threads $TH start $(date -Is)"
s=$(date +%s); python step2_ladder.py --manifest $R/vqa_data/step1/manifest.csv --out $O --shard $SH --workers $WK --threads $TH $SKIP
echo "TIMING ladder shard $SH: $(( $(date +%s) - s ))s"
s=$(date +%s); python step2_postdrop.py --manifest $R/vqa_data/step1/manifest.csv --out $O --shard $SH --workers $(( WK * 2 )) --threads $TH $SKIP
echo "TIMING postdrop shard $SH: $(( $(date +%s) - s ))s"
echo "done $(date -Is)"
