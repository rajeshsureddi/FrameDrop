#!/bin/bash
#SBATCH -J cands
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00
#SBATCH -A ASC22049
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_cand/logs/%x_%j.out
# usage: sbatch step10_cands.sbatch "<name> <args>" "<name> <args>" ...   (one per GPU, max 3)
M=/scratch/09997/rs64952/FrameDrop-VQA/method
g=0
for spec in "$@"; do
  $M/step10_cand_run.sh ${spec%% *} $g $( [ "$spec" != "${spec#* }" ] && echo ${spec#* } ) &
  g=$((g+1))
done
wait
echo ALL_DONE $(date)
