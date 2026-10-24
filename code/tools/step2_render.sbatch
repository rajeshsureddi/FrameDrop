#!/bin/bash
#SBATCH -J step2_render
#SBATCH -p gpu-a100
#SBATCH -A ASC22049
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/logs/%x.%j.out
# usage: sbatch step2_render.sbatch <first shard> <n shards>   (3 shards per node, one per GPU)
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; M=$R/method; O=$R/vqa_data/step2; cd $M
K=$1; N=$2
echo "node $(hostname) GPUs $(nvidia-smi -L | wc -l) shards $K..$((K+2))/$N start $(date -Is)"
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g python step2_capture.py render --manifest $R/vqa_data/step1/manifest.csv --out $O \
     --shard $((K+g))/$N > $O/logs/render_shard$((K+g))of$N.log 2>&1 &
done
wait
echo "done $(date -Is)"
