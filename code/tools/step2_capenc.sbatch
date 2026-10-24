#!/bin/bash
#SBATCH -J step2_capenc
#SBATCH -p normal
#SBATCH -A ASC22049
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 48:00:00
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/logs/%x.%j.out
# usage: sbatch step2_capenc.sbatch <shard k/n> [workers] [threads]
# encodes rendered 2c intermediates of its shard as they appear; exits when all 9 render shards are done
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step2; cd $R/method
echo "node $(hostname) shard $1 start $(date -Is)"
python step2_capture.py encode --manifest $R/vqa_data/step1/manifest.csv --out $O --shard ${1:-0/1} \
   --workers ${2:-16} --threads ${3:-4} --loop --render-shards 9
echo "done $(date -Is)"
