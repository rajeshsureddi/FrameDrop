#!/bin/bash
#SBATCH -J step1_smoke
#SBATCH -p development
#SBATCH -A ASC22049
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 01:00:00
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step1/logs/%x.%j.out
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step1; cd $R/method
echo "node $(hostname) cores $(nproc) affinity $(taskset -pc $$)"; lscpu | grep 'Model name'
for cfg in "128 1" "32 4"; do set -- $cfg
  s=$(date +%s)
  python build_manifest.py --data LMM-10K=$O/smoke_inputs/LMM-10K --data Inter4K=$O/smoke_inputs/Inter4K \
    --exclude $O/exclude_study_links --out $O/smoke_w$1_t$2 --workers $1 --decode-threads $2 --segment 5
  echo "CONFIG workers=$1 threads=$2 wall=$(( $(date +%s) - s ))s"
done
s=$(date +%s); python build_manifest.py --data LMM-10K=$O/smoke_inputs/LMM-10K --data Inter4K=$O/smoke_inputs/Inter4K \
    --exclude $O/exclude_study_links --out $O/smoke_w128_t1 --workers 128 --decode-threads 1 --segment 5 > /dev/null
echo "RERUN cached wall=$(( $(date +%s) - s ))s"
cmp $O/smoke_w128_t1/manifest.csv $O/smoke_w32_t4/manifest.csv && echo "manifests identical across configs"
