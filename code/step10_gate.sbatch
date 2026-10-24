#!/bin/bash
#SBATCH -J dd_gate
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 12:00:00
#SBATCH -A ASC22049
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/logs/%x_%j.out
# 10a gate: v1 / v2 / v3 (frozen backups, read only) on DropDev. Evaluation only; no study data.
source /work/09997/rs64952/ls6/miniconda3/etc/profile.d/conda.sh
conda activate /work/09997/rs64952/ls6/miniconda3/envs/vlm-r1
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; V1=$R/frozen_v1_20261017; V2=$R/frozen_v2_20261017; V3=$R/frozen_v3_20261018
L=$R/step10_dropdev/logs
[ -f $R/step10_dropdev/pairs.csv ] || { echo "no pairs.csv"; exit 1; }
ext() {  # name s1 s2
  for g in 0 1 2; do
    python step10_eval.py extract --name $1 --s1 $2 --s2 $3 --gpu $g --part $g --parts 3 --workers 14 >> $L/extract_$1_g$g.log 2>&1 &
  done
  wait
}
ext v1enc $V1/stage1_s_mae_final_avg4.pt $V1/stage2_v_mae_final_avg4.pt
ext v3enc $V3/s1.pt $V3/s2.pt
python step10_eval.py score --name v1 --tok v1enc --fusion $V1/stage3_fusion_best.pt
python step10_eval.py score --name v2 --tok v1enc --fusion $V2/stage3_fusion_v2_mos_pairs_final.pt
python step10_eval.py score --name v2_mos_only --tok v1enc --fusion $V2/stage3_fusion_v2_mos_only_final.pt
python step10_eval.py score --name v3 --tok v3enc --fusion $V3/fusion.pt
for m in v1 v2 v2_mos_only v3; do python step10_eval.py metrics --name $m; done
python step10_eval.py compare --of v2 v3
echo GATE_DONE $(date)
# 10a rule (fixed in step10_eval.py before any DropDev number was seen): only if DropDev reproduces v2 > v3
# (Q, 95% paired bootstrap CI > 0) submit 10c: v2_90, then 5 end-to-end runs on two 3-GPU nodes.
if python -c "import json,sys; sys.exit(0 if json.load(open('$R/step10_dropdev/eval/compare_v2_vs_v3.json'))['gap_reproduced_Q'] else 1)"; then
  M=/scratch/09997/rs64952/FrameDrop-VQA/method
  ssh -o BatchMode=yes login1.ls6.tacc.utexas.edu "cd $M; J=\$(sbatch --parsable step10_v2_90.sbatch 2>/dev/null | tail -1); \
    sbatch --dependency=afterok:\$J step10_cands.sbatch 'v3_90' 'c1 --pair-every 2' 'c2 --pair-every 2 --drop-weight 3'; \
    sbatch --dependency=afterok:\$J step10_cands.sbatch 'c3 --freeze-motion' 'c4 --consist-weight 1'; echo V2_90 \$J" 2>&1 | grep -E "V2_90|Submitted"
  echo "10c SUBMITTED $(date)"
else
  echo "GAP NOT REPRODUCED: stop (benchmark needs fixing before it can select)"
fi
# A100 end-to-end speed with the benchmark package (non-study 4K60 clips; v3 weights; timing is weight-independent)
cd step10_speedbench
mkdir -p $R/step10_dropdev/speed
python -I bench.py --gpu 0 --weights $V3 --workers 16 --threads 8 --out $R/step10_dropdev/speed/speed_A100_v3.json > $L/speed_A100.log 2>&1
echo SPEED_DONE $(date)
