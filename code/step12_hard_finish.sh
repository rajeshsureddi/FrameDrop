#!/bin/bash
# 10-21: DropDev-hard tail. pairs.csv and v1enc / v3enc / d1enc tokens are complete (35,559 clips each).
# Extract v3_90enc over 5 GPUs, then score v1 / v2 / v2_mos_only / v3 / d1 / v2_90 / v3_90, metrics,
# and the pre-fixed v2 vs v3 comparison. Evaluation only; no study data.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 DROPDEV=hard
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; L=$R/step10_dropdev_hard/logs; C=$R/step10_cand
V1=$R/frozen_v1_20261017; V2=$R/frozen_v2_20261017; V3=$R/frozen_v3_20261018; D1=$C/runs/d1; E=$C/runs/v3_90
GPUS="c317-012:0:16 c317-012:1:16 c317-012:2:16 v330-021:0:12 v330-007:0:12"
spec=($GPUS); k=0
for w in "${spec[@]}"; do
  IFS=: read h g c <<< "$w"
  ssh -o BatchMode=yes $h "cd $M; export PATH=$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 DROPDEV=hard; \
    python step10_eval.py extract --name v3_90enc --s1 $E/s1.pt --s2 $E/s2.pt --gpu $g --part $k --parts ${#spec[@]} --workers $c" \
    >> $L/extract_v3_90enc_w$k.log 2>&1 &
  k=$((k + 1))
done
python step10_eval.py score --name v1 --tok v1enc --fusion $V1/stage3_fusion_best.pt --gpu 0 --workers 8
python step10_eval.py score --name v2 --tok v1enc --fusion $V2/stage3_fusion_v2_mos_pairs_final.pt --gpu 1 --workers 8
python step10_eval.py score --name v2_mos_only --tok v1enc --fusion $V2/stage3_fusion_v2_mos_only_final.pt --gpu 2 --workers 8
python step10_eval.py score --name v3 --tok v3enc --fusion $V3/fusion.pt --gpu 0 --workers 8
python step10_eval.py score --name d1 --tok d1enc --fusion $D1/fusion.pt --gpu 1 --workers 8
python step10_eval.py score --name v2_90 --tok v1enc --fusion $R/step8_lsvq/runs/v2_90/final.pt --gpu 2 --workers 8
wait
python step10_eval.py score --name v3_90 --tok v3_90enc --fusion $E/fusion.pt --gpu 0 --workers 12
for m in v1 v2 v2_mos_only v3 d1 v2_90 v3_90; do python step10_eval.py metrics --name $m; done
python step10_eval.py compare --of v2 v3
echo "HARD_DONE $(date -Is)"
