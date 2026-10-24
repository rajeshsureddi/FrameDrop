#!/bin/bash
# DropDev-hard (10-19) on the idev nodes, after the d1 token stage has released them.
#   step10h_run.sh encode <workers> <threads>     every node: re-encodes (per-clip locks, shared with ddh_enc batch jobs)
#   step10h_run.sh gate                            c306-001: wait for all encodes -> measure -> collect -> v1enc / v3enc
#                                                  tokens on GPUS -> score v1 / v2 / v2_mos_only / v3 -> metrics -> compare
# Same metric and fixed 10a rule as the base DropDev (step10_eval.py); no study data; no candidate is submitted here.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 DROPDEV=hard
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; H=$R/step10_dropdev_hard; L=$H/logs; mkdir -p $L
V1=$R/frozen_v1_20261017; V2=$R/frozen_v2_20261017; V3=$R/frozen_v3_20261018
GPUS=${GPUS:-"c306-001:0:30 c306-001:1:30 c306-001:2:30 v330-021:0:24 v330-007:0:24"}

while [ ! -f $R/step11_dino/tok/.lsvq_done ]; do sleep 60; done     # d1 tokens first

if [ $1 = encode ]; then
  python step10_dropdev.py encode --workers $2 --threads $3
  exit
fi

# gate: all 17,781 re-encodes ok?
n_ok() { python -c "
import json, glob
print(sum(json.load(open(f)).get('ok', False) for f in glob.glob('$H/clips/*.json')))"; }
until [ "$(n_ok)" -ge 17781 ]; do sleep 300; done
echo "=== encodes done $(date -Is)"
until [ "$(ls $H/measure | grep -c 'json$')" -ge 2223 ]; do sleep 120; done   # 10-20: sharded measure (c306-001 died)
[ -f $H/pairs.csv ] || { python step10_dropdev.py measure --workers 100 && python step10_dropdev.py collect; }
ext() {   # name s1 s2
  spec=($GPUS); k=0
  for w in "${spec[@]}"; do
    IFS=: read h g c <<< "$w"
    ssh -o BatchMode=yes $h "cd $M; export PATH=$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 DROPDEV=hard; \
      python step10_eval.py extract --name $1 --s1 $2 --s2 $3 --gpu $g --part $k --parts ${#spec[@]} --workers $c" \
      >> $L/extract_$1_w$k.log 2>&1 &
    k=$((k + 1))
  done
  wait
}
ext v1enc $V1/stage1_s_mae_final_avg4.pt $V1/stage2_v_mae_final_avg4.pt
ext v3enc $V3/s1.pt $V3/s2.pt
D1=$R/step10_cand/runs/d1                                            # Step 11 candidate (10-20), reported next to v1-v3
ext d1enc $D1/s1.pt $D1/s2.pt
python step10_eval.py score --name v1 --tok v1enc --fusion $V1/stage3_fusion_best.pt --gpu 0 --workers 12 &
python step10_eval.py score --name v2 --tok v1enc --fusion $V2/stage3_fusion_v2_mos_pairs_final.pt --gpu 1 --workers 12 &
python step10_eval.py score --name v2_mos_only --tok v1enc --fusion $V2/stage3_fusion_v2_mos_only_final.pt --gpu 2 --workers 12 &
python step10_eval.py score --name v3 --tok v3enc --fusion $V3/fusion.pt --gpu 0 --workers 12 &
python step10_eval.py score --name d1 --tok d1enc --fusion $D1/fusion.pt --gpu 1 --workers 12 &
python step10_eval.py score --name v2_90 --tok v1enc --fusion $R/step8_lsvq/runs/v2_90/final.pt --gpu 2 --workers 12 &
wait
for m in v1 v2 v2_mos_only v3 d1 v2_90; do python step10_eval.py metrics --name $m; done
python step10_eval.py compare --of v2 v3
echo "GATE_HARD_DONE $(date -Is)"
