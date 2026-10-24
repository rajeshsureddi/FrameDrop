#!/bin/bash
# Step 11 candidate d1: Stage 1 initialised from DINOv2-S (patch 14, 252 centre crop of the 256 tiles),
# then the v1 -> v2 -> v3 chain with the same recipes, all on LSVQ train90; Stage 2 = v1 motion encoder.
#   1. wait for step4/runs/s_dino/final_avg4.pt   (Tile-MAE + L_rank, 50k steps, started 10-19 14:16)
#   2. tokens with d1 S1 + v1 S2, on 8 GPUs (WORKERS_SPEC):
#      Step 6 synthetic pairs (same shards / seeds as v1: valid 0-3 x 500, train 0-99 x 600)
#      LSVQ train (step7_eval --tok-root; train90 / val10 split is applied by step8 --holdout 2806)
#   3. fusion on the pairs (v1 recipe, 30k steps, best.pt on synthetic valid pairs as in v1)
#   4. v2 recipe on train90 (step8_finetune --holdout 2806, 20k steps, final.pt)
#   5. end-to-end v3 recipe on train90 + DropDev / LSVQ val10 eval (step10_cand_run.sh d1)
# Evaluation only on DropDev and LSVQ val10 (10b rule); no study, KoNViD or LSVQ test here.
# Tokens run on all GPUs of the 4 idev nodes (WORKERS_SPEC). Every step skips finished work, so the script can be re-run after a crash.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; D=$R/step11_dino; L=$D/logs; mkdir -p $L
S1=$R/step4/runs/s_dino/final_avg4.pt
S2=$R/frozen_v1_20261017/stage2_v_mae_final_avg4.pt
# token workers "host:gpu:cpu_workers"; Step 6 shards are claimed with mkdir (atomic on the shared fs), LSVQ train split statically
WORKERS_SPEC=${WORKERS_SPEC:-"c317-012:0:30 c317-012:1:30 c317-012:2:30 c306-001:0:36 c306-001:1:36 c306-001:2:36 v330-021:0:26 v330-007:0:26"}

while [ ! -f $S1 ]; do sleep 60; done
echo "=== d1 S1 final $(date -Is)"

worker() {   # $1 = worker id, $2 = gpu, $3 = cpu workers, $4 = number of workers
  mkdir -p $D/step6/claims
  for job in valid:0 valid:1 valid:2 valid:3 $(seq -f "train:%g" 0 99); do
    sp=${job%%:*}; s=${job#*:}
    mkdir $D/step6/claims/${sp}_$s 2>/dev/null || continue
    n=600; [ $sp = valid ] && n=500
    python step6_features.py pairs --split $sp --n $n --shard $s --gpu $2 --workers $3 --s1 $S1 --s2 $S2 --out $D/step6 \
      || rmdir $D/step6/claims/${sp}_$s
  done
  python step7_eval.py extract --dataset lsvq_train --gpu $2 --part $1 --parts $4 --workers $3 --s1 $S1 --s2 $S2 --tok-root $D/tok
}
if [ "$(ls $D/step6/train/pairs_*.npz 2>/dev/null | grep -vc tmp)" -lt 100 ] || [ ! -f $D/tok/.lsvq_done ]; then
  rm -rf $D/step6/claims                           # stale claims of a crashed run (finished shards are skipped)
  spec=($WORKERS_SPEC); k=0
  for w in "${spec[@]}"; do
    IFS=: read h g c <<< "$w"
    ssh -o BatchMode=yes $h "$(declare -p M R D L S1 S2); export PATH=$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1; cd $M; \
      $(declare -f worker); worker $k $g $c ${#spec[@]}" > $L/tokens_w$k.log 2>&1 &
    k=$((k + 1))
  done
  wait
  n=$(ls $D/step6/train/pairs_*.npz | grep -vc tmp)
  echo "=== tokens done $(date -Is): $n train shards, $(ls $D/tok/lsvq_train/tokens | grep -c npz) LSVQ train clips"
  [ $n -ge 100 ] || { echo "MISSING step6 shards"; exit 1; }
  touch $D/tok/.lsvq_done
fi

F3=$D/step6/runs/fusion/best.pt
if [ ! -f $D/step6/runs/fusion/.done ]; then      # step6_train has no resume: a crash restarts the 30k steps
  rm -f $D/step6/runs/fusion/train_log.jsonl
  python step6_train.py --name fusion --root $D/step6 --gpu 0 > $L/fusion.log 2>&1 && touch $D/step6/runs/fusion/.done
fi
[ -f $F3 ] || { echo "FUSION FAILED"; exit 1; }
echo "=== fusion done $(date -Is)"

F8=$D/step8/runs/d1_v2_90/final.pt
[ -f $F8 ] || python step8_finetune.py --name d1_v2_90 --holdout 2806 --gpu 0 --steps 20000 --pair-every 2 --pool 8 --workers 8 \
  --init $F3 --out $D/step8 --pairs-root $D/step6 --tok-root $D/tok > $L/v2_90.log 2>&1
[ -f $F8 ] || { echo "V2_90 FAILED"; exit 1; }
echo "=== d1 v2_90 done $(date -Is)"

$M/step10_cand_run.sh d1 0 --init-s1 $S1 --init-fusion $F8
echo "=== D1_FINISHED $(date -Is)"
