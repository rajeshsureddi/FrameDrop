#!/bin/bash
# One Step 10 model on one GPU (resumable): train -> export -> DropDev tokens/score/metrics -> LSVQ val10 tokens/score/metrics.
#   step10_cand_run.sh <name> <gpu> [train args...]       name = v3_90 | c1 | c2 | c3 | c4
#   step10_cand_run.sh v2_90 <gpu>                        fusion-only v2 recipe on train90 (step8_finetune)
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
N=$1; G=$2; shift 2
if [ $N = d1 ]; then   # Step 11 d1 (10-19): extraction over all idev GPUs; report LSVQ test / 1080p / KoNViD (not the study)
  FANOUT=${FANOUT:-"c317-012:0:30 c317-012:1:30 c317-012:2:30 c306-001:0:36 c306-001:1:36 c306-001:2:36 v330-021:0:26 v330-007:0:26"}
  REPORT_SETS=${REPORT_SETS:-"lsvq_test lsvq_1080p konvid1k"}
fi
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; C=$R/step10_cand; L=$C/logs; mkdir -p $L
echo "$N start $(date)"
if [ $N = v2_90 ]; then
  F=$R/step8_lsvq/runs/v2_90/final.pt
  [ -f $F ] || python step8_finetune.py --name v2_90 --holdout 2806 --gpu $G --steps 20000 --pair-every 2 --pool 8 --workers 8 >> $L/train_v2_90.log 2>&1
  [ -f $F ] || { echo "TRAIN FAILED $N"; exit 1; }
  TOK=v1enc; TR=$C/tokens_v1enc; V1=$R/frozen_v1_20261017
  python step7_eval.py extract --dataset lsvq_val10 --gpu $G --workers 12 --s1 $V1/stage1_s_mae_final_avg4.pt \
    --s2 $V1/stage2_v_mae_final_avg4.pt --tok-root $TR >> $L/extract_val_v2_90.log 2>&1
else
  D=$C/runs/$N; F=$D/fusion.pt
  [ -f $D/final.pt ] || python step10_train.py train --name $N --gpu $G "$@" >> $L/train_$N.log 2>&1
  [ -f $D/final.pt ] || { echo "TRAIN FAILED $N"; exit 1; }
  [ -f $F ] || python step10_train.py export --name $N >> $L/train_$N.log 2>&1
  TOK=${N}enc; TR=$C/tokens_$N
  if [ -n "${FANOUT:-}" ]; then   # "host:gpu:workers ..." -> extraction split over these GPUs (d1, 10-19)
    spec=($FANOUT); k=0
    for w in "${spec[@]}"; do
      IFS=: read h g c <<< "$w"
      ssh -o BatchMode=yes $h "cd $PWD; export PATH=$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1; \
        python step10_eval.py extract --name $TOK --s1 $D/s1.pt --s2 $D/s2.pt --gpu $g --part $k --parts ${#spec[@]} --workers $c; \
        for ds in lsvq_val10 $REPORT_SETS; do python step7_eval.py extract --dataset \$ds --gpu $g --part $k --parts ${#spec[@]} \
          --workers $c --s1 $D/s1.pt --s2 $D/s2.pt --tok-root $TR; done" > $L/extract_${N}_w$k.log 2>&1 &
      k=$((k + 1))
    done
    wait
  else
    python step10_eval.py extract --name $TOK --s1 $D/s1.pt --s2 $D/s2.pt --gpu $G --workers 12 >> $L/extract_dd_$N.log 2>&1
    python step7_eval.py extract --dataset lsvq_val10 --gpu $G --workers 12 --s1 $D/s1.pt --s2 $D/s2.pt --tok-root $TR >> $L/extract_val_$N.log 2>&1
  fi
fi
echo "$N trained $(date)"
python step10_eval.py score --name $N --tok $TOK --fusion $F --gpu $G >> $L/eval_$N.log 2>&1
python step10_eval.py metrics --name $N >> $L/eval_$N.log 2>&1
o=$C/eval/$N/lsvq_val10
python step7_eval.py score --dataset lsvq_val10 --fusion $F --tok-root $TR --out $o --gpu $G >> $L/eval_$N.log 2>&1
python step7_metrics.py --dataset lsvq_val10 --out $o > $o.metrics.log 2>&1
for ds in ${REPORT_SETS:-}; do   # test sets: reported, never used for selection
  o=$C/eval/$N/$ds
  python step7_eval.py score --dataset $ds --fusion $F --tok-root $TR --out $o --gpu $G >> $L/eval_$N.log 2>&1
  python step7_metrics.py --dataset $ds --out $o > $o.metrics.log 2>&1
done
echo "$N DONE $(date)"
