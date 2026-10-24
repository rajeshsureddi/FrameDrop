#!/bin/bash
# Step 8 driver (resumable; rerun the same command after an interruption):
#   1. finish LSVQ token extraction on this node (waits for other nodes' extract jobs to stop first)
#   2. fine-tune v2 (MOS + synthetic pairs, main) and the MOS-only ablation, on all 28,056 LSVQ train clips
#   3. score v1 / v2 / ablation on LSVQ test_1080p, LSVQ test, KoNViD-1k, study; metrics
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data
S8=$R/step8_lsvq; L=$S8/logs; mkdir -p $L
WORKERS=${WORKERS:-28}
echo "start $(date)"
# 1. extraction
while pgrep -f "step8_extract[.]sh" > /dev/null; do sleep 60; done                  # this node's earlier round
while ssh -o ConnectTimeout=10 c316-013 'pgrep -f "step8_extract[.]sh"' > /dev/null 2>&1; do sleep 60; done
for ds in lsvq_1080p lsvq_test lsvq_train; do
  python step7_eval.py extract --dataset $ds --gpu 0 --workers $WORKERS >> $L/extract_final.log 2>&1
done
for ds in lsvq_1080p lsvq_test lsvq_train; do
  echo "$ds tokens $(ls $S8/$ds/tokens | grep -c 'npz$') errors $(ls $S8/$ds/tokens | grep -c 'err$')"
done
echo "extraction done $(date)"
# 2. training (both on GPU 0, concurrently)
run() {  # name, args...
  n=$1; shift
  [ -f $S8/runs/$n/final.pt ] && return
  python step8_finetune.py --name $n --gpu 0 "$@" >> $L/train_$n.log 2>&1
}
run v2_mos_pairs --steps 20000 --pair-every 2 --pool 8 --workers 8 &
run v2_mos_only --steps 10000 --pair-every 0 --workers 8 &              # same number of MOS batches
wait
echo "training done $(date)"
# 3. evaluation
declare -A FUS=([v1]=$R/frozen_v1_20261017/stage3_fusion_best.pt
                [v2_mos_pairs]=$S8/runs/v2_mos_pairs/final.pt [v2_mos_only]=$S8/runs/v2_mos_only/final.pt)
for m in v1 v2_mos_pairs v2_mos_only; do
  for ds in lsvq_1080p lsvq_test konvid1k study; do
    o=$S8/eval/$m/$ds
    [ -f $o/metrics.json ] && continue
    python step7_eval.py score --dataset $ds --fusion ${FUS[$m]} --out $o >> $L/eval.log 2>&1
    python step7_metrics.py --dataset $ds --out $o > $o.metrics.log 2>&1
  done
  python step7_extra.py --out $S8/eval/$m/study >> $L/eval.log 2>&1
done
echo "STEP8_DONE $(date)"
