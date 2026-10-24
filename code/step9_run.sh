#!/bin/bash
# Step 9 driver (resumable): valid pairs -> end-to-end training (v3) -> export -> v3 tokens -> score -> metrics.
# Writes only to vqa_data/step9_e2e (v1/v2 frozen dirs and their token dirs are never touched).
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0
S9=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step9_e2e; L=$S9/logs; mkdir -p $L
N=${NAME:-v3_e2e}; R=$S9/runs/$N
echo "start $(date)"
[ $(ls $S9/valid_pairs/pair_*.npz 2>/dev/null | wc -l) -ge 150 ] || python step9_e2e.py make-valid --tiles 16 --workers 28 >> $L/make_valid.log 2>&1
echo "valid pairs $(ls $S9/valid_pairs/pair_*.npz | wc -l) $(date)"
[ -f $R/final.pt ] || python step9_e2e.py train --name $N $TRAIN_ARGS >> $L/train_$N.log 2>&1
[ -f $R/final.pt ] || { echo "TRAIN FAILED $(date)"; exit 1; }
echo "training done $(date)"
[ -f $R/fusion.pt ] || python step9_e2e.py export --name $N >> $L/train_$N.log 2>&1
TR=$S9/tokens_$N
for ds in lsvq_1080p lsvq_test konvid1k study; do
  python step7_eval.py extract --dataset $ds --gpu 0 --workers 28 --s1 $R/s1.pt --s2 $R/s2.pt --tok-root $TR >> $L/extract_$N.log 2>&1
  o=$S9/eval/$N/$ds
  [ -f $o/metrics.json ] && continue
  python step7_eval.py score --dataset $ds --fusion $R/fusion.pt --tok-root $TR --out $o >> $L/eval_$N.log 2>&1
  python step7_metrics.py --dataset $ds --out $o > $o.metrics.log 2>&1
done
python step7_extra.py --out $S9/eval/$N/study >> $L/eval_$N.log 2>&1
echo "STEP9_DONE $(date)"
