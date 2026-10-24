#!/bin/bash
# Step 15 study eval (user 10-23: "also study (with base lines table) to decide"): zero-shot, logged in study_eval_log.csv.
#   v4  new inference: v4-encoder study tokens -> score -> Protocol B metrics, extra, Protocol A matched folds
#   v3  no new inference: eval-5 scores / pooled features re-scored against the current GT (mos_visibility_per_video.csv)
#   GPU=0 WORKERS=24 setsid nohup bash step15_study.sh > $S/logs/study.out 2>&1 < /dev/null &
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
ROOT=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; S=$ROOT/step15_v4; R=$S/runs/v4; TR=$S/tokens_v4; L=$S/logs
G=${GPU:-0}; W=${WORKERS:-24}
echo "$(date -Is) start on $(hostname) gpu$G"
o=$S/eval/v3/study; mkdir -p $o
[ -f $o/scores.csv ] || cp $ROOT/step9_e2e/eval/v3_e2e/study/{scores.csv,pooled_features.npz} $o/
[ -f $o/metrics.json ] || { python step7_metrics.py --dataset study --out $o > $o.metrics.log 2>&1; python step7_extra.py --out $o >> $o.metrics.log 2>&1; }
echo "$(date -Is) v3 rescored"
o=$S/eval/v4/study
if [ ! -f $o/metrics.json ]; then
  python step7_eval.py extract --dataset study --gpu $G --workers $W --s1 $R/s1.pt --s2 $R/s2.pt --tok-root $TR >> $L/extract_v4_study.log 2>&1
  python step7_eval.py score --dataset study --fusion $R/fusion.pt --tok-root $TR --out $o >> $L/score_v4.log 2>&1
  python step7_metrics.py --dataset study --out $o > $o.metrics.log 2>&1
  python step7_extra.py --out $o >> $o.metrics.log 2>&1
fi
echo "$(date -Is) v4 scored"
[ -f $ROOT/step10_protA/v4/summary.json ] || python step10_protA.py --model v4=$o/pooled_features.npz --out $ROOT/step10_protA/v4 > $L/study_protA_v4.log 2>&1
echo "$(date -Is) STEP15_STUDY_DONE"
