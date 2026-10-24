#!/bin/bash
# Step 15 eval driver (resumable, one GPU): v4 on the sets it never trained on. No study data.
#   held-out MOS  ytugc split0 test, LIVE-YT-Gaming val            (v3 and v4, same code: step15_v4.py evalmos)
#   Protocol 1    lsvq_test, lsvq_1080p, konvid1k, livevqc          (v4 tokens -> score -> step7_metrics, as v3)
#   DropDev       base + hard                                       (step10_eval extract/score/metrics, compare v3 v4)
#   GPU=0 WORKERS=24 setsid nohup bash step15_eval.sh > $S/logs/eval.out 2>&1 < /dev/null &
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
ROOT=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; S=$ROOT/step15_v4; R=$S/runs/v4; TR=$S/tokens_v4; L=$S/logs
V3=$ROOT/frozen_v3_20261018; G=${GPU:-0}; W=${WORKERS:-24}; mkdir -p $L
echo "$(date -Is) start on $(hostname) gpu$G"
for m in v4 v3; do
  [ -f $S/eval/$m/mos_heldout.json ] || python step15_v4.py evalmos --name $m --gpu $G --val-workers $W >> $L/evalmos_$m.log 2>&1
  echo "$(date -Is) evalmos $m: $(python -c "import json;r=json.load(open('$S/eval/$m/mos_heldout.json'));print({k:round(r[k]['SROCC'],3) for k in ('ytugc_test','gaming_val')})" 2>&1)"
done
for ds in lsvq_test lsvq_1080p konvid1k livevqc; do
  o=$S/eval/v4/$ds
  [ -f $o/metrics.json ] && continue
  python step7_eval.py extract --dataset $ds --gpu $G --workers $W --s1 $R/s1.pt --s2 $R/s2.pt --tok-root $TR >> $L/extract_v4.log 2>&1
  python step7_eval.py score --dataset $ds --fusion $R/fusion.pt --tok-root $TR --out $o >> $L/score_v4.log 2>&1
  python step7_metrics.py --dataset $ds --out $o > $o.metrics.log 2>&1
  echo "$(date -Is) v4 $ds done"
done
o=$S/eval/v3/livevqc                                   # v3 has LIVE-VQC tokens (step9) but no Protocol 1 row on disk
[ -f $o/metrics.json ] || { python step7_eval.py score --dataset livevqc --fusion $V3/fusion.pt --tok-root $ROOT/step9_e2e/tokens_v3_e2e --out $o >> $L/score_v3.log 2>&1
  python step7_metrics.py --dataset livevqc --out $o > $o.metrics.log 2>&1; }
for dd in base hard; do
  export DROPDEV=$dd
  python step10_eval.py extract --name v4enc --s1 $R/s1.pt --s2 $R/s2.pt --gpu $G --workers $W >> $L/dropdev_${dd}.log 2>&1
  python step10_eval.py score --name v4 --tok v4enc --fusion $R/fusion.pt --gpu $G --workers 12 >> $L/dropdev_${dd}.log 2>&1
  python step10_eval.py metrics --name v4 >> $L/dropdev_${dd}.log 2>&1
  python step10_eval.py compare --of v3 v4 >> $L/dropdev_${dd}.log 2>&1
  echo "$(date -Is) dropdev $dd done"
done
echo "$(date -Is) STEP15_EVAL_DONE"
