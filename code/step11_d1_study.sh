#!/bin/bash
# 10d study evaluation of d1 (user 10-20: "evaluate on all datasets including FrameDropVQA"). Single run: refuses to
# run again once the output exists. Same pipeline as the v1-v3 study evals: tokens (d1 S1/S2) -> score (d1 fusion)
# -> step7_metrics (mos_visibility_per_video.csv, visibility metrics fixed pre-10d) -> Protocol A on the baselines' 20 splits.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; C=$R/step10_cand; D=$C/runs/d1; TR=$C/tokens_d1; O=$C/eval/d1/study; L=$C/logs
[ -e $O/scores.csv ] && { echo "d1 study already evaluated: $O"; exit 1; }
for g in 0 1 2; do
  python step7_eval.py extract --dataset study --gpu $g --part $g --parts 3 --workers 14 --s1 $D/s1.pt --s2 $D/s2.pt --tok-root $TR > $L/study_extract_g$g.log 2>&1 &
done
wait
echo "=== study tokens $(date -Is)"
python step7_eval.py score --dataset study --fusion $D/fusion.pt --tok-root $TR --out $O --gpu 0
python step7_metrics.py --dataset study --out $O > $O.metrics.log 2>&1
python step10_protA.py --model d1=$O/pooled_features.npz --out $R/step10_protA/d1 > $L/study_protA_d1.log 2>&1
echo "D1_STUDY_DONE $(date -Is)"
