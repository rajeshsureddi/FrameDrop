#!/bin/bash
# Step 12 study eval (eval 9): the selected ens(d1, v3_90) (SELECTION_RULE.md). Single run: refuses once the output exists.
# v3_90 is scored on the study as an ensemble member only (no v3_90-alone metrics). Ensemble Q/V = mean of member z-scores
# (step12_ensemble.py); Protocol A features = concatenation of the members' pooled features.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; C=$R/step10_cand; E=$C/runs/v3_90; TR=$C/tokens_v3_90; L=$R/step12/logs
MEM=$R/step12/members/v3_90/study; O=$R/step12/eval/ens_d1_v3_90/study
[ -e $O/scores.csv ] && { echo "already evaluated: $O"; exit 1; }
for g in 0 1 2; do
  python step7_eval.py extract --dataset study --gpu $g --part $g --parts 3 --workers 14 --s1 $E/s1.pt --s2 $E/s2.pt --tok-root $TR > $L/study_v3_90_extract_g$g.log 2>&1 &
done
wait
python step7_eval.py score --dataset study --fusion $E/fusion.pt --tok-root $TR --out $MEM --gpu 0 > $L/study_v3_90_score.log 2>&1
python step12_ensemble.py --members $C/eval/d1/study $MEM --out $O
python - <<PY
import numpy as np
a, b = np.load("$C/eval/d1/study/pooled_features.npz"), np.load("$MEM/pooled_features.npz")
fb = dict(zip(b["clips"].tolist(), b["x"]))
np.savez("$O/pooled_features.npz", clips=a["clips"], x=np.concatenate([a["x"], np.stack([fb[c] for c in a["clips"].tolist()])], 1))
PY
python step7_metrics.py --dataset study --out $O > $O.metrics.log 2>&1
python step10_protA.py --model ens_d1_v3_90=$O/pooled_features.npz --out $R/step10_protA/ens_d1_v3_90 > $L/study_protA_ens.log 2>&1
echo "ENS_STUDY_DONE $(date -Is)"
