#!/bin/bash
# 10-20 (user: "add LIVE-VQC evaluation too"): LIVE-VQC (585 clips, eval only, native res/fps) for v2, v3, d1, v3_90.
# Waits for the videos (mos_datasets/LIVE-VQC/Video/*.mp4, official LIVE release) and for v3_90. Cross-dataset protocol
# of Table 1 (arXiv 2609.16946): LSVQ-trained models tested as-is; reported only, never used for selection.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; C=$R/step10_cand; O=$R/step12/eval; L=$R/step12/logs; mkdir -p $L
V1=$R/frozen_v1_20261017; V2=$R/frozen_v2_20261017; V3=$R/frozen_v3_20261018
VID=/scratch/09997/rs64952/FrameDrop-VQA/mos_datasets/LIVE-VQC/Video
until [ "$(ls $VID 2>/dev/null | grep -c 'mp4$')" -ge 585 ]; do sleep 120; done
until grep -q V3_90_DONE $C/logs/v3_90_ddp.out 2>/dev/null; do sleep 120; done
run() {   # name gpu s1 s2 fusion tok_root
  python step7_eval.py extract --dataset livevqc --gpu $2 --workers 24 --s1 $3 --s2 $4 --tok-root $6 &&
  python step7_eval.py score --dataset livevqc --fusion $5 --tok-root $6 --out $O/$1/livevqc --gpu $2 &&
  python step7_metrics.py --dataset livevqc --out $O/$1/livevqc > $O/$1/livevqc.metrics.log 2>&1
}
mkdir -p $O/v2 $O/v3 $O/d1 $O/v3_90
( run v2 0 $V1/stage1_s_mae_final_avg4.pt $V1/stage2_v_mae_final_avg4.pt $V2/stage3_fusion_v2_mos_pairs_final.pt $C/tokens_v1enc
  run v3_90 0 $C/runs/v3_90/s1.pt $C/runs/v3_90/s2.pt $C/runs/v3_90/fusion.pt $C/tokens_v3_90 ) > $L/livevqc_g0.log 2>&1 &
run v3 1 $V3/s1.pt $V3/s2.pt $V3/fusion.pt $R/step9_e2e/tokens_v3_e2e > $L/livevqc_g1.log 2>&1 &
run d1 2 $C/runs/d1/s1.pt $C/runs/d1/s2.pt $C/runs/d1/fusion.pt $C/tokens_d1 > $L/livevqc_g2.log 2>&1 &
wait
echo "LIVEVQC_DONE $(date -Is)"
