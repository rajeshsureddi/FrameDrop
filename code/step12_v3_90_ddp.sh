#!/bin/bash
# 10-20 (user: "use all idle GPUs to speed up v3_90"): v3_90 resumed from last.pt (step 1000) on 4 GPUs,
# gradient-equivalent to the 1-GPU recipe (step10_train.py --world 4; global batch 8 clips / 4 pairs unchanged).
# Ranks 0-2 on c317-012 GPUs 0-2, rank 3 on v330-007 GPU 0. Then step10_cand_run.sh v3_90 (skips training) exports and
# evaluates (DropDev + LSVQ val10 + report sets) over c317-012's 3 GPUs.
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
ENV="export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:\$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1"
eval $ENV
C=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_cand; L=$C/logs
URL=tcp://c317-012:29611; ARGS="train --name v3_90 --world 4 --dist-url $URL --workers 6 --pair-workers 3"
for r in 0 1 2; do python step10_train.py $ARGS --rank $r --gpu $r >> $L/train_v3_90_r$r.log 2>&1 & done
ssh -o BatchMode=yes v330-007 "cd $M; $ENV; python step10_train.py $ARGS --rank 3 --gpu 0" >> $L/train_v3_90_r3.log 2>&1 &
wait
[ -f $C/runs/v3_90/final.pt ] || { echo "DDP TRAIN FAILED $(date -Is)"; exit 1; }
echo "=== v3_90 trained $(date -Is)"
FANOUT="c317-012:0:30 c317-012:1:30 c317-012:2:30" REPORT_SETS="lsvq_test lsvq_1080p konvid1k" $M/step10_cand_run.sh v3_90 0
echo "V3_90_DONE $(date -Is)"
