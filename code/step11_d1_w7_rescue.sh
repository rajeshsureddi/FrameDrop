#!/bin/bash
# 10-20: d1 eval worker 7 (v330-007) hung at 400/4271 after /dev/shm filled (DropDev-hard encodes share the node).
# Redo part 7/8 on c317-012 (finished clips are skipped), then kill the hung worker so step10_cand_run.sh proceeds to scoring.
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
C=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_cand; D=$C/runs/d1; TR=$C/tokens_d1; L=$C/logs
python step10_eval.py extract --name d1enc --s1 $D/s1.pt --s2 $D/s2.pt --gpu 0 --part 7 --parts 8 --workers 30 > $L/rescue_w7_dd.log 2>&1 &
( for ds in lsvq_val10 lsvq_test; do python step7_eval.py extract --dataset $ds --gpu 1 --part 7 --parts 8 --workers 30 --s1 $D/s1.pt --s2 $D/s2.pt --tok-root $TR; done ) > $L/rescue_w7_a.log 2>&1 &
( for ds in lsvq_1080p konvid1k; do python step7_eval.py extract --dataset $ds --gpu 2 --part 7 --parts 8 --workers 30 --s1 $D/s1.pt --s2 $D/s2.pt --tok-root $TR; done ) > $L/rescue_w7_b.log 2>&1 &
wait
echo "RESCUE_EXTRACT_DONE $(date -Is)"
ssh -o BatchMode=yes v330-007 "kill 2941440; sleep 2; kill 2941519; sleep 5; kill -9 2941519 2>/dev/null; true"
echo "KILLED_W7 $(date -Is)"
