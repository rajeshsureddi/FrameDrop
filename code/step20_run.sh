#!/bin/bash
# Step 20 (user 10-23): pooling-query ablation of the fusion. Same recipe as v2_90 (v1 encoders frozen, init v1 fusion,
# LSVQ train90 (holdout 2806 = val10) + synthetic pairs every 2nd step, 20k steps, final.pt), only --n-q differs.
# Selection/dev metrics only: val10 holdout SROCC (logged in training), DropDev + DropDev-hard (v1enc tokens).
# Usage: step20_run.sh <n_q> <gpu>
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
D=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; O=$D/step20_pooling; k=$1; g=$2; n=pool_k$k
mkdir -p $O/logs
[ -f $O/runs/$n/final.pt ] || python step8_finetune.py --name $n --n-q $k --holdout 2806 --steps 20000 --pair-every 2 \
  --pool 8 --workers 8 --gpu $g --init $D/frozen_v1_20261017/stage3_fusion_best.pt --out $O > $O/logs/train_$n.log 2>&1
python step10_eval.py score --name $n --tok v1enc --fusion $O/runs/$n/final.pt --gpu $g --workers 12 > $O/logs/dd_$n.log 2>&1
python step10_eval.py metrics --name $n >> $O/logs/dd_$n.log 2>&1
DROPDEV=hard python step10_eval.py score --name $n --tok v1enc --fusion $O/runs/$n/final.pt --gpu $g --workers 12 > $O/logs/ddh_$n.log 2>&1
DROPDEV=hard python step10_eval.py metrics --name $n >> $O/logs/ddh_$n.log 2>&1
echo "STEP20_DONE $n $(date -Is)" >> $O/logs/done.log
