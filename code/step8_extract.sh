#!/bin/bash
# Extract frozen Stage 1/2 tokens for LSVQ (test sets first) on GPUs $GPUS of this node.
# Usage: step8_extract.sh "<gpu ids>" <workers per gpu> [part offset] [total parts]
# Resumable: skips clips with tokens; clips whose video is not unpacked yet are skipped and picked up next run.
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
L=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step8_lsvq/logs; mkdir -p $L
GPUS=($1); W=$2; OFF=${3:-0}; TOT=${4:-${#GPUS[@]}}
for k in "${!GPUS[@]}"; do
  ( for ds in lsvq_1080p lsvq_test lsvq_train; do
      python step7_eval.py extract --dataset $ds --gpu ${GPUS[$k]} --part $((OFF + k)) --parts $TOT --workers $W
    done ) > $L/extract_$(hostname -s)_part$((OFF + k)).log 2>&1 &
done
wait
echo EXTRACT_ROUND_DONE
