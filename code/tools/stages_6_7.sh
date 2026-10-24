#!/bin/bash
# c316-013: once Stage 1 and Stage 2 final checkpoints exist, run
#   Step 6: frozen-token extraction (valid 4 x 500 pairs, train 150 x 600 pairs; 3 GPUs)
#           -> fusion training (30k steps, 1 GPU)
#   Step 7: token extraction for study + KoNViD-1k (as is) -> scores
# Every part skips work already done, so the script can simply be re-run.
set -u
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
D=../vqa_data
S1=$D/step4/runs/s_mae/final_avg4.pt
S2=$D/step5/runs/v_mae/final_avg4.pt
while [ ! -f $S1 ] || [ ! -f $S2 ]; do sleep 120; done
echo "=== encoders final $(date -Is)"
W=${WORKERS:-36}
gpu_worker() {   # $1 = gpu
    for s in $(seq $1 3 3); do python step6_features.py pairs --split valid --n 500 --shard $s --gpu $1 --workers $W; done
    for s in $(seq $1 3 149); do python step6_features.py pairs --split train --n 600 --shard $s --gpu $1 --workers $W; done
}
for g in 0 1 2; do gpu_worker $g > $D/step6/logs/extract_gpu$g.log 2>&1 & done
wait
echo "=== step6 extraction done $(date -Is): $(ls $D/step6/train/pairs_*.npz | wc -l) train shards"
python step6_train.py --name fusion --gpu 0 > $D/step6/logs/fusion.log 2>&1 &
FUS=$!
for g in 1 2; do
    for ds in study konvid1k; do python step7_eval.py extract --dataset $ds --gpu $g --part $((g - 1)) --parts 2 --workers $((W / 2 * 3)); done \
        > $D/step7/logs/extract_gpu$g.log 2>&1 &
done
wait
echo "=== fusion + step7 extraction done $(date -Is)"
for ds in study konvid1k; do python step7_eval.py score --dataset $ds --gpu 0; done
echo "=== FINISHED stages 6-7 (scores) $(date -Is)"
