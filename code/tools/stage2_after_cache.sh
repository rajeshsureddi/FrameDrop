#!/bin/bash
# c316-013: wait for this node's low-res cache build, redo failed cache items
# (fixed md5 check), then start Stage 2 training on the 3 GPUs (shared with Stage 1).
set -u
cd /scratch/09997/rs64952/FrameDrop-VQA/method
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
D=../vqa_data
while pgrep -u "$USER" -f "step3_lowres.py --workers 72" > /dev/null; do sleep 60; done
echo "=== cache build on c316 done $(date -Is); retrying failed items"
find $D/step3/lowres -name '*.err' -printf '%p\n' > $D/step3/logs/lowres_retry_list.txt
find $D/step3/lowres -name '*.err' -delete
python step3_lowres.py --workers 32 --order-seed 3
echo "=== cache retry done $(date -Is): $(find $D/step3/lowres -name '*.json' | wc -l) items, $(find $D/step3/lowres -name '*.err' | wc -l) err"
while [ ! -f $D/step5/val_pairs.npy ]; do sleep 60; done
echo "=== starting Stage 2 $(date -Is)"
exec torchrun --standalone --nproc_per_node 3 --master_port 29611 step5_train.py --name v_mae --steps 100000 \
    --workers-single 8 --workers-pair 8 > $D/step5/logs/v_mae.log 2>&1
