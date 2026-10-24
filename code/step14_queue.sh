#!/bin/bash
# Step 14 worker: claims (init dataset split) tasks from a shared queue (mkdir lock) and runs them on one GPU.
#   step14_queue.sh <gpu> <init> [datasets regex, default all] [deadline: unix time; a task starts only if its worst-case time fits before it]
# Order: the 20 study splits first (GAMIVAL comparison, approved 10-21), then KoNViD / LIVE-VQC splits 1-9, split 0 last
# (the idev runs may already have done it). Tasks with result.json are skipped.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step14_finetune; L=$R/logs; Q=$R/queue; mkdir -p $L $Q
g=$1; init=$2; only=${3:-.}; deadline=${4:-0}
TASKS="$(for s in $(seq 0 19); do echo study:$s; done) $(for s in 1 2 3 4 5 6 7 8 9 0; do echo konvid1k:$s livevqc:$s; done) $(for s in $(seq 0 9); do echo ytugc:$s; done)"
for t in $TASKS; do
  ds=${t%%:*}; s=${t##*:}
  echo $ds | grep -Eq "$only" || continue
  need=12600; [ $ds = study ] && need=21600                   # worst case on a100-sm: public ~3 h, study ~5.5 h (10-23)
  [ $deadline -gt 0 ] && [ $(( $(date +%s) + need )) -gt $deadline ] && { echo "deadline: $ds needs ${need}s, skip"; continue; }
  w=10; pw=0; [ $ds = study ] && w=36                         # 4K decode: ~15 s per training window
  # small nodes (36 GB /dev/shm, 70 GB RAM): val/test loader next to the live train loader ran out of shm / OOM (10-22)
  [ $ds = study ] && [ $(df -k --output=size /dev/shm | tail -1) -lt 100000000 ] && { w=30; pw=6; }
  {
    [ -e $R/runs/$init/$ds/split$s/result.json ] && continue
    mkdir $Q/${init}_${ds}_$s 2>/dev/null || continue          # claimed by another worker
    echo "$(hostname) gpu$g $(date -Is)" > $Q/${init}_${ds}_$s/owner
    python step14_finetune.py train --dataset $ds --split $s --init $init --gpu $g --workers $w --pred-workers $pw > $L/${init}_${ds}_s$s.log 2>&1
    echo "$(date -Is) ${init} $ds split$s exit $?"
  }
done
echo "WORKER_DONE gpu$g $(date -Is)"
