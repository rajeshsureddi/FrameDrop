#!/bin/bash
# DropDev-hard (10-20): early token extraction on idle GPUs while the re-encodes are still running.
#   step10h_tok.sh <gpu> <part> <parts> <cpu_workers> [wait_file]
# Passes with --skip-missing (raw clips + finished re-encodes) for v1enc, v3enc, d1enc; finished clips are skipped,
# so each pass only adds new ones. Stops when every clip has tokens for all three. The gate's ext step then only fills the rest.
set -u
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 DROPDEV=hard
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; H=$R/step10_dropdev_hard
V1=$R/frozen_v1_20261017; V3=$R/frozen_v3_20261018; D1=$R/step10_cand/runs/d1
[ -n "${5:-}" ] && while ! grep -q RESCUE_EXTRACT_DONE $5 2>/dev/null; do sleep 60; done
while true; do
  python step10_eval.py extract --name v1enc --s1 $V1/stage1_s_mae_final_avg4.pt --s2 $V1/stage2_v_mae_final_avg4.pt --gpu $1 --part $2 --parts $3 --workers $4 --skip-missing
  python step10_eval.py extract --name v3enc --s1 $V3/s1.pt --s2 $V3/s2.pt --gpu $1 --part $2 --parts $3 --workers $4 --skip-missing
  python step10_eval.py extract --name d1enc --s1 $D1/s1.pt --s2 $D1/s2.pt --gpu $1 --part $2 --parts $3 --workers $4 --skip-missing
  n=$(for t in v1enc v3enc d1enc; do ls $H/tokens/$t | grep -c '^[^.].*npz$'; done | sort -n | head -n 1)
  echo "=== pass done $(date -Is): min tokens $n / 35559"
  [ $n -ge 35559 ] && break
  sleep 1200
done
echo "TOK_DONE $(date -Is)"
