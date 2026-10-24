#!/bin/bash
# Step 19 finish: score every new token set with its model's fusion -> pooled features -> public Protocol A
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH PYTHONPATH=$M/_vendor:$M OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
D=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; O=$D/step19_public
for t in v3:ytugc v3:ytugc_half v3:konvid1k_half v3:livevqc_half v4:konvid1k_half v4:livevqc_half; do
  m=${t%%:*}; ds=${t##*:}; [ $m = v3 ] && F=$D/frozen_v3_20261018/fusion.pt || F=$D/step15_v4/runs/v4/fusion.pt
  python step7_eval.py score --dataset $ds --gpu ${1:-0} --fusion $F --tok-root $O/tokens_$m --out $O/eval/$m/$ds
done
python step19_public_protA.py
echo STEP19_DONE
