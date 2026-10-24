#!/bin/bash
# Step 18 finish: score half-res tokens with the v4 fusion -> 1024-d multi-scale features -> 20-split Protocol A
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH PYTHONPATH=$M/_vendor:$M OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
D=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data; S=$D/step18_multiscale
python step7_eval.py score --dataset study_half --gpu ${1:-0} --fusion $D/step15_v4/runs/v4/fusion.pt --tok-root $S/tokens_v4 --out $S/eval/v4_half/study
python step18_finish.py
python step17_protA.py --feat v4_full=$D/step15_v4/eval/v4/study/pooled_features.npz --feat v4_half=$S/v4_half.npz \
  --feat v4_full_half=$S/v4_full_half.npz --out $S/protA
echo STEP18_DONE
