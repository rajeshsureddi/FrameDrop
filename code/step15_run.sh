#!/bin/bash
# Step 15: train v4 on both H100s of c318-002 once the study splits 2/3 (step14) on this node have finished.
#   ssh c318-002 'cd method; setsid nohup bash step15_run.sh > ../vqa_data/step15_v4/logs/run.out 2>&1 &'
# Resume: rerun (step15_v4.py resumes from runs/v4/last.pt).
M=/scratch/09997/rs64952/FrameDrop-VQA/method; cd $M
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1
L=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step15_v4/logs
while pgrep -f "step14_finetune.py train" > /dev/null; do sleep 60; done
echo "$(date -Is) GPUs free, starting v4"
for r in 0 1; do
  python step15_v4.py train --name v4 --world 2 --rank $r --gpu $r --dist-url tcp://localhost:29515 > $L/v4_rank$r.log 2>&1 &
done
wait
echo "$(date -Is) training exited"
python step15_v4.py export --name v4 && echo "$(date -Is) V4_EXPORTED"
