#!/bin/bash
# 10-20: d1 eval fan-out workers 3-5 (c306-001) died of CUDA OOM at start (40 GB GPUs were still occupied), so DropDev parts 3-5
# and 881 LSVQ-1080p clips had no tokens and the first d1 DropDev score covered half the pairs. Redo the d1 eval tail on
# c306-001's 3 GPUs: step10_cand_run.sh skips training and finished clips, re-extracts the rest, then scores + metrics.
# Then restart the DropDev-hard early-token workers on this node (paused for this).
M=/scratch/09997/rs64952/FrameDrop-VQA/method; L=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev_hard/logs
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data
FANOUT="c306-001:0:30 c306-001:1:30 c306-001:2:30" $M/step10_cand_run.sh d1 0 --init-s1 $R/step4/runs/s_dino/final_avg4.pt \
  --init-fusion $R/step11_dino/step8/runs/d1_v2_90/final.pt
echo "D1_REFILL_DONE $(date -Is)"
for g in 0 1 2; do setsid nohup $M/step10h_tok.sh $g $g 6 12 >> $L/tok_p$g.log 2>&1 < /dev/null & done
