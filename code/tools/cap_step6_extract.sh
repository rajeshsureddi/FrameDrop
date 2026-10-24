#!/bin/bash
# Stop Step 6 train-pair extraction once N shards exist (time budget), so the
# orchestrator (tools/stages_6_7.sh, PID $1) continues with fusion training + Step 7.
O=$1; N=${2:-100}
D=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step6
until [ $(ls $D/train/ 2>/dev/null | grep -cE '^pairs_[0-9]{5}\.npz$') -ge $N ]; do sleep 60; done
echo "cap reached $(date -Is)"
W=$(pgrep -P $O)                         # the per-GPU extraction loops (subshells of the orchestrator)
kill -STOP $W
for w in $W; do pkill -TERM -P $w; done  # their running extraction processes
sleep 5
kill -TERM $W; kill -CONT $W
rm -f $D/train/*.tmp.npz
echo "stopped loops $W; $(ls $D/train | grep -cE '^pairs_[0-9]{5}\.npz$') train shards kept"
