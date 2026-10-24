#!/bin/bash
# one-shot: wait until idev job $1 is running, then start capture encode + ladders on its node (32-core A100)
J=$1; miss=0; L=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/logs; T=/scratch/09997/rs64952/FrameDrop-VQA/method/tools
while true; do
  s=$(ssh -n -o BatchMode=yes login1.ls6.tacc.utexas.edu "squeue -h -j $J -o '%T %N'" 2>/dev/null)
  st=${s%% *}; node=${s#* }
  if [ "$st" = RUNNING ]; then break; fi
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { echo "job $J gone from queue - not launching"; exit 1; }; else miss=0; fi
  case "$st" in PENDING|CONFIGURING|"") sleep 120;; *) echo "job $J state '$s' - not launching"; exit 1;; esac
done
sleep 60
timeout 60 ssh -n -o BatchMode=yes $node "setsid nohup $T/capenc_h100.sh 4 4 > $L/capenc_$node.log 2>&1 < /dev/null &" ; true
timeout 60 ssh -n -o BatchMode=yes $node "setsid nohup $T/ladder_h100.sh 2 3 > $L/ladder_$node.log 2>&1 < /dev/null &" ; true
sleep 30
echo "launched on $node at $(date -Is):"; ssh -n -o BatchMode=yes $node "pgrep -af 'step2_(capture|ladder)' | cut -c1-120"
