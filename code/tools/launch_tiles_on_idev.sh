#!/bin/bash
# one-shot: wait until idev job $1 is running, then start the Step 3a tile store on its node
# (workers = 3/4 of its cores, leaving room for loader tests)
J=$1; miss=0; L=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step3/logs; T=/scratch/09997/rs64952/FrameDrop-VQA/method/tools
while true; do
  s=$(ssh -n -o BatchMode=yes login1.ls6.tacc.utexas.edu "squeue -h -j $J -o '%T %N'" 2>/dev/null)
  st=${s%% *}; node=${s#* }
  if [ "$st" = RUNNING ]; then break; fi
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { echo "job $J gone from queue - not launching"; exit 1; }; else miss=0; fi
  case "$st" in PENDING|CONFIGURING|"") sleep 120;; *) echo "job $J state '$s' - not launching"; exit 1;; esac
done
sleep 60
n=$(timeout 60 ssh -n -o BatchMode=yes $node "taskset -pc \$\$ | sed 's/.*: //'" 2>/dev/null)
cores=$(python3 -c "s='$n';print(sum(int(b)-int(a)+1 if '-' in p else 1 for p in s.split(',') for a,b in [p.split('-') if '-' in p else (p,p)]))" 2>/dev/null)
[ -z "$cores" ] && cores=128
W=$((cores*3/4))
timeout 60 ssh -n -o BatchMode=yes $node "setsid nohup $T/tiles_node.sh $W 2 > $L/tiles_$node.log 2>&1 < /dev/null &" ; true
sleep 30
echo "launched tiles on $node ($cores cores, $W workers) at $(date -Is):"; tail -2 $L/tiles_$node.log
