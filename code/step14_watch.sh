#!/bin/bash
# Step 14 idle-GPU watcher (user 10-21: "arm yourself if any idev session activates, do not keep idle").
# Every 5 min: for each of my RUNNING idev jobs (name idv*), start a step14_queue.sh worker on every GPU that has
# no worker and no other process (memory.used < 1 GiB), if unclaimed tasks remain for it.
#   big nodes (>= 36 cores per GPU): default queue order (study first, then public)
#   small nodes (gpu-a100-sm, 32 cores): public datasets first; study only once no public task is unclaimed
# Also frees locks of unfinished tasks whose owner node is no longer one of my running jobs (job ended mid-task).
# Never submits or resubmits jobs.
#   nohup setsid bash step14_watch.sh v3 > $R/logs/watch.out 2>&1 &
set -u
init=${1:-v3}
M=/scratch/09997/rs64952/FrameDrop-VQA/method
R=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step14_finetune; Q=$R/queue; L=$R/logs
PUB='konvid1k|livevqc|ytugc'
all_tasks() { for s in $(seq 0 19); do echo study:$s; done; for d in konvid1k livevqc ytugc; do for s in $(seq 0 9); do echo $d:$s; done; done; }
unclaimed() {  # count tasks matching regex $1 with no result and no lock
  local n=0 t ds s
  for t in $(all_tasks); do ds=${t%%:*}; s=${t##*:}
    echo $ds | grep -Eq "$1" || continue
    [ -e $R/runs/$init/$ds/split$s/result.json ] || [ -d $Q/${init}_${ds}_$s ] || n=$((n+1))
  done; echo $n; }
log() { echo "$(date -Is) $*"; }

while true; do
  sq=$(ssh -o ConnectTimeout=20 -o BatchMode=yes login1.ls6.tacc.utexas.edu \
       "squeue -u rs64952 -h -t R -o '%i %j %N %e'" 2>/dev/null | grep -E '^[0-9]+ ')
  if [ -z "$sq" ]; then log "squeue failed/empty"; sleep 300; continue; fi
  running_nodes=$(echo "$sq" | awk '{print $3}')

  # stale locks: owner node gone, no result
  for d in $Q/${init}_*; do [ -d $d ] || continue
    b=$(basename $d); rest=${b#${init}_}; ds=${rest%_*}; s=${rest##*_}
    [ -e $R/runs/$init/$ds/split$s/result.json ] && continue
    host=$(awk '{print $1}' $d/owner 2>/dev/null); host=${host%%.*}
    [ -n "$host" ] || continue
    echo "$running_nodes" | grep -qx "$host" && continue
    log "stale lock $b (owner $host not running) -> freed"; rm -rf $d
  done

  echo "$sq" | while read -r jid name node endt; do
    case $name in idv*) ;; *) continue;; esac
    grep -qx "$node" $R/watch_skip 2>/dev/null && continue        # node reserved for other work (e.g. v4 on c318-002)
    dl=$(date -d "$endt" +%s 2>/dev/null) || continue
    info=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes $node \
      'nproc --all; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; echo PS; pgrep -af "step14_queue.sh" | grep -v pgrep; true' 2>/dev/null) || { log "$node: ssh failed"; continue; }
    ncpu=$(echo "$info" | head -1)
    gpus=$(echo "$info" | sed -n '2,/^PS$/p' | grep -v '^PS$')
    procs=$(echo "$info" | sed -n '/^PS$/,$p' | tail -n +2)
    ngpu=$(echo "$gpus" | grep -c .)
    [ $ngpu -gt 0 ] || continue
    if [ $((ncpu / ngpu)) -ge 36 ]; then rx='.'
    elif [ $(unclaimed "$PUB") -gt 0 ]; then rx=$PUB
    else rx='study'; fi
    echo "$gpus" | while IFS=', ' read -r g mem; do
      echo "$procs" | grep -q "step14_queue.sh $g " && continue        # worker already on this GPU
      [ "$mem" -lt 1024 ] || continue                                   # GPU used by something else
      [ $(unclaimed "$rx") -gt 0 ] || continue
      log "$node (job $jid, $name) gpu$g idle -> worker regex '$rx' deadline $endt"
      ssh -n -o BatchMode=yes $node "cd $M; nohup setsid bash step14_queue.sh $g $init '$rx' $dl >> $L/idev_${node}_g$g.out 2>&1 < /dev/null &"
    done
  done
  [ $(unclaimed '.') -eq 0 ] && [ -z "$(ls $Q 2>/dev/null | while read b; do rest=${b#${init}_}; [ -e $R/runs/$init/${rest%_*}/split${rest##*_}/result.json ] || echo x; done)" ] \
    && { log "all step14 tasks done; watcher exits"; exit 0; }
  sleep 300
done
