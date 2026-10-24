#!/bin/bash
# exits (= wakes Claude) on the first noteworthy event, or after $1 hours as a heartbeat
L=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/logs; E=${L%/logs}/encodes; HB=$(( ${1:-3} * 3600 )); t0=$(date +%s)
q() { ssh -n -o BatchMode=yes login1.ls6.tacc.utexas.edu 'squeue -u rs64952 -h -o "%i %T"' 2>/dev/null | sort; }
base=$(q); [ -z "$base" ] && { echo "squeue failed at start"; exit 2; }
while true; do
  sleep 300
  now=$(date +%s)
  cur=$(q)
  if [ -n "$cur" ] && [ "$cur" != "$base" ]; then echo "QUEUE CHANGE $(date '+%m-%d %H:%M')"; diff <(echo "$base") <(echo "$cur") | grep '^[<>]'; exit 0; fi
  bad=$(grep -h -E ' [1-9][0-9]* bad|Traceback|Error|Killed|No space' $L/step2_full[123].35040[23]?.out $L/step2_capenc.35040[23]?.out $L/capenc_h100.log $L/capenc_v*.log $L/capenc_c3*.log 2>/dev/null | grep -v 'TIME LIMIT' | sort -u | head -5)
  # shard-0 postdrop process predates the 10-13 fix (false "bad" on tail drops; revalidated later): only crashes count
  bad=$(printf '%s\n%s' "$bad" "$(grep -h -E 'Traceback|Killed|No space' $L/step2_full0.3504029.out)" | grep -v '^$' | grep -v 'TIME LIMIT' | head -5)
  [ -n "$bad" ] && { echo "ERROR LINES $(date '+%m-%d %H:%M')"; echo "$bad"; exit 0; }
  done_l=$(grep -h -l -E 'pass done|ladder pass done|postdrop pass done|all done' $L/*.log $L/step2_*.350402?.out $L/step2_*.35040[23]?.out 2>/dev/null)
  [ "$done_l" != "${prev_done-$done_l}" ] && { echo "PHASE DONE $(date '+%m-%d %H:%M'): $done_l"; exit 0; }; prev_done=$done_l
  if [ $(( now - ${tc:-0} )) -ge 1800 ]; then  # output-count progress, every 30 min
    c=$(find $E -path '*_quarantine*' -prune -o \( -name 's*_r*.json' -o -name '*postdrop*.json' \) -print 2>/dev/null | wc -l)
    if [ "$c" != "${pc:-}" ]; then pc=$c; tpc=$now; elif [ $(( now - tpc )) -ge 5400 ]; then echo "NO NEW OUTPUTS for 90 min (count $c) $(date '+%m-%d %H:%M')"; exit 0; fi
    tc=$now
  fi
  [ $(( now - t0 )) -ge $HB ] && { echo "HEARTBEAT $(date '+%m-%d %H:%M')"; exit 0; }
done
