#!/bin/bash
# Wait (max $1 s, default 3600) for a new event line in the pipeline logs; print new events.
# Remembers line counts in $STATE so each call reports only what is new.
D=/scratch/09997/rs64952/FrameDrop-VQA/vqa_data
STATE=${STATE:-/tmp/claude-893466/-scratch-09997-rs64952-FrameDrop-VQA-method/a9428130-a9ba-4036-9bc0-fa8da38a3e35/scratchpad/watch_state}
LOGS="$D/step4/logs/s_mae.log $D/step5/logs/chain_c316.log $D/step5/logs/v_mae.log $D/step6/logs/orchestrator.log $D/step6/logs/fusion.log $D/step6/logs/extract_gpu0.log $D/step7/logs/extract_gpu1.log $D/step3/logs/lowres_v330-016.log"
PAT='Traceback|Error|error:|FINISHED|^===|Killed|ERR |OutOfMemory|NCCL'
touch $STATE
end=$(( $(date +%s) + ${1:-3600} ))
while [ $(date +%s) -lt $end ]; do
    out=""
    for f in $LOGS; do
        [ -f $f ] || continue
        n=$(wc -l < $f); old=$(command grep -F "$f " $STATE | awk '{print $2}'); old=${old:-0}
        if [ $n -gt $old ]; then
            new=$(tail -n +$((old + 1)) $f | head -n $((n - old)) | command grep -E "$PAT" | command grep -v Warning | command grep -vE "^ERR .*/orig$" | cut -c1-700)
            [ -n "$new" ] && out="$out\n## $(basename $f)\n$new"
            command grep -vF "$f " $STATE > $STATE.tmp; echo "$f $n" >> $STATE.tmp; mv $STATE.tmp $STATE
        fi
    done
    if [ -n "$out" ]; then date; echo -e "$out"; exit 0; fi
    sleep 60
done
date; echo "no new events"
