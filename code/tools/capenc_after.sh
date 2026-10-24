#!/bin/bash
# wait for process $1 (e.g. the ladder/postdrop driver) to exit, then run capture encoding with $2 workers x $3 threads
while kill -0 $1 2>/dev/null; do sleep 60; done
echo "pid $1 gone, starting capenc $(date -Is)"
exec /scratch/09997/rs64952/FrameDrop-VQA/method/tools/capenc_h100.sh $2 $3
