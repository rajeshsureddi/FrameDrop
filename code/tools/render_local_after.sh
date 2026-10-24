#!/bin/bash
# wait for the running local render loop, then render more shards on this GPU
while kill -0 $1 2>/dev/null; do sleep 60; done
shift
exec /scratch/09997/rs64952/FrameDrop-VQA/method/tools/render_local.sh "$@"
