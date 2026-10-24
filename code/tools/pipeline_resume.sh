#!/bin/bash
# Run by the queued batch job (tools/pipeline_job.sbatch, 48 h on one A100 node).
# Executes work orders: every executable tools/batch_orders/*.sh in name order,
# each once (renamed to .done / .failed). Waits for new orders while none exist,
# and exits after 6 h with nothing to do.
set -u
cd /scratch/09997/rs64952/FrameDrop-VQA/method
O=tools/batch_orders
mkdir -p $O
echo "pipeline host $(hostname) started $(date -Is)"
idle=0
while [ $idle -lt 72 ]; do
    next=$(ls $O/*.sh 2>/dev/null | sort | head -1)
    if [ -z "$next" ]; then sleep 300; idle=$((idle + 1)); continue; fi
    idle=0
    echo "=== running $next $(date -Is)"
    mv "$next" "$next.running"
    if bash "$next.running"; then mv "$next.running" "${next%.sh}.done"; else mv "$next.running" "${next%.sh}.failed"; fi
    echo "=== finished $next $(date -Is)"
done
echo "no orders for 6 h, exiting $(date -Is)"
