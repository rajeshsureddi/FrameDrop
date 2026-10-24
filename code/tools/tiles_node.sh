#!/bin/bash
# usage: tiles_node.sh WORKERS ORDER_SEED
# Step 3a tile store on this node (nice 10), then the Step 3 low-res cache.
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
cd /scratch/09997/rs64952/FrameDrop-VQA/method
nice -n 10 python step3_tiles.py --workers $1 --order-seed $2
echo "=== tiles driver exited $(date -Is); starting low-res cache"
exec nice -n 10 python step3_lowres.py --workers $(( $1 * 3 / 4 )) --order-seed $2
