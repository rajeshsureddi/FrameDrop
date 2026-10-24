#!/bin/bash
# Waits for the 6 manual extract runs (started 10-19 13:25), then runs the 10a gate script in place (same steps as dd_gate).
for f in /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/logs/extract_v1enc_g{0,1,2}.log /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step10_dropdev/logs/extract_v3enc_g{0,1,2}.log; do
  until [ "$(tail -n 1 $f)" = "FINISHED extract" ]; do sleep 60; done
done
bash /scratch/09997/rs64952/FrameDrop-VQA/method/step10_gate.sbatch
