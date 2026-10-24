#!/bin/bash
# low-priority QC PSNR/VMAF of the 2% clean-ladder sample (cached per encode), sharing a node with encodes
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; cd $R/method
exec nice -n 19 python step2_qc.py --manifest $R/vqa_data/step1/manifest.csv --out $R/vqa_data/step2 --quality-only --workers ${1:-12}
