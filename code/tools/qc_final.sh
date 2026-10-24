#!/bin/bash
# final Step 2 QC: post-drop revalidation, then full step2_qc.py (PSNR/VMAF from the per-encode cache)
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step2; cd $R/method
echo "start $(date -Is) on $(hostname -s)"
python tools/revalidate_postdrop.py $O/encodes
s=$(date +%s)
python step2_qc.py --manifest $R/vqa_data/step1/manifest.csv --out $O --workers 32 --throughput $O/throughput.txt
echo "qc exit $? after $(( $(date +%s) - s ))s; done $(date -Is)"
