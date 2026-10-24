#!/bin/bash
# Fetch the LSVQ parts missing locally (ia-batch1-24, backup, backup2) from the HF mirror teowu/LSVQ-videos.
D=/scratch/09997/rs64952/FrameDrop-VQA/mos_datasets/LSVQ_PatchVQ/dl_ia
cd $D
names="backup backup2 $(seq -f 'ia-batch%g' 1 24)"
printf '%s\n' $names | xargs -P 6 -I{} sh -c 'curl -sSL -C - --retry 10 --retry-delay 10 -o {}.tar.gz https://huggingface.co/datasets/teowu/LSVQ-videos/resolve/main/{}.tar.gz && echo done {}'
echo ALL_DONE
