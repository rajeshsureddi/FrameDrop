#!/bin/bash
# The local yfcc-batch*.zip are password-protected; fetch the unencrypted tarballs from the HF mirror.
D=/scratch/09997/rs64952/FrameDrop-VQA/mos_datasets/LSVQ_PatchVQ/dl_yfcc
mkdir -p $D; cd $D
seq -f 'yfcc-batch%g' 1 25 | xargs -P 6 -I{} sh -c 'curl -sSL -C - --retry 10 --retry-delay 10 -o {}.tar.gz https://huggingface.co/datasets/teowu/LSVQ-videos/resolve/main/{}.tar.gz && echo done {}'
echo ALL_DONE
