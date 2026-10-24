#!/bin/bash
# Untar each LSVQ tarball as soon as its download finishes -> LSVQ_PatchVQ/videos/<batch>/<id>.mp4
D=/scratch/09997/rs64952/FrameDrop-VQA/mos_datasets/LSVQ_PatchVQ
V=$D/videos; mkdir -p $V $D/unpacked
while true; do
  for log in $D/dl_ia/download.log $D/dl_yfcc.log; do
    sub=$([ $log = $D/dl_yfcc.log ] && echo dl_yfcc || echo dl_ia)
    for n in $(grep '^done ' $log | cut -d' ' -f2); do
      [ -e $D/unpacked/$n ] && continue
      touch $D/unpacked/$n
      (tar xzf $D/$sub/$n.tar.gz -C $V && echo "untar $n $(date +%H:%M)" || echo "FAIL $n") &
    done
  done
  wait
  grep -q ALL_DONE $D/dl_ia/download.log && grep -q ALL_DONE $D/dl_yfcc.log && [ $(ls $D/unpacked | wc -l) -ge 51 ] && break
  sleep 30
done
echo UNPACK_DONE; find $V -name '*.mp4' | wc -l
