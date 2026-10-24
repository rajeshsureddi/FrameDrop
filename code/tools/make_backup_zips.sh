#!/bin/bash
# Zip the essential, hard-to-recreate small set for UT Box (each zip < 15 GB).
# Bulk data (encodes, tiles, low-res cache, features) is NOT included.
set -eu
R=/scratch/09997/rs64952/FrameDrop-VQA
B=${1:-$R/backup_utbox_20261016}
mkdir -p $B
cd $R
z() { echo "=== $1 $(date +%T)"; rm -f "$B/$1"; }
[ -n "${SKIP_CODE:-}" ] || { z 01_code.zip; zip -qr $B/01_code.zip method README.md REORG_MANIFEST.txt -x '*/__pycache__/*' 'method/_vendor/*'; }
z 02_metadata_manifests_logs.zip
# small files directly; the ~280k per-item JSONs of step1 / step2 / low-res cache
# are packed into JSONL files first (parallel reads; restore with tools/unpack_jsonl.py)
S=$B/_staging; mkdir -p $S/packed
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
python method/tools/pack_json_dirs.py $S/packed/step1_json.jsonl vqa_data/step1
python method/tools/pack_json_dirs.py $S/packed/step2_json.jsonl vqa_data/step2
python method/tools/pack_json_dirs.py $S/packed/step3_lowres_json.jsonl vqa_data/step3/lowres
{ find vqa_data/step1 vqa_data/step2 -type f ! -name '*.json' ! -name '*.mp4' ! -name '*.mkv'
  echo vqa_data/step3/capture_plans.json
  find vqa_data/step3/logs vqa_data/step3/review -type f
  find vqa_data/step3/tiles -maxdepth 1 -name '*.json'
  find vqa_data/step4 vqa_data/step5 vqa_data/step6 vqa_data/step7 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.md' -o -name '*.csv' \) 2>/dev/null
  find vqa_data/eval vqa_data/logs_batch -type f
} | zip -q $B/02_metadata_manifests_logs.zip -@
(cd $S && zip -qr $B/02_metadata_manifests_logs.zip packed)
rm -rf $S
z 03_checkpoints_stage1.zip
zip -q0 $B/03_checkpoints_stage1.zip vqa_data/step4/runs/s_mae/final_avg4.pt vqa_data/step4/runs/s_mae/ckpt_0100000.pt
z 04_valsets_step4.zip
zip -q0 $B/04_valsets_step4.zip vqa_data/step4/val_pairs.npz vqa_data/step4/val_clean.npz
z 05_valsets_step5_pairs.zip
zip -q0 $B/05_valsets_step5_pairs.zip vqa_data/step5/val_pairs.npy vqa_data/step5/val_pairs.json
z 06_valsets_step5_singles.zip
zip -q0 $B/06_valsets_step5_singles.zip vqa_data/step5/val_singles.npy vqa_data/step5/val_singles.json
for f in $B/*.zip; do unzip -tq "$f" > /dev/null && echo "OK $(basename $f)" || echo "BAD $(basename $f)"; done
ls -la $B; du -sh $B
echo "=== DONE $(date +%T)"
