#!/bin/bash
#SBATCH -J step1_spans
#SBATCH -p normal
#SBATCH -A ASC22049
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 04:00:00
#SBATCH -o /scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step1/logs/%x.%j.out
set -e
export PATH=/work/09997/rs64952/ls6/miniconda3/envs/vlm-r1/bin:$PATH
R=/scratch/09997/rs64952/FrameDrop-VQA; O=$R/vqa_data/step1; M=$R/method; cd $M
# the study folder has moved twice: refuse to run with broken exclusion links
nl=$(ls $O/exclude_study_links | wc -l); nb=$(find -L $O/exclude_study_links -type l | wc -l)
echo "exclusion links: $nl, broken: $nb"; [ "$nl" -eq 35 ] && [ "$nb" -eq 0 ] || { echo "ABORT: exclusion links"; exit 1; }
DATA="--data LMM-10K=$R/4k_datasets/LMM-10K --data Inter4K=$R/data_extracted/Inter4K/60fps/UHD"
ARGS="$DATA --exclude $O/exclude_study_links --out $O --workers 32 --decode-threads 4"
echo "node $(hostname) start $(date -Is)"
s=$(date +%s); python build_manifest.py $ARGS; echo "TIMING full_run=$(( $(date +%s) - s ))s"
cp $O/manifest.csv $O/logs/manifest_run1.csv; cp $O/report.txt $O/logs/report_run1.txt
s=$(date +%s); python build_manifest.py $ARGS > $O/logs/rerun.out; echo "TIMING cached_rerun=$(( $(date +%s) - s ))s"
python tools/step1_checks.py $O/manifest.csv $O/logs/manifest_run1.csv LMM-10K=$R/4k_datasets/LMM-10K Inter4K=$R/data_extracted/Inter4K/60fps/UHD | tee $O/logs/acceptance.txt || true
# exclusion positive control: study variants as a dataset vs the bare study clips (frames only)
C=$O/exclusion_control
s=$(date +%s); python build_manifest.py --data STUDY_CONTROL=$C/inputs --exclude $O/exclude_study_links --out $C \
   --workers 32 --decode-threads 4 > $C/stdout.txt; echo "TIMING control=$(( $(date +%s) - s ))s"
mv -f $C/manifest.csv $C/control_manifest_DO_NOT_TRAIN.csv
python - <<PY | tee $O/logs/exclusion_control.txt
import csv
rows = list(csv.DictReader(open("$C/control_manifest_DO_NOT_TRAIN.csv")))
hit = [r for r in rows if "matches_excluded_source" in r["flags"]]
print(f"exclusion control: {len(hit)}/{len(rows)} study variants matched their source")
for r in rows:
    print(("MATCH " if r in hit else "MISS  ") + r["path"].split("/")[-1] + "  " + r["excl_match"])
PY
rm -rf $O/step1_review
s=$(date +%s); python step1_review.py --manifest $O/manifest.csv --out $O/step1_review --n 10; echo "TIMING review=$(( $(date +%s) - s ))s"
W=/work/09997/rs64952/ls6/FrameDrop-VQA; mkdir -p $W/method $W/vqa_data/step1
rsync -a --exclude __pycache__ --exclude '*.zip' $M/ $W/method/
rsync -a --exclude 'smoke_*' --exclude exclude_study_links --exclude 'probe_cache_*' --exclude inputs $O/ $W/vqa_data/step1/
rsync -a $O/probe_cache_span240/ $W/vqa_data/step1/probe_cache_span240/
echo "done $(date -Is)"
