"""Step 1 acceptance checks on a finished manifest.
python tools/step1_checks.py <manifest.csv> <rerun manifest.csv> NAME=root [NAME=root ...]"""
import csv
import filecmp
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_manifest import HARD, HDR_TRC, list_videos

man, rerun, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = list(csv.DictReader(open(man)))
fails = []


def check(cond, msg):
    print(("[x] " if cond else "[ ] FAIL ") + msg)
    if not cond:
        fails.append(msg)


paths = Counter(r["path"] for r in rows)
want = set()
for spec in specs:
    name, root = spec.split("=", 1)
    want |= {os.path.abspath(p) for p in list_videos(root)}
check(want == set(paths) and max(paths.values()) == 1,
      f"every video appears exactly once ({len(want)} on disk, {len(rows)} rows, "
      f"missing {len(want - set(paths))}, extra {len(set(paths) - want)}, "
      f"duplicated {sum(v > 1 for v in paths.values())})")
kept = [r for r in rows if r["keep"] == "True"]
bad = [r for r in kept if r["transfer"] in HDR_TRC or int(r["width"]) < 3840 or int(r["height"]) < 2160
       or float(r["fps"]) < 50 or r["vfr"] == "True"
       or any(f in HARD or f == "matches_excluded_source" for f in r["flags"].split("|") if f)]
check(not bad, f"no kept clip is HDR / <3840x2160 / <50 fps / VFR / hard-flagged ({len(bad)} bad of {len(kept)})")
def spans_ok(r):
    sp = json.loads(r["spans"] or "[]")
    n = int(r["frames_scanned"])
    return bool(sp) and all(0 <= a < b <= n and b - a >= 240 for a, b in sp) \
        and sum(b - a for a, b in sp) == int(r["usable_frames"])


bad = [r for r in kept if not spans_ok(r)]
check(not bad, f"every kept clip has >= 1 span of >= 240 frames inside its decoded range, "
               f"all spans valid ({len(bad)} bad)")
bad = [r for r in kept if len(json.loads(r["active_xywh"] or "[]")) != 4]
check(not bad, f"every kept clip has an active_xywh ({len(bad)} missing)")
study = [r for r in rows if "subjective_study" in r["path"] or "exclude_study_links" in r["path"]]
check(not study, f"no study-folder path in the manifest ({len(study)} found)")
bad = [r for r in kept if float(r["bpp"] or 0) and float(r["bpp"]) < 0.03]
check(not bad, f"no kept clip below --min-bpp 0.03 ({len(bad)} found)")
spl = defaultdict(set)
for r in rows:
    if r["split"]:
        spl[r["group_id"]].add(r["split"])
check(all(len(v) == 1 for v in spl.values()),
      f"no group_id in more than one split ({sum(len(v) > 1 for v in spl.values())} bad of {len(spl)} groups)")
ex = [r for r in rows if "matches_excluded_source" in r["flags"]]
check(all(r["keep"] == "False" for r in ex), f"all {len(ex)} excluded-source matches dropped")
check(filecmp.cmp(man, rerun, shallow=False), "re-run reuses the cache and gives an identical manifest.csv")
print("ALL ACCEPTANCE CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
