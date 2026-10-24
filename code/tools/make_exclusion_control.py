"""Positive control for the study exclusion (frames only, never scores).
Links 2 random compressed/frame-dropped variants per study scene (fixed seed)
into <out>; build_manifest.py is then run with these as a dataset and the bare
<Scene>.mp4 files as --exclude: every control clip must come out as
matches_excluded_source. Only the scene name is read from file names.
The resulting manifest is a test artifact: never train on it.
python tools/make_exclusion_control.py <study clip dir> <out dir>"""
import os, random, re, sys
from collections import defaultdict
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
by_scene = defaultdict(list)
for f in sorted(os.listdir(src)):
    if not f.endswith(".mp4"):
        continue
    scene = re.split(r"_(?:4k|hd)_|_(?:consecutive|random)_drop", f[:-4])[0]
    if scene != f[:-4]:                       # skip the bare <Scene>.mp4 files
        by_scene[scene].append(f)
rng = random.Random(0)
n = 0
for scene in sorted(by_scene):
    for f in rng.sample(by_scene[scene], 2):
        d = os.path.join(out, f)
        if not os.path.lexists(d):
            os.symlink(os.path.join(src, f), d)
        n += 1
print(f"{len(by_scene)} scenes, {n} control clips linked into {out}")
