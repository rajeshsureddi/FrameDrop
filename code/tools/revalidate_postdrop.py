"""Mark pre-fix post-drop records ok when they failed only on the decoder's broken-pipe exit."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import step2_postdrop as P
fixed = still_bad = 0
for j in glob.glob(os.path.join(sys.argv[1], "*", "s*_postdrop_v*.json")):
    d = json.load(open(j))
    if d.get("ok") or "dec_complete" in d:
        continue
    if P._valid_output(d.get("ffmpeg_rc"), d.get("probe", {}), d["span_frames"]):
        d.update(ok=True, revalidated="decoder rc ignored (pipe closed early after tail drop)")
        with open(j + ".tmp", "w") as f:
            json.dump(d, f)
        os.replace(j + ".tmp", j)
        fixed += 1
    else:
        still_bad += 1
        print("still bad:", j)
print(f"revalidated {fixed}, still bad {still_bad}")
