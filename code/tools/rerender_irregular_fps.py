"""Re-render 2c capture variants of sources with irregular frame timing.
Before the -fps_mode passthrough fix, the render decode used ffmpeg's default CFR
output, which can duplicate/drop frames when a source's timestamps are irregular
(average fps not 60 / 59.94). Existing capture intermediates and final capture
encodes of those sources are moved (not deleted) to encodes/_quarantine_fps/,
then the 4 variants are rendered again with the fixed decode.
python tools/rerender_irregular_fps.py"""
import csv, glob, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import step2_capture as cap
from step2_ladder import kept_sources
OUT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2"
rows = kept_sources("/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step1/manifest.csv")
odd = [r for r in rows if min(abs(float(r["fps"]) - 60), abs(float(r["fps"]) - 59.94)) > 0.02]
print(f"{len(odd)} sources with irregular average fps", flush=True)
Q = os.path.join(OUT, "encodes", "_quarantine_fps")
for i, r in enumerate(odd, 1):
    d = os.path.join(OUT, "encodes", r["source_id"])
    q = os.path.join(Q, r["source_id"])
    os.makedirs(q, exist_ok=True)
    for f in glob.glob(os.path.join(d, "tmp_*_cap_*")) + glob.glob(os.path.join(d, "s*_cap_*_r*.*")):
        shutil.move(f, os.path.join(q, os.path.basename(f)))
    t = time.time()
    recs = cap.render_source(r, OUT, "cuda")
    print(f"  {i}/{len(odd)} {r['source_id']} fps {r['fps']}: {len(recs)} renders, ok "
          f"{sum(x['ok'] for x in recs)}, frames {[x['frames_written'] for x in recs]} "
          f"expected {[x['render_frames'] for x in recs]} ({time.time() - t:.0f}s)", flush=True)
