"""Check random stored tiles against an independent full decode (no select filter)."""
import json, os, sys, tarfile, numpy as np, cv2, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import step3_common as c3
from step2_capture import yuv_to_rgb
tiles_dir, sid, n_check = sys.argv[1], sys.argv[2], int(sys.argv[3])
idx = json.load(open(os.path.join(tiles_dir, sid + ".json")))
enc = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data/step2/encodes/" + sid
groups = {}
with tarfile.open(os.path.join(tiles_dir, idx["tar"])) as tf:
    for m in tf:
        key, ext = m.name.split(".", 1)
        groups.setdefault(key, {})[ext] = tf.extractfile(m).read()
keys = sorted(groups); print(len(keys), "groups,", sum(len(g) - 1 for g in groups.values()), "tiles")
rng = np.random.default_rng(1)
bad = 0
for v in rng.choice(idx["versions"], n_check, replace=False):
    path = c3.load_json(os.path.join(enc, "s0_clean_r0.json"))["source_path"] if v["encode_id"] == "orig" else os.path.join(enc, v["encode_id"] + ".mp4")
    trim = idx["span"] if v["encode_id"] == "orig" else None
    frames = [b for b in c3.iter_frames(c3.decode_cmd(path, 4, trim))]
    gk = [k for k in keys if any(x["v"] == v["v"] for x in json.loads(groups[k]["json"])["versions"])]
    for k in rng.choice(gk, 6, replace=False):
        meta = json.loads(groups[k]["json"]); vv = [x for x in meta["versions"] if x["v"] == v["v"]][0]
        Y, U, V = c3.yuv_planes(frames[vv["enc_frame"]], 3840, 2160)
        ref = yuv_to_rgb(*[torch.from_numpy(a.copy())[None] for a in (Y, U, V)])[0]
        ref8 = torch.round(ref * 255).to(torch.uint8).numpy()
        x0, y0 = meta["x"] - 64, meta["y"] - 64
        t = cv2.imdecode(np.frombuffer(groups[k][v["v"] + ".png"], np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
        d = np.abs(t.astype(int) - ref8[y0:y0 + 384, x0:x0 + 384].astype(int)).max()
        bad += d > 0
        print(v["encode_id"], k, "frame", meta["frame"], "enc_frame", vv["enc_frame"], "maxdiff", d)
print("BAD", bad)
