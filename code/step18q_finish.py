"""Step 18 quarter scale: build multi-scale feature sets (ours only; GAMIVAL dropped by the user 10-23) for Protocol A.
  python step18q_finish.py  ->  vqa_data/step18_multiscale/{v4_quarter, v4_full_half_quarter}.npz
"""
import numpy as np

D = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
S = f"{D}/step18_multiscale"
src = dict(full=f"{D}/step15_v4/eval/v4/study/pooled_features.npz", half=f"{S}/eval/v4_half/study/pooled_features.npz",
           quarter=f"{S}/eval/v4_quarter/study/pooled_features.npz")
f = {k: dict(zip(np.load(p)["clips"].tolist(), np.load(p)["x"])) for k, p in src.items()}
names = sorted(f["full"])
assert len(names) == 1823 and all(set(v) == set(names) for v in f.values())
X = {k: np.stack([v[n] for n in names]).astype(np.float64) for k, v in f.items()}
for out, parts in (("v4_quarter", ["quarter"]), ("v4_full_half_quarter", ["full", "half", "quarter"])):
    x = np.concatenate([X[p] for p in parts], 1)
    np.savez(f"{S}/{out}.npz", clips=np.array(names), x=x)
    print(out, x.shape)
