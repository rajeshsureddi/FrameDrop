"""Step 18: half-res pooled features (v4 fusion over study_half tokens) + full-res v4 pooled features -> 1024-d.
  python step18_finish.py   ->  vqa_data/step18_multiscale/{v4_half.npz, v4_full_half.npz}
"""
import numpy as np

D = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
full = np.load(f"{D}/step15_v4/eval/v4/study/pooled_features.npz")
half = np.load(f"{D}/step18_multiscale/eval/v4_half/study/pooled_features.npz")
ff, fh = dict(zip(full["clips"].tolist(), full["x"])), dict(zip(half["clips"].tolist(), half["x"]))
names = sorted(ff)
assert len(names) == 1823 and set(names) == set(fh), (len(names), len(fh))
Xh = np.stack([fh[n] for n in names])
np.savez(f"{D}/step18_multiscale/v4_half.npz", clips=np.array(names), x=Xh)
np.savez(f"{D}/step18_multiscale/v4_full_half.npz", clips=np.array(names),
         x=np.concatenate([np.stack([ff[n] for n in names]), Xh], 1))
print("half", Xh.shape, "full+half", (len(names), 2 * Xh.shape[1]))
