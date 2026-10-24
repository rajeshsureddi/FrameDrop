"""Step 17 (user 10-23: "add the same hand crafted features of GAMIVAL to our features and then do same regressor").
GAMIVAL NSS features (user-provided baselines/GAMIVAL_nss_features/FrameDrop_GAMIVAL_feats.mat: feats_mat 1823 x 1156
= 680 spatial + 476 temporal noisy-NSS; no NDNetGaming CNN part) -> npz files for step10_protA.py:
  gamival_nss.npz      GAMIVAL NSS alone (sanity check vs the official GAMIVAL row)
  v4_gamival_nss.npz   v4 frozen pooled features (512) concatenated with GAMIVAL NSS (1156)
Row order: the mat's `filelist` table (MATLAB opaque object; scipy can't decode it) -> the 1823 file paths are read in
order from its __function_workspace__ bytes and checked (unique, == the study GT set).
"""
import csv
import os
import re

import numpy as np
import scipy.io as sio

F = "/scratch/09997/rs64952/FrameDrop-VQA"
MAT = f"{F}/baselines/GAMIVAL_nss_features/FrameDrop_GAMIVAL_feats.mat"
V4 = f"{F}/vqa_data/step15_v4/eval/v4/study/pooled_features.npz"
GT = f"{F}/mos_datasets/framedrop_vqa_subjective_study/ground_truth/mos_visibility_per_video.csv"
OUT = f"{F}/vqa_data/step17_gamival_concat"

m = sio.loadmat(MAT)
X = m["feats_mat"]
ws = bytes(m["__function_workspace__"].astype(np.uint8).ravel())
paths = [p.decode() for p in re.findall(rb"/[ -~]*?\.mp4", ws)]
names = [os.path.basename(p) for p in paths]
gt = {r["video_filename"] for r in csv.DictReader(open(GT))}
assert len(names) == len(set(names)) == X.shape[0] == 1823 and set(names) == gt, (len(names), X.shape)
print("gamival", X.shape, "first", names[0], "last", names[-1])
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/gamival_nss.npz", clips=np.array(names), x=X)
v = np.load(V4)
fx = dict(zip(v["clips"].tolist(), v["x"]))
Xc = np.concatenate([np.stack([fx[n] for n in names]).astype(np.float64), X], 1)
np.savez(f"{OUT}/v4_gamival_nss.npz", clips=np.array(names), x=Xc)
print("v4+gamival", Xc.shape)
with open(f"{OUT}/gamival_row_order.txt", "w") as f:
    f.write("\n".join(names) + "\n")
