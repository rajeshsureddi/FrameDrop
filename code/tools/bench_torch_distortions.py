"""ms/frame per op at 3840x2160 on the GPU (video_distortions_torch.render).
Frames are already on the GPU as uint8 (decode/upload not included); each op
runs at severity 1, const envelope, global support, over a batch of frames.
python tools/bench_torch_distortions.py [out.txt]"""
import os, subprocess, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video_distortions as vd, video_distortions_torch as vt
SRC = "/scratch/09997/rs64952/FrameDrop-VQA/data_extracted/Inter4K/60fps/UHD/1.mp4"
T = 8
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", SRC, "-frames:v", str(T), "-pix_fmt", "rgb24",
                      "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
F = torch.as_tensor(np.frombuffer(raw, np.uint8).reshape(T, 2160, 3840, 3).copy(), device="cuda")
idx = np.arange(T)
lines = [f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}, 3840x2160, batch {T} frames, "
         f"uint8 frames already on GPU, severity 1, const envelope, global support, fp32 (no TF32)"]
res = {}
for name in vd.REGISTRY:
    op = vd.sample_op(np.random.default_rng(1), name, 1.0, severity=1.0)
    op.envelope, op.support, op.bipolar = vd.Envelope(), vd.Support(), False
    plan = vd.Plan([op], 60.0, 0)
    vt.render(F[:2], idx[:2], plan)                       # warm-up
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(2):
        vt.render(F, idx, plan)
    torch.cuda.synchronize()
    res[name] = (time.perf_counter() - t0) / (2 * T) * 1000
base = vd.Plan([vd.Op("low_contrast", 0.0, {"c": 0.5}, 0)], 60.0, 0)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(2):
    vt.render(F, idx, base)
torch.cuda.synchronize()
ovh = (time.perf_counter() - t0) / (2 * T) * 1000
lines.append(f"render overhead (uint8->float, clamp; no op active): {ovh:.1f} ms/frame")
for name, ms in sorted(res.items(), key=lambda kv: -kv[1]):
    lines.append(f"{name:22s} {ms:7.1f} ms/frame{'   <-- over 20 ms' if ms > 20 else ''}")
over = [n for n, ms in res.items() if ms > 20]
lines.append(f"ops over 20 ms/frame: {len(over)} of {len(res)}: {', '.join(over) or 'none'}")
txt = "\n".join(lines)
print(txt)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(txt + "\n")
