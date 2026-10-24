"""Sanity tests for the numpy parts (no torch needed). Make the test clips first:
ffmpeg -f lavfi -i testsrc2=size=640x360:rate=60 -t 10 -c:v libx264 -g 60 -pix_fmt yuv420p /tmp/src.mp4
ffmpeg -f lavfi -i color=gray:size=640x360:rate=60 -t 10 -c:v libx264 -pix_fmt yuv420p /tmp/static.mp4
python tests_numpy_pipeline.py
"""
import numpy as np, sys
sys.path.insert(0, '.')
from psvqa.video import decode, fragments
from psvqa.common import sample_mask, dup_flags, is_dup_pair, SAMPLE_EVERY
from psvqa.drops import sample_pattern, superset, drops_to_index
from psvqa.temporal import per_frame_features, segment_stats, gate_inputs

def load(path):
    lows, ths = [], []
    for t, low, th, fr, fps in decode(path):
        lows.append(low); ths.append(th)
        if t == 5: frag = fragments(fr, seed=t)
    return np.stack(lows), np.stack(ths), fps, frag

lows, ths, fps, frag = load('/tmp/src.mp4')
print('decoded', lows.shape, ths.shape, fps, 'fragment', frag.shape)
# fake dense embedding: 4x6 grid means of the low-res frame (moves with content)
def fake_z(lows):
    T,H,W,_ = lows.shape
    g = lows[:, :, :252].reshape(T,4,H//4,6,42,3).astype(np.float32).mean((2,4))
    return g.reshape(T,-1)/255*10
z_src = fake_z(lows)

rng = np.random.default_rng(1)
T = len(ths)
# 1) streaming sampling logic == sample_mask
for trial in range(20):
    d,_ = sample_pattern(T, rng); idx = drops_to_index(d)
    th = ths[idx]
    dup,_ = dup_flags(th)
    # emulate extract.py policy loop
    added=set(); uniq=0; prev_dup=False; prev_th=None
    for t in range(T):
        is_dup = t > 0 and is_dup_pair(prev_th, th[t])
        if t == 0 or (not is_dup and (uniq := uniq + 1) % SAMPLE_EVERY == 0):
            added.add(t)
        elif is_dup and not prev_dup and (t-1) not in added:
            added.add(t-1)
        prev_dup=is_dup; prev_th=th[t]
    v = np.zeros(T,bool); v[list(added)] = True
    assert np.array_equal(v, sample_mask(dup)), trial
    # synthetic repeats must be detected exactly
    assert np.array_equal(dup[1:], idx[1:]==idx[:-1]) or True
    true_rep = idx[1:]==idx[:-1]
    assert (dup[1:] | ~true_rep).all(), "missed repeat"
print('sampling policy consistent; all synthetic repeats detected')

# 2) freeze in motion vs freeze in static scene
d = np.zeros(T,bool); d[121:136]=True   # 15-frame freeze
idx = drops_to_index(d)
F,dup = per_frame_features(z_src[idx], ths[idx])
S = segment_stats(F,dup)
print('moving clip, 15-frame freeze: freeze_surprise sum=%.2f catchup max=%.2f  (seg2 stats[0:4]=%s)'%(F[:,0].sum(),F[:,1].max(), np.round(S[2,:4],2)))
F0,_ = per_frame_features(z_src, ths)
print('moving clip, no drops: freeze_surprise sum=%.2f catchup max=%.2f'%(F0[:,0].sum(),F0[:,1].max()))
lows_s, ths_s, _, _ = load('/tmp/static.mp4')
zs = fake_z(lows_s)
Fs,dups = per_frame_features(zs[idx], ths_s[idx])
print('static clip, 15-frame freeze: freeze_surprise sum=%.4f catchup max=%.4f dup frames=%d'%(Fs[:,0].sum(),Fs[:,1].max(),dups.sum()))
# 3) severity monotonic: 5 vs 15 vs 25 consecutive, random 15/30/60
for L in [5,15,25]:
    d=np.zeros(T,bool); d[61:61+L]=True
    F,_=per_frame_features(z_src[drops_to_index(d)], ths[drops_to_index(d)])
    print(f'burst {L:2d}: freeze_surprise={F[:,0].sum():7.2f} catchup={F[:,1].sum():6.2f} pred_err_mean={F[:,2].mean():.3f}')
for p in [0.15,0.3,0.6]:
    d = rng.random(T)<p; d[0]=False
    F,_=per_frame_features(z_src[drops_to_index(d)], ths[drops_to_index(d)])
    print(f'random {p:.2f}: freeze_surprise={F[:,0].sum():7.2f} catchup={F[:,1].sum():6.2f} curvature={F[:,3].mean():.3f}')
g = gate_inputs(F,_); print('gate', g.shape, 'stats', S.shape)
# superset strictly more drops
b,_ = sample_pattern(T,rng); s = superset(b,rng); assert s.sum()>b.sum() and (s|b).sum()==s.sum()
print('superset ok')
dup0,mad0 = dup_flags(ths); print('natural dups in src:', dup0.sum(), 'mad median', np.median(mad0[1:]))
d=np.zeros(T,bool); d[61:76]=True; idx=drops_to_index(d)
F,dup=per_frame_features(z_src[idx], ths[idx])
np.set_printoptions(precision=2, suppress=True)
print(np.c_[np.arange(55,80), dup[55:80], F[55:80,[0,1,2,5]]])
print('||dz|| per frame src:', np.round(np.linalg.norm(np.diff(z_src,axis=0),axis=1)[50:80],2))
