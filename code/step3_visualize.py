"""Step 3 review figures -> vqa_data/step3/review/*.png

  1 tile_groups.png       all stored versions of a few tile groups (256 cores)
  2 tile_pairs.png        loader pair samples (better | worse), type, order, ops
  3 tile_ops.png          one clean-source tile under every on-the-fly op
  4 drop_strips.png       low-res strips of drop pairs around the added drops
  5 video_tiles.png       a displayed frame with the 32 sampled 9x15 cells + tiles
  6 mos_display.png       public-MOS clip as is (native res/fps, no 4K conversion)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_tile_loader as tl  # noqa: E402
import step3_video_loader as vl  # noqa: E402
import video_distortions as vd  # noqa: E402
from psvqa import drops as dr  # noqa: E402

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
TILES = f"{ROOT}/step3/tiles"
OUT = f"{ROOT}/step3/review"
MANIFEST = f"{ROOT}/step2/encodes_manifest.csv"
PLANS = f"{ROOT}/step3/capture_plans.json"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=80, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def label(v):
    if v["kind"] == "orig":
        return "orig (source)"
    if v["kind"] == "clean":
        return f"clean r{v['rank']}\n{v['size_bytes'] / 1e6:.1f} MB"
    return f"cap {v['encode_id'].split('_cap_')[1]}\n{v['size_bytes'] / 1e6:.1f} MB"


def fig_groups(shards, rng, n=3):
    rows = []
    for sh in rng.choice(shards, n, replace=False):
        g = tl.iter_groups(sh)
        for _ in range(int(rng.integers(0, 300))):
            meta, pngs = next(g)
        rows.append((meta, pngs))
    ncol = max(len(m["versions"]) for m, _ in rows)
    fig, ax = plt.subplots(n, ncol, figsize=(1.9 * ncol, 2.2 * n))
    for r, (meta, pngs) in enumerate(rows):
        for c in range(ncol):
            a = ax[r, c]
            a.axis("off")
            if c < len(meta["versions"]):
                v = meta["versions"][c]
                a.imshow(tl.center(tl.decode_png(pngs[v["v"]])))
                a.set_title(label(v), fontsize=7)
        ax[r, 0].text(-20, 128, f"{meta['source_id']}\nframe {meta['frame']}\n(x,y)={meta['x']},{meta['y']}",
                      fontsize=7, ha="right", va="center")
    fig.suptitle("Tile store: one group = same source/frame/position across all stored versions (256 core shown)")
    save(fig, "tile_groups.png")


def fig_pairs(shards, n=16):
    ds = tl.TileDataset(TILES, shards=list(shards[:40]), open_per_worker=40, seed=7)
    it = iter(ds)
    want = {"rate": 4, "capture_rate": 3, "severity": 3, "otf_severity": 4, "none": 2}
    got = []
    while sum(want.values()) > 0:
        s = next(it)
        if want.get(s["ptype"], 0) > 0:
            want[s["ptype"]] -= 1
            got.append(s)
    fig, ax = plt.subplots(len(got) // 2, 4, figsize=(11, 2.9 * len(got) // 2))
    for k, s in enumerate(got):
        r, c = k // 2, 2 * (k % 2)
        for j in range(2):
            ax[r, c + j].imshow(s["x"][j])
            ax[r, c + j].axis("off")
        tag = {1: "left better", 0: "no order"}[s["order"]]
        ax[r, c].set_title(f"{s['ptype']} ({tag})\n{s['versions'][0]}", fontsize=7)
        ax[r, c + 1].set_title(f"{s['ops'][:40]}\n{s['versions'][1]}", fontsize=7)
    fig.suptitle("Tile loader pair samples (after flip/jitter/on-the-fly ops, center-cropped)")
    save(fig, "tile_pairs.png")


def fig_ops(shards, rng):
    g = tl.iter_groups(shards[0])
    for _ in range(int(rng.integers(50, 400))):
        meta, pngs = next(g)
    t = tl.decode_png(pngs["v00"])
    x0, y0 = meta["x"] - 64, meta["y"] - 64
    names = list(tl.OPS_ANY)
    ncol = 7
    nrow = int(np.ceil((len(names) + 1) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2.2 * nrow))
    ax = ax.ravel()
    ax[0].imshow(tl.center(t))
    ax[0].set_title("clean source", fontsize=8)
    r = np.random.default_rng(1)
    for k, n in enumerate(names, 1):
        op = vd.sample_op(r, n, 5.0, severity=1.0)
        op = vd.replace(op, envelope=vd.Envelope(), support=vd.Support())
        plan = vd.Plan([op], 60.0, 1)
        m = plan.margin(2160)
        y = tl.render_tile(t, plan, meta["frame"], x0, y0)
        ax[k].imshow(tl.center(y))
        ax[k].set_title(f"{n}{' (margin>64: skipped)' if m > 64 else ''}", fontsize=7)
    for a in ax:
        a.axis("off")
    fig.suptitle("On-the-fly ops on one stored clean-source tile (severity 1.0, constant envelope)")
    save(fig, "tile_ops.png")


def fig_drops(n=3):
    ds = vl.VideoDataset(MANIFEST, mode="pair", tiles=False, pair_types=("drops",), plans_json=PLANS,
                         seed=11, lowres_dir=f"{ROOT}/step3/lowres")
    rng = np.random.default_rng(11)
    got = []
    while len(got) < n:
        s = ds.pair_sample(ds.sources[int(rng.integers(len(ds.sources)))], rng)
        if s is not None:
            got.append(s)
    L = 14
    fig, ax = plt.subplots(3 * n, L, figsize=(1.6 * L, 1.25 * 3 * n))
    for k, s in enumerate(got):
        P, Q = s["meta"][0]["drop_mask"], s["meta"][1]["drop_mask"]
        added = np.where(Q & ~P)[0]
        t0 = int(max(0, min(added[0] - 3, 240 - L)))
        for j in range(L):
            t = t0 + j
            for r, (img, m) in enumerate(((s["better"]["low"], P), (s["worse"]["low"], Q))):
                a = ax[3 * k + r, j]
                a.imshow(img[t])
                a.set_xticks([]), a.set_yticks([])
                if m[t]:
                    for sp in a.spines.values():
                        sp.set_color("red" if (r == 1 and not P[t]) else "black")
                        sp.set_linewidth(3)
                if j == 0:
                    a.set_ylabel(("P" if r == 0 else "superset") + f"\n{int(m.sum())} drops", fontsize=7)
            a = ax[3 * k + 2, j]
            a.imshow(np.abs(s["better"]["low"][t].astype(int) - s["worse"]["low"][t].astype(int)).sum(-1),
                     cmap="magma", vmin=0, vmax=60)
            a.set_xticks([]), a.set_yticks([])
            a.set_xlabel(f"tick {t}", fontsize=6)
            if j == 0:
                a.set_ylabel("|diff|", fontsize=7)
        ax[3 * k, 0].set_title(f"{s['meta'][0]['encode']} {s['source_id'][:28]}  family {s['meta'][0]['drop_family']}"
                               f"  window start {s['meta'][0]['start']}  mean|diff| {s['mean_abs_diff']:.2f}",
                               fontsize=8, loc="left")
    fig.suptitle("Drop pairs (384x216 displayed frames). Box: dropped tick (black in both, red = added in superset)")
    save(fig, "drop_strips.png")


def fig_video_tiles():
    ds = vl.VideoDataset(MANIFEST, mode="single", tiles=True, plans_json=PLANS, seed=5)
    rng = np.random.default_rng(5)
    s = None
    while s is None:
        s = ds.single_sample(ds.sources[int(rng.integers(len(ds.sources)))], rng)
    k = 4
    d = s["tile_ticks"][k]
    fig = plt.figure(figsize=(16, 7))
    a = fig.add_axes([0, 0.3, 0.5, 0.7])
    a.imshow(s["low"][d], extent=(0, 3840, 2160, 0))
    for r, c in s["cells"][k]:
        a.add_patch(plt.Rectangle((c * 256, r * 238), 256, 256, fill=False, ec="yellow", lw=1))
    a.set_title(f"{s['meta']['encode']} tick {d}: 32 of 135 grid cells (row stride 238, col 256)", fontsize=9)
    for j in range(12):
        b = fig.add_axes([0.52 + (j % 6) * 0.08, 0.62 - (j // 6) * 0.3, 0.075, 0.28])
        b.imshow(s["tiles"][k, j])
        b.set_title(f"cell {tuple(int(x) for x in s['cells'][k, j])}", fontsize=7)
        b.axis("off")
    save(fig, "video_tiles.png")


def fig_mos():
    kon = f"{ROOT}/../mos_datasets/KonViD/KoNViD_1k_videos"
    paths = sorted(os.listdir(kon))[:3]
    items = [dict(path=os.path.join(kon, p), dataset="KoNViD-1k") for p in paths]
    fig, ax = plt.subplots(len(items), 3, figsize=(15, 3.2 * len(items)))
    for i, it in enumerate(items):
        s = vl.mos_native(it, start=0, n=48, seed=2)
        pb = it["probe"]
        ax[i, 0].imshow(s["low"][0])
        ax[i, 0].set_title(f"{os.path.basename(it['path'])} {pb['w']}x{pb['h']} @ {pb['fps']:.3f} fps, as is\n"
                           f"low-res input (fit 384x216, downscale only); tile frames {s['tile_frames'].tolist()}",
                           fontsize=8)
        x, y = (int(v) for v in s["tile_xy"][0, 0])
        ax[i, 1].imshow(s["tiles"][0, 0])
        ax[i, 1].set_title(f"native-resolution 256 tile at (x,y)=({x},{y}), no upscale", fontsize=8)
        rep = [int((s["low"][t] == s["low"][t - 1]).all()) for t in range(1, 25)]
        ax[i, 2].bar(range(1, 25), rep)
        ax[i, 2].set_ylim(0, 1)
        ax[i, 2].set_title("frame t identical to t-1? (should be 0: native frame rate, no repetition)", fontsize=8)
        ax[i, 0].axis("off"), ax[i, 1].axis("off")
    fig.suptitle("Public-MOS clips are used as is (native resolution and frame rate)")
    save(fig, "mos_display.png")


def main():
    rng = np.random.default_rng(0)
    shards = tl.list_shards(TILES, ("train",))
    which = sys.argv[1:] or ["groups", "pairs", "ops", "drops", "video", "mos"]
    for w in which:
        dict(groups=lambda: fig_groups(shards, rng), pairs=lambda: fig_pairs(shards),
             ops=lambda: fig_ops(shards, rng), drops=fig_drops, video=fig_video_tiles, mos=fig_mos)[w]()


if __name__ == "__main__":
    main()
