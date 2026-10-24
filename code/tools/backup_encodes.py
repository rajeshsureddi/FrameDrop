"""Back up the Step 2 encodes as self-contained zips (< 15 GB each) for UT Box.

Each part holds whole source folders vqa_data/step2/encodes/<source_id>/ (all
mp4 encodes + their JSON records), stored without recompression (mp4 is
already compressed). encodes/_quarantine_fps (Step 2 leftovers awaiting
deletion) is excluded. Writes encodes_parts_index.csv (source_id -> part),
then zips N parts in parallel and tests each with `unzip -t`.

    python tools/backup_encodes.py [--out DIR] [--jobs 6] [--max-gb 14]
Re-runnable: parts whose zip already tested OK (.ok marker) are skipped.
"""
import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA"
ENC = "vqa_data/step2/encodes"


def dir_size(d):
    return sum(e.stat().st_size for e in os.scandir(d) if e.is_file())


def plan(max_bytes):
    srcs = sorted(e.name for e in os.scandir(os.path.join(ROOT, ENC)) if e.is_dir() and not e.name.startswith("_"))
    with ThreadPoolExecutor(32) as ex:
        sizes = list(ex.map(lambda s: dir_size(os.path.join(ROOT, ENC, s)), srcs))
    parts, cur, cur_b = [], [], 0
    for s, b in zip(srcs, sizes):
        if cur and cur_b + b > max_bytes:
            parts.append((cur, cur_b))
            cur, cur_b = [], 0
        cur.append(s)
        cur_b += b
    if cur:
        parts.append((cur, cur_b))
    return parts, dict(zip(srcs, sizes))


def make(k, srcs, out):
    z = os.path.join(out, f"step2_encodes_part{k:03d}.zip")
    if os.path.exists(z + ".ok"):
        return k, "skip"
    if os.path.exists(z):
        os.remove(z)
    lst = "\n".join(f"{ENC}/{s}" for s in srcs) + "\n"
    p = subprocess.run(["zip", "-q", "-0", "-r", z, "-@", "-x", "*.mkv"], input=lst.encode(), cwd=ROOT,
                       capture_output=True)
    if p.returncode:
        return k, "ZIP FAILED " + p.stderr.decode()[-300:]
    t = subprocess.run(["unzip", "-tq", z], capture_output=True)
    if t.returncode:
        return k, "TEST FAILED " + t.stdout.decode()[-300:]
    open(z + ".ok", "w").close()
    return k, f"ok {os.path.getsize(z) / 2**30:.1f} GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/backup_utbox_20261016/step2_encodes")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=14.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts, sizes = plan(a.max_gb * 2**30)
    with open(os.path.join(a.out, "encodes_parts_index.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "part", "bytes"])
        for k, (srcs, _) in enumerate(parts):
            for s in srcs:
                w.writerow([s, f"step2_encodes_part{k:03d}.zip", sizes[s]])
    print(f"{len(parts)} parts, {sum(b for _, b in parts) / 2**40:.2f} TiB, {len(sizes)} sources", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, msg in ex.map(lambda kv: make(kv[0], kv[1][0], a.out), enumerate(parts)):
            print(f"part {k:03d}: {msg}", flush=True)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
