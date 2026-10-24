"""Pack many small JSON files into one JSONL (for backups): each line is
{"path": <path relative to the project root>, "text": <exact file text>}.
Reads in parallel (small-file opens are slow on /scratch). Restore with
tools/unpack_jsonl.py.

    python tools/pack_json_dirs.py OUT.jsonl DIR [DIR ...]
"""
import json
import os
import sys
from multiprocessing import Pool

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA"


def read(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return p, f.read()


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    paths = []
    for d in dirs:
        for dp, _, fs in os.walk(os.path.join(ROOT, d)):
            paths += [os.path.relpath(os.path.join(dp, f), ROOT) for f in fs if f.endswith(".json")]
    paths.sort()
    n = 0
    with Pool(32) as pool, open(out + ".tmp", "w", encoding="utf-8") as fo:
        for p, text in pool.imap(read, paths, chunksize=64):
            fo.write(json.dumps({"path": p, "text": text}) + "\n")
            n += 1
    os.replace(out + ".tmp", out)
    print(f"{out}: {n} files from {dirs}", flush=True)


if __name__ == "__main__":
    main()
