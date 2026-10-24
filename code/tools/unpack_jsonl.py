"""Restore files packed by tools/pack_json_dirs.py.

    python tools/unpack_jsonl.py PACK.jsonl DEST_ROOT
"""
import json
import os
import sys


def main():
    pack, dest = sys.argv[1], sys.argv[2]
    n = 0
    with open(pack, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p = os.path.join(dest, r["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fo:
                fo.write(r["text"])
            n += 1
    print(f"restored {n} files under {dest}")


if __name__ == "__main__":
    main()
