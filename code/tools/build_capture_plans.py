"""plan_id -> ops [(name, base severity)] for every capture plan (from tmp_*_sev100.json)."""
import json, os, sys
from multiprocessing import Pool
enc = sys.argv[1]; out = sys.argv[2]
def one(sid):
    res = {}
    d = os.path.join(enc, sid)
    for f in os.listdir(d):
        if f.startswith("tmp_") and f.endswith("_sev100.json"):
            j = json.load(open(os.path.join(d, f)))
            res[j["plan_id"]] = dict(ops=[(o["name"], o["severity"], o["P"]) for o in j["plan"]["ops"]])
    return res
sids = sorted(s for s in os.listdir(enc) if not s.startswith("_"))
allp = {}
with Pool(24) as p:
    for r in p.imap_unordered(one, sids, chunksize=16):
        allp.update(r)
json.dump(allp, open(out, "w"))
print(len(sids), "sources", len(allp), "plans")
