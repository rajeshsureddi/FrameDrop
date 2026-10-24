"""Step 15 decision tables (user 10-23: "a table like HFVQA comparison, and also study (with base lines table) to decide").

Reads every number from disk and writes vqa_data/step15_v4/STEP15_COMPARISON.md:
  Table 1  HFVQA Table 1 layout (arXiv 2609.16946): LSVQ-trained intra / cross-dataset + per-dataset fine-tune,
           public baselines (baselines_lsvq_public.csv) + ours v3, v4
  Table 2  FrameDrop-VQA study, official baselines (baselines_framedrop_vqa.csv) + our logged study evals

    python step15_tables.py
"""
import csv
import glob
import json
import os
import statistics

ROOT = "/scratch/09997/rs64952/FrameDrop-VQA/vqa_data"
S15 = f"{ROOT}/step15_v4"
S14 = f"{ROOT}/step14_finetune/runs/v3"
P1_DIRS = dict(v3=f"{ROOT}/frozen_v3_20261018/eval", v4=f"{S15}/eval/v4")
P1_SETS = ["lsvq_test", "lsvq_1080p", "konvid1k", "livevqc"]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f3(x):
    return "—" if x is None or x == "" else f"{float(x):.3f}"


def p1(model, ds):
    d = P1_DIRS[model]
    if model == "v3" and ds == "livevqc":
        d = f"{S15}/eval/v3"                           # v3 LIVE-VQC row scored in step 15 from step9 tokens
    m = jload(f"{d}/{ds}/metrics.json")
    return (m["Q_vs_MOS"]["SROCC"], m["Q_vs_MOS"]["PLCC"]) if m else (None, None)


def ft(ds):
    s = jload(f"{S14}/{ds}/summary.json")
    return (s["median"]["SROCC"], s["median"]["PLCC"], s["splits_done"]) if s else (None, None, 0)


def table1():
    rows = list(csv.DictReader(l for l in open(f"{ROOT}/baselines_lsvq_public.csv") if not l.startswith("#")))
    cols = ["lsvq_test", "lsvq1080p", "konvid_x", "livevqc_x", "konvid_ft", "livevqc_ft", "ytugc_ft", "lbvd_ft"]
    out = ["| Method | LSVQ test | LSVQ 1080p | KoNViD-1k (cross) | LIVE-VQC (cross) | KoNViD-1k (FT) | LIVE-VQC (FT) | YouTube-UGC (FT) | LBVD (FT) |",
           "|---|" + "---|" * len(cols)]
    for r in rows:
        out.append(f"| {r['method']} | " + " | ".join(f"{f3(r[c + '_srocc'])} / {f3(r[c + '_plcc'])}" if r[c + "_srocc"] else "— / —"
                                                    for c in cols) + " |")
    kf, lf, yf = ft("konvid1k"), ft("livevqc"), ft("ytugc")
    for m, label in (("v3", "**Ours v3** (LSVQ only)"), ("v4", "**Ours v4**† (LSVQ + YT-UGC/Gaming train)")):
        cells = [f"{f3(s)} / {f3(p)}" for s, p in (p1(m, d) for d in P1_SETS)]
        if m == "v3":
            cells += [f"{f3(kf[0])} / {f3(kf[1])}", f"{f3(lf[0])} / {f3(lf[1])}", f"{f3(yf[0])} / {f3(yf[1])}", "— / —"]
        else:
            cells += ["not run", "not run", "n/a‡", "— / —"]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out), (kf[2], lf[2], yf[2])


def protA(name):
    for p in (f"{ROOT}/step10_protA/{name}/summary.json", f"{ROOT}/step10_protA/summary.json"):
        s = jload(p)
        if s and name in s:
            return s[name]
    return None


def study_zero(path):
    m = jload(f"{path}/metrics.json")
    if not m:
        return None
    q = m["Q_vs_MOS"]
    return dict(PLCC=q["PLCC"], SROCC=q["SROCC"], KROCC=q["KROCC"], RMSE=q["RMSE"])


def study_ft():
    rs = [json.load(open(f))["test"] for f in sorted(glob.glob(f"{S14}/study/split*/result.json"))]
    if not rs:
        return None, 0
    return {k: statistics.median(r[k] for r in rs) for k in ("PLCC", "SROCC", "KROCC", "RMSE")}, len(rs)


def table2():
    base = list(csv.DictReader(open(f"{ROOT}/baselines_framedrop_vqa.csv")))
    rows = [dict(method=b["method"], PLCC=float(b["PLCC"]), SROCC=float(b["SROCC"]), KROCC=float(b["KROCC"]),
                 RMSE=float(b["RMSE"]), ours=False,
                 kind="zero-shot" if "zero-shot" in b["method"] else ("trained (20 splits)" if b["method"] not in ("NIQE", "BRISQUE") else "opinion-unaware"))
            for b in base]
    zs = [("Ours v3 (zero-shot)", f"{S15}/eval/v3/study"), ("Ours v4† (zero-shot)", f"{S15}/eval/v4/study"),
          ("Ours d1 (zero-shot, eval 8)", f"{ROOT}/step10_cand/eval/d1/study"),
          ("Ours ens(d1, v3_90) (zero-shot, eval 9)", f"{ROOT}/step12/eval/ens_d1_v3_90/study")]
    for name, p in zs:
        r = study_zero(p)
        if r:
            rows.append(dict(method=name, ours=True, kind="zero-shot", **r))
    for name, key in (("v3", "v3"), ("v4†", "v4")):
        a = protA(key)
        if a:
            for reg, lab in (("ridge", "ridge"), ("svr_cv", "SVR")):
                rows.append(dict(method=f"Ours {name} frozen features + {lab} (Protocol A)", ours=True,
                                 kind="trained (20 splits)", **a[reg]["median"]))
    f, n = study_ft()
    if f:
        rows.append(dict(method=f"Ours v3 fine-tuned per split (Step 14, {n}/20 splits)", ours=True,
                         kind=f"trained ({n}/20 splits)", **f))
    rows.sort(key=lambda r: -abs(r["SROCC"]))
    out = ["| # | Method | Protocol | PLCC | SROCC | KROCC | RMSE |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        b = "**" if r["ours"] else ""
        out.append(f"| {i} | {b}{r['method']}{b} | {r['kind']} | {f3(r['PLCC'])} | {f3(r['SROCC'])} | {f3(r['KROCC'])} | {f3(r['RMSE'])} |")
    return "\n".join(out), n


def main():
    t1, (nk, nl, ny) = table1()
    t2, ns = table2()
    md = f"""# Step 15 comparison tables (v3 vs v4)

Generated by `method/step15_tables.py`. Each cell is SROCC / PLCC.

## Table 1: HFVQA Table 1 layout (arXiv 2609.16946)

How the columns were measured:
- **Intra and cross columns:** train on LSVQ train, test as-is. We use the 4-parameter logistic for PLCC (step 7 metrics).
- **FT columns:** fine-tune per dataset over 10 random 70:10:20 splits and report the median test score (Step 14). Our PLCC here uses raw predictions.
- **Our FT runs:**
  - Init: v3. Encoders frozen for epochs 1-4, then all trained, 12 epochs.
  - Splits done: KoNViD {nk}/10, LIVE-VQC {nl}/10, YT-UGC {ny}/10.
  - YT-UGC has 977 clips: the 403 videos no longer in the bucket are missing.

{t1}

- † **v4** also trained on YT-UGC split0 train (684 clips) and LIVE-YT-Gaming train (430 clips). Its intra/cross columns are therefore not LSVQ-only, so flag this if v4 goes into the HFVQA table.
- ‡ v4 cannot be fine-tuned on YT-UGC under Protocol 2, because it has already trained on YT-UGC split0 train.
- v4 Protocol 2 on KoNViD and LIVE-VQC has not been run (about 10 GPU-hours).

## Table 2: FrameDrop-VQA study (1823 videos), official baselines + ours, sorted by |SROCC|

How each protocol was measured:
- **Zero-shot:** 4-parameter logistic over all 1823 videos; RMSE on the MOS scale.
- **Trained:** median over the baselines' 20 GroupShuffleSplit(test_size=.2, random_state=0) splits, raw predictions.
- **Ground truth:** mos_visibility_per_video.csv, the same GT as the baseline table.
- **Ours v3 zero-shot:** eval-5 predictions re-scored against the current GT. The logged eval 5 (.745) used the pre-10-18 MOS.
- **NIQE / BRISQUE:** their negative SROCC/KROCC are kept as in the official table.

{t2}

† v4 = v3 + YT-UGC/Gaming MOS (Step 15). No study data was used to train any zero-shot row.
"""
    open(f"{S15}/STEP15_COMPARISON.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
