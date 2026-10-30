#!/usr/bin/env python3
"""Smoke test for the RQ1 extras in results/rearranged/ (read-only; writes nothing).

qrels_official/, calibration_r6/, metrics_human_only/, task2_r5/, formula_overlap_r7/.
Scores are recomputed with a pure-Python trec_eval (no pytrec_eval) from the copied run and
qrel files; published figures and the pre-existing result files are used as external anchors.

Usage:
    python experiments/rearrange/smoke_test_rq1_extras.py      # exit 1 on any failure
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    CONFIGS, OFFICIAL_QRELS, OFFICIAL_QRELS_T2, OUT, ROOT, SRC_R6_RUN, SRC_T2_CFT, SRC_T2_OURS, read_qrels, sha256,
)

FAILS: list[str] = []
R = OUT


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------- an independent trec_eval ----------------------------------------------------------
def load_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    run: dict[str, dict[str, float]] = defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        q, d, s = (p[0], p[2], float(p[4])) if p[1].upper() == "Q0" else (p[0], p[1], float(p[3]))
        run[q][d] = s
    return {q: list(ds.items()) for q, ds in run.items()}


def te_order(docs: list[tuple[str, float]]) -> list[str]:          # trec_eval: score desc, docno desc
    return [d for d, _ in sorted(docs, key=lambda x: (x[1], x[0]), reverse=True)]


def py_order(docs: list[tuple[str, float]]) -> list[str]:          # hole_rate's ranking: score desc, stable
    return [d for d, _ in sorted(docs, key=lambda x: -x[1])]


def measures(docs, qrel_t, prime: bool, ks=(5, 10, 20)) -> dict[str, float]:
    order = te_order([(d, s) for d, s in docs if (d in qrel_t or not prime)])
    gains = [max(qrel_t.get(d, 0), 0) for d in order]
    ideal = sorted((g for g in qrel_t.values() if g > 0), reverse=True)
    out = {}
    for k in ks:
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]) if g > 0)
        idc = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        out[f"ndcg@{k}"] = dcg / idc if idc else 0.0
        out[f"P@{k}"] = sum(1 for g in gains[:k] if g >= 2) / k
    nrel = sum(1 for g in qrel_t.values() if g >= 2)
    hits, ap = 0, 0.0
    for i, g in enumerate(gains, 1):
        if g >= 2:
            hits += 1
            ap += hits / i
    out["map"] = ap / nrel if nrel else 0.0
    return out


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def verify_folder(folder: Path) -> None:
    sums = folder / "SHA256SUMS"
    listed = {}
    for ln in sums.read_text().splitlines():
        h, name = ln.split("  ", 1)
        listed[name] = h
    bad = [n for n, h in listed.items() if sha256(folder / n) != h]
    present = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    check(f"{folder.name}/SHA256SUMS verifies, no unlisted files",
          not bad and present == set(listed), f"{len(listed)} files" + (f"; bad {bad}" if bad else "")
          + (f"; unlisted {sorted(present - set(listed))}" if present != set(listed) else ""))


def main() -> int:
    q1, q2 = read_qrels(OFFICIAL_QRELS), read_qrels(OFFICIAL_QRELS_T2)

    print("1. Layout and checksums")
    for name in ("qrels_official", "calibration_r6", "metrics_human_only", "task2_r5", "formula_overlap_r7"):
        verify_folder(R / name)
    bad = [ln for ln in (R / "SHA256SUMS").read_text().splitlines()
           if sha256(R / ln.split("  ", 1)[1]) != ln.split("  ", 1)[0]]
    check("root SHA256SUMS still verifies (README section appended)", not bad, str(bad))

    print("2. Copies of inputs are byte-identical to their sources")
    pairs = [(R / "qrels_official" / OFFICIAL_QRELS.name, OFFICIAL_QRELS),
             (R / "qrels_official" / OFFICIAL_QRELS_T2.name, OFFICIAL_QRELS_T2),
             (R / "calibration_r6" / SRC_R6_RUN.name, SRC_R6_RUN),
             (R / "task2_r5" / "runs" / SRC_T2_OURS.name, SRC_T2_OURS),
             (R / "task2_r5" / "runs" / SRC_T2_CFT.name, SRC_T2_CFT)]
    diff = [str(a.relative_to(R)) for a, b in pairs if sha256(a) != sha256(b)]
    check("qrels, R6 run, both Task 2 runs", not diff, str(diff))
    qall = read_qrels(ROOT / "data/raw/qrels/qrel_task1_2022_all.tsv")
    check("Task 1 official: 78 topics, identical to qrel_task1_2022_all on those topics",
          len(q1) == 78 and all(qall[t] == q1[t] for t in q1))
    check("Task 2 official: 76 topics", len(q2) == 76)

    print("3. R6 calibration")
    run = load_run(R / "calibration_r6" / SRC_R6_RUN.name)
    pt = {t: measures(run.get(t, []), q1[t], prime=True, ks=(10, 1000)) for t in q1}
    ours = {"ndcg'": sum(v["ndcg@1000"] for v in pt.values()) / 78,
            "map'": sum(v["map"] for v in pt.values()) / 78,
            "precision'@10": sum(v["P@10"] for v in pt.values()) / 78}
    rows = {r["metric"]: r for r in read_csv(R / "calibration_r6" / "calibration_task1_tfidf_terrier.csv")}
    err = max(abs(ours[m] - float(rows[m]["ours"])) for m in ours)
    check("file == pure-Python recomputation", err < 1e-9, f"max |diff| {err:.1e}")
    pub = {"ndcg'": 0.272, "map'": 0.064, "precision'@10": 0.124}
    check("matches ARQMath-3 Table 3 at published precision (|diff| <= 0.0005)",
          all(abs(ours[m] - pub[m]) <= 0.0005 for m in pub),
          ", ".join(f"{m} {ours[m]:.4f} vs {pub[m]}" for m in pub))

    print("4. human_only and hole rates, 78 topics")
    per = {(r["config_name"], r["topic_id"]): r for r in read_csv(R / "metrics_human_only" / "metrics_per_topic.csv")}
    agg = {r["config_name"]: r for r in read_csv(R / "metrics_human_only" / "metrics_aggregate.csv")}
    check("15 configs x 78 topics", len(agg) == 15 and len(per) == 15 * 78 and all(a["n_topics"] == "78" for a in agg.values()))
    err = aerr = 0.0
    out_of_range = 0
    for _, _, cfg in CONFIGS:
        run = load_run(R / "runs_task1" / f"{cfg}.tsv")
        sums = defaultdict(float)
        for t in q1:
            std, prm = measures(run[t], q1[t], prime=False), measures(run[t], q1[t], prime=True)
            order = py_order(run[t])
            vals = {}
            for k in (5, 10, 20):
                vals[f"ndcg@{k}"], vals[f"precision@{k}"] = std[f"ndcg@{k}"], std[f"P@{k}"]
                vals[f"ndcg'@{k}"], vals[f"precision'@{k}"] = prm[f"ndcg@{k}"], prm[f"P@{k}"]
                top = order[:k]
                vals[f"hole@{k}"] = sum(1 for d in top if d not in q1[t]) / len(top)
            row = per[(cfg, t)]
            for c, v in vals.items():
                err = max(err, abs(v - float(row[c])))
                out_of_range += not (0.0 <= float(row[c]) <= 1.0)
                sums[c] += v
        aerr = max(aerr, max(abs(sums[c] / 78 - float(agg[cfg][c])) for c in sums))
    check("per-topic file == pure-Python recomputation", err < 1e-9, f"max |diff| {err:.1e}")
    check("aggregate == mean of the 78 topics", aerr < 1e-9, f"max |diff| {aerr:.1e}")
    check("every value in [0, 1]", out_of_range == 0)
    # regression anchor: the same scorer on the old 87-topic qrels reproduces results/track_b human_only
    reg = 0.0
    for _, _, cfg in CONFIGS:
        old = [r for r in read_csv(ROOT / "results/track_b" / f"{cfg}_metrics.csv") if r["channel"] == "human_only"]
        if not old:
            continue
        old = old[0]
        run = load_run(R / "runs_task1" / f"{cfg}.tsv")
        s = defaultdict(float)
        for t in qall:
            std, prm = measures(run.get(t, []), qall[t], False), measures(run.get(t, []), qall[t], True)
            for k in (5, 10, 20):
                s[f"ndcg@{k}"] += std[f"ndcg@{k}"]
                s[f"precision@{k}"] += std[f"P@{k}"]
                s[f"ndcg'@{k}"] += prm[f"ndcg@{k}"]
                s[f"precision'@{k}"] += prm[f"P@{k}"]
        reg = max(reg, max(abs(v / len(qall) - float(old[c])) for c, v in s.items()))
    check("same scorer on the 87 'all' topics reproduces results/track_b human_only (15 configs, 12 measures)",
          reg < 1e-9, f"max |diff| {reg:.1e}")

    print("5. Task 2 (R5.1 and TangentCFT2)")
    col = load_run(R / "task2_r5" / "runs" / "DPRL-Task2-CFT-auto-math-A.visual_id_collapsed.tsv")
    dup = [t for t, ds in col.items() if len({d for d, _ in ds}) != len(ds)]
    by_t = defaultdict(list)
    for line in open(R / "task2_r5" / "runs" / "DPRL-Task2-CFT-auto-math-A.visual_id_collapsed.tsv"):
        p = line.split("\t")
        by_t[p[0]].append(int(p[3]))
    ranks_ok = all(v == list(range(1, len(v) + 1)) for v in by_t.values())
    check("collapsed TangentCFT2: one row per visual id per topic, ranks 1..n", not dup and ranks_ok and len(col) == 100)
    m2 = {r["run"]: r for r in read_csv(R / "task2_r5" / "metrics_task2.csv")}
    h2 = {r["run"]: r for r in read_csv(R / "task2_r5" / "hole_rates_task2.csv")}
    runs = {"R5.1 (this thesis)": load_run(R / "task2_r5" / "runs" / SRC_T2_OURS.name), "TangentCFT2": col}
    err = herr = 0.0
    got = {}
    for name, run in runs.items():
        pts = {t: measures(run.get(t, []), q2[t], prime=True, ks=(5, 10, 20, 1000)) for t in q2}
        got[name] = {"ndcg'@1000": sum(v["ndcg@1000"] for v in pts.values()) / 76,
                     "map'": sum(v["map"] for v in pts.values()) / 76,
                     "precision'@10": sum(v["P@10"] for v in pts.values()) / 76,
                     "ndcg'@5": sum(v["ndcg@5"] for v in pts.values()) / 76,
                     "ndcg'@10": sum(v["ndcg@10"] for v in pts.values()) / 76,
                     "precision'@5": sum(v["P@5"] for v in pts.values()) / 76}
        err = max(err, max(abs(v - float(m2[name][c])) for c, v in got[name].items()))
        for k in (5, 10, 20, 100, 1000):
            hr = sum(sum(1 for d in py_order(run[t])[:k] if d not in q2[t]) / len(py_order(run[t])[:k]) for t in q2) / 76
            herr = max(herr, abs(hr - float(h2[name][f"hole@{k}"])))
    check("metrics_task2.csv == pure-Python recomputation", err < 1e-9, f"max |diff| {err:.1e}")
    check("hole_rates_task2.csv == pure-Python recomputation", herr < 1e-9, f"max |diff| {herr:.1e}")
    c = got["TangentCFT2"]
    check("TangentCFT2 reproduces ARQMath-3 Table 4 (0.641 / 0.419 / 0.534)",
          abs(c["ndcg'@1000"] - 0.641) <= 5e-4 and abs(c["map'"] - 0.419) <= 5e-4 and abs(c["precision'@10"] - 0.534) <= 5e-4,
          f"{c['ndcg' + chr(39) + '@1000']:.4f} / {c['map' + chr(39)]:.4f} / {c['precision' + chr(39) + '@10']:.4f}")
    old = {r["metric"]: float(r["value"]) for r in read_csv(ROOT / "results/track_a/a3_slt_opt_slt_type_rrf_k60_n1000_2022_20260902.csv")}
    o = got["R5.1 (this thesis)"]
    e = max(abs(o[k] - old[k]) for k in ("ndcg'@1000", "map'", "precision'@10", "ndcg'@5", "ndcg'@10", "precision'@5"))
    check("R5.1 == existing results/track_a file", e < 1e-9, f"max |diff| {e:.1e}")
    man = {"R5.1 (this thesis)": {5: 0.263, 10: 0.355, 20: 0.455, 100: 0.731, 1000: 0.943}, "TangentCFT2": {10: 0.026, 1000: 0.921}}
    check("hole rates match the manifest's R5 figures (3 d.p.)",
          all(abs(float(h2[n][f"hole@{k}"]) - v) <= 5e-4 for n, d in man.items() for k, v in d.items()))
    lens = [len(col[t]) for t in col]
    print(f"  info: collapsed TangentCFT2 length, mean over 100 topics {sum(lens) / len(lens):.1f}, "
          f"over the 76 assessed {sum(len(col[t]) for t in q2) / 76:.1f} (manifest quotes 747.7)")

    print("6. R7 formula overlap")
    summ = read_csv(R / "formula_overlap_r7" / "overlap_summary.csv")
    bt = read_csv(R / "formula_overlap_r7" / "overlap_by_topic.csv")
    check("6 summary rows, all n_topics == 78", len(summ) == 6 and all(r["n_topics"] == "78" for r in summ))
    check("per-topic rows cover exactly the 78 official topics", {r["topic_id"] for r in bt} == set(q1))

    def recompute(rows):
        g = defaultdict(dict)
        for r in rows:
            if r["relevance_def"] == "high_medium":
                g[(r["query_strategy"], r["representation"], r["topic_id"])][r["group"]] = r
        out = {}
        for (s, rep) in {(k[0], k[1]) for k in g}:
            tops = sorted(t for (s2, rep2, t) in g if (s2, rep2) == (s, rep) and "relevant" in g[(s, rep, t)])
            row = {"n_topics": len(tops)}
            for st, col_ in (("mean", "mean_containment"), ("max", "max_containment"), ("top5_mean", "top5_mean_containment")):
                rel = [float(g[(s, rep, t)]["relevant"][col_]) for t in tops]
                non = [float(g[(s, rep, t)]["non_relevant"][col_]) for t in tops]
                rnd = [float(g[(s, rep, t)]["random"][col_]) for t in tops]
                row[f"{st}_relevant"], row[f"{st}_non_relevant"], row[f"{st}_random"] = (
                    sum(rel) / len(tops), sum(non) / len(tops), sum(rnd) / len(tops))
                row[f"n_topics_{st}_relevant_gt_non_relevant"] = sum(a > b for a, b in zip(rel, non))
                row[f"n_topics_{st}_relevant_gt_random"] = sum(a > b for a, b in zip(rel, rnd))
            out[(s, rep)] = row
        return out

    rc = recompute(bt)
    worst = max(abs(float(r[c]) - float(rc[(r["query_strategy"], r["representation"])][c]))
                for r in summ for c in rc[(r["query_strategy"], r["representation"])])
    check("summary == recomputation from overlap_by_topic.csv", worst < 1e-12, f"max |diff| {worst:.1e}")

    def near_ties(rows):
        g = defaultdict(dict)
        for r in rows:
            if r["relevance_def"] == "high_medium":
                g[(r["query_strategy"], r["representation"], r["topic_id"])][r["group"]] = r
        return {(s, rep, t, c) for (s, rep, t), gr in g.items() if "relevant" in gr and "non_relevant" in gr
                for c in ("mean_containment", "max_containment", "top5_mean_containment")
                if float(gr["relevant"][c]) != float(gr["non_relevant"][c])
                and abs(float(gr["relevant"][c]) - float(gr["non_relevant"][c])) < 1e-9}

    check("78-topic run has no floating-point near-ties that could flip a win count", not near_ties(bt))
    val = {(r["query_strategy"], r["representation"]): r for r in read_csv(R / "formula_overlap_r7" / "validation_87_topics" / "overlap_summary.csv")}
    orig = {(r["query_strategy"], r["representation"]): r for r in read_csv(ROOT / "results/formula_overlap/overlap_results.csv")}
    fl = max(abs(float(orig[k][c]) - float(val[k][c])) for k in orig for c in orig[k]
             if c not in ("query_strategy", "representation") and not c.startswith("n_topics"))
    check("validation: 87-topic re-run == original overlap_results.csv on every mean/max/top-5 value", fl < 1e-12, f"max |diff| {fl:.1e}")
    cnt = [(k, c, orig[k][c], val[k][c]) for k in orig for c in orig[k] if c.startswith("n_topics_") and orig[k][c] != val[k][c]]
    vt = near_ties(read_csv(R / "formula_overlap_r7" / "validation_87_topics" / "overlap_by_topic.csv"))
    explained = all(any(s == k[0] and rep == k[1] and c.replace("n_topics_", "").replace("_relevant_gt_non_relevant", "") in col_
                        for (s, rep, _, col_) in vt) for k, c, _, _ in cnt)
    check("validation: every differing win count sits on a floating-point near-tie", explained,
          f"{len(cnt)} counts differ by 1: {[(k[0], k[1], c.split('_')[2]) for k, c, _, _ in cnt]}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
