#!/usr/bin/env python3
"""Smoke test for results/rearranged/ (read-only; writes nothing).

Every check is independent of the code that built the files where that is
possible: labels are re-read from the original audit CSVs, run files from
data/runs/task1/, and the metrics are recomputed with a pure-Python nDCG/P
that shares no code with pytrec_eval.

Usage:
    python experiments/rearrange/smoke_test_rearranged.py      # exit 1 on any failure
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    CONFIGS, CUTOFFS, OFFICIAL_QRELS, OUT, PER_RUN_SUFFIX, POOL_DEPTH, POOLED_QREL_NAME,
    SRC_AUDIT_DIR, SRC_RUNS_DIR, SRC_TRACK_V_DIR, ranking, read_qrels, read_run, sha256,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def audit(cfg: str) -> list[dict]:
    with open(SRC_AUDIT_DIR / f"{cfg}_answers_audit.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["llm_label"] = None if r["llm_label"] in ("", "null", "None") else int(r["llm_label"])
        r["retrieval_score"] = float(r["retrieval_score"])
    return rows


def trec_order(docs: list[tuple[str, float]]) -> list[str]:
    """trec_eval's ordering: score descending, ties broken by docno descending."""
    return [d for d, _ in sorted(docs, key=lambda x: (x[1], x[0]), reverse=True)]


def ndcg(gains: list[int], ideal: list[int], k: int) -> float:
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]) if g > 0)
    idc = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:k]) if g > 0)
    return dcg / idc if idc > 0 else 0.0


def prec(gains: list[int], k: int) -> float:
    return sum(1 for g in gains[:k] if g >= 2) / k


def main() -> int:
    R = OUT
    cfgs = [c for _, _, c in CONFIGS]
    official78 = set(read_qrels(OFFICIAL_QRELS))

    print("1. Layout and checksums")
    exp = ([R / "runs_task1" / f"{c}.tsv" for c in cfgs]
           + [R / "judge_qrels_per_run" / f"{c}{PER_RUN_SUFFIX}" for c in cfgs]
           + [R / "judge_qrels_pooled" / n for n in (POOLED_QREL_NAME, "pool_per_topic.csv", "excluded_unparsed.tsv")]
           + [R / "metrics_judge_pool20" / n for n in ("metrics_aggregate.csv", "metrics_per_topic.csv", "idcg_per_topic.csv", "SHA256SUMS")]
           + [R / "README.md", R / "SHA256SUMS"])
    missing = [str(p.relative_to(R)) for p in exp if not p.exists()]
    check("all expected files present", not missing, f"missing {missing}" if missing else f"{len(exp)} files")
    # only the folders this build owns; the RQ1 extras have their own smoke test
    owned = ("runs_task1", "judge_qrels_per_run", "judge_qrels_pooled", "metrics_judge_pool20", "track_v")
    actual = {p for p in R.rglob("*") if p.is_file() and (p.parent == R or p.relative_to(R).parts[0] in owned)}
    tv_src = sorted(p for p in SRC_TRACK_V_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    extra = actual - set(exp) - {R / "track_v" / p.name for p in tv_src}
    check("no unexpected files", not extra, str(sorted(map(str, extra))[:5]))
    for sums, base in ((R / "SHA256SUMS", R), (R / "metrics_judge_pool20" / "SHA256SUMS", R / "metrics_judge_pool20")):
        bad = [ln.split("  ", 1)[1] for ln in sums.read_text().splitlines()
               if sha256(base / ln.split("  ", 1)[1]) != ln.split("  ", 1)[0]]
        check(f"{sums.relative_to(R)} verifies", not bad, f"bad: {bad}" if bad else f"{len(sums.read_text().splitlines())} files")

    print("2. Copies are byte-identical to their sources")
    diff = [c for c in cfgs if sha256(R / "runs_task1" / f"{c}.tsv") != sha256(SRC_RUNS_DIR / f"{c}.tsv")]
    check("15 run files == data/runs/task1/", not diff, str(diff))
    diff = [p.name for p in tv_src if sha256(R / "track_v" / p.name) != sha256(p)]
    check(f"{len(tv_src)} Track V files == results/track_v/", not diff, str(diff))

    print("3. Qrel format (16 files)")
    qfiles = [R / "judge_qrels_per_run" / f"{c}{PER_RUN_SUFFIX}" for c in cfgs] + [R / "judge_qrels_pooled" / POOLED_QREL_NAME]
    problems = []
    for qf in qfiles:
        text = qf.read_text()
        if not text.endswith("\n"):
            problems.append(f"{qf.name}: no trailing newline")
        seen = set()
        for i, ln in enumerate(text.split("\n")[:-1], 1):
            p = ln.split("\t")
            if len(p) != 4 or p[1] != "0" or p[0] not in official78 or p[3] not in {"0", "1", "2", "3"} or not p[2].isdigit():
                problems.append(f"{qf.name}:{i}: {ln!r}")
            if (p[0], p[2]) in seen:
                problems.append(f"{qf.name}:{i}: duplicate pair")
            seen.add((p[0], p[2]))
    check("4 tab-separated fields, iter 0, official topic, grade 0-3, numeric id, no duplicates",
          not problems, "; ".join(problems[:3]))

    print("4. Labels round-trip against the original audit CSVs")
    per_run = {c: read_qrels(R / "judge_qrels_per_run" / f"{c}{PER_RUN_SUFFIX}") for c in cfgs}
    audits = {c: audit(c) for c in cfgs}
    bad = []
    for c in cfgs:
        want = {(r["topic_id"], r["answer_id"], r["llm_label"]) for r in audits[c]
                if r["topic_id"] in official78 and r["llm_label"] is not None}
        got = {(t, d, g) for t, ds in per_run[c].items() for d, g in ds.items()}
        if want != got or set(per_run[c]) != official78:
            bad.append(c)
    check("each per-run qrel == its audit labels on the 78 topics, all 78 topics present", not bad, str(bad))
    dropped_topics = {r["topic_id"] for c in cfgs for r in audits[c]} - official78
    check("non-official topics dropped", len(dropped_topics) == 22, f"{len(dropped_topics)} topics dropped (100 - 78)")

    print("5. Pooled qrel == union of the 15 per-run qrels")
    pooled = read_qrels(R / "judge_qrels_pooled" / POOLED_QREL_NAME)
    union, conflicts = {}, 0
    for c in cfgs:
        for t, ds in per_run[c].items():
            for d, g in ds.items():
                if (t, d) in union and union[(t, d)] != g:
                    conflicts += 1
                union[(t, d)] = g
    got = {(t, d): g for t, ds in pooled.items() for d, g in ds.items()}
    check("same pairs, same grades", got == union, f"{len(got)} pairs")
    check("no grade conflicts between runs", conflicts == 0)
    check("78 topics, every IDCG > 0", set(pooled) == official78 and all(max(ds.values()) > 0 for ds in pooled.values()))

    print("6. Coverage: every top-20 answer is in the pool, except the listed unparsed ones")
    excluded = set()
    with open(R / "judge_qrels_pooled" / "excluded_unparsed.tsv") as f:
        next(f)
        excluded = {tuple(ln.split("\t")[:2]) for ln in f if ln.strip()}
    want_excl = {(r["topic_id"], r["answer_id"]) for c in cfgs for r in audits[c]
                 if r["topic_id"] in official78 and r["llm_label"] is None} - set(got)
    check("excluded_unparsed.tsv == unparsed audit pairs", excluded == want_excl, f"{len(excluded)} pairs")
    runs = {c: read_run(R / "runs_task1" / f"{c}.tsv") for c in cfgs}
    bad_cov, bad_top = [], []
    for c in cfgs:
        rk = ranking(runs[c])
        top = {(t, d) for t in official78 for d in rk[t][:POOL_DEPTH]}
        if top != {(r["topic_id"], r["answer_id"]) for r in audits[c] if r["topic_id"] in official78}:
            bad_top.append(c)
        if top - set(got) - excluded:
            bad_cov.append(c)
    check("each run file's top 20 == the answers its audit labelled", not bad_top, str(bad_top))
    check("each run's top 20 is fully judged in the pool (minus unparsed)", not bad_cov, str(bad_cov))

    print("7. Metrics recomputed independently (pure Python, no pytrec_eval)")
    with open(R / "metrics_judge_pool20" / "metrics_aggregate.csv", newline="") as f:
        agg = {r["config_name"]: r for r in csv.DictReader(f)}
    with open(R / "metrics_judge_pool20" / "metrics_per_topic.csv", newline="") as f:
        pt = {(r["config_name"], r["topic_id"]): r for r in csv.DictReader(f)}
    with open(R / "metrics_judge_pool20" / "idcg_per_topic.csv", newline="") as f:
        idcg_file = {r["topic_id"]: r for r in csv.DictReader(f)}
    check("15 configs x 78 topics", len(agg) == 15 and len(pt) == 15 * 78 and all(a["n_topics"] == "78" for a in agg.values()))
    cols = [f"ndcg'@{k}" for k in CUTOFFS] + [f"precision'@{k}" for k in CUTOFFS]
    in_range = all(0.0 <= float(r[c]) <= 1.0 for r in list(pt.values()) + list(agg.values()) for c in cols)
    check("every value in [0, 1]", in_range)

    max_err = max_perr = max_ierr = max_full = 0.0
    implied_bad = 0
    for c in cfgs:
        rk = ranking(runs[c])
        sums = defaultdict(float)
        for t in official78:
            top = [(d, runs[c][t][d]) for d in rk[t][:POOL_DEPTH] if d in pooled[t]]
            order = trec_order(top)
            gains = [pooled[t][d] for d in order]
            ideal = list(pooled[t].values())
            # P' from the ORIGINAL audit labels -- independent of every file in results/rearranged
            alab = {r["answer_id"]: r for r in audits[c] if r["topic_id"] == t and r["llm_label"] is not None}
            a_gains = [alab[d]["llm_label"] for d in trec_order([(d, r["retrieval_score"]) for d, r in alab.items()])]
            # untruncated run, for the depth-20 sensitivity note
            full = trec_order([(d, s) for d, s in runs[c][t].items() if d in pooled[t]])
            for k in CUTOFFS:
                n, p = ndcg(gains, ideal, k), prec(gains, k)
                row = pt[(c, t)]
                max_err = max(max_err, abs(n - float(row[f"ndcg'@{k}"])), abs(p - float(row[f"precision'@{k}"])))
                max_perr = max(max_perr, abs(prec(a_gains, k) - float(row[f"precision'@{k}"])))
                max_full = max(max_full, abs(ndcg([pooled[t][d] for d in full], ideal, k) - n))
                sums[f"ndcg'@{k}"] += n
                sums[f"precision'@{k}"] += p
                # the ideal each run was divided by: DCG / nDCG' must equal the topic's single IDCG
                idc = float(idcg_file[t][f"idcg@{k}"])
                dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]) if g > 0)
                max_ierr = max(max_ierr, abs(idc - sum(g / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:k]) if g > 0)))
                nd = float(row[f"ndcg'@{k}"])
                if nd > 0 and abs(dcg / nd - idc) > 1e-9 * idc:
                    implied_bad += 1
        for col, s in sums.items():
            max_err = max(max_err, abs(s / 78 - float(agg[c][col])))
    check("pytrec_eval == pure Python (per topic and aggregate)", max_err < 1e-9, f"max |diff| = {max_err:.2e}")
    check("P' from the original audit labels == P' from the new qrel", max_perr < 1e-12, f"max |diff| = {max_perr:.2e}")
    check("idcg_per_topic.csv == IDCG recomputed from the pooled qrel", max_ierr < 1e-12)
    check("all 15 runs divided by the same IDCG on every topic (DCG / nDCG' == IDCG)", implied_bad == 0,
          f"{implied_bad} mismatches over 15 x 78 x {len(CUTOFFS)}")

    print("8. Information only")
    print(f"  depth-20 cut vs scoring the full 1000-deep run: max |nDCG' diff| = {max_full:.4f}")
    sizes = [len(ds) for ds in pooled.values()]
    print(f"  pool: {len(got)} pairs, {min(sizes)}-{max(sizes)} per topic; "
          f"topics with no grade>=2: {sorted(t for t, ds in pooled.items() if max(ds.values()) < 2)}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
