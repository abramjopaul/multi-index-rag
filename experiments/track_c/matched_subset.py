#!/usr/bin/env python3
"""Track C: re-aggregate every generation run over the oracle-comparable topic subset.

The oracle retrieval ceiling (``C-oracle_retrieval_ceiling-k5``) can only run on
topics that have strictly more than ``k`` High-labelled answers: ``k`` of them
become the context, and at least one must remain to build the reference, or the
context would *be* the reference and the run would collapse into the reference
ceiling. At ``k = 5`` that leaves 53 of the 100 ARQMath-3 Task 1 topics.

That exclusion is principled but it is not neutral. The 47 dropped topics are the
thinly judged ones -- 23 have no High-labelled answer at all -- so every system
scores higher on the 53 that remain. Comparing an oracle mean over 53 topics with
a system mean over 100 therefore overstates the ceiling gap, and on
``factual_correctness_precision`` it reverses the sign.

This script re-computes each run's means over the matched subset from the
per-topic answer audits that already exist. No generation and no judging is
re-run; it is a pure re-aggregation.

Usage:
    poetry run python experiments/track_c/matched_subset.py
    poetry run python experiments/track_c/matched_subset.py --subset-run <run> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO_ROOT / "results" / "per_topic"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "track_c"
DEFAULT_SUBSET_RUN = "C-oracle_retrieval_ceiling-k5-20260821-215539"
DEFAULT_REFERENCE_RUN = "C-bm25-k5-20260821-094243"

METRICS = (
    "faithfulness",
    "factual_correctness_precision",
    "factual_correctness_recall",
)


def audit_path(run: str) -> Path:
    return AUDIT_DIR / f"{run}_answers_audit.csv"


def read_audit(run: str) -> list[dict[str, str]]:
    with audit_path(run).open(newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value: str | None) -> float | None:
    """Parse a metric cell; blank / 'null' / 'nan' all mean 'not scored'."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return None if parsed != parsed else parsed  # drop NaN


def mean_over(
    rows: list[dict[str, str]], metric: str, subset: set[str] | None
) -> tuple[float | None, int]:
    if metric not in (rows[0] if rows else {}):
        return None, 0
    values = [
        v
        for r in rows
        if subset is None or r["topic_id"] in subset
        for v in (as_float(r.get(metric)),)
        if v is not None
    ]
    return (statistics.mean(values) if values else None), len(values)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-run",
        default=DEFAULT_SUBSET_RUN,
        help="Run whose topic set defines the matched subset (default: the k=5 oracle ceiling).",
    )
    parser.add_argument(
        "--reference-run",
        default=DEFAULT_REFERENCE_RUN,
        help="Run whose topic set defines the full population (default: the BM25 k=5 run).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--out-name",
        default="oracle_matched53_metrics.csv",
        help="Output filename written under --out-dir.",
    )
    args = parser.parse_args()

    subset = {r["topic_id"] for r in read_audit(args.subset_run)}
    population = {r["topic_id"] for r in read_audit(args.reference_run)}
    if not subset <= population:
        raise SystemExit(
            f"subset run has {len(subset - population)} topics absent from the population run"
        )

    runs = sorted(
        p.name.removesuffix("_answers_audit.csv")
        for p in AUDIT_DIR.glob("C-*_answers_audit.csv")
    )
    if not runs:
        raise SystemExit(f"no audits found under {AUDIT_DIR}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(
            [
                "run",
                "metric",
                "mean_all",
                "n_all",
                "mean_matched",
                "n_matched",
                "delta_matched_minus_all",
            ]
        )
        for run in runs:
            rows = read_audit(run)
            for metric in METRICS:
                all_mean, n_all = mean_over(rows, metric, None)
                sub_mean, n_sub = mean_over(rows, metric, subset)
                delta = (
                    None if all_mean is None or sub_mean is None else sub_mean - all_mean
                )
                writer.writerow(
                    [run, metric, fmt(all_mean), n_all, fmt(sub_mean), n_sub, fmt(delta)]
                )

    print(f"matched subset: {len(subset)} of {len(population)} topics")
    print(f"defined by    : {args.subset_run}")
    print(f"runs          : {len(runs)}")
    print(f"written       : {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
