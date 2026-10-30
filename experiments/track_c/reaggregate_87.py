#!/usr/bin/env python3
"""Re-aggregate every Track C generation run over the 87 ARQMath-3 qrels topics.

Why this exists
---------------
The Track C metrics CSVs are aggregated over inconsistent topic sets. Chapter 5's
retrieval sections are all scored on the 87 topics of `qrel_task1_2022_all`, and
those 87 are exactly the topics for which a reference answer exists (asserted
below), so re-aggregating every generation run over the same 87 puts the whole
chapter on one topic set.

It also separates two things that scoring a refusal as zero conflates:

    coverage   = fraction of topics the system answered
    selective  = metric averaged over answered topics only
    composite  = metric averaged over all topics, refusals counted as scored

A refusal is scored 0 on every metric, so
    composite == selective * coverage
holds exactly and is verified per row. RAGAS as shipped awards a refusal partial
credit on factual-correctness recall -- a refusal restates the question and so
shares content with the reference -- which would break the identity for that one
metric and make its composite uninterpretable. Refusal rows are therefore zeroed
on all three metrics here; see ZERO_ON_REFUSAL.

Finally, for each retrieval run it scores the no-retrieval floor on *that run's*
answered topics, so a selective score can be compared against the floor on the
same topics rather than against the floor's all-topic mean. Systems refuse on
harder topics, so the two comparisons differ substantially.

No generation and no judging is re-run: everything derives from the per-topic
audits, so this is reproducible from the repository alone.

Usage:
    python experiments/track_c/reaggregate_87.py
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

QRELS = REPO_ROOT / "data" / "raw" / "qrels" / "qrel_task1_2022_all.tsv"
AUDIT_GLOB = str(REPO_ROOT / "results" / "per_topic" / "C-*_answers_audit.csv")
OUT_CSV = REPO_ROOT / "results" / "track_c" / "topics87_metrics.csv"
OUT_CSV_53 = REPO_ROOT / "results" / "track_c" / "topics53_metrics.csv"

ORACLE_RUN = "C-oracle_retrieval_ceiling-k5-20260821-215539"

FLOOR_RUN = "C-no_rag-k0-20260821-173043"
METRICS = ("faithfulness", "factual_correctness_precision", "factual_correctness_recall")
# A refusal contains no answer, so it is scored 0 on every metric. Faithfulness and
# factual-correctness precision already come out of RAGAS as 0; recall does not, and
# is overridden here so that all three metrics decompose the same way.
ZERO_ON_REFUSAL = METRICS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_qrels_topics(path: Path) -> set[str]:
    topics: set[str] = set()
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 4:
                topics.add(parts[0])
    return topics


def to_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def is_refusal(row: dict) -> bool:
    return str(row.get("is_refusal", "")).strip().lower() in {"true", "1"}


def load_audit(path: str, topics: set[str]) -> list[dict]:
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if r["topic_id"] in topics]


def run_name(path: str) -> str:
    return os.path.basename(path).replace("_answers_audit.csv", "")


def metric_value(row: dict, metric: str) -> float | None:
    """A refusal scores 0 on every metric, whatever RAGAS recorded for it."""
    if metric in ZERO_ON_REFUSAL and is_refusal(row):
        return 0.0
    return to_float(row.get(metric))


def subset_mean(rows: list[dict], metric: str) -> tuple[float | None, int]:
    """Mean of a metric over rows that carry a value for it."""
    values = [metric_value(r, metric) for r in rows]
    values = [v for v in values if v is not None]
    return (mean(values) if values else None), len(values)


def main() -> int:
    topics = load_qrels_topics(QRELS)
    logger.info("qrels topics: %d", len(topics))

    audits = sorted(glob.glob(AUDIT_GLOB))
    if not audits:
        logger.error("no audits matched %s", AUDIT_GLOB)
        return 1

    by_run = {run_name(p): load_audit(p, topics) for p in audits}

    if FLOOR_RUN not in by_run:
        logger.error("floor run %s not found among audits", FLOOR_RUN)
        return 1
    floor_by_topic = {r["topic_id"]: r for r in by_run[FLOOR_RUN]}

    # Sanity check the premise of this script: the topics with a usable reference
    # are exactly the qrels topics.
    ref_topics = {
        r["topic_id"]
        for r in by_run[FLOOR_RUN]
        if to_float(r.get("factual_correctness_precision")) is not None
        and to_float(r.get("factual_correctness_recall")) is not None
    }
    if ref_topics != topics:
        logger.warning(
            "reference-bearing topics differ from qrels topics "
            "(%d vs %d; only_ref=%s only_qrels=%s)",
            len(ref_topics), len(topics),
            sorted(ref_topics - topics)[:5], sorted(topics - ref_topics)[:5],
        )
    else:
        logger.info("reference-bearing topics == qrels topics (%d) -- premise holds", len(topics))

    def aggregate(topic_set: set[str]) -> list[dict]:
        rows_out: list[dict] = []
        floor_rows_all = load_audit(
            str(REPO_ROOT / "results" / "per_topic" / f"{FLOOR_RUN}_answers_audit.csv"),
            topic_set,
        )
        floor_map = {r["topic_id"]: r for r in floor_rows_all}
        for run, all_rows in sorted(by_run.items()):
            rows = [r for r in all_rows if r["topic_id"] in topic_set]
            if not rows:
                continue
            answered = [r for r in rows if not is_refusal(r)]
            n_topics, n_answered = len(rows), len(answered)
            coverage = n_answered / n_topics
            for metric in METRICS:
                if metric not in rows[0]:
                    continue
                selective, n_sel = subset_mean(answered, metric)
                composite, n_comp = subset_mean(rows, metric)
                product = (selective * coverage) if selective is not None else None
                drift = (
                    abs(product - composite)
                    if (product is not None and composite is not None)
                    else None
                )
                floor_on_answered = None
                if run != FLOOR_RUN and metric != "faithfulness":
                    fr = [floor_map[r["topic_id"]] for r in answered if r["topic_id"] in floor_map]
                    floor_on_answered, _ = subset_mean(fr, metric)
                rows_out.append({
                    "run": run, "metric": metric,
                    "n_topics": n_topics, "n_answered": n_answered,
                    "coverage": round(coverage, 4),
                    "selective": None if selective is None else round(selective, 4),
                    "n_selective": n_sel,
                    "composite": None if composite is None else round(composite, 4),
                    "n_composite": n_comp,
                    "product_check_abs_diff": None if drift is None else round(drift, 4),
                    "floor_on_answered_topics": (
                        None if floor_on_answered is None else round(floor_on_answered, 4)
                    ),
                })
        return rows_out

    def write(path: Path, rows_out: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        worst = max((r["product_check_abs_diff"] or 0.0) for r in rows_out)
        logger.info("wrote %d rows to %s (max product drift %.4f)", len(rows_out), path, worst)

    write(OUT_CSV, aggregate(topics))

    # The oracle-comparable subset: the topics on which a retrieval ceiling can be
    # defined at all (it needs more High answers than fit in the context). Taken
    # from the oracle run's own topic list rather than recomputed, so the subset is
    # exactly the one the ceiling was scored on.
    if ORACLE_RUN in by_run:
        oracle_topics = {r["topic_id"] for r in by_run[ORACLE_RUN]}
        logger.info("oracle-comparable subset: %d topics", len(oracle_topics))
        write(OUT_CSV_53, aggregate(oracle_topics))
    else:
        logger.warning("oracle run %s not found -- skipping 53-topic aggregation", ORACLE_RUN)

    return 0


if __name__ == "__main__":
    sys.exit(main())
