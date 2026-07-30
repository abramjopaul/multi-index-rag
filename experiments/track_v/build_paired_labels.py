#!/usr/bin/env python3
"""Track V: join human qrels with judge labels into one PairedLabel per pair.

This is the raw material every kappa/tau number is derived from, and the
primary human-readable audit artifact -- every disagreement can be inspected
here without re-running the judge. Every human-judged pair from
build_validation_set.py's output must appear exactly once; the join is
asserted lossless.

Usage:
    poetry run python experiments/track_v/build_paired_labels.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.judge_config import JudgeConfigManager  # noqa: E402
from multirag.config.path_configs import (
    EVAL_CONFIG_DIR,  # noqa: E402
    RESULTS_TRACK_V_DIR,
)
from multirag.eval.schema import (
    CSV_COLUMNS,
    PairedLabel,  # noqa: E402
    paired_label_to_csv_row,
)
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_paired_labels(
    validation_rows: list[dict], judgement_rows: list[dict]
) -> list[PairedLabel]:
    judge_by_key = {(j["topic_id"], j["answer_id"]): j for j in judgement_rows}

    paired: list[PairedLabel] = []
    missing: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in validation_rows:
        key = (row["topic_id"], row["answer_id"])
        if key in seen:
            raise ValueError(
                f"Duplicate (topic_id, answer_id) in validation set: {key}"
            )
        seen.add(key)

        judgement = judge_by_key.get(key)
        if judgement is None:
            missing.append(key)
            continue

        human_label = row["human_label"]
        judge_label = judgement["label"]  # None if parse failed
        human_bin = row["human_bin"]
        judge_bin = None if judge_label is None else (1 if judge_label >= 2 else 0)

        paired.append(
            PairedLabel(
                topic_id=row["topic_id"],
                answer_id=row["answer_id"],
                human_label=human_label,
                judge_label=judge_label,
                human_bin=human_bin,
                judge_bin=judge_bin,
                agree_graded=(
                    None if judge_label is None else (human_label == judge_label)
                ),
                agree_binary=None if judge_bin is None else (human_bin == judge_bin),
                delta=None if judge_label is None else (judge_label - human_label),
                topic_type=None,  # not available in this checkout -- see plan's Context section
                multi_approach=False,  # no local flagged-topic list; see plan's Context section
                question=row["question"],
                answer_text=row["answer_text"],
                judge_raw_response=judgement.get("raw_response", ""),
            )
        )

    if missing:
        raise AssertionError(
            f"{len(missing)} human-judged pair(s) missing a judge label -- the join must be "
            f"lossless. Run run_judge_over_validation.py again first. Examples: {missing[:10]}"
        )

    return paired


def main() -> int:
    parser = argparse.ArgumentParser(description="Join human qrels with judge labels")
    parser.add_argument("--config", default=str(EVAL_CONFIG_DIR / "judge.yaml"))
    parser.add_argument(
        "--validation-set", default=str(RESULTS_TRACK_V_DIR / "validation_set.jsonl")
    )
    parser.add_argument(
        "--judgements", default=str(RESULTS_TRACK_V_DIR / "judgements.jsonl")
    )
    parser.add_argument(
        "--output-jsonl", default=str(RESULTS_TRACK_V_DIR / "paired_labels.jsonl")
    )
    parser.add_argument(
        "--output-csv", default=str(RESULTS_TRACK_V_DIR / "paired_labels.csv")
    )
    args = parser.parse_args()

    config = JudgeConfigManager.from_yaml(args.config)

    validation_rows = _load_jsonl(Path(args.validation_set))
    all_judgement_rows = _load_jsonl(Path(args.judgements))
    meta_row = next((r for r in all_judgement_rows if r.get("_meta")), {})
    judgement_rows = [r for r in all_judgement_rows if not r.get("_meta")]

    paired = build_paired_labels(validation_rows, judgement_rows)

    n_pairs = len(paired)
    n_parse_fail = sum(1 for p in paired if p.judge_label is None)
    n_scored = n_pairs - n_parse_fail
    graded_agreement = (
        sum(1 for p in paired if p.agree_graded) / n_scored if n_scored else 0.0
    )
    binary_agreement = (
        sum(1 for p in paired if p.agree_binary) / n_scored if n_scored else 0.0
    )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        for p in paired:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    output_csv = Path(args.output_csv)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for p in paired:
            writer.writerow(paired_label_to_csv_row(p))

    print(f"\n{'=' * 60}")
    print("Track V paired labels")
    print(f"{'=' * 60}")
    print(f"  n pairs:            {n_pairs}")
    print(f"  parse failures:     {n_parse_fail}")
    print(f"  graded agreement:   {graded_agreement:.4f} (of {n_scored} scored pairs)")
    print(f"  binary agreement:   {binary_agreement:.4f} (of {n_scored} scored pairs)")
    print(f"Written to {output_jsonl}")
    print(f"Written to {output_csv}")
    print(f"{'=' * 60}\n")

    # --- W&B: resume the run opened by run_judge_over_validation.py -------
    exp_logger = ExperimentLogger(config.wandb)
    exp_logger.start_run(
        config=meta_row,
        run_name=meta_row.get("run_name", "V-paired-labels"),
        run_id=meta_row.get("wandb_run_id"),
    )
    exp_logger.log_metrics(
        {
            "paired_labels/n_pairs": n_pairs,
            "paired_labels/n_parse_fail": n_parse_fail,
            "paired_labels/graded_agreement": graded_agreement,
            "paired_labels/binary_agreement": binary_agreement,
        }
    )
    disagreements = [
        p for p in paired if p.judge_label is not None and not p.agree_graded
    ]
    agreements = [p for p in paired if p.judge_label is not None and p.agree_graded]
    sample_size = min(50, len(agreements))
    sample = disagreements + (
        random.sample(agreements, sample_size) if sample_size else []
    )
    exp_logger.log_table(
        "paired_labels_sample",
        [
            {
                "topic_id": p.topic_id,
                "answer_id": p.answer_id,
                "human_label": p.human_label,
                "judge_label": p.judge_label,
                "agree_graded": p.agree_graded,
                "delta": p.delta,
                "question": p.question[:300],
                "answer_text": p.answer_text[:300],
            }
            for p in sample
        ],
    )
    exp_logger.log_artifact(output_jsonl, artifact_type="paired_labels")
    exp_logger.log_artifact(output_csv, artifact_type="paired_labels")
    # Intentionally no finish() here -- agreement_analysis.py / report.py
    # continue logging into this same run; report.py closes it.

    return 0


if __name__ == "__main__":
    sys.exit(main())
