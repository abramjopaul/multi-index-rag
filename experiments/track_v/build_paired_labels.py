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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.judge_config import JudgeConfigManager  # noqa: E402
from multirag.config.path_configs import EVAL_CONFIG_DIR  # noqa: E402
from multirag.config.path_configs import RESULTS_TRACK_V_DIR
from multirag.eval.schema import PairedLabel  # noqa: E402
from multirag.eval.schema import CSV_COLUMNS, paired_label_to_csv_row
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


def _write_paired_files(
    paired: list[PairedLabel], jsonl_path: Path, csv_path: Path
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for p in paired:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for p in paired:
            writer.writerow(paired_label_to_csv_row(p))


def _infer_year(qrel_source: str) -> str | None:
    """Same substring-match pattern as build_validation_set.py's
    _infer_topics_xml -- qrel_source is the qrel filename, e.g.
    "qrel_task1_2020_all", which unambiguously names its year.
    """
    for year in ("2020", "2021", "2022"):
        if year in qrel_source:
            return year
    return None


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
                year=_infer_year(row.get("qrel_source", "")),
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
    output_csv = Path(args.output_csv)
    _write_paired_files(paired, output_jsonl, output_csv)

    # Split by year (from validation_set's qrel_source, threaded through as
    # PairedLabel.year) -- only years actually present get written, no empty
    # placeholder files for years not judged yet.
    by_year: dict[str, list[PairedLabel]] = {}
    for p in paired:
        if p.year:
            by_year.setdefault(p.year, []).append(p)

    year_paths: dict[str, tuple[Path, Path]] = {}
    for year, rows in sorted(by_year.items()):
        year_jsonl = output_jsonl.with_name(
            f"{output_jsonl.stem}_{year}{output_jsonl.suffix}"
        )
        year_csv = output_csv.with_name(f"{output_csv.stem}_{year}{output_csv.suffix}")
        _write_paired_files(rows, year_jsonl, year_csv)
        year_paths[year] = (year_jsonl, year_csv)

    print(f"\n{'=' * 60}")
    print("Track V paired labels")
    print(f"{'=' * 60}")
    print(f"  n pairs:            {n_pairs}")
    print(f"  parse failures:     {n_parse_fail}")
    print(f"  graded agreement:   {graded_agreement:.4f} (of {n_scored} scored pairs)")
    print(f"  binary agreement:   {binary_agreement:.4f} (of {n_scored} scored pairs)")
    print(
        "  per-year:           "
        + (
            ", ".join(f"{y}={len(rows)}" for y, rows in sorted(by_year.items()))
            or "(none)"
        )
    )
    print(f"Written to {output_jsonl}")
    print(f"Written to {output_csv}")
    for year, (yj, yc) in sorted(year_paths.items()):
        print(f"Written to {yj}")
        print(f"Written to {yc}")
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
    # Log the FULL paired-label set as one table (compact columns, matching
    # paired_labels.csv exactly) -- not a disagreement-biased sample. A
    # disagreements-plus-50-agreements sample looks overwhelmingly negative
    # regardless of the real agreement rate (e.g. 11,773 disagreements + 50
    # agreements reads as "almost nothing agrees" even at 70% real
    # agreement) -- every row here is real, so the table is exactly as
    # representative as the aggregate numbers above. Plus one table + one
    # file pair per year with data, so 2020/2021/2022 can each be browsed or
    # downloaded independently.
    exp_logger.log_table(
        "paired_labels_overall", [paired_label_to_csv_row(p) for p in paired]
    )
    exp_logger.save_file(output_jsonl)
    exp_logger.save_file(output_csv)
    for year, rows in sorted(by_year.items()):
        exp_logger.log_table(
            f"paired_labels_{year}", [paired_label_to_csv_row(p) for p in rows]
        )
        yj, yc = year_paths[year]
        exp_logger.save_file(yj)
        exp_logger.save_file(yc)
    # Intentionally no finish() here -- agreement_analysis.py / report.py
    # continue logging into this same run; report.py closes it.

    return 0


if __name__ == "__main__":
    sys.exit(main())
