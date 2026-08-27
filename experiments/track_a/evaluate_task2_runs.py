#!/usr/bin/env python3
"""Track A: evaluate ARQMath Task 2 (formula retrieval) TREC run files
against their qrels, log each as a separate run in W&B group "Track A :
Task 2".

Reuses the same evaluation pipeline already used for Track B
(multirag.evaluation.metrics.evaluate_run, pytrec_eval-based) -- Task 2's
qrels/run files are the same TREC format, and evaluate_run already computes
prime (judged-only) variants of every cutoff, which is what nDCG', MAP',
P'@10 are: ndcg'@1000 (uncut nDCG', since these runs are depth<=1000),
map' (uncut MAP'), precision'@10.

Usage:
    poetry run python experiments/track_a/evaluate_task2_runs.py
    poetry run python experiments/track_a/evaluate_task2_runs.py \\
        --trec-file data/runs/track_a/foo.tsv --qrels-file data/raw/qrels/qrel_task2_2021_all.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.judge_config import WandbConfig  # noqa: E402
from multirag.config.path_configs import (  # noqa: E402
    QREL_TASK2_2021_ALL,
    QREL_TASK2_2022_OFFICIAL,
    RESULTS_TRACK_A_DIR,
    TRACK_A_RUNS_DIR,
)
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402
from multirag.evaluation.metrics import evaluate_run  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

DEFAULT_PAIRS = [
    (TRACK_A_RUNS_DIR / "a3_slt_opt_slt_type_rrf_k60_n2000_2021_20260702.tsv", QREL_TASK2_2021_ALL),
    (TRACK_A_RUNS_DIR / "a3_slt_opt_slt_type_rrf_k60_n2000_2022_20260702.tsv", QREL_TASK2_2022_OFFICIAL),
]

# The three ARQMath-official comparison metrics, called out explicitly since
# they're the ones any published-baseline comparison actually uses -- all
# already produced by evaluate_run(), nothing extra to compute.
HEADLINE_METRICS = {"nDCG'": "ndcg'@1000", "MAP'": "map'", "P'@10": "precision'@10"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Track A (ARQMath Task 2) run files, log to W&B")
    parser.add_argument("--trec-file", action="append", default=None, help="Repeatable; pairs by position with --qrels-file")
    parser.add_argument("--qrels-file", action="append", default=None, help="Repeatable; pairs by position with --trec-file")
    parser.add_argument("--wandb-group", default="Track A : Task 2")
    parser.add_argument("--output-dir", default=str(RESULTS_TRACK_A_DIR))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.trec_file and args.qrels_file:
        if len(args.trec_file) != len(args.qrels_file):
            parser.error("--trec-file and --qrels-file must be given the same number of times")
        pairs = [(Path(t), Path(q)) for t, q in zip(args.trec_file, args.qrels_file)]
    else:
        pairs = DEFAULT_PAIRS

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for trec_path, qrels_path in pairs:
        run_name = trec_path.stem
        logger.info(f"=== {run_name} ===  (qrels: {qrels_path.name})")
        results = evaluate_run(qrels_path, trec_path)

        headline = {label: results.get(key) for label, key in HEADLINE_METRICS.items()}
        print(f"\n{'=' * 70}")
        print(f"{run_name}  (qrels={qrels_path.name})")
        print(f"{'=' * 70}")
        for label, value in headline.items():
            print(f"  {label:<8} {value:.4f}" if value is not None else f"  {label:<8} MISSING")
        print(f"{'-' * 70}")
        for k in sorted(results):
            print(f"  {k:<16} {results[k]:.4f}")
        print(f"{'=' * 70}\n")

        run_out_dir = out_dir / run_name
        run_out_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(run_out_dir / "metrics.json", "w") as f:
            json.dump({"trec_file": str(trec_path), "qrels_file": str(qrels_path),
                       "headline": headline, "all_metrics": results}, f, indent=2)

        if not args.no_wandb:
            try:
                wandb_config = WandbConfig(
                    group=args.wandb_group,
                    job_type="track-a-task2-eval",
                    tags=["track-a", "task2", "arqmath", "formula-retrieval"],
                )
                exp_logger = ExperimentLogger(wandb_config)
                exp_logger.start_run(
                    config={
                        "trec_file": str(trec_path),
                        "qrels_file": str(qrels_path),
                        "run_name": run_name,
                    },
                    run_name=run_name,
                )
                exp_logger.log_table(
                    "metrics",
                    [{"metric": k, "value": v} for k, v in sorted(results.items())],
                )
                exp_logger.log_metrics(results)
                exp_logger.log_metrics({f"headline/{label}": v for label, v in headline.items() if v is not None})
                exp_logger.save_file(trec_path)
                exp_logger.save_file(qrels_path)
                exp_logger.save_file(run_out_dir / "metrics.json")
                exp_logger.finish()
                logger.info(f"  logged -> group={args.wandb_group!r} run={run_name!r}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  W&B logging failed (results already written locally): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
