#!/usr/bin/env python3
"""Calibrate our evaluation pipeline against a published ARQMath baseline.

Runs evaluate_run() (the same pytrec_eval-based scoring every Track B config
uses) on a known official baseline run file, and diffs the result against
the published numbers for that exact baseline in the ARQMath-3 (2022)
overview paper (Table 3, ARQMath-3 column, "TF-IDF(Terrier)" row):
https://ceur-ws.org/Vol-3180/paper-01.pdf -- nDCG'=0.272, MAP'=0.064,
P'@10=0.124, over 78 topics.

Table 3's columns are all prime (judged-only) metrics with H+M binarization
(qrels label>=2 => relevant, matching evaluate()'s hardcoded relevance_level=2):
nDCG' is reported uncut, which we take as ndcg'@1000 (the run file is depth-1000
per topic, so ndcg_cut_1000 == uncut nDCG here); MAP' is the bare, uncut map'
measure (not map'@k); P'@10 is precision'@10.

The paper's Table 3 numbers were computed against the OFFICIAL qrels (78
topics) -- NOT qrel_task1_2022_all.tsv (87 topics, used elsewhere in this
repo), which additionally includes qrel_task1_2022_additional.tsv's pooled
judgments. Evaluating against "all" would not be comparable to the paper, so
this script uses the official qrels by default and reports "all" separately,
clearly labeled, for reference only.

Usage:
    poetry run python experiments/track_b/calibrate_baseline.py
    poetry run python experiments/track_b/calibrate_baseline.py \\
        --run-file data/runs/track_b/baseline_task1_tf_idf_terrier.tsv
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
    QREL_TASK1_2022_ALL,
    QREL_TASK1_2022_OFFICIAL,
    TRACK_B_RUNS_DIR,
)
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402
from multirag.evaluation.metrics import evaluate_run  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

# Table 3, ARQMath-3 (2022) column, "TF-IDF(Terrier)" baseline row, 78 topics.
PUBLISHED_ARQMATH3_TFIDF_TERRIER = {
    "ndcg'@1000": 0.272,
    "map'": 0.064,
    "precision'@10": 0.124,
}
PUBLISHED_N_TOPICS = 78
PUBLISHED_SOURCE = "https://ceur-ws.org/Vol-3180/paper-01.pdf, Table 3, ARQMath-3 column"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate evaluate_run() against a published ARQMath baseline")
    parser.add_argument(
        "--run-file",
        default=str(TRACK_B_RUNS_DIR / "baseline_task1_tf_idf_terrier.tsv"),
    )
    parser.add_argument("--official-qrels", default=str(QREL_TASK1_2022_OFFICIAL))
    parser.add_argument("--all-qrels", default=str(QREL_TASK1_2022_ALL))
    parser.add_argument("--wandb-group", default="Calibration : ARQMath Published Baselines")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    run_file = Path(args.run_file)
    logger.info("Evaluating %s against official qrels (%s)...", run_file, args.official_qrels)
    official_metrics = evaluate_run(args.official_qrels, run_file)

    logger.info("Also evaluating against qrel_task1_2022_all.tsv, for reference only (not paper-comparable)...")
    all_metrics = evaluate_run(args.all_qrels, run_file)

    comparison_rows = []
    print(f"\n{'=' * 78}")
    print(f"Calibration: {run_file.name}  vs.  {PUBLISHED_SOURCE}")
    print(f"{'=' * 78}")
    print(f"{'metric':<14} {'ours (official qrels)':>22} {'published':>11} {'diff':>9}")
    for metric, published in PUBLISHED_ARQMATH3_TFIDF_TERRIER.items():
        ours = official_metrics.get(metric)
        diff = (ours - published) if ours is not None else None
        print(f"{metric:<14} {ours:>22.4f} {published:>11.3f} {diff:>+9.4f}")
        comparison_rows.append({
            "metric": metric,
            "ours_official_qrels": ours,
            "published": published,
            "diff": diff,
            "published_n_topics": PUBLISHED_N_TOPICS,
        })
    print(f"{'=' * 78}\n")

    if not args.no_wandb:
        try:
            wandb_config = WandbConfig(
                group=args.wandb_group,
                job_type="baseline-calibration",
                tags=["calibration", "arqmath-3-task1", "tf-idf-terrier"],
            )
            exp_logger = ExperimentLogger(wandb_config)
            exp_logger.start_run(
                config={
                    "run_file": str(run_file),
                    "official_qrels": args.official_qrels,
                    "all_qrels": args.all_qrels,
                    "published_source": PUBLISHED_SOURCE,
                    "published_n_topics": PUBLISHED_N_TOPICS,
                },
                run_name=f"calibrate-{run_file.stem}",
            )
            exp_logger.log_table("published_comparison", comparison_rows)
            exp_logger.log_table(
                "full_metrics_official_qrels",
                [{"metric": k, "value": v} for k, v in sorted(official_metrics.items())],
            )
            exp_logger.log_table(
                "full_metrics_all_qrels_reference_only",
                [{"metric": k, "value": v} for k, v in sorted(all_metrics.items())],
            )
            exp_logger.log_metrics({f"official/{k}": v for k, v in official_metrics.items()})
            exp_logger.log_metrics({f"all_qrels_reference/{k}": v for k, v in all_metrics.items()})
            exp_logger.save_file(run_file)
            exp_logger.finish()
            logger.info("Logged to W&B group %r", args.wandb_group)
        except Exception as e:  # noqa: BLE001
            logger.warning("W&B logging failed: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
