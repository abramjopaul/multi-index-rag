#!/usr/bin/env python3
"""Sweep summary for A3 SLT+OPT fusion experiments.

Pulls all W&B runs tagged "a3" from project "one-last-run", builds a
combined results DataFrame sorted by ndcg'@10 descending, and logs it as
a wandb.Table to a dedicated summary run.

Usage:
    poetry run python experiments/a3_summarize.py
    poetry run python experiments/a3_summarize.py --year 2021
    poetry run python experiments/a3_summarize.py --project one-last-run --year 2021
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

WANDB_PROJECT = "one-last-run"


def _extract_row(run) -> dict | None:
    cfg = run.config
    summary = run.summary._json_dict

    # Only include runs that have metrics (finished runs)
    ndcg = summary.get("ndcg'@10") or summary.get("ndcg'") or summary.get("ndcg_prime_10")
    if ndcg is None:
        return None

    return {
        "run_name": run.name,
        "year": cfg.get("year", "?"),
        "representations": cfg.get("representation", "?"),
        "n": cfg.get("n"),
        "rrf_k": cfg.get("rrf_k"),
        "fusion_method": cfg.get("fusion_method", "single"),
        "weights": str(cfg.get("channel_weights")),
        "slt_type": "slt_type" in str(cfg.get("representation", "")),
        "ndcg'@10": ndcg,
        "map'": summary.get("map'"),
        "P'@10": summary.get("P'@10"),
        "recall'@1000": summary.get("recall'@1000"),
        "channel_overlap": summary.get("channel_overlap_mean"),
        "state": run.state,
        "url": run.url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 fusion sweep summary")
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument(
        "--year",
        default=None,
        help="Filter by year (2021 or 2022). Default: include all.",
    )
    args = parser.parse_args()

    api = wandb.Api()

    filters: dict = {"tags": {"$in": ["a3"]}}
    if args.year:
        filters["config.year"] = args.year

    logger.info(f"Fetching A3 runs from project '{args.project}' with filters={filters}")
    runs = api.runs(args.project, filters=filters)

    rows = []
    for run in runs:
        row = _extract_row(run)
        if row is not None:
            rows.append(row)
        else:
            logger.debug(f"Skipping run {run.name} (no metrics or still running)")

    if not rows:
        logger.warning("No completed A3 runs found. Nothing to summarize.")
        return

    df = pd.DataFrame(rows).sort_values("ndcg'@10", ascending=False)

    print("\n" + "=" * 90)
    print(f"A3 Sweep Summary — {len(df)} runs")
    print("=" * 90)
    print(
        df[["run_name", "year", "n", "rrf_k", "fusion_method", "weights", "ndcg'@10", "map'", "recall'@1000"]]
        .to_string(index=False)
    )
    print("=" * 90)
    print(f"\nBest config: {df.iloc[0]['run_name']}  ndcg'@10={df.iloc[0]['ndcg'@10']:.4f}")

    year_tag = f"_{args.year}" if args.year else ""
    summary_run_name = f"a3_sweep_summary{year_tag}_{datetime.now().strftime('%Y%m%d')}"

    logger.info(f"Logging summary table to W&B as run '{summary_run_name}'...")
    with wandb.init(
        project=args.project,
        name=summary_run_name,
        tags=["a3", "summary"],
        job_type="summary",
    ):
        table = wandb.Table(dataframe=df.drop(columns=["url"]))
        wandb.log({f"a3{year_tag}_sweep_results": table})

    logger.info(f"Summary logged. Best: {df.iloc[0]['run_name']}")


if __name__ == "__main__":
    main()
