#!/usr/bin/env python3
"""
TREC Run Fusion Script — Fuse multiple TREC run files using RRF, CombSUM, and CombMNZ.

Usage:
    poetry run python experiments/fuse_runs.py <run1.tsv> <run2.tsv> [run3.tsv ...] \
        [--output-dir OUTDIR] [--techniques TECH1,TECH2,...] [--rrf-k K] [--top-k K] [--dry-run]

Example:
    poetry run python experiments/fuse_runs.py \
        data/runs/bm25_baseline.tsv \
        data/runs/dense_mpnet_20260611.tsv \
        data/runs/dense_mpnet_20260612.tsv \
        --output-dir experiments/runs/ \
        --techniques rrf,combsum,combmnz

With W&B logging:
    poetry run python experiments/fuse_runs.py \
        data/runs/bm25_baseline.tsv \
        data/runs/dense_mpnet_20260611.tsv \
        data/runs/dense_mpnet_20260612.tsv

Dry-run (skip W&B):
    poetry run python experiments/fuse_runs.py \
        data/runs/bm25_baseline.tsv \
        data/runs/dense_mpnet_20260611.tsv \
        data/runs/dense_mpnet_20260612.tsv \
        --dry-run
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import wandb

from multirag.evaluation.metrics import _parse_qrels, _parse_run, evaluate_run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed qrels path
QRELS_PATH = Path("data/raw/qrels/qrel_task1_2022_all.tsv")


def validate_run_files(run_paths: list[Path]) -> None:
    """Validate that all run files exist and are readable."""
    for path in run_paths:
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Run file is empty: {path}")


def fuse_rrf(
    runs: list[dict[str, dict[str, float]]],
    run_names: list[str],
    k: int = 60,
    top_k: int | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    Reciprocal Rank Fusion (RRF).

    For each query, combines rankings from multiple runs:
        score(doc) = sum over runs of 1 / (rank + k)

    Args:
        runs: List of run dicts {query_id: {doc_id: score}}.
        run_names: Names of runs (for reference).
        k: RRF constant (default 60).
        top_k: Number of top results to return per query (None = keep all).

    Returns:
        Dict {query_id: [(doc_id, fused_score), ...]} sorted by score descending.
    """
    if not runs:
        return {}

    # Collect all query IDs across all runs
    all_queries = set()
    for run in runs:
        all_queries.update(run.keys())

    fused: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for query_id in all_queries:
        for run_idx, run in enumerate(runs):
            if query_id not in run:
                continue

            # Sort docs by original score descending, extract rank
            ranked_docs = sorted(
                run[query_id].items(), key=lambda x: -x[1]
            )
            for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
                fused[query_id][doc_id] += 1.0 / (rank + k)

    # Convert to output format and apply top_k cutoff
    result: dict[str, list[tuple[str, float]]] = {}
    for query_id in all_queries:
        if query_id in fused:
            sorted_docs = sorted(
                fused[query_id].items(), key=lambda x: -x[1]
            )
            if top_k:
                sorted_docs = sorted_docs[:top_k]
            result[query_id] = sorted_docs

    return result


def fuse_combsum(
    runs: list[dict[str, dict[str, float]]],
    run_names: list[str],
    top_k: int | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    CombSUM Fusion.

    Normalizes scores in each run to [0, 1] using min-max scaling,
    then sums across runs:
        score(doc) = sum over runs of (score_i(doc) - min_i) / (max_i - min_i)

    Args:
        runs: List of run dicts {query_id: {doc_id: score}}.
        run_names: Names of runs (for reference).
        top_k: Number of top results to return per query (None = keep all).

    Returns:
        Dict {query_id: [(doc_id, fused_score), ...]} sorted by score descending.
    """
    if not runs:
        return {}

    # Collect all query IDs across all runs
    all_queries = set()
    for run in runs:
        all_queries.update(run.keys())

    fused: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for query_id in all_queries:
        for run_idx, run in enumerate(runs):
            if query_id not in run:
                continue

            scores = list(run[query_id].values())
            if not scores:
                continue

            # Min-max normalization
            min_score = min(scores)
            max_score = max(scores)
            range_score = max_score - min_score

            for doc_id, score in run[query_id].items():
                if range_score > 0:
                    normalized = (score - min_score) / range_score
                else:
                    normalized = 0.5  # All scores equal; use midpoint
                fused[query_id][doc_id] += normalized

    # Convert to output format and apply top_k cutoff
    result: dict[str, list[tuple[str, float]]] = {}
    for query_id in all_queries:
        if query_id in fused:
            sorted_docs = sorted(
                fused[query_id].items(), key=lambda x: -x[1]
            )
            if top_k:
                sorted_docs = sorted_docs[:top_k]
            result[query_id] = sorted_docs

    return result


def fuse_combmnz(
    runs: list[dict[str, dict[str, float]]],
    run_names: list[str],
    top_k: int | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    CombMNZ Fusion.

    Combines CombSUM with a count-based multiplier:
        score(doc) = count(doc in runs) * CombSUM_score(doc)

    This prioritizes documents that appear in more retrieval systems.

    Args:
        runs: List of run dicts {query_id: {doc_id: score}}.
        run_names: Names of runs (for reference).
        top_k: Number of top results to return per query (None = keep all).

    Returns:
        Dict {query_id: [(doc_id, fused_score), ...]} sorted by score descending.
    """
    if not runs:
        return {}

    # Collect all query IDs across all runs
    all_queries = set()
    for run in runs:
        all_queries.update(run.keys())

    fused: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    doc_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for query_id in all_queries:
        for run_idx, run in enumerate(runs):
            if query_id not in run:
                continue

            scores = list(run[query_id].values())
            if not scores:
                continue

            # Min-max normalization (same as CombSUM)
            min_score = min(scores)
            max_score = max(scores)
            range_score = max_score - min_score

            for doc_id, score in run[query_id].items():
                if range_score > 0:
                    normalized = (score - min_score) / range_score
                else:
                    normalized = 0.5
                fused[query_id][doc_id] += normalized
                doc_counts[query_id][doc_id] += 1

    # Apply MNZ multiplier (count of runs containing doc)
    result: dict[str, list[tuple[str, float]]] = {}
    for query_id in all_queries:
        if query_id in fused:
            mnz_scores = {}
            for doc_id, combsum_score in fused[query_id].items():
                count = doc_counts[query_id].get(doc_id, 1)
                mnz_scores[doc_id] = combsum_score * count

            sorted_docs = sorted(mnz_scores.items(), key=lambda x: -x[1])
            if top_k:
                sorted_docs = sorted_docs[:top_k]
            result[query_id] = sorted_docs

    return result


def write_trec_run(
    fused_results: dict[str, list[tuple[str, float]]],
    output_path: Path,
    run_name: str,
) -> None:
    """
    Write fused results to TREC format.

    Format (tab-separated):
        query_id Q0 doc_id rank score run_name

    Args:
        fused_results: Dict {query_id: [(doc_id, score), ...]}.
        output_path: Path to write TREC file.
        run_name: Name identifier for this run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for query_id in sorted(fused_results.keys()):
            doc_list = fused_results[query_id]
            for rank, (doc_id, score) in enumerate(doc_list, start=1):
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")


def save_metrics_tsv(
    eval_results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """
    Save evaluation results as TSV file for easy inspection.

    Args:
        eval_results: Dict {run_name: {metric: value}}.
        output_path: Path to write TSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all metrics across all runs
    all_metrics = set()
    for metrics in eval_results.values():
        all_metrics.update(metrics.keys())

    # Sort metrics by key name
    sorted_metrics = sorted(all_metrics)

    # Write TSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Run"] + sorted_metrics)

        for run_name in sorted(eval_results.keys()):
            metrics = eval_results[run_name]
            row = [run_name]
            for metric in sorted_metrics:
                value = metrics.get(metric, "N/A")
                row.append(f"{value:.6f}" if isinstance(value, float) else str(value))
            writer.writerow(row)


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Fuse multiple TREC run files using RRF, CombSUM, and CombMNZ.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "run_files",
            nargs="+",
            type=Path,
            help="Input TREC run file paths (minimum 2 files)",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("experiments/runs"),
            help="Output directory for fused TREC files (default: experiments/runs)",
        )
        parser.add_argument(
            "--techniques",
            type=str,
            default="rrf,combsum,combmnz",
            help="Comma-separated fusion techniques to apply (default: rrf,combsum,combmnz)",
        )
        parser.add_argument(
            "--rrf-k",
            type=int,
            default=60,
            help="RRF constant K (default: 60)",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=None,
            help="Number of top results per query to keep (default: None = keep all)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Dry run: skip W&B logging (useful for testing)",
        )

        args = parser.parse_args()

        # Validate input
        if len(args.run_files) < 2:
            logger.error("Minimum 2 run files required for fusion.")
            sys.exit(1)

        run_paths = [Path(p).resolve() for p in args.run_files]
        validate_run_files(run_paths)

        # Check qrels exists
        qrels_path = Path(QRELS_PATH).resolve()
        if not qrels_path.exists():
            logger.error(f"Qrels file not found at {qrels_path}")
            sys.exit(1)

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        techniques = [t.strip().lower() for t in args.techniques.split(",")]
        valid_techniques = {"rrf", "combsum", "combmnz"}
        for tech in techniques:
            if tech not in valid_techniques:
                logger.error(f"Unknown technique '{tech}'. Valid: {valid_techniques}")
                sys.exit(1)

        # Generate run name from input files
        run_name = f"fusion_{len(run_paths)}runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize W&B (unless dry-run)
        if not args.dry_run:
            logger.info("Initializing Weights & Biases...")
            wandb.init(
                project="multi-index-rag",
                name=run_name,
                group="fusion",
                tags=techniques,
                config={
                    "input_runs": [str(p) for p in run_paths],
                    "rrf_k": args.rrf_k,
                    "top_k": args.top_k,
                    "techniques": techniques,
                },
            )
            logger.info(f"W&B initialized: {wandb.run.url}")  # type: ignore
        else:
            logger.info("Dry-run mode: W&B logging disabled")

        logger.info(f"\n{'=' * 80}")
        logger.info("TREC Run Fusion")
        logger.info(f"{'=' * 80}")
        logger.info(f"Input runs: {len(run_paths)}")
        for i, p in enumerate(run_paths, 1):
            logger.info(f"  {i}. {p.name}")
        logger.info(f"Techniques: {', '.join(techniques)}")
        logger.info(f"RRF-K: {args.rrf_k}")
        logger.info(f"Top-K: {args.top_k if args.top_k else 'all'}")
        logger.info(f"Qrels: {qrels_path}")
        logger.info(f"Output: {output_dir}")

        # Parse all run files
        logger.info("\nParsing run files...")
        runs: list[dict[str, dict[str, float]]] = []
        run_names: list[str] = []
        for path in run_paths:
            run = _parse_run(path)
            runs.append(run)
            run_names.append(path.stem)
            logger.info(f"  ✓ {path.name}: {len(run)} queries")

        # Parse qrels
        logger.info(f"\nParsing qrels from {qrels_path.name}...")
        qrels = _parse_qrels(qrels_path)
        logger.info(f"  ✓ {len(qrels)} queries with judgments")

        # Perform fusion
        logger.info(f"\nApplying fusion techniques...")
        fusion_results: dict[str, dict[str, list[tuple[str, float]]]] = {}

        if "rrf" in techniques:
            logger.info(f"  • RRF (k={args.rrf_k})...")
            fusion_results["rrf"] = fuse_rrf(runs, run_names, k=args.rrf_k, top_k=args.top_k)

        if "combsum" in techniques:
            logger.info(f"  • CombSUM...")
            fusion_results["combsum"] = fuse_combsum(runs, run_names, top_k=args.top_k)

        if "combmnz" in techniques:
            logger.info(f"  • CombMNZ...")
            fusion_results["combmnz"] = fuse_combmnz(runs, run_names, top_k=args.top_k)

        # Write fused TREC files
        logger.info(f"\nWriting fused TREC files...")
        written_files: dict[str, Path] = {}
        for technique, results in fusion_results.items():
            output_path = output_dir / f"fused_{technique}.tsv"
            write_trec_run(results, output_path, f"fused_{technique}")
            written_files[technique] = output_path
            total_docs = sum(len(docs) for docs in results.values())
            logger.info(f"  ✓ {output_path.name}: {len(results)} queries, {total_docs} docs")

        # Evaluate all runs (individual + fused)
        logger.info(f"\nEvaluating runs against qrels...")
        eval_results: dict[str, dict[str, float]] = {}

        # Evaluate individual runs
        for i, run_path in enumerate(run_paths):
            run_name_eval = run_names[i]
            try:
                metrics = evaluate_run(qrels_path, run_path)
                eval_results[f"[input] {run_name_eval}"] = metrics
                logger.info(f"  ✓ {run_name_eval}")
            except Exception as e:
                logger.error(f"  ✗ {run_name_eval}: {e}")

        # Evaluate fused runs
        for technique, output_path in written_files.items():
            try:
                metrics = evaluate_run(qrels_path, output_path)
                eval_results[f"[fused] {technique}"] = metrics
                logger.info(f"  ✓ fused_{technique}")
            except Exception as e:
                logger.error(f"  ✗ fused_{technique}: {e}")

        # Generate comparison report
        logger.info(f"\n{'=' * 80}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'=' * 80}")

        # Select key metrics for display
        key_metrics = [
            "ndcg@10",
            "ndcg@100",
            "precision@5",
            "precision@10",
            "recall@10",
            "recall@100",
            "map",
            "bpref",
        ]

        # Print table
        header = ["Run"] + key_metrics
        logger.info(f"{header[0]:<25} " + " ".join(f"{m:>10}" for m in header[1:]))
        logger.info("-" * (25 + len(key_metrics) * 12))

        for run_name_tbl, metrics in eval_results.items():
            values = [run_name_tbl[:25]]
            for metric in key_metrics:
                if metric in metrics:
                    values.append(f"{metrics[metric]:>10.4f}")
                else:
                    values.append(f"{'N/A':>10}")
            logger.info(" ".join(f"{v:>10}" if v != run_name_tbl[:25] else f"{v:<25}" for v in values))

        # Save metrics TSV
        metrics_tsv_path = output_dir / "fusion_metrics.tsv"
        save_metrics_tsv(eval_results, metrics_tsv_path)
        logger.info(f"\n✓ Metrics TSV saved to {metrics_tsv_path}")

        # Create W&B tables and log
        if not args.dry_run:
            logger.info("\nLogging to Weights & Biases...")

            log_dict: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "qrels_path": str(qrels_path),
                "num_input_runs": len(run_paths),
                "num_techniques": len(techniques),
                "total_queries": len(qrels),
            }

            # One table per input run (Metric, Value format — matches run_experiment.py)
            for run_name_wb, metrics in eval_results.items():
                if "[input]" not in run_name_wb:
                    continue
                table_key = run_name_wb.replace("[input] ", "input_") + "_metrics"
                t = wandb.Table(columns=["Metric", "Value"])
                for metric_name, metric_value in sorted(metrics.items()):
                    t.add_data(metric_name, metric_value)
                log_dict[table_key] = t

            # One table per fusion technique (Metric, Value format)
            for run_name_wb, metrics in eval_results.items():
                if "[fused]" not in run_name_wb:
                    continue
                technique = run_name_wb.replace("[fused] ", "")
                t = wandb.Table(columns=["Metric", "Value"])
                for metric_name, metric_value in sorted(metrics.items()):
                    t.add_data(metric_name, metric_value)
                log_dict[f"fused_{technique}_metrics"] = t

            wandb.log(log_dict)

            # Upload input run files as artifacts
            logger.info("Uploading artifacts to W&B...")
            for run_path_artifact in run_paths:
                wandb.save(str(run_path_artifact), base_path=str(run_path_artifact.parent))

            # Upload fused TREC files and metrics summary
            wandb.save(str(metrics_tsv_path), base_path=str(output_dir))
            for technique, output_path in written_files.items():
                wandb.save(str(output_path), base_path=str(output_dir))

            logger.info(f"W&B logging complete: {wandb.run.url}")  # type: ignore

        # Save JSON report
        report_path = output_dir / "fusion_results.json"
        report = {
            "metadata": {
                "input_runs": [str(p) for p in run_paths],
                "run_names": run_names,
                "techniques": list(fusion_results.keys()),
                "qrels_path": str(qrels_path),
                "output_dir": str(output_dir),
                "parameters": {
                    "rrf_k": args.rrf_k,
                    "top_k": args.top_k,
                },
            },
            "evaluation_results": {
                run_name: metrics for run_name, metrics in eval_results.items()
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n✓ JSON report saved to {report_path}")

        logger.info(f"\n{'=' * 80}")
        logger.info("Fusion complete!")
        logger.info(f"{'=' * 80}\n")

    except Exception as e:
        logger.error(f"Error during fusion: {e}", exc_info=True)
        try:
            if not args.dry_run:
                wandb.log({"error": str(e), "error_type": type(e).__name__})
                wandb.finish()
        except Exception as wbe:
            logger.error(f"Failed to log error to W&B: {wbe}")
        sys.exit(1)
    else:
        # Success: finalize W&B
        if not args.dry_run:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finalize W&B: {e}")


if __name__ == "__main__":
    main()
