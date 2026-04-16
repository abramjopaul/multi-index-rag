"""Retrieval evaluation metrics for ARQMath Task 1 using ranx."""

import json
from pathlib import Path
from typing import Any

from ranx import Qrels, Run, evaluate

from multirag.indexing.base import BaseIndexer


def generate_run_file(
    indexer: BaseIndexer,
    topics_path: str | Path,
    output_path: str | Path,
    run_name: str,
    k: int = 1000,
) -> None:
    """Generate a TREC run file by batch-searching all topics.

    Loads topics from a JSONL file, runs batch search on the indexer,
    and writes results in standard TREC run format.

    Args:
        indexer: BaseIndexer instance (must support batch_search if available).
        topics_path: Path to topics.jsonl file (one JSON topic per line).
                     Each topic should have: topic_id, title, question fields.
        output_path: Path to write the TREC run file (TSV format).
        run_name: Name/identifier for this run (written in column 6).
        k: Number of results to retrieve per topic (default: 1000).

    Returns:
        None (writes run file as side effect).

    Raises:
        FileNotFoundError: If topics_path does not exist.
        AttributeError: If indexer doesn't support batch_search.
    """
    topics_path = Path(topics_path)
    output_path = Path(output_path)

    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}")

    # Load all topics from JSONL
    queries: list[tuple[str, str]] = []
    with open(topics_path) as f:
        for line in f:
            if not line.strip():
                continue
            topic = json.loads(line)
            topic_id = topic["topic_id"]
            title = topic.get("title", "")
            question = topic.get("question", "")
            query_text = f"{title} {question}".strip()
            if query_text:
                queries.append((topic_id, query_text))

    if not queries:
        raise ValueError(f"No valid topics found in {topics_path}")

    # Use batch_search for efficiency
    if not hasattr(indexer, "batch_search"):
        raise AttributeError(
            f"Indexer {type(indexer).__name__} does not support batch_search method. "
            "Use an indexer that implements batch_search for better performance."
        )

    results_dict = indexer.batch_search(queries, k=k)

    # Write TREC run file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for topic_id, results in results_dict.items():
            # results is a list of dicts with "doc_id"/"id" and "score" keys
            for rank, result in enumerate(results, start=1):
                # Handle both "doc_id" and "id" keys (implementation-dependent)
                doc_id = result.get("doc_id") or result.get("id")
                score = result["score"]
                # TREC format: query_id Q0 doc_id rank score run_id
                line = f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_name}\n"
                f.write(line)


def evaluate_run(
    qrels_path: str | Path,
    run_path: str | Path,
    metrics: list[str] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate a TREC run file using standard IR metrics.

    Computes metrics for ARQMath Task 1 evaluation:
    - nDCG@k: Normalized Discounted Cumulative Gain (primary metric, graded relevance)
    - MAP: Mean Average Precision (secondary metric)
    - Recall@k: Recall at k (secondary metric, coverage)
    - Precision@k: Precision at k (secondary metric, early precision)

    Args:
        qrels_path: Path to qrels file (TREC format).
        run_path: Path to run file (TREC format).
        metrics: List of metric strings for ranx (e.g., ["ndcg@5", "map", "recall@10"]).
                If None, uses default: ["map", "ndcg@5", "ndcg@10", "ndcg@100", "ndcg@1000",
                "recall@5", "recall@10", "recall@100", "recall@1000",
                "precision@5", "precision@10", "precision@100", "precision@1000"]
        k_values: Deprecated. Use metrics parameter instead.

    Returns:
        Dictionary with metric results. Keys are metric names like:
        "ndcg@5", "map", "recall@10", etc.

    Raises:
        FileNotFoundError: If qrels_path or run_path do not exist.
    """
    qrels_path = Path(qrels_path)
    run_path = Path(run_path)

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    # Default metrics if not provided
    if metrics is None:
        if k_values is None:
            k_values = [5, 10, 100, 1000]
        metrics = ["map"]
        for k in k_values:
            metrics.append(f"ndcg@{k}")
            metrics.append(f"recall@{k}")
            metrics.append(f"precision@{k}")

    # Load qrels and run files using ranx
    qrels = Qrels.from_file(str(qrels_path), kind="trec")
    run = Run.from_file(str(run_path), kind="trec")

    # Evaluate using ranx
    results = evaluate(qrels=qrels, run=run, metrics=metrics, make_comparable=True)

    # Ensure results is a dictionary (ranx may return dict or aggregated value)
    if isinstance(results, dict):
        return results
    else:
        # If ranx returns a single value, wrap it
        return {"aggregate": float(results)}


def print_evaluation_report(
    qrels_path: str | Path,
    run_path: str | Path,
    run_name: str = "Run",
    metrics: list[str] | None = None,
    k_values: list[int] | None = None,
) -> None:
    """Print a formatted evaluation report for a TREC run.

    Args:
        qrels_path: Path to qrels file (TREC format).
        run_path: Path to run file (TREC format).
        run_name: Name of the run (for display).
        metrics: List of metric strings for ranx (e.g., ["ndcg@5", "map"]).
                If None, uses k_values to generate default metrics.
        k_values: List of k values for @k metrics (only used if metrics is None).
                 Deprecated: prefer passing metrics directly.
    """
    if metrics is None and k_values is None:
        k_values = [5, 10, 100, 1000]

    results = evaluate_run(
        qrels_path=qrels_path, run_path=run_path, metrics=metrics, k_values=k_values
    )

    print(f"\n{'='*70}")
    print(f"Evaluation Report: {run_name}")
    print(f"{'='*70}")
    print(results)
    print(f"{'='*70}\n")
