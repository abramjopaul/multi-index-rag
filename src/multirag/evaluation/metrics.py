"""Retrieval evaluation metrics for ARQMath Task 1 using pytrec_eval."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytrec_eval

from multirag.indexing.base import BaseIndexer

# Canonical metric set: all evaluated at k=5,10,100,1000.
# bpref and map are query-level aggregates with no @k cutoff.
_EVAL_METRICS = {
    "ndcg_cut.5,10,100,1000",
    "P.5,10,100,1000",
    "recall.5,10,100,1000",
    "map_cut.5,10,100,1000",
    "map",
    "bpref",
}

# Map pytrec_eval internal key → display name
_RENAME = {
    "ndcg_cut_5": "ndcg@5",
    "ndcg_cut_10": "ndcg@10",
    "ndcg_cut_100": "ndcg@100",
    "ndcg_cut_1000": "ndcg@1000",
    "P_5": "precision@5",
    "P_10": "precision@10",
    "P_100": "precision@100",
    "P_1000": "precision@1000",
    "recall_5": "recall@5",
    "recall_10": "recall@10",
    "recall_100": "recall@100",
    "recall_1000": "recall@1000",
    "map_cut_5": "map@5",
    "map_cut_10": "map@10",
    "map_cut_100": "map@100",
    "map_cut_1000": "map@1000",
    "map": "map",
    "bpref": "bpref",
}


def _parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, score = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def _parse_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Standard TREC (6 cols): qid Q0 doc_id rank score run_name
            # ARQMath (5 cols):       qid doc_id rank score run_name
            if parts[1].upper() == "Q0":
                qid, doc_id, score = parts[0], parts[2], float(parts[4])
            else:
                qid, doc_id, score = parts[0], parts[1], float(parts[3])
            run.setdefault(qid, {})[doc_id] = score
    return run


def _to_judged_only(
    run: dict[str, dict[str, float]],
    qrel: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Remove unjudged docs from each query's ranking — converts metrics to prime variants."""
    return {
        qid: {doc: score for doc, score in docs.items() if doc in qrel.get(qid, {})}
        for qid, docs in run.items()
    }


def _avg_over_queries(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average per-query metric values across all queries."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for per_query in results.values():
        for key, val in per_query.items():
            totals[key] += val
            counts[key] += 1
    return {key: totals[key] / counts[key] for key in totals}


# ---------------------------------------------------------------------------
# Formula selection helpers
# ---------------------------------------------------------------------------

# Matches single variables (a-z, A-Z, 0-9, up to 2 chars) or lone LaTeX
# commands with no arguments (\infty, \alpha, \pi …).
_SINGLE_SYMBOL_RE = re.compile(r"^[a-zA-Z0-9]{1,2}$|^\\[a-zA-Z]+$")


def _strip_latex_delimiters(latex: str) -> str:
    """Remove outer LaTeX math delimiters, returning bare content."""
    s = latex.strip()
    s = re.sub(r"\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}", r"\1", s, flags=re.DOTALL)
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    elif s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2]
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s.strip()


def _is_clearly_trivial(latex: str) -> bool:
    """True for single variables, digits, and lone LaTeX commands (\\infty, \\alpha)."""
    return bool(_SINGLE_SYMBOL_RE.match(_strip_latex_delimiters(latex)))


def _build_tuple_counts(latex_set: set[str]) -> dict[str, int]:
    """Batch LaTeX → CMML → OPT tuple count used to rank formula complexity.

    convert_batch2 outputs Content MathML, so we pair it with tree_type="OPT".
    Using "SLT" here would silently return empty lists (SLT needs Presentation MathML).
    The count is used only for ranking candidates, not for retrieval itself.
    """
    from multirag.formula_search.latex_mml import LatexToMathML
    from multirag.formula_search.tuple_extraction import extract_tuples_from_mathml_direct

    unique = list(latex_set)
    mathml_list = LatexToMathML.convert_batch2(unique)
    return {
        lx: len(extract_tuples_from_mathml_direct(mml or "", tree_type="OPT"))
        for lx, mml in zip(unique, mathml_list)
    }


def _select_formula_for_topic(
    topic: dict,
    tuple_counts: dict[str, int],
) -> str | None:
    """Pick the single best query formula for a topic.

    Priority:
    1. Highest-tuple-count non-trivial title formula.
    2. Highest-tuple-count non-trivial question formula (if no title candidates).
    3. Longest (by stripped length) formula when everything is clearly trivial.
    """
    title = topic.get("title", "")
    formulas = topic.get("formulas", [])
    if not formulas:
        return None

    seen: set[str] = set()
    unique: list[dict] = []
    for f in formulas:
        if f["latex"] not in seen:
            seen.add(f["latex"])
            unique.append(f)

    def _in_title(f: dict) -> bool:
        return f.get("in_title", f["latex"] in title)

    title_candidates = [
        f["latex"] for f in unique if _in_title(f) and not _is_clearly_trivial(f["latex"])
    ]
    question_candidates = [
        f["latex"] for f in unique if not _in_title(f) and not _is_clearly_trivial(f["latex"])
    ]

    def _best(pool: list[str]) -> str | None:
        return max(pool, key=lambda lx: tuple_counts.get(lx, 0)) if pool else None

    return (
        _best(title_candidates)
        or _best(question_candidates)
        or max((f["latex"] for f in unique), key=lambda lx: len(_strip_latex_delimiters(lx)))
    )


def generate_run_file(
    indexer: BaseIndexer,
    topics_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
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
    # Inline import to avoid circular dependency at module level
    from multirag.indexing.formula.faiss_scalar_quantizer import (
        FormulaFAISSIndexerIVFScalarQuantizer,
    )

    topics_path = Path(topics_path)
    output_path = Path(output_path)

    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}")

    # Load all topics from JSONL (always include formulas field)
    topics_list: list[dict[str, Any]] = []
    with open(topics_path) as f:
        for line in f:
            if not line.strip():
                continue
            topics_list.append(json.loads(line))

    if not topics_list:
        raise ValueError(f"No valid topics found in {topics_path}")

    # Formula indexer path: per-formula batch_search + within-topic RRF merge
    if isinstance(indexer, FormulaFAISSIndexerIVFScalarQuantizer):
        # Phase 1: batch-count OPT tuples for all non-trivial formula candidates
        candidates = {
            f["latex"]
            for topic in topics_list
            for f in topic.get("formulas", [])
            if not _is_clearly_trivial(f["latex"])
        }
        tuple_counts = _build_tuple_counts(candidates)

        # Phase 2: heuristic selection — one formula per topic
        formula_queries: list[tuple[str, str]] = []
        qid_to_topic: dict[str, str] = {}
        for topic in topics_list:
            latex = _select_formula_for_topic(topic, tuple_counts)
            if latex:
                uqid = f"{topic['topic_id']}_f0"
                formula_queries.append((uqid, latex))
                qid_to_topic[uqid] = topic["topic_id"]

        if not formula_queries:
            raise ValueError("No formulas found in topics — cannot run formula index search.")

        # Single batch_search call with all (unique_qid, latex) pairs
        raw = indexer.batch_search(formula_queries, k=k)

        # RRF merge: accumulate scores per (topic_id, doc_id) across formula ranked lists
        RRF_K = 60
        topic_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for uqid, hits in raw.items():
            topic_id = qid_to_topic[uqid]
            for rank, hit in enumerate(hits):
                doc_id = hit["doc_id"]
                topic_scores[topic_id][doc_id] += 1.0 / (rank + 1 + RRF_K)

        # Sort each topic's results by RRF score descending and take top-k
        merged: dict[str, list[tuple[str, float]]] = {
            tid: sorted(scores.items(), key=lambda x: -x[1])[:k]
            for tid, scores in topic_scores.items()
        }

        # Write TREC run file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for topic_id, doc_scores in merged.items():
                for rank, (doc_id, score) in enumerate(doc_scores, start=1):
                    f.write(f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")

        return

    # Text indexer path (sparse / dense): single query string per topic
    queries: list[tuple[str, str]] = []
    for topic in topics_list:
        topic_id = topic["topic_id"]
        query_text = f"{topic.get('title', '')} {topic.get('question', '')}".strip()
        if query_text:
            queries.append((topic_id, query_text))

    if not queries:
        raise ValueError(f"No valid topics found in {topics_path}")

    results_dict = indexer.batch_search(queries, k=k)

    # Write TREC run file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for topic_id, results in results_dict.items():
            for rank, result in enumerate(results, start=1):
                doc_id = result["doc_id"]
                score = result["score"]
                f.write(f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_name}\n")


def evaluate_run(
    qrels_path: str | os.PathLike[str],
    run_path: str | os.PathLike[str],
) -> dict[str, float]:
    """Evaluate a TREC run file using pytrec_eval.

    Computes standard and prime (judged-only) variants of:
      ndcg@5/10/100/1000, precision@5/10/100/1000,
      recall@5/10/100/1000, map@5/10/100/1000, map, bpref

    Prime metrics (unjudged docs removed before scoring) use the same names
    with a trailing apostrophe, e.g. ndcg'@10.

    Args:
        qrels_path: Path to qrels file (TREC format).
        run_path: Path to run file (TREC format).

    Returns:
        Flat dict of averaged metric values, e.g. {"ndcg@10": 0.42, "ndcg'@10": 0.51, ...}.

    Raises:
        FileNotFoundError: If qrels_path or run_path do not exist.
    """
    qrels_path = Path(qrels_path)
    run_path = Path(run_path)

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    qrel = _parse_qrels(qrels_path)
    run = _parse_run(run_path)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, _EVAL_METRICS, relevance_level=2)

    avg_standard = _avg_over_queries(evaluator.evaluate(run))
    avg_prime = _avg_over_queries(evaluator.evaluate(_to_judged_only(run, qrel)))

    output: dict[str, float] = {}
    for internal_key, display_name in _RENAME.items():
        if internal_key in avg_standard:
            output[display_name] = avg_standard[internal_key]
        if internal_key in avg_prime:
            # Insert apostrophe before @ if present, otherwise append
            if "@" in display_name:
                base, k = display_name.split("@", 1)
                output[f"{base}'@{k}"] = avg_prime[internal_key]
            else:
                output[f"{display_name}'"] = avg_prime[internal_key]

    return output


def print_evaluation_report(
    qrels_path: str | os.PathLike[str],
    run_path: str | os.PathLike[str],
    run_name: str = "Run",
) -> None:
    """Print a formatted evaluation report for a TREC run.

    Args:
        qrels_path: Path to qrels file (TREC format).
        run_path: Path to run file (TREC format).
        run_name: Name of the run (for display).
    """
    results = evaluate_run(qrels_path=qrels_path, run_path=run_path)

    standard = {k: v for k, v in results.items() if "'" not in k}
    prime = {k: v for k, v in results.items() if "'" in k}

    print(f"\n{'=' * 70}")
    print(f"Evaluation Report: {run_name}")
    print(f"{'=' * 70}")
    print("Standard metrics:")
    for k, v in sorted(standard.items()):
        print(f"  {k:<20} {v:.4f}")
    print("Prime metrics (judged-only):")
    for k, v in sorted(prime.items()):
        print(f"  {k:<20} {v:.4f}")
    print(f"{'=' * 70}\n")
