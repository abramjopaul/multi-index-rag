"""
Formula IVFFlat index sanity check.

Builds an index from the first 1000 answers, then queries it with:
  - Formulas taken directly from those same indexed answers
  - The most complex formulas from topic A.301 in topics.jsonl

Usage:  python test/test_formula_ivfflat.py
"""

import json
import tempfile
from pathlib import Path

from multirag.config.path_configs import ANSWERS_JSONL, FORMULA_INDEX_DIR, TOPICS_JSONL
from multirag.indexing.formula.faiss_ivfflat import FormulaFAISSIndexerIVFFlat

ANSWERS_LIMIT = 1000
REPRESENTATION = "slt"
TOP_K = 10


def load_answer_formulas(n: int = 5) -> list[tuple[str, str, str]]:
    """Return (answer_id, formula_id, latex) for the first n non-trivial indexed formulas."""
    results = []
    with open(ANSWERS_JSONL) as f:
        for i, line in enumerate(f):
            if i >= ANSWERS_LIMIT:
                break
            doc = json.loads(line)
            for fml in doc.get("formulas", []):
                latex = fml.get("latex", "")
                if len(latex.strip()) > 5:
                    results.append((doc["id"], fml["formula_id"], latex))
                if len(results) >= n:
                    return results
    return results


def load_topic_formulas(n: int = 3) -> list[tuple[str, str, str]]:
    """Return (topic_id, formula_id, latex) for the n most complex formulas from topic A.301."""
    with open(TOPICS_JSONL) as f:
        topic = json.loads(f.readline())

    formulas = sorted(topic.get("formulas", []), key=lambda x: len(x.get("latex", "")), reverse=True)
    return [(topic["topic_id"], fml["formula_id"], fml["latex"]) for fml in formulas[:n]]


def print_hits(label: str, latex: str, hits: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"Query  : {latex[:100]}")
    print(f"Label  : {label}")
    print(f"{'='*70}")
    if not hits:
        print("  (no results)")
        return
    print(f"  {'rank':<6} {'post_id':<12} {'formula_id':<16} {'score'}")
    print(f"  {'-'*55}")
    for rank, hit in enumerate(hits, 1):
        print(f"  {rank:<6} {hit['doc_id']:<12} {hit['formula_id']:<16} {hit['score']:.6f}")


def main():
    index_dir = Path(tempfile.mkdtemp(prefix="formula_ivfflat_"))
    print(f"Index directory: {index_dir}")

    indexer = FormulaFAISSIndexerIVFFlat(
        index_path=index_dir,
        corpus_path=ANSWERS_JSONL,
        embedding_dir=str(FORMULA_INDEX_DIR),
        representation=REPRESENTATION,
        nprobe=16,
    )

    print(f"\nBuilding index (first {ANSWERS_LIMIT} answers, representation={REPRESENTATION})...")
    indexer.index(limit=ANSWERS_LIMIT)
    print(f"Index built: {indexer._faiss_index.ntotal} formula vectors")

    # --- Queries from indexed answers (expect self-hits near the top) ---
    print("\n\n>>> QUERIES FROM INDEXED ANSWERS")
    for answer_id, formula_id, latex in load_answer_formulas(n=5):
        hits = indexer.search(latex, k=TOP_K)
        print_hits(f"answer_id={answer_id}  formula_id={formula_id}", latex, hits)

    # --- Queries from topics (cross-query) ---
    print("\n\n>>> QUERIES FROM TOPIC FORMULAS (A.301)")
    for topic_id, formula_id, latex in load_topic_formulas(n=3):
        hits = indexer.search(latex, k=TOP_K)
        print_hits(f"topic_id={topic_id}  formula_id={formula_id}", latex, hits)


if __name__ == "__main__":
    main()
