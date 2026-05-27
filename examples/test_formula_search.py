#!/usr/bin/env python3
"""Test formula search functionality for both indexers."""

import json
from pathlib import Path

from multirag.indexing.formula.faiss_ivfflat import FormulaFAISSIndexerIVFFlat
from multirag.indexing.formula.faiss_scalar_quantizer import (
    FormulaFAISSIndexerIVFScalarQuantizer,
)

# Paths
data_dir = Path("data/formula-indexing")
index_dir = Path("data/indices/formula/test_slt_only")
index_dir_sq = Path("data/indices/formula/test_slt_sq")
answers_path = Path("data/processed/collection/answers.jsonl")


def load_sample_formulas(limit: int = 5) -> list[str]:
    """Load sample formulas from the corpus for testing."""
    formulas = []
    with open(answers_path) as f:
        for line in f:
            try:
                answer = json.loads(line)
                for formula_obj in answer.get("formulas", []):
                    latex = formula_obj.get("latex", "")
                    if latex:
                        formulas.append(latex)
                        if len(formulas) >= limit:
                            return formulas
            except json.JSONDecodeError:
                continue
    return formulas


def test_ivfflat_search():
    """Test IVFFlat search methods."""
    print("\n" + "=" * 80)
    print("Testing IVFFlat Search Methods")
    print("=" * 80)

    indexer = FormulaFAISSIndexerIVFFlat(
        index_path=index_dir,
        corpus_path=answers_path,
        embedding_dir=str(data_dir),
        representation="slt",
        nprobe=32,
    )

    # Load sample formulas for testing
    test_formulas = load_sample_formulas(limit=3)
    if not test_formulas:
        print("❌ No sample formulas found for testing")
        return

    print(f"✓ Loaded {len(test_formulas)} sample formulas for testing")

    # Test 1: Single search
    print("\n--- Test 1: Single Query Search ---")
    query = test_formulas[0]
    print(f"Query: {query}")
    results = indexer.search(query, k=5)
    print(f"Results: {len(results)} hits")
    for hit in results[:3]:
        print(f"  Rank {hit['rank']}: formula_id={hit['formula_id']}, distance={hit['distance']:.4f}")

    # Test 2: Batch search
    print("\n--- Test 2: Batch Search ---")
    batch_queries = [(f"q{i}", formula) for i, formula in enumerate(test_formulas)]
    print(f"Batch queries: {len(batch_queries)} queries")
    batch_results = indexer.batch_search(batch_queries, k=3)
    print("\n--- Test 2: Batch Search Results ---")
    print(batch_results)
    for qid, hits in batch_results.items():
        print(f"  {qid}: {len(hits)} hits")


def test_ivfscalarquantizer_search():
    """Test IVFScalarQuantizer search methods."""
    print("\n" + "=" * 80)
    print("Testing IVFScalarQuantizer Search Methods")
    print("=" * 80)

    indexer = FormulaFAISSIndexerIVFScalarQuantizer(
        index_path=index_dir_sq,
        corpus_path=answers_path,
        embedding_dir=str(data_dir),
        representation="slt",
        quantizer_bits="fp16",
        nprobe=32,
    )

    # Load sample formulas for testing
    test_formulas = load_sample_formulas(limit=3)
    if not test_formulas:
        print("❌ No sample formulas found for testing")
        return

    print(f"✓ Loaded {len(test_formulas)} sample formulas for testing")

    # Test 1: Single search
    print("\n--- Test 1: Single Query Search ---")
    query = test_formulas[0]
    print(f"Query: {query}")
    results = indexer.search(query, k=5)
    print(f"Results: {len(results)} hits")
    for hit in results[:3]:
        print(f"  Rank {hit['rank']}: formula_id={hit['formula_id']}, distance={hit['distance']:.4f}")

    # Test 2: Batch search
    print("\n--- Test 2: Batch Search ---")
    batch_queries = [(f"q{i}", formula) for i, formula in enumerate(test_formulas)]
    print(f"Batch queries: {len(batch_queries)} queries")
    batch_results = indexer.batch_search(batch_queries, k=3)
    print("\n--- Test 2: Batch Search Results ---")
    print(batch_results)
    for qid, hits in batch_results.items():
        print(f"  {qid}: {len(hits)} hits")

    # Test 3: search_by_representation
    # print("\n--- Test 3: search_by_representation ---")
    # query = test_formulas[0]
    # print(f"Query: {query}")
    # results = indexer.search_by_representation(query, k=5)
    # print(f"Results: {results}")


if __name__ == "__main__":
    print("Testing Formula Search Implementation")

    try:
        test_ivfflat_search()
    except Exception as e:
        print(f"❌ IVFFlat test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_ivfscalarquantizer_search()
    except Exception as e:
        print(f"❌ IVFScalarQuantizer test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✓ Search implementation tests complete")
    print("=" * 80)
