"""Formula-aware retrieval indexing module using TangentCFT and FAISS.

Provides two FAISS-based indexing strategies for evaluating trade-offs:
1. FormulaFAISSIndexerIVFFlat: High accuracy, high memory (~94.5 GB for 28M formulas)
2. FormulaFAISSIndexerIVFScalarQuantizer: Compressed, low memory (~2 GB for 28M formulas)

Usage:
    from multirag.indexing.formula import create_formula_indexer

    # Create and build IVFFlat index
    indexer = create_formula_indexer(
        index_type="ivfflat",
        embedding_dir="data/models/v1"
    )
    indexer.index()

    # Or use IVFScalarQuantizer for memory efficiency
    indexer = create_formula_indexer(
        index_type="scalar_quantizer",
        embedding_dir="data/models/v1"
    )
    indexer.index()

    # Compare both indices
    from multirag.indexing.formula import FormulaIndexComparator

    comparator = FormulaIndexComparator(
        index_path_ivfflat="data/indices/formula/ivfflat",
        index_path_sq="data/indices/formula/sq",
        embedding_dir="data/models/v1"
    )
    stats = comparator.build_both_indices()
    comparison_results = comparator.compare_search_results(query_embeddings)
    comparator.print_comparison_report(stats, comparison_results)
"""


from .faiss_ivfflat import FormulaFAISSIndexerIVFFlat
from .faiss_scalar_quantizer import FormulaFAISSIndexerIVFScalarQuantizer

__all__ = [
    "FormulaFAISSIndexerIVFFlat",
    "FormulaFAISSIndexerIVFScalarQuantizer"
]
