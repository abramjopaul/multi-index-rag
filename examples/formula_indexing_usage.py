#!/usr/bin/env python3
"""
Usage examples for IVFFlat and IVFScalarQuantizer indexers with single representation.

Examples:
    # Index only SLT representation
    indexer = FormulaFAISSIndexerIVFFlat(
        index_path="data/indices/formula/ivfflat_slt",
        embedding_dir="data/formula-indexing",
        representation="slt",
        force_rebuild=True
    )
    indexer.index(limit=10000)  # For testing with 10k answers

    # Index OPT representation
    indexer = FormulaFAISSIndexerIVFFlat(
        index_path="data/indices/formula/ivfflat_opt",
        embedding_dir="data/formula-indexing",
        representation="opt",
        force_rebuild=True
    )
    indexer.index(limit=50000)  # For testing with 50k answers

    # Index SLT_type representation
    indexer = FormulaFAISSIndexerIVFFlat(
        index_path="data/indices/formula/ivfflat_slt_type",
        embedding_dir="data/formula-indexing",
        representation="slt_type"  # Explicitly specify
    )
    indexer.index()  # Full indexing without limit
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multirag.indexing.formula import (
    FormulaFAISSIndexerIVFFlat,
    FormulaFAISSIndexerIVFScalarQuantizer,
)

if __name__ == "__main__":
    print(__doc__)

    # # Example 1: Index only SLT with IVFFlat
    # print("\n" + "=" * 80)
    # print("EXAMPLE 1: Index SLT representation with IVFFlat")
    # print("=" * 80)

    # indexer_ivfflat = FormulaFAISSIndexerIVFFlat(
    #     index_path="data/indices/formula/test_slt_only",
    #     embedding_dir="data/formula-indexing",
    #     representation="slt",
    #     force_rebuild=True,
    # )
    # indexer_ivfflat.batch_search([("q1", "p^2")], k=2)  # Test search without indexing

    # print(f"Representations: {indexer_ivfflat.get_representations()}")
    # indexer_ivfflat.index(limit=500)

    # # Example 2: Index SLT with IVFScalarQuantizer
    # print("\n" + "=" * 80)
    # print("EXAMPLE 2: Index SLT representation with IVFScalarQuantizer (8-bit)")
    # print("=" * 80)

    # indexer_ivfscalarquant = FormulaFAISSIndexerIVFScalarQuantizer(
    #     index_path="data/indices/formula/test_slt_sq",
    #     embedding_dir="data/formula-indexing",
    #     representation="slt",
    #     quantizer_bits="8bit",
    #     force_rebuild=False,
    # )
    # print(f"Representations: {indexer_ivfscalarquant.get_representations()}")
    # indexer_ivfscalarquant.index(limit=500)

    # # Example 3: Index OPT with IVFScalarQuantizer (fp16)
    # print("\n" + "=" * 80)
    # print("EXAMPLE 3: Index OPT representation with IVFScalarQuantizer (fp16)")
    # print("=" * 80)

    indexer_opt = FormulaFAISSIndexerIVFScalarQuantizer(
        index_path="data/indices/formula/test_opt_sq",
        embedding_dir="data/formula-indexing",
        representation="opt",
        quantizer_bits="fp16",
        force_rebuild=True,
    )
    print(f"Representations: {indexer_opt.get_representations()}")
    # indexer_opt.index(limit=500)
    indexer_opt.index()

    
    print("\n✓ Uncomment indexer.index() calls above to run actual indexing")
