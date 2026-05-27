#!/usr/bin/env python3
"""
Test script for single-representation FAISS indexers.

This script validates that the simplified architecture works correctly
with one representation per indexer instance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multirag.indexing.formula import (
    FormulaFAISSIndexerIVFFlat,
    FormulaFAISSIndexerIVFScalarQuantizer,
)


def test_ivfflat_single_rep():
    """Test IVFFlat with single representation."""
    print("\n" + "=" * 80)
    print("TEST: IVFFlat with SLT representation")
    print("=" * 80)
    
    indexer = FormulaFAISSIndexerIVFFlat(
        index_path="data/indices/formula/test_ivfflat_slt",
        embedding_dir="data/formula-indexing",
        representation="slt",
        force_rebuild=False,
    )
    
    # Verify representation
    reps = indexer.get_representations()
    print(f"Representations: {reps}")
    assert reps == ["slt"], f"Expected ['slt'], got {reps}"
    print("✓ get_representations() returns correct value")
    
    # Verify single representation attribute
    assert indexer.representation == "slt"
    print(f"✓ representation attribute = {indexer.representation}")
    
    print("✓ IVFFlat single-rep test passed")


def test_ivfscalarquantizer_single_rep():
    """Test IVFScalarQuantizer with single representation."""
    print("\n" + "=" * 80)
    print("TEST: IVFScalarQuantizer with OPT representation (8-bit)")
    print("=" * 80)
    
    indexer = FormulaFAISSIndexerIVFScalarQuantizer(
        index_path="data/indices/formula/test_sq_opt",
        embedding_dir="data/formula-indexing",
        representation="opt",
        quantizer_bits="8bit",
        force_rebuild=False,
    )
    
    # Verify representation
    reps = indexer.get_representations()
    print(f"Representations: {reps}")
    assert reps == ["opt"], f"Expected ['opt'], got {reps}"
    print("✓ get_representations() returns correct value")
    
    # Verify single representation attribute
    assert indexer.representation == "opt"
    print(f"✓ representation attribute = {indexer.representation}")
    
    # Verify quantizer setup
    assert indexer.quantizer_bits == "8bit"
    print(f"✓ quantizer_bits = {indexer.quantizer_bits}")
    
    print("✓ IVFScalarQuantizer single-rep test passed")


def test_invalid_representation():
    """Test that invalid representations are rejected."""
    print("\n" + "=" * 80)
    print("TEST: Invalid representation rejection")
    print("=" * 80)
    
    try:
        indexer = FormulaFAISSIndexerIVFFlat(
            index_path="data/indices/formula/test_invalid",
            embedding_dir="data/formula-indexing",
            representation="invalid_rep",
        )
        print("✗ Should have raised ValueError for invalid representation")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")


def test_all_representations():
    """Test that all valid representations work."""
    print("\n" + "=" * 80)
    print("TEST: All valid representations")
    print("=" * 80)
    
    for rep in ["slt", "opt", "slt_type"]:
        indexer = FormulaFAISSIndexerIVFFlat(
            index_path=f"data/indices/formula/test_{rep}",
            embedding_dir="data/formula-indexing",
            representation=rep,
        )
        assert indexer.representation == rep
        assert indexer.get_representations() == [rep]
        print(f"✓ {rep} representation works")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Testing Single-Representation FAISS Indexers")
    print("=" * 80)
    
    try:
        test_ivfflat_single_rep()
        test_ivfscalarquantizer_single_rep()
        test_invalid_representation()
        test_all_representations()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
