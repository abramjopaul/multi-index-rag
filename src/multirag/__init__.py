"""
Multi-Index RAG: Retrieval-Augmented Generation with Formula-Aware Retrieval

This package provides tools for building and querying multi-modal retrieval systems
with support for:
    - Dense embeddings
    - Sparse (BM25) retrieval
    - Formula-aware retrieval using Symbol Layout Trees (SLT) and Operator Trees (OPT)

Modules:
    - formula_search: SLT/OPT generation from LaTeX formulas
    - embedding: Formula embedding generation using FastText
    - indexing: Index creation and management (dense, sparse, formula)
    - preprocessing: Data preprocessing utilities
    - evaluation: Evaluation metrics and tools
    - config: Configuration management
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from . import embedding, formula_search

__version__ = "0.1.0"
__author__ = "multirag"

__all__ = [
    "formula_search",
]
