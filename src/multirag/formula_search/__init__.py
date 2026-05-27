"""
Formula Search Module

This module provides functionality to generate Symbol Layout Trees (SLT) and Operator Trees (OPT)
from LaTeX mathematical formulas. These tree representations are used for efficient formula retrieval
and similarity computation.

Main Classes:
    - SLTGenerator: Generate Symbol Layout Trees (visual/spatial representation)
    - OPTGenerator: Generate Operator Trees (semantic/operational representation)
    - SymbolTree: Tree structure wrapping SLT or OPT roots

Example:
    >>> from multirag.formula_search import SLTGenerator, OPTGenerator
    >>>
    >>> # Generate SLT
    >>> slt_gen = SLTGenerator()
    >>> slt = slt_gen.generate(r"x^2 + y")
    >>> if slt:
    ...     tuples = slt.get_pairs(window=2, eob=True)
    >>>
    >>> # Generate OPT
    >>> opt_gen = OPTGenerator()
    >>> opt = opt_gen.generate(r"x^2 + y")
    >>> if opt:
    ...     tuples = opt.get_pairs(window=2, eob=True)
"""

from multirag.formula_search.exceptions import UnknownTagException
from multirag.formula_search.layout_symbol import LayoutSymbol
from multirag.formula_search.math_extractor import MathExtractor
from multirag.formula_search.opt_generator import OPTGenerator
from multirag.formula_search.semantic_symbol import SemanticSymbol
from multirag.formula_search.slt_generator import SLTGenerator
from multirag.formula_search.symbol_tree import SymbolTree
from multirag.formula_search.tuple_extraction import (
    encode_tuples,
    extract_tuples_from_latex_subprocess,
    extract_tuples_from_mathml_direct,
)
from multirag.formula_search.tuple_tokenizer import (
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
)

__all__ = [
    "SLTGenerator",
    "OPTGenerator",
    "SymbolTree",
    "LayoutSymbol",
    "SemanticSymbol",
    "MathExtractor",
    "UnknownTagException",
    "TupleTokenizationMode",
    "TokenIDManager",
    "TupleTokenizer",
    "FormulaTokenizerPipeline",
    "extract_tuples_from_mathml_direct",
    "extract_tuples_from_latex_subprocess",
    "encode_tuples",
]

__version__ = "0.1.0"
__author__ = "multirag"
