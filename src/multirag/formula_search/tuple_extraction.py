# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Tuple extraction and encoding utilities for mathematical formulas.
#
# Handles:
# 1. Extraction of tuples from pre-computed MathML (direct approach, no subprocess)
# 2. Encoding tuples into Unicode token sequences
#
# Distinguishes between:
# - Direct MathML approach (this module) — pre-computed MathML from TSV files
# - LaTeX subprocess approach (FormulaTrainer) — LaTeX → subprocess → MathML → trees

import logging
from typing import List, Literal

from multirag.formula_search.math_extractor import MathExtractor
from multirag.formula_search.opt_generator import OPTGenerator
from multirag.formula_search.slt_generator import SLTGenerator
from multirag.formula_search.symbol_tree import SymbolTree
from multirag.formula_search.tuple_tokenizer import TupleTokenizer

logger = logging.getLogger(__name__)


def extract_tuples_from_mathml_direct(
    mathml: str,
    tree_type: Literal["SLT", "OPT", "SLT-TYPE"] = "SLT",
) -> List[str]:
    """
    Extract tuples directly from pre-computed MathML (no subprocess conversion).

    This is the DIRECT approach used by FormulaTrainerDirect, which reads
    pre-computed MathML from TSV files. In contrast, FormulaTrainer converts
    LaTeX to MathML via latexmlmath subprocess before tuple extraction.

    Args:
        mathml: MathML string (Presentation for SLT, Content for OPT)
        tree_type: "SLT" (Symbol Layout Tree), "OPT" (Operator Tree), or "SLT-TYPE"

    Returns:
        List of tab-separated tuples (empty list if parsing failed)

    Example:
        >>> mathml = '<math><mrow><mn>2</mn></mrow></math>'
        >>> tuples = extract_tuples_from_mathml_direct(mathml, tree_type="SLT")
        >>> len(tuples) > 0
        True

    Implementation:
    1. Isolate MathML markup (remove annotations)
    2. Parse to LayoutSymbol or SemanticSymbol tree
    3. Extract tuples using SymbolTree.get_pairs(window=2, eob=True)
    4. Return list of tab-separated tuples
    """
    try:
        if tree_type == "OPT":
            # Content MathML → Operator Tree
            cmml = MathExtractor.isolate_cmml(mathml)
            symbol_root = MathExtractor.convert_to_semanticsymbol(cmml)
        else:
            # Presentation MathML → Symbol Layout Tree (SLT and SLT-TYPE)
            pmml = MathExtractor.isolate_pmml(mathml)
            symbol_root = MathExtractor.convert_to_layoutsymbol(pmml)

        if symbol_root is None:
            return []

        # Extract tuples from SymbolTree
        tree = SymbolTree(symbol_root)
        tuples = tree.get_pairs(window=2, eob=True)
        return tuples if tuples else []

    except Exception as e:
        logger.debug(f"Failed to parse MathML for {tree_type}: {e}")
        return []


def extract_tuples_from_latex_subprocess(
    latex: str,
    tree_type: Literal["SLT", "OPT", "SLT-TYPE"] = "SLT",
) -> List[str]:
    """
    Extract tuples from LaTeX formula using latexmlmath subprocess conversion.

    This is the SUBPROCESS approach used by FormulaTrainer, which converts
    LaTeX → MathML via latexmlmath subprocess before tuple extraction.
    In contrast, extract_tuples_from_mathml_direct() reads pre-computed MathML.

    Process:
    1. Convert LaTeX to MathML using latexmlmath (subprocess call)
    2. Parse MathML to LayoutSymbol or SemanticSymbol tree
    3. Extract tuples using SymbolTree.get_pairs(window=2, eob=True)
    4. Return list of tab-separated tuples

    Note: SLT-TYPE uses SLT tree generation with Type-only tokenization.

    Args:
        latex: LaTeX formula string (e.g., "x^2 + y" or "\\frac{a}{b}")
        tree_type: "SLT" (Symbol Layout Tree), "OPT" (Operator Tree), or "SLT-TYPE"

    Returns:
        List of tab-separated tuples (empty list if conversion or parsing failed)

    Example:
        >>> latex = r"x^2 + y"
        >>> tuples = extract_tuples_from_latex_subprocess(latex, tree_type="SLT")
        >>> len(tuples) > 0
        True

    Performance Note:
        This approach calls latexmlmath subprocess for each formula, which is slower
        than extract_tuples_from_mathml_direct() but doesn't require pre-computed MathML.
        For batch processing, consider pre-computing MathML when possible.
    """
    try:
        # Use SLTGenerator or OPTGenerator (both handle LaTeX → MathML → tree conversion)
        if tree_type == "OPT":
            generator = OPTGenerator()
            tree = generator.generate(latex)
        else:
            # SLT or SLT-TYPE both use SLT tree generation
            generator = SLTGenerator()
            tree = generator.generate(latex)

        if tree is None:
            return []

        # Extract tuples from SymbolTree
        tuples = tree.get_pairs(window=2, eob=True)
        return tuples if tuples else []

    except Exception as e:
        logger.debug(f"Failed to parse LaTeX for {tree_type}: {e}")
        return []


def encode_tuples(
    tuples: List[str],
    tuple_tokenizer: TupleTokenizer,
) -> str:
    """
    Encode a list of tuples into a whitespace-separated token string.

    Shared utility used by both trainer approaches (FormulaTrainer and FormulaTrainerDirect).

    Process:
    1. Parse each tuple (tab-separated components)
    2. Tokenize using TupleTokenizer
    3. Map tokens to numeric IDs via TokenIDManager (encoder_maps)
    4. Convert IDs to Unicode characters using chr()
    5. Join all encoded tokens with whitespace

    Args:
        tuples: List of tab-separated tuples
                Format: "N!2\tV!q\tn\t-" or similar
        tuple_tokenizer: TupleTokenizer instance with configured TokenIDManager

    Returns:
        Whitespace-separated encoded tokens (one token per tuple)
        Used as LineSentence format for FastText training

    Example:
        >>> from multirag.formula_search import TokenIDManager, TupleTokenizer
        >>> tokenizer = TupleTokenizer(TokenIDManager(), ...)
        >>> tuples = ["N!2\\tV!q\\tn\\t-", "V!q\\tN!2\\ta\\tn"]
        >>> encoded = encode_tuples(tuples, tokenizer)
        >>> isinstance(encoded, str)
        True
        >>> len(encoded.split(" ")) == len(tuples)
        True
    """
    encoded_tokens = []

    for tuple_str in tuples:
        encoded = tuple_tokenizer.tokenize_tuple(tuple_str)
        if encoded:
            encoded_tokens.append(encoded)

    return " ".join(encoded_tokens)  # Whitespace-separated for LineSentence
