"""
Operator Tree (OPT) Generator.

Generate semantic/operational representations of mathematical formulas using latexmlmath and TangentS.

This module provides a clean, high-level API for generating Operator Trees from LaTeX formulas.
Built on top of adapted TangentCFT components:

Based on TangentCFT (Tangent Combined FastText)
Original repository: https://github.com/BehroozMansouri/TangentCFT

Original authors: Behrooz Mansouri, Richard Zanibbi, and contributors
Module author: multirag
"""

import logging
from typing import Optional

from multirag.formula_search.math_extractor import MathExtractor
from multirag.formula_search.symbol_tree import SymbolTree

logger = logging.getLogger(__name__)


class OPTGenerator:
    """
    Generate Operator Trees (OPT) from LaTeX mathematical formulas.

    OPTs represent the semantic/operational structure of mathematical formulas,
    encoding the mathematical operators and their operands (e.g., addition, multiplication, etc.).

    Example:
        >>> gen = OPTGenerator()
        >>> tree = gen.generate(r"x^2 + y")
        >>> print(tree)  # SymbolTree object or None if parsing failed
    """

    def __init__(self):
        """Initialize OPTGenerator. Checks latexmlmath availability."""
        self._check_latexmlmath_available()

    @staticmethod
    def _check_latexmlmath_available():
        """
        Check if latexmlmath is available on the system.
        Raises an exception if not found.
        """
        import shutil
        import subprocess

        if shutil.which("latexmlmath") is None:
            logger.error(
                "latexmlmath not found on system. Install it via: apt-get install latexml"
            )
            raise RuntimeError(
                "latexmlmath is not available. Please install LaTeXML package."
            )

    def generate(self, latex_formula: str) -> Optional[SymbolTree]:
        """
        Generate an Operator Tree from a LaTeX formula.

        :param latex_formula: LaTeX string representing a mathematical formula
        :type latex_formula: str
        :return: SymbolTree with SemanticSymbol root, or None if parsing fails
        :rtype: Optional[SymbolTree]
        :raises: Logs errors but does not raise exceptions (returns None on error)

        Example:
            >>> gen = OPTGenerator()
            >>> tree = gen.generate(r"\\frac{a}{b}")
            >>> if tree:
            ...     pairs = tree.get_pairs(window=2, eob=True)
            ...     print(pairs)
        """
        try:
            if not latex_formula or not isinstance(latex_formula, str):
                logger.warning(f"Invalid LaTeX formula provided: {type(latex_formula)}")
                return None

            # Parse LaTeX to OPT
            tree = MathExtractor.parse_from_tex_opt(latex_formula)

            # Validate the result
            if tree is None or tree.root is None:
                logger.warning(f"Failed to generate OPT for formula: {latex_formula}")
                return None

            return tree

        except Exception as e:
            logger.error(
                f"Error generating OPT for formula '{latex_formula}': {str(e)}"
            )
            return None

    def generate_tuples(self, latex_formula: str, window: int = 2, eob: bool = True):
        """
        Generate tuple representations of symbol pairs from a LaTeX formula.

        This is a convenience method that generates an OPT and extracts tuples in one step.

        :param latex_formula: LaTeX string
        :type latex_formula: str
        :param window: Maximum distance between symbols in pair (default 2)
        :type window: int
        :param eob: Include end-of-baseline pairs (default True)
        :type eob: bool
        :return: List of tuples as strings, or empty list if parsing fails
        :rtype: list[str]

        Example:
            >>> gen = OPTGenerator()
            >>> tuples = gen.generate_tuples(r"x + y")
            >>> print(tuples)  # Tuples representing semantic operations
        """
        tree = self.generate(latex_formula)

        if tree is None:
            return []

        try:
            return tree.get_pairs(window=window, eob=eob)
        except Exception as e:
            logger.error(
                f"Error extracting tuples from OPT for formula '{latex_formula}': {str(e)}"
            )
            return []
