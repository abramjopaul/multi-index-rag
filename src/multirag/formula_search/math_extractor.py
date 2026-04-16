"""
Math formula extractor for generating Symbol Layout Trees (SLT) and Operator Trees (OPT).

Minimal LaTeX-only extraction from TangentCFT MathExtractor.

Based on TangentCFT (Tangent Combined FastText)
Original repository: https://github.com/BehroozMansouri/TangentCFT

Original authors: Nidhin Pattaniyil, Frank Wm. Tompa, Kenny Davila Castellanos
Adapted for multirag formula_search module:
  - Removed XML/file parsing (LaTeX-only input)
  - Removed data reader dependencies
  - Simplified to core LaTeX → SLT/OPT conversion pipeline
  - Updated to use relative imports
"""

import io
import re
import sys
import xml.etree.ElementTree

from bs4 import BeautifulSoup

from multirag.formula_search.exceptions import UnknownTagException
from multirag.formula_search.latex_mml import LatexToMathML
from multirag.formula_search.layout_symbol import LayoutSymbol
from multirag.formula_search.semantic_symbol import SemanticSymbol
from multirag.formula_search.symbol_tree import SymbolTree

__author__ = "Nidhin, FWTompa, KDavila"


class MathExtractor:
    """
    Extract Symbol Layout Trees (SLT) or Operator Trees (OPT) from LaTeX formulas.
    Minimal extraction - LaTeX input only, no XML parsing.
    """

    def __init__(self):
        pass

    @classmethod
    def isolate_pmml(cls, tree):
        """
        Extract the Presentation MathML from a MathML expr

        :param tree: MathML expression
        :type tree: string
        :return: Presentation MathML
        :rtype: string
        """
        parsed_xml = BeautifulSoup(tree, "lxml")

        math_root = parsed_xml.find("math")  # namespaces have been removed (FWT)
        application_tex = math_root.find(
            "annotation", {"encoding": "application/x-tex"}
        )

        if application_tex:
            application_tex.decompose()

        pmml_markup = math_root.find(
            "annotation-xml", {"encoding": "MathML-Presentation"}
        )
        if pmml_markup:
            pmml_markup.name = "math"
        else:
            pmml_markup = math_root
            cmml_markup = math_root.find(
                "annotation-xml", {"encoding": "MathML-Content"}
            )
            if cmml_markup:
                cmml_markup.decompose()  # delete any Content MML
        pmml_markup["xmlns"] = (
            "http://www.w3.org/1998/Math/MathML"  # set the default namespace
        )
        return str(pmml_markup)

    @classmethod
    def isolate_cmml(cls, tree):
        """
        Extract the Content MathML from a MathML expr

        :param tree: MathML expression
        :type tree: string
        :return: Content MathML
        :rtype: string
        """
        parsed_xml = BeautifulSoup(tree, "lxml")

        math_root = parsed_xml.find("math")  # namespaces have been removed (FWT)
        application_tex = math_root.find(
            "annotation", {"encoding": "application/x-tex"}
        )

        if application_tex:
            application_tex.decompose()

        cmml_markup = math_root.find("annotation-xml", {"encoding": "MathML-Content"})
        if cmml_markup:
            cmml_markup.name = "math"
        else:
            cmml_markup = math_root
            pmml_markup = math_root.find(
                "annotation-xml", {"encoding": "MathML-Presentation"}
            )
            if pmml_markup:
                pmml_markup.decompose()  # delete any Presentation MML

        cmml_markup["xmlns"] = (
            "http://www.w3.org/1998/Math/MathML"  # set the default namespace
        )
        return str(cmml_markup)

    @classmethod
    def convert_to_layoutsymbol(cls, elem):
        """
        Parse expression from Presentation-MathML

        :param elem: mathml
        :type elem: string
        :rtype: LayoutSymbol or None
        :return: root of symbol tree
        """
        if len(elem) == 0:
            return None

        elem_content = io.StringIO(elem)  # treat the string as if a file
        root = xml.etree.ElementTree.parse(elem_content).getroot()
        return LayoutSymbol.parse_from_mathml(root)

    @classmethod
    def convert_to_semanticsymbol(cls, elem):
        """
        Parse expression from Content-MathML

        :param elem: mathml
        :type elem: string
        :rtype: SemanticSymbol or None
        :return: root of symbol tree
        """
        if len(elem) == 0:
            return None

        elem_content = io.StringIO(elem)  # treat the string as if a file
        root = xml.etree.ElementTree.parse(elem_content).getroot()
        return SemanticSymbol.parse_from_mathml(root)

    @classmethod
    def parse_from_tex(cls, tex, file_id=-1, position=None):
        """
        Parse expression from LaTeX string to Symbol Layout Tree (SLT).
        Uses latexmlmath to convert to presentation markup language.

        :param tex: LaTeX string
        :type tex: string
        :param file_id: file identifier (default -1)
        :type file_id: int
        :param position: position in document (default [0])
        :type position: list
        :rtype: SymbolTree
        :return: SymbolTree with LayoutSymbol root
        """
        if position is None:
            position = [0]

        mathml = LatexToMathML.convert_to_mathml(tex)
        pmml = cls.isolate_pmml(mathml)
        symbol_root = cls.convert_to_layoutsymbol(pmml)
        return SymbolTree(symbol_root, file_id, position)

    @classmethod
    def parse_from_tex_opt(cls, tex, file_id=-1, position=None):
        """
        Parse expression from LaTeX string to Operator Tree (OPT).
        Uses latexmlmath to convert to content markup language.

        :param tex: LaTeX string
        :type tex: string
        :param file_id: file identifier (default -1)
        :type file_id: int
        :param position: position in document (default None)
        :type position: list
        :rtype: SymbolTree
        :return: SymbolTree with SemanticSymbol root
        """
        mathml = LatexToMathML.convert_to_mathml2(tex)
        cmml = cls.isolate_cmml(mathml)
        symbol_root = cls.convert_to_semanticsymbol(cmml)
        return SymbolTree(symbol_root, file_id, position)
