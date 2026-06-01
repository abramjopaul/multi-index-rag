"""
Share Element Resolver for LaTeXML Content MathML.

Resolves <share href="#..."/> elements in LaTeXML's latexmlmath --cmml=- output.

LaTeXML generates unresolved shares for chained operations (a = b = c). The href IDs
are never stamped on elements in the output, but shares always reference the RHS of
the previous pair in a chained structure.

Pattern LaTeXML generates for chained relations (a = b = c = d):

  <apply>
    <and/>
    <apply> <eq/> [A]      [B] </apply>   pair 0 : B becomes sh1
    <apply> <eq/> <share/> [C] </apply>   pair 1 : share → B; C becomes sh2
    <apply> <eq/> <share/> [D] </apply>   pair 2 : share → C; D becomes sh3
  </apply>

This module uses structural pattern matching based on LaTeXML's generation pattern:
- Share at position N refers to RHS of pair N-1
- Pattern works because LaTeXML always orders shares this way
- Doesn't require explicit IDs

Based on resolve_shares.py by the original author.
"""

import logging
import xml.etree.ElementTree as ET
from copy import deepcopy

logger = logging.getLogger(__name__)

MML = "http://www.w3.org/1998/Math/MathML"

# Pre-register namespace to preserve it in output
ET.register_namespace("", MML)


class ShareResolver:
    """Resolve unresolved <share> elements in LaTeXML Content MathML output."""

    SHARE_TAG = f"{{{MML}}}share"
    CHAIN_HEADS = {
        f"{{{MML}}}and",
        f"{{{MML}}}or",
        f"{{{MML}}}eq",
        f"{{{MML}}}neq",
        f"{{{MML}}}lt",
        f"{{{MML}}}leq",
        f"{{{MML}}}gt",
        f"{{{MML}}}geq",
    }

    @staticmethod
    def resolve(mathml_str: str) -> str:
        """
        Resolve all <share> elements in MathML string.

        Tries structural resolution first (works for most LaTeXML output),
        falls back to ID-based resolution if available.

        Args:
            mathml_str: MathML string potentially containing unresolved <share> elements

        Returns:
            MathML string with shares replaced by cloned nodes
        """
        if not mathml_str or len(mathml_str) == 0:
            return mathml_str

        try:
            root = ET.fromstring(mathml_str)
        except Exception as e:
            logger.debug(f"Failed to parse MathML for share resolution: {e}")
            return mathml_str

        # Try structural resolution first
        ShareResolver._resolve_tree(root)

        # If shares remain, try ID-based resolution
        if ShareResolver._has_shares(root):
            ShareResolver._resolve_by_id(root)

        # Convert back to string
        try:
            result = ET.tostring(root, encoding="unicode")
            # Remove self-closing space: /> → />
            result = result.replace(" />", "/>")
            return result
        except Exception as e:
            logger.debug(f"Failed to serialize resolved MathML: {e}")
            return mathml_str

    @staticmethod
    def _has_shares(root) -> bool:
        """Check if any share elements remain."""
        return any(elem.tag == ShareResolver.SHARE_TAG for elem in root.iter())

    @staticmethod
    def _resolve_tree(root):
        """Structurally resolve shares based on LaTeXML generation pattern."""
        ShareResolver._walk(root)

    @staticmethod
    def _walk(node):
        """Recursively walk and resolve chained operations."""
        children = list(node)
        if not children:
            return

        # Check if first child is a chain operator
        if children[0].tag in ShareResolver.CHAIN_HEADS:
            pairs = children[1:]
            ShareResolver._resolve_chain(pairs)

        # Continue walking
        for child in list(node):
            ShareResolver._walk(child)

    @staticmethod
    def _resolve_chain(pairs):
        """Resolve shares in a chain of pairs (e.g., a = b = c = d)."""
        for i in range(1, len(pairs)):
            pair = pairs[i]
            p_kids = list(pair)

            if len(p_kids) < 2:
                continue

            share = p_kids[1]  # LHS of this pair
            if share.tag != ShareResolver.SHARE_TAG:
                continue

            # Get the previous pair's RHS
            prev = pairs[i - 1]
            prev_kids = list(prev)
            if len(prev_kids) < 3:
                continue

            rhs_of_prev = prev_kids[2]

            # Clone the previous RHS and replace the share
            clone = deepcopy(rhs_of_prev)
            clone.attrib.pop("id", None)  # Remove ID from cloned node

            # Preserve whitespace/tail
            clone.tail = share.tail

            # Replace share with clone
            idx = list(pair).index(share)
            pair[idx] = clone

            # Recursively resolve any nested shares in the clone
            ShareResolver._walk(clone)

    @staticmethod
    def _resolve_by_id(root):
        """ID-based resolver for shares where structural resolution fails."""
        # Build ID map
        id_map = {}
        for elem in root.iter():
            elem_id = elem.get("id")
            if elem_id:
                id_map[elem_id] = elem

        def _replace(parent):
            """Recursively replace shares using ID references."""
            for i, child in enumerate(list(parent)):
                if child.tag == ShareResolver.SHARE_TAG:
                    ref = child.get("href", "").lstrip("#")
                    if ref in id_map:
                        clone = deepcopy(id_map[ref])
                        clone.attrib.pop("id", None)  # Remove ID from clone
                        clone.tail = child.tail  # Preserve whitespace
                        parent[i] = clone
                        _replace(clone)  # Recursively resolve nested shares
                else:
                    _replace(child)

        _replace(root)


__all__ = ["ShareResolver"]
