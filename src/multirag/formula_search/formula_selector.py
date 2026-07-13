"""Query-formula selection layer for Task 1 formula retrieval.

Two strategies:
  fanout   – query with ALL non-trivial formulas in the topic, then RRF-fuse.
  heuristic – query with ONE formula per topic (title-first, most-tuples proxy).

Triviality is assessed on OPT tuples (Content MathML semantic structure), not on
raw LaTeX length or regex — this catches structurally empty formulas that a string
check would miss (e.g. bare set names, multi-char variables with no operators).
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# OPT tag prefixes that indicate a non-trivial operator or structural node
_OPERATOR_PREFIXES: frozenset[str] = frozenset(["O!", "U!", "F!", "+!", "M!"])
# OPT tag prefixes for content/leaf nodes (variables, numbers, constants, text)
_CONTENT_PREFIXES: frozenset[str] = frozenset(["V!", "N!", "C!", "T!"])
# End-of-baseline marker inserted by SymbolTree.get_pairs for leaf nodes
_EOB_MARKER = "0!"


def is_trivial(tuples: list[str], tree_type: str = "OPT") -> bool:
    """Return True if a formula carries no structural/relational information.

    Definition:
      - Empty tuple set → trivial (parsing failed: OCR junk, bare labels like "a)")
      - OPT (Content MathML): zero operator/relation nodes AND ≤1 distinct content symbol
      - SLT / SLT-TYPE (Presentation MathML): ≤1 distinct non-EOB node
        (^ _ appear as spatial *relations* in the third tuple field, not as node tags)

    Tuple format from SymbolTree.get_pairs: "s1 \\t s2 \\t relationship \\t location"
    OPT node tag prefixes: V!=variable, N!=number, C!=constant, T!=text (content);
                           O!/U!/F!/+!/M! = operator/structural (non-trivial).
    """
    if not tuples:
        return True

    if tree_type == "OPT":
        content_nodes: set[str] = set()
        has_operator = False
        for t in tuples:
            parts = t.split("\t")
            if len(parts) < 2:
                continue
            for tag in (parts[0], parts[1]):
                if tag == _EOB_MARKER:
                    continue
                prefix = tag[:2]
                if prefix in _OPERATOR_PREFIXES:
                    has_operator = True
                elif prefix in _CONTENT_PREFIXES:
                    content_nodes.add(tag)
        return not has_operator and len(content_nodes) <= 1

    else:
        # SLT / SLT-TYPE: all node tags are math symbols (letters, numbers, operators like =, +).
        # Spatial layout relations (superscript, subscript) appear in the *relationship* field
        # (third tab), never as node labels — so x^n has two real nodes {x, n} → not trivial.
        real_nodes: set[str] = set()
        for t in tuples:
            parts = t.split("\t")
            if len(parts) >= 1 and parts[0] != _EOB_MARKER:
                real_nodes.add(parts[0])
            if len(parts) >= 2 and parts[1] != _EOB_MARKER:
                real_nodes.add(parts[1])
        return len(real_nodes) <= 1


def _batch_extract_tuples(latex_list: list[str]) -> dict[str, list[str]]:
    """Batch-convert LaTeX → Content MathML → OPT tuples.

    Always uses OPT (Content MathML) regardless of the active retrieval representation.
    This mirrors _build_tuple_counts() but returns full tuple lists instead of counts,
    enabling both tuple-based triviality checks and complexity ranking.

    Args:
        latex_list: Unique LaTeX strings to process.

    Returns:
        {latex: [tuple_strings]} — empty list if MathML conversion or parsing failed.
    """
    from multirag.formula_search.latex_mml import LatexToMathML
    from multirag.formula_search.tuple_extraction import extract_tuples_from_mathml_direct

    if not latex_list:
        return {}

    mathml_list = LatexToMathML.convert_batch2(list(latex_list))
    return {
        lx: extract_tuples_from_mathml_direct(mml or "", tree_type="OPT")
        for lx, mml in zip(latex_list, mathml_list)
    }


def _strip_latex_delimiters(latex: str) -> str:
    """Remove outer LaTeX math delimiters, returning bare content."""
    import re
    s = latex.strip()
    s = re.sub(r"\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}", r"\1", s, flags=re.DOTALL)
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    elif s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2]
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s.strip()


def select_fanout(
    topic: dict,
    tuple_map: dict[str, list[str]],
    trivial_filter: bool = True,
) -> list[str]:
    """Return all non-trivial, deduplicated LaTeX formulas for this topic.

    Flow:
    1. Dedup by stripped LaTeX string (preserves original form for batch_search).
    2. Drop trivial formulas (via is_trivial on OPT tuples) if trivial_filter is True.
    3. Fallback: if all formulas are trivial, return the one with the most tuples
       (most structural complexity) and log a WARNING.

    Args:
        topic: Topic dict with 'topic_id' and 'formulas' list.
        tuple_map: {latex: [tuple_strings]} from _batch_extract_tuples.
        trivial_filter: Whether to apply is_trivial filtering.

    Returns:
        Non-empty list of LaTeX strings to use as queries.
    """
    topic_id = topic.get("topic_id", "?")
    formulas = topic.get("formulas", [])
    if not formulas:
        logger.warning(f"topic {topic_id} — no formulas found")
        return []

    # Dedup by stripped LaTeX (keep first occurrence's original form)
    seen: set[str] = set()
    unique: list[str] = []
    for f in formulas:
        key = _strip_latex_delimiters(f["latex"])
        if key not in seen:
            seen.add(key)
            unique.append(f["latex"])

    if not trivial_filter:
        return unique

    non_trivial = [
        lx for lx in unique
        if not is_trivial(tuple_map.get(lx, []), tree_type="OPT")
    ]

    if non_trivial:
        return non_trivial

    # All trivial fallback
    fallback = max(unique, key=lambda lx: len(tuple_map.get(lx, [])))
    logger.warning(
        f"topic {topic_id} — all {len(unique)} formula(s) trivial; "
        f"querying with most-complex fallback"
    )
    return [fallback]


def select_heuristic(
    topic: dict,
    tuple_map: dict[str, list[str]],
    trivial_filter: bool = True,
    title_preference: bool = True,
) -> list[str]:
    """Return a single-element list with the best query formula for this topic.

    Priority (title_preference=True):
    1. Non-trivial title formula with most tuples.
    2. Non-trivial body formula with most tuples.
    3. Any formula with most tuples (fallback when all trivial).

    With title_preference=False: pick from all formulas by most tuples.

    Args:
        topic: Topic dict with 'topic_id', 'formulas' list.
        tuple_map: {latex: [tuple_strings]} from _batch_extract_tuples.
        trivial_filter: Whether to apply is_trivial filtering.
        title_preference: Whether to prefer title formulas over body formulas.

    Returns:
        Single-element list with selected LaTeX string, or empty list if no formulas.
    """
    formulas = topic.get("formulas", [])
    if not formulas:
        return []

    # Dedup
    seen: set[str] = set()
    unique: list[dict] = []
    for f in formulas:
        if f["latex"] not in seen:
            seen.add(f["latex"])
            unique.append(f)

    def _complexity(lx: str) -> int:
        return len(tuple_map.get(lx, []))

    def _not_trivial(f: dict) -> bool:
        return not trivial_filter or not is_trivial(tuple_map.get(f["latex"], []), tree_type="OPT")

    def _best(pool: list[dict]) -> str | None:
        candidates = [f["latex"] for f in pool if _not_trivial(f)]
        return max(candidates, key=_complexity) if candidates else None

    if title_preference:
        title_formulas = [f for f in unique if f.get("in_title", False)]
        body_formulas = [f for f in unique if not f.get("in_title", False)]
        selected = _best(title_formulas) or _best(body_formulas)
    else:
        selected = _best(unique)

    if selected is None:
        # All trivial fallback
        selected = max((f["latex"] for f in unique), key=_complexity)
        logger.warning(
            f"topic {topic.get('topic_id', '?')} — all formulas trivial (heuristic); "
            f"falling back to most-complex"
        )

    return [selected]
