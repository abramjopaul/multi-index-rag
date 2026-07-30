"""Hole rate: a free, no-API util for the ARQMath "hole" problem.

Counts unjudged documents at a rank cutoff -- pure set difference against a
qrels dict, no judge calls. Forward-compat contract item 3
(track_v_implementation_plan.md): Track B reports hole rate @10/@100/@1000
without touching the judge API.

Accepts the same qrels dict shape multirag.evaluation.metrics._parse_qrels
already produces (dict[qid][doc_id] = int(relevance)) so it composes with
that module without duplicating a parser.
"""

from __future__ import annotations


def hole_rate(
    ranking: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> float:
    """Mean, over queries present in `ranking`, of the fraction of the top-k
    ranked docs that have no qrels judgement at all (neither relevant nor
    non-relevant).

    Args:
        ranking: qid -> ranked list of doc_ids (best first).
        qrels: qid -> {doc_id: relevance_label}, as produced by
            multirag.evaluation.metrics._parse_qrels.
        k: rank cutoff.

    Returns:
        Mean hole rate in [0, 1]. 0.0 if `ranking` is empty or every query's
        top-k slice is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    rates: list[float] = []
    for qid, docs in ranking.items():
        top_k = docs[:k]
        if not top_k:
            continue
        judged = qrels.get(qid, {})
        n_holes = sum(1 for doc_id in top_k if doc_id not in judged)
        rates.append(n_holes / len(top_k))

    if not rates:
        return 0.0
    return sum(rates) / len(rates)
