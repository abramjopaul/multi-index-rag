"""Reference construction for Track C generation evaluation.

Per topic, built from human qrels only -- never LLM labels. The judge stays
confined to Track B hole-filling; using it here to build the reference the
same judge later scores against would make the evaluation circular.

ARQMath topics admit multiple valid solution methods, so the reference is
the UNION of up to `max_answers` answers rather than a single best answer --
a single-answer reference would mark a correct answer wrong for choosing a
different method than the one reference happened to use.

High(3)-labelled answers are always preferred and taken first, sorted by
(community score desc, answer_id desc) -- deterministic and stable. If High
alone doesn't reach `max_answers`, the remainder is filled from Medium(2)
answers, same sort, so every topic's reference tops out at max_answers
whenever enough judged answers of either label exist -- Medium is a filler
for topics with too few High answers, not an all-or-nothing fallback used
only when High is completely empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ReferenceResult:
    reference_text: str | None
    answer_ids_used: list[str] = field(default_factory=list)
    n_high_available: int = 0
    n_high_used: int = 0
    fallback_used: bool = False
    reference_char_len: int = 0
    reference_token_len: int = 0


def _select_candidates(
    topic_qrels: dict[str, int],
    answer_lookup: dict[str, tuple[str, int]],
    label: int,
) -> list[str]:
    """Answer ids at exactly `label`, present in answer_lookup, sorted by
    (community score desc, answer_id desc) -- deterministic and stable, so a
    topic with more than max_answers candidates yields the same chosen
    answers on every run.
    """
    candidates = [
        aid for aid, lbl in topic_qrels.items() if lbl == label and aid in answer_lookup
    ]
    candidates.sort(key=lambda aid: (-answer_lookup[aid][1], -int(aid)))
    return candidates


def select_references(
    topic_id: str,
    qrels: dict[str, dict[str, int]],
    answer_lookup: dict[str, tuple[str, int]],
    max_answers: int = 10,
    token_counter: Callable[[str], int] | None = None,
) -> ReferenceResult:
    """Select and concatenate the reference answers for one topic.

    Args:
        token_counter: optional callable (e.g. wrapping the judge's
            client.models.count_tokens) to measure reference_token_len
            against the judge's own tokenizer -- this exists to flag topics
            approaching the judge's context limit, not the generator's.
    """
    topic_qrels = qrels.get(topic_id, {})

    high = _select_candidates(topic_qrels, answer_lookup, label=3)
    n_high_available = len(high)

    high_used = high[:max_answers]
    remaining = max_answers - len(high_used)
    medium_used: list[str] = []
    if remaining > 0:
        medium = _select_candidates(topic_qrels, answer_lookup, label=2)
        medium_used = medium[:remaining]

    used = high_used + medium_used
    fallback_used = bool(medium_used)

    if not used:
        return ReferenceResult(
            reference_text=None,
            n_high_available=n_high_available,
            fallback_used=False,
        )

    texts = [answer_lookup[aid][0] for aid in used]
    reference_text = "\n\n---\n\n".join(texts)

    return ReferenceResult(
        reference_text=reference_text,
        answer_ids_used=used,
        n_high_available=n_high_available,
        n_high_used=len(high_used),
        fallback_used=fallback_used,
        reference_char_len=len(reference_text),
        reference_token_len=token_counter(reference_text) if token_counter else 0,
    )
