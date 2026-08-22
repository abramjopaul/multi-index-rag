"""Context source abstraction for Track C.

Each source implements get_contexts(topic) -> list[str].
The generator and RAGAS harness only call this interface — swapping
retrieval strategies (no-RAG, BM25, Dense, Block, oracle) never requires
changing the harness, only the context source.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Literal

logger = logging.getLogger(__name__)


class ContextSource(ABC):
    @abstractmethod
    def get_contexts(self, topic: dict) -> list[str]:
        """Return ordered list of passage strings for a topic.

        Args:
            topic: Dict with at least "topic_id" and "question" keys (from topics.jsonl).

        Returns:
            Ordered list of passage strings (most relevant first).
            Empty list = no context (no-RAG mode).
        """


class NoRagSource(ContextSource):
    """No-context source for C0.1 baseline. Always returns an empty list."""

    def get_contexts(self, topic: dict) -> list[str]:
        return []


class RunFileSource(ContextSource):
    """C1: Reads a TREC run file + a pre-built answer lookup, returns top-k
    passages in raw retrieval rank order, UNMODIFIED.

    No relevance filtering, re-ranking, or dropping of non-relevant passages —
    non-relevant retrieved passages are the real operating point this source
    is used to measure judge behavior against. Building context from qrels
    or filtering by relevance grade belongs to OracleSource, not this one.
    """

    def __init__(
        self,
        run_path: str,
        answer_lookup: dict[str, tuple[str, int]],
        k: int = 5,
    ) -> None:
        self._answer_lookup = answer_lookup
        self._k = k
        self._topic_doc_ids = self._parse_run_file(run_path)

    @staticmethod
    def _parse_run_file(run_path: str) -> dict[str, list[str]]:
        """Parse a TREC run .tsv into {topic_id: [doc_id, ...]}, sorted by the
        file's own rank column ascending (defensive: doesn't change which
        docs are top-k, just doesn't blindly trust file ordering).
        """
        rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
        with open(run_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                topic_id, doc_id, rank = parts[0], parts[2], parts[3]
                rows[topic_id].append((int(rank), doc_id))

        topic_doc_ids: dict[str, list[str]] = {}
        for topic_id, entries in rows.items():
            entries.sort(key=lambda x: x[0])
            topic_doc_ids[topic_id] = [doc_id for _, doc_id in entries]
        logger.info(f"RunFileSource: parsed {len(topic_doc_ids)} topics from {run_path}")
        return topic_doc_ids

    def get_contexts(self, topic: dict) -> list[str]:
        topic_id = topic["topic_id"]
        doc_ids = self._topic_doc_ids.get(topic_id, [])[: self._k]

        contexts: list[str] = []
        missing: list[str] = []
        for doc_id in doc_ids:
            entry = self._answer_lookup.get(doc_id)
            if entry is None:
                missing.append(doc_id)
                continue
            body_text, _score = entry
            contexts.append(body_text)

        if missing:
            logger.warning(
                f"RunFileSource: topic {topic_id!r}: {len(missing)}/{len(doc_ids)} "
                f"doc_id(s) from run file not found in answer_lookup: {missing}"
            )
        return contexts


# ---------------------------------------------------------------------------
# Ceiling experiments: build contexts directly from qrels, not a retrieval run.
# ---------------------------------------------------------------------------


class OracleSource(ContextSource):
    """Ceiling-experiment context source. Builds contexts directly from
    qrels, not from a retrieval run file.

    mode="reference": context = the exact answer_ids_used already selected
      by references.select_references() for this topic (High, or
      Medium-filled per its own fallback rule) -- the instrument ceiling.
      Guarantees the context IS the reference the judge compares answers
      against, so precision isn't testing retrieval at all -- whatever this
      scores is the practical maximum this pipeline can read (1024-token
      compression, ragas judge/claim-decomposition noise, etc., not the
      retriever).

    mode="disjoint": context = the next `k` High(3)-labelled answers ranked
      immediately after the ones already consumed by the reference (same
      score-desc, answer_id-desc sort as select_references) -- the retrieval
      ceiling: genuinely relevant content the judge does not already hold as
      the answer key. Topics without enough surplus High answers return an
      empty list (no Medium fallback -- deliberately narrower than the
      reference's own fallback, so this stays a clean "perfect retrieval"
      condition rather than mixing in lower-quality content).
    """

    def __init__(
        self,
        mode: Literal["reference", "disjoint"],
        qrels: dict[str, dict[str, int]],
        answer_lookup: dict[str, tuple[str, int]],
        max_reference_answers: int = 5,
        k: int = 5,
    ) -> None:
        if mode not in ("reference", "disjoint"):
            raise ValueError(f"OracleSource mode must be 'reference' or 'disjoint', got {mode!r}")
        self._mode = mode
        self._qrels = qrels
        self._answer_lookup = answer_lookup
        self._max_reference_answers = max_reference_answers
        self._k = k

    def get_contexts(self, topic: dict) -> list[str]:
        from multirag.eval.references import _select_candidates, select_references

        topic_id = topic["topic_id"]
        if self._mode == "reference":
            ref = select_references(
                topic_id, self._qrels, self._answer_lookup, self._max_reference_answers
            )
            return [self._answer_lookup[aid][0] for aid in ref.answer_ids_used]

        # mode == "disjoint"
        high = _select_candidates(self._qrels.get(topic_id, {}), self._answer_lookup, label=3)
        surplus = high[self._max_reference_answers : self._max_reference_answers + self._k]
        return [self._answer_lookup[aid][0] for aid in surplus]


def build_context_source(
    config,
    answer_lookup: dict[str, tuple[str, int]] | None = None,
    qrels: dict[str, dict[str, int]] | None = None,
    max_reference_answers: int = 5,
) -> ContextSource:
    """Factory: build the right ContextSource from a ContextSourceConfig."""
    from multirag.config.generation_config import ContextSourceConfig

    if isinstance(config, dict):
        config = ContextSourceConfig(**config)

    if config.type == "no_rag":
        return NoRagSource()
    if config.type == "run_file":
        if not config.run_path:
            raise ValueError("context_source.type == 'run_file' requires run_path to be set.")
        if answer_lookup is None:
            raise ValueError(
                "context_source.type == 'run_file' requires answer_lookup to be passed to "
                "build_context_source() — build one via sample_builder.build_answer_lookup() "
                "and pass it through."
            )
        return RunFileSource(run_path=config.run_path, answer_lookup=answer_lookup, k=config.k)
    if config.type == "oracle":
        if qrels is None or answer_lookup is None:
            raise ValueError(
                "context_source.type == 'oracle' requires both qrels and answer_lookup to be "
                "passed to build_context_source()."
            )
        if config.oracle_mode not in ("reference", "disjoint"):
            raise ValueError(
                f"context_source.oracle_mode must be 'reference' or 'disjoint', "
                f"got {config.oracle_mode!r}."
            )
        return OracleSource(
            mode=config.oracle_mode,
            qrels=qrels,
            answer_lookup=answer_lookup,
            max_reference_answers=max_reference_answers,
            k=config.k,
        )
    raise ValueError(
        f"Unknown context_source.type: {config.type!r}. Supported: 'no_rag', 'run_file', 'oracle'."
    )
