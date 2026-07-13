"""Context source abstraction for Track C.

Each source implements get_contexts(topic) -> list[str].
The generator and RAGAS harness only call this interface — swapping
retrieval strategies (no-RAG, BM25, Dense, Block, oracle) never requires
changing the harness, only the context source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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


# ---------------------------------------------------------------------------
# Stubs for later phases — same interface, different retrieval backends.
# Implement these when the corresponding phase is built; the harness never changes.
# ---------------------------------------------------------------------------

# class RunFileSource(ContextSource):
#     """C1: Reads a TREC run file + answers.jsonl and returns top-k passages."""
#     def __init__(self, run_path: str, answers_path: str, k: int = 5): ...
#     def get_contexts(self, topic: dict) -> list[str]: ...

# class OracleSource(ContextSource):
#     """C-Oracle: Builds contexts from qrels at a fixed relevance level."""
#     def __init__(self, qrels_path: str, answers_path: str,
#                  relevance_level: int, k: int = 5): ...
#     def get_contexts(self, topic: dict) -> list[str]: ...


def build_context_source(config) -> ContextSource:
    """Factory: build the right ContextSource from a ContextSourceConfig."""
    from multirag.config.generation_config import ContextSourceConfig

    if isinstance(config, dict):
        config = ContextSourceConfig(**config)

    if config.type == "no_rag":
        return NoRagSource()
    raise ValueError(
        f"Unknown context_source.type: {config.type!r}. "
        "Supported: 'no_rag'. (run_file and oracle are not yet implemented.)"
    )
