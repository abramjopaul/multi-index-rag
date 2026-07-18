"""Versioned, byte-frozen prompt templates for Track C.

Template versions are frozen: once defined, the template string NEVER changes.
The SHA256 of the template strings (not the rendered output) is stored in every
result file, providing byte-level reproducibility across all Track C phases.

Different phases use different template versions, selected via each config's
prompt_template_version field:
  - v1: no-RAG (C0.1/C0.2) — open-book, answers from the model's own knowledge
        when contexts is empty.
  - v2: retrieval (C1+) — answers ONLY from retrieved context, with an explicit
        refusal instruction when context is insufficient. Always expects
        non-empty contexts; not used for no-RAG runs.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod


class PromptTemplate(ABC):
    version: str

    @abstractmethod
    def render(self, question: str, contexts: list[str]) -> list[dict]:
        """Return a list of chat message dicts (role/content pairs).

        Args:
            question: Full question text (title + body, may contain LaTeX).
            contexts: Retrieved passages. Empty list = no-RAG mode.

        Returns:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """

    @property
    def sha256(self) -> str:
        """SHA256 of the raw template strings (frozen; independent of rendered content)."""
        raw = self._system_template + "\n---\n" + self._user_template
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    @abstractmethod
    def _system_template(self) -> str: ...

    @property
    @abstractmethod
    def _user_template(self) -> str: ...


class PromptTemplateV1(PromptTemplate):
    """Version 1 — frozen 2026-07-13. Do not modify; create v2 instead."""

    version = "v1"

    @property
    def _system_template(self) -> str:
        return "You are a helpful mathematics assistant."

    @property
    def _user_template(self) -> str:
        return (
            "Question:\n"
            "{question}\n"
            "\n"
            "{context_block}"
            "Answer the question concisely and accurately. "
            "Use LaTeX for math expressions."
        )

    def render(self, question: str, contexts: list[str]) -> list[dict]:
        if contexts:
            context_block = "Context:\n" + "\n\n".join(contexts) + "\n\n"
        else:
            context_block = ""

        user_content = self._user_template.format(
            question=question,
            context_block=context_block,
        )
        return [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": user_content},
        ]


class PromptTemplateV2(PromptTemplate):
    """Version 2 — frozen 2026-07-14. Do not modify; create v3 instead.

    RAG-only: instructs the model to answer strictly from retrieved context
    and refuse when context is insufficient. Not used for no-RAG runs.
    """

    version = "v2"

    @property
    def _system_template(self) -> str:
        return (
            "You are a mathematics question-answering assistant. "
            "Answer the question using ONLY the information in the retrieved "
            "context below. Base every mathematical statement, formula, and "
            "step strictly on the retrieved context. Do not use outside "
            "knowledge. If the context does not contain enough information to "
            'answer, say "The provided context does not contain enough '
            'information to answer this question." Do not guess.'
        )

    @property
    def _user_template(self) -> str:
        return (
            "Retrieved context:\n"
            "{context}\n"
            "\n"
            "Question:\n"
            "{question}\n"
            "\n"
            "Answer:"
        )

    def render(self, question: str, contexts: list[str]) -> list[dict]:
        user_content = self._user_template.format(
            question=question,
            context="\n\n".join(contexts),
        )
        return [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": user_content},
        ]


_REGISTRY: dict[str, PromptTemplate] = {
    "v1": PromptTemplateV1(),
    "v2": PromptTemplateV2(),
}


def get_template(version: str) -> PromptTemplate:
    if version not in _REGISTRY:
        raise ValueError(
            f"Unknown prompt template version: {version!r}. "
            f"Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[version]
