"""Versioned, byte-frozen prompt templates for Track C.

Template versions are frozen: once defined, the template string NEVER changes.
The SHA256 of the template strings (not the rendered output) is stored in every
result file, providing byte-level reproducibility across all Track C phases.

All phases use the same template version — only the context_block differs:
  - no-RAG (C0.1):  context_block = ""
  - retrieval (C1+): context_block = "Context:\\n" + "\\n\\n".join(passages) + "\\n\\n"
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


_REGISTRY: dict[str, PromptTemplate] = {
    "v1": PromptTemplateV1(),
}


def get_template(version: str) -> PromptTemplate:
    if version not in _REGISTRY:
        raise ValueError(
            f"Unknown prompt template version: {version!r}. "
            f"Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[version]
