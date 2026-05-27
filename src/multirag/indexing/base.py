"""Base interface for retrieval indexers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseIndexer(ABC):
    """Abstract base class for document indexers."""

    @abstractmethod
    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Convert raw corpus document to indexer-specific format."""
        ...

    @abstractmethod
    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build index from corpus. Corpus path configured in __init__.

        Args:
            force: If True, force rebuild index even if it exists.
            limit: Maximum number of documents to index. If None, index all documents.
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search index and return top-k results."""
        ...

    @abstractmethod
    def batch_search(
        self,
        queries: list[tuple[str, str]],
        k: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search multiple (qid, query) pairs. Returns {qid: [hit, ...]}."""
        ...

    def get_representations(self) -> list[str]:
        """Return list of representations supported by this indexer.

        For multi-representation indices (e.g., formula indices with SLT/OPT/SLT_type),
        return the representation names. Single-representation indices return empty list.

        Returns:
            List of representation names (e.g., ['slt', 'opt', 'slt_type']).
                Empty list for single-representation indexers.
        """
        return []

    def search_by_representation(
        self, query: str, k: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """Search index and return top-k results grouped by representation.

        For multi-representation indices, return results per representation.
        For single-representation indices, return results under empty string key.

        Args:
            query: Query string
            k: Number of results to return per representation

        Returns:
            dict mapping representation name → list of hits.
            For single-representation: {'' or 'default': [hits]}.
            For multi-representation: {'slt': [...], 'opt': [...], 'slt_type': [...]}.
        """
        # Default implementation: call search() and return under empty key
        return {"": self.search(query, k)}
