"""Pyserini-based sparse (BM25) indexer."""

import json
import shutil
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

from multirag.indexing.base import BaseIndexer


class PyseriniSparseIndexer(BaseIndexer):
    """Sparse BM25 indexer using Pyserini. Expects prepared docs: {id, contents}."""

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path,
        k1: float = 0.9,
        b: float = 0.4,
    ):
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.k1 = k1
        self.b = b
        self._indexer: LuceneIndexer | None = None
        self._searcher: LuceneSearcher | None = None

    @property
    def index_exists(self) -> bool:
        """True if index directory exists and has content."""
        return self.index_path.is_dir() and any(self.index_path.iterdir())

    def _get_indexer(self, append: bool = False) -> LuceneIndexer:
        """Get LuceneIndexer for building. Creates index if not present."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._indexer = LuceneIndexer(str(self.index_path), append=append)
        return self._indexer

    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Convert raw doc to Pyserini format: {id, contents}."""
        if "contents" in raw_doc:
            return {"id": str(raw_doc["id"]), "contents": raw_doc["contents"]}
        # From answers.jsonl: body_text is HTML (p, a, span.math-container, ul, li, etc.)
        body = raw_doc.get("body_text", "")
        soup = BeautifulSoup(body, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return {"id": str(raw_doc["id"]), "contents": text}

    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build Lucene index from JSONL corpus. Skips if index exists unless force=True.

        Args:
            force: If True, rebuild index even if it exists.
            limit: Maximum number of documents to index. If None, index all documents.
        """
        if self.index_exists and not force:
            return
        if self.index_exists and force:
            shutil.rmtree(self.index_path)

        indexer = self._get_indexer(append=False)
        batch_size = 1000

        # Count lines upfront so tqdm can show total (or use limit if set)
        total_docs = sum(1 for line in open(self.corpus_path) if line.strip())
        total_docs = min(total_docs, limit) if limit else total_docs

        with open(self.corpus_path) as f:
            batch = []
            indexed_count = 0
            with tqdm(total=total_docs, desc="Indexing", unit="doc") as pbar:
                for line in f:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    doc = self.prepare_document(raw)
                    batch.append(doc)
                    indexed_count += 1

                    if len(batch) >= batch_size:
                        indexer.add_batch_dict(batch)
                        pbar.update(len(batch))
                        batch = []

                    # Stop if limit is reached
                    if limit and indexed_count >= limit:
                        break

                if batch:
                    indexer.add_batch_dict(batch)
                    pbar.update(len(batch))

        indexer.close()
        self._indexer = None

    def _get_searcher(self) -> LuceneSearcher:
        if self._searcher is None:
            self._searcher = LuceneSearcher(str(self.index_path))
            self._searcher.set_bm25(k1=self.k1, b=self.b)
        return self._searcher

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search and return top-k hits."""
        searcher = self._get_searcher()
        hits = searcher.search(query, k=k)
        return [{"id": hit.docid, "score": hit.score} for hit in hits]

    def batch_search(
        self,
        queries: list[tuple[str, str]],
        k: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search multiple (qid, query) pairs. Returns {qid: [hit, ...]}.

        Uses pyserini's native batch_search for efficiency.
        """
        searcher = self._get_searcher()
        # Extract qids and query strings as parallel lists
        qids = [qid for qid, _ in queries]
        query_texts = [query for _, query in queries]
        # Use pyserini's batch_search which is more efficient
        hits_dict = searcher.batch_search(qids=qids, queries=query_texts, k=k)
        # Convert result format to match our interface
        results = {}
        for qid, hits in hits_dict.items():
            results[qid] = [{"doc_id": hit.docid, "score": hit.score} for hit in hits]
        return results
