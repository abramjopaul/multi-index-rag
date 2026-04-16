"""Pyserini-based dense (embedding) indexer using FAISS and Pyserini encoders."""

import os
import random
import shutil
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import orjson as json
from bs4 import BeautifulSoup
from pyserini.encode import AnceDocumentEncoder
from pyserini.search.faiss import FaissSearcher
from tqdm import tqdm

from multirag.indexing.base import BaseIndexer


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducible results across numpy, torch, and Python stdlib.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set torch seeds if available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class PyseriniDenseIndexer(BaseIndexer):
    """Dense embedding-based indexer using Pyserini AnceDocumentEncoder and FAISS.

    Optimized for speed with:
    - Reduced max_length (192 vs 256) for faster encoding
    - Smaller batch_size (48 vs 64) for better CPU cache utilization
    - orjson for faster JSON parsing
    - Single-pass FAISS indexing (no temp embedding files)

    Builds a FAISS index with files compatible with Pyserini's FaissSearcher:
    - {index_path}/index: Binary FAISS index file
    - {index_path}/docid: Text file with document IDs (one per line)
    """

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path,
        embedding_model: str = "castorini/ance-msmarco-passage",
        batch_size: int = 48,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self._embedder: AnceDocumentEncoder | None = None
        self._searcher: FaissSearcher | None = None

    @property
    def index_exists(self) -> bool:
        """True if index directory exists and has FAISS index file."""
        index_file = self.index_path / "index"
        docid_file = self.index_path / "docid"
        return index_file.exists() and docid_file.exists()

    def _get_embedder(self) -> AnceDocumentEncoder:
        """Lazy load and cache the document encoder model."""
        if self._embedder is None:
            self._embedder = AnceDocumentEncoder(
                model_name=self.embedding_model,
                device=self.device,
            )
        return self._embedder

    def _get_searcher(self) -> FaissSearcher:
        """Lazy load and cache the FaissSearcher."""
        if self._searcher is None:
            self._searcher = FaissSearcher(str(self.index_path), self.embedding_model)
        return self._searcher

    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Convert raw doc to dense format: {id, contents}.

        Parse HTML to plain text, similar to sparse indexing.
        """
        if "contents" in raw_doc:
            return {"id": str(raw_doc["id"]), "contents": raw_doc["contents"]}

        # From answers.jsonl: body_text is HTML
        body = raw_doc.get("body_text", "")
        soup = BeautifulSoup(body, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return {"id": str(raw_doc["id"]), "contents": text}

    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build FAISS index from corpus using dense embeddings.

        Creates files in index_path:
        - index: Binary FAISS index
        - docid: Text file with document IDs (one per line)

        Args:
            force: If True, rebuild index even if it exists.
            limit: Maximum number of documents to index. If None, index all documents.
        """
        # Set seeds for reproducibility
        set_seed(self.seed)

        if self.index_exists and not force:
            self._get_searcher()
            return

        if self.index_exists and force:
            shutil.rmtree(self.index_path)

        self.index_path.mkdir(parents=True, exist_ok=True)

        import faiss

        embedder = self._get_embedder()

        # Count total documents
        total_docs = sum(1 for line in open(self.corpus_path) if line.strip())
        total_docs = min(total_docs, limit) if limit else total_docs

        docid_path = str(self.index_path / "docid")

        docid_file: TextIO | None = None
        faiss_index = None
        embedding_dim = None

        try:
            with open(self.corpus_path) as f:
                batch_texts = []
                batch_docids = []
                indexed_count = 0

                with tqdm(
                    total=total_docs, desc="Encoding documents", unit="doc"
                ) as pbar:
                    for line in f:
                        if not line.strip():
                            continue

                        # Parse and prepare document
                        if hasattr(json, "loads"):
                            raw = json.loads(line)
                        else:
                            raw = json.loads(line.encode())
                        doc = self.prepare_document(raw)

                        batch_texts.append(doc["contents"])
                        batch_docids.append(doc["id"])
                        indexed_count += 1

                        if len(batch_texts) >= self.batch_size:
                            # Encode batch (max_length reduced from 256 to 192)
                            embeddings = embedder.encode(batch_texts, max_length=192)

                            # Initialize FAISS index on first batch
                            if faiss_index is None:
                                embedding_dim = embeddings.shape[1]
                                faiss_index = faiss.IndexFlatIP(embedding_dim)
                                docid_file = open(docid_path, "w")

                            # Add embeddings directly to FAISS (single-pass, no temp file)
                            faiss_index.add(np.ascontiguousarray(embeddings.astype(np.float32)))  # type: ignore

                            # Write document IDs
                            for docid in batch_docids:
                                docid_file.write(f"{docid}\n")  # type: ignore
                            docid_file.flush()  # type: ignore

                            pbar.update(len(batch_texts))
                            batch_texts = []
                            batch_docids = []

                        # Stop if limit is reached
                        if limit and indexed_count >= limit:
                            break

                    # Process remaining documents
                    if batch_texts:
                        embeddings = embedder.encode(batch_texts, max_length=192)

                        if faiss_index is None:
                            embedding_dim = embeddings.shape[1]
                            faiss_index = faiss.IndexFlatIP(embedding_dim)
                            docid_file = open(docid_path, "w")

                        faiss_index.add(np.ascontiguousarray(embeddings.astype(np.float32)))  # type: ignore
                        for docid in batch_docids:
                            docid_file.write(f"{docid}\n")  # type: ignore
                        docid_file.flush()  # type: ignore

                        pbar.update(len(batch_texts))

            # Close docid file
            if docid_file is not None:
                docid_file.close()

            # Save FAISS index
            if faiss_index is not None:
                index_path = str(self.index_path / "index")
                faiss.write_index(faiss_index, index_path)

        finally:
            if docid_file is not None and not docid_file.closed:
                docid_file.close()

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search and return top-k hits.

        Returns results in format: [{"id": docid, "score": score}, ...]
        """
        searcher = self._get_searcher()
        hits = searcher.search(query, k=k)
        return [{"id": hit.docid, "score": hit.score} for hit in hits]  # type: ignore

    def batch_search(
        self,
        queries: list[tuple[str, str]],
        k: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search multiple (qid, query) pairs. Returns {qid: [hit, ...]}.

        Returns results in format: {qid: [{"doc_id": docid, "score": score}, ...]}
        """
        searcher = self._get_searcher()

        # Extract qids and query strings
        qids = [qid for qid, _ in queries]
        query_texts = [query for _, query in queries]

        # Use FaissSearcher's batch_search
        hits_dict = searcher.batch_search(query_texts, qids, k=k)

        # Convert result format to match sparse indexer
        results = {}
        for qid, hits in hits_dict.items():  # type: ignore
            results[qid] = [{"doc_id": hit.docid, "score": hit.score} for hit in hits]
        return results
