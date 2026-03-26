"""Pyserini-based dense (embedding) indexer using FAISS and Pyserini encoders."""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from bs4 import BeautifulSoup
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AnceDocumentEncoder
from tqdm import tqdm

from multirag.indexing.base import BaseIndexer


class PyseriniDenseIndexer(BaseIndexer):
    """Dense embedding-based indexer using Pyserini AnceDocumentEncoder and FAISS.
    
    Builds a FAISS index with files compatible with Pyserini's FaissSearcher:
    - {index_path}/index: Binary FAISS index file
    - {index_path}/docid: Text file with document IDs (one per line)
    """

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path,
        embedding_model: str = "castorini/ance-msmarco-passage",
        batch_size: int = 64,
        device: str = "mps",
    ):
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = device
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
            self._searcher = FaissSearcher(
                str(self.index_path),
                self.embedding_model
            )
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
        
        Creates two files in index_path:
        - index: Binary FAISS index
        - docid: Text file with document IDs (one per line)

        Args:
            force: If True, rebuild index even if it exists.
            limit: Maximum number of documents to index. If None, index all documents.
        """
        if self.index_exists and not force:
            self._get_searcher()
            return
        
        if self.index_exists and force:
            shutil.rmtree(self.index_path)

        self.index_path.mkdir(parents=True, exist_ok=True)

        embedder = self._get_embedder()

        # Count total documents
        total_docs = sum(1 for line in open(self.corpus_path) if line.strip())
        total_docs = min(total_docs, limit) if limit else total_docs

        # Temporary disk file for embeddings (avoids O(n²) FAISS reallocation)
        embeddings_tmp_path = self.index_path / ".embeddings.tmp"
        docid_path = str(self.index_path / "docid")
        
        embeddings_tmp_file = None
        docid_file = None
        embedding_dim = None
        total_embeddings = 0

        try:
            with open(self.corpus_path) as f:
                batch_texts = []
                batch_docids = []
                indexed_count = 0

                with tqdm(total=total_docs, desc="Encoding documents", unit="doc") as pbar:
                    for line in f:
                        if not line.strip():
                            continue

                        raw = json.loads(line)
                        doc = self.prepare_document(raw)
                        docid = doc["id"]
                        text = doc["contents"]

                        batch_texts.append(text)
                        batch_docids.append(docid)
                        indexed_count += 1

                        if len(batch_texts) >= self.batch_size:
                            # Encode batch
                            embeddings = embedder.encode(
                                batch_texts, max_length=256
                            )
                            
                            # Initialize temp files on first batch
                            if embeddings_tmp_file is None:
                                embedding_dim = embeddings.shape[1]
                                embeddings_tmp_file = open(embeddings_tmp_path, "wb")
                                docid_file = open(docid_path, "w")
                            
                            # Stream embeddings to disk (avoids O(n²) FAISS reallocation)
                            np.save(embeddings_tmp_file, embeddings.astype(np.float32))
                            
                            # Write docids
                            for docid in batch_docids:
                                docid_file.write(f"{docid}\n")
                            
                            total_embeddings += len(embeddings)
                            pbar.update(len(batch_texts))
                            batch_texts = []
                            batch_docids = []

                        # Stop if limit is reached
                        if limit and indexed_count >= limit:
                            break

                    # Process remaining documents
                    if batch_texts:
                        embeddings = embedder.encode(
                            batch_texts, max_length=256
                        )
                        
                        if embeddings_tmp_file is None:
                            embedding_dim = embeddings.shape[1]
                            embeddings_tmp_file = open(embeddings_tmp_path, "wb")
                            docid_file = open(docid_path, "w")
                        
                        np.save(embeddings_tmp_file, embeddings.astype(np.float32))
                        for docid in batch_docids:
                            docid_file.write(f"{docid}\n")
                        
                        total_embeddings += len(embeddings)
                        pbar.update(len(batch_texts))
            
            # Close temp files
            if embeddings_tmp_file is not None:
                embeddings_tmp_file.close()
            if docid_file is not None:
                docid_file.close()
            
            # Build FAISS index from disk (O(n) single pass)
            if embedding_dim is not None:
                self._build_faiss_index_from_disk(
                    embeddings_tmp_path, total_embeddings, embedding_dim
                )
        
        finally:
            if embeddings_tmp_file is not None and not embeddings_tmp_file.closed:
                embeddings_tmp_file.close()
            if docid_file is not None and not docid_file.closed:
                docid_file.close()

    def _build_faiss_index_from_disk(
        self, embeddings_tmp_path: Path, total_embeddings: int, embedding_dim: int
    ) -> None:
        """Build FAISS index from disk-saved embeddings in a single O(n) pass.
        
        Args:
            embeddings_tmp_path: Path to temporary file with numpy arrays
            total_embeddings: Total number of embeddings to expect
            embedding_dim: Dimensionality of embeddings
        """
        import faiss
        
        # Create index
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        
        # Read all embeddings from disk and add to index (single allocation)
        embeddings_tmp_path = Path(embeddings_tmp_path)
        with open(embeddings_tmp_path, "rb") as f:
            with tqdm(
                total=total_embeddings,
                desc="Building FAISS index",
                unit="doc",
            ) as pbar:
                while True:
                    try:
                        embeddings_batch = np.load(f, allow_pickle=False)
                        faiss_index.add(embeddings_batch.astype(np.float32))
                        pbar.update(len(embeddings_batch))
                    except (ValueError, EOFError):
                        # End of file
                        break
        
        # Save final index
        index_path = str(self.index_path / "index")
        faiss.write_index(faiss_index, index_path)
        
        # Clean up temp file
        embeddings_tmp_path.unlink()

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search and return top-k hits.
        
        Returns results in format: [{"id": docid, "score": score}, ...]
        """
        searcher = self._get_searcher()
        hits = searcher.search(query, k=k)
        return [{"id": hit.docid, "score": hit.score} for hit in hits]

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
        for qid, hits in hits_dict.items():
            results[qid] = [{"doc_id": hit.docid, "score": hit.score} for hit in hits]
        return results