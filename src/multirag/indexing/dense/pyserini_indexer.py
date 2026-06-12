"""Dense (embedding) indexer using SentenceTransformers and FAISS."""

import os
import random
import shutil
from pathlib import Path
from typing import Any, TextIO

import faiss
import numpy as np
import orjson as json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from multirag.indexing.base import BaseIndexer


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducible results across numpy, torch, and Python stdlib."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class PyseriniDenseIndexer(BaseIndexer):
    """Dense embedding indexer using SentenceTransformers and FAISS (dot-product).

    Builds a FAISS IndexFlatIP index compatible with dot-product scoring:
    - {index_path}/index: Binary FAISS index file
    - {index_path}/docid: Text file with document IDs (one per line)

    Default model: multi-qa-mpnet-base-dot-v1 — QA-optimized, trained on
    MS MARCO, Natural Questions, TriviaQA, SQuAD. Designed for dot-product retrieval.
    """

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path,
        embedding_model: str = "multi-qa-mpnet-base-dot-v1",
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
        self._model: SentenceTransformer | None = None
        self._faiss_index: faiss.Index | None = None
        self._docids: list[str] = []

    @property
    def index_exists(self) -> bool:
        """True if index directory exists and has FAISS index file."""
        return (self.index_path / "index").exists() and (self.index_path / "docid").exists()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model, device=self.device)
        return self._model

    def _load_index(self) -> tuple[faiss.Index, list[str]]:
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(str(self.index_path / "index"))
            with open(self.index_path / "docid") as f:
                self._docids = [line.strip() for line in f]
        return self._faiss_index, self._docids #type: ignore

    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Convert raw doc to indexable format: {id, contents}."""
        if "contents" in raw_doc:
            return {"id": str(raw_doc["id"]), "contents": raw_doc["contents"]}
        return {"id": str(raw_doc["id"]), "contents": raw_doc.get("body_text", "")}

    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build FAISS index from corpus using dense embeddings.

        Args:
            force: If True, rebuild index even if it exists.
            limit: Maximum number of documents to index. If None, index all documents.
        """
        set_seed(self.seed)

        if self.index_exists and not force:
            self._load_index()
            return

        if self.index_exists and force:
            shutil.rmtree(self.index_path)

        self.index_path.mkdir(parents=True, exist_ok=True)

        model = self._get_model()
        docid_path = str(self.index_path / "docid")

        docid_file: TextIO | None = None
        faiss_index: faiss.Index | None = None

        try:
            with open(self.corpus_path) as f:
                batch_texts: list[str] = []
                batch_docids: list[str] = []
                indexed_count = 0

                with tqdm(total=limit, desc="Encoding documents", unit="doc") as pbar:
                    for line in f:
                        if not line.strip():
                            continue

                        doc = self.prepare_document(json.loads(line))
                        batch_texts.append(doc["contents"])
                        batch_docids.append(doc["id"])
                        indexed_count += 1

                        if len(batch_texts) >= self.batch_size:
                            embeddings = model.encode(
                                batch_texts,
                                convert_to_numpy=True,
                                show_progress_bar=False,
                            ).astype(np.float32)

                            if faiss_index is None:
                                faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                                docid_file = open(docid_path, "w")

                            faiss_index.add(np.ascontiguousarray(embeddings)) #type: ignore
                            for docid in batch_docids:
                                docid_file.write(f"{docid}\n")  # type: ignore
                            docid_file.flush()  # type: ignore

                            pbar.update(len(batch_texts))
                            batch_texts = []
                            batch_docids = []

                        if limit and indexed_count >= limit:
                            break

                    # Flush remaining batch
                    if batch_texts:
                        embeddings = model.encode(
                            batch_texts,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        ).astype(np.float32)

                        if faiss_index is None:
                            faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                            docid_file = open(docid_path, "w")

                        faiss_index.add(np.ascontiguousarray(embeddings))   #type: ignore
                        for docid in batch_docids:
                            docid_file.write(f"{docid}\n")  # type: ignore
                        docid_file.flush()  # type: ignore

                        pbar.update(len(batch_texts))

            if docid_file is not None:
                docid_file.close()

            if faiss_index is not None:
                faiss.write_index(faiss_index, str(self.index_path / "index"))

        finally:
            if docid_file is not None and not docid_file.closed:
                docid_file.close()

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search and return top-k hits."""
        model = self._get_model()
        index, docids = self._load_index()
        q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
        scores, indices = index.search(q_emb, k) # type: ignore
        return [
            {"doc_id": docids[idx], "score": float(scores[0][i])}
            for i, idx in enumerate(indices[0])
            if idx >= 0
        ]

    def batch_search(
        self,
        queries: list[tuple[str, str]],
        k: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search multiple (qid, query) pairs. Returns {qid: [hit, ...]}."""
        model = self._get_model()
        index, docids = self._load_index()
        qids = [qid for qid, _ in queries]
        texts = [text for _, text in queries]
        q_embs = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        ).astype(np.float32)
        all_scores, all_indices = index.search(q_embs, k) #type: ignore
        return {
            qid: [
                {"doc_id": docids[idx], "score": float(all_scores[i][j])}
                for j, idx in enumerate(all_indices[i])
                if idx >= 0
            ]
            for i, qid in enumerate(qids)
        }
