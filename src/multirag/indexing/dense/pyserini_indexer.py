"""Dense (embedding) indexer using SentenceTransformers and FAISS."""

import json as _json
import os
import queue
import random
import shutil
import threading
from pathlib import Path
from typing import Any, TextIO

import faiss
import numpy as np
import orjson as json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from multirag.indexing.base import BaseIndexer

_CHECKPOINT_EVERY = 100_000


def _auto_device() -> str:
    """Detect the best available device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


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

    Builds a FAISS IndexFlatIP index:
    - {index_path}/index:       Binary FAISS index file
    - {index_path}/docid:       Text file with document IDs (one per line)
    - {index_path}/checkpoint.json: Resume state (indexed_count)

    Default model: multi-qa-mpnet-base-dot-v1 — QA-optimized, 768-dim, dot-product.
    Default device: "auto" — picks cuda > mps > cpu automatically.
    """

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path,
        embedding_model: str = "multi-qa-mpnet-base-dot-v1",
        batch_size: int = 256,
        device: str = "auto",
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
        return (self.index_path / "index").exists() and (self.index_path / "docid").exists()

    def _resolved_device(self) -> str:
        return _auto_device() if self.device == "auto" else self.device

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model, device=self._resolved_device())
        return self._model

    def _load_index(self) -> tuple[faiss.Index, list[str]]:
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(str(self.index_path / "index"))
            with open(self.index_path / "docid") as f:
                self._docids = [line.strip() for line in f]
        return self._faiss_index, self._docids  # type: ignore

    def _checkpoint_path(self) -> Path:
        return self.index_path / "checkpoint.json"

    def _load_checkpoint(self) -> int:
        """Return number of docs already indexed (0 if no checkpoint)."""
        cp = self._checkpoint_path()
        if cp.exists():
            with open(cp) as f:
                return _json.load(f).get("indexed_count", 0)
        return 0

    def _save_checkpoint(self, indexed_count: int) -> None:
        with open(self._checkpoint_path(), "w") as f:
            _json.dump({"indexed_count": indexed_count}, f)

    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Convert raw doc to indexable format: {id, contents}."""
        if "contents" in raw_doc:
            return {"id": str(raw_doc["id"]), "contents": raw_doc["contents"]}
        return {"id": str(raw_doc["id"]), "contents": raw_doc.get("body_text", "")}

    def _producer(
        self,
        q: queue.Queue,
        skip: int,
        limit: int | None,
    ) -> None:
        """Background thread: read corpus → batch docs → put into queue."""
        batch: list[dict[str, Any]] = []
        count = 0
        try:
            with open(self.corpus_path) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    if i < skip:
                        continue
                    batch.append(self.prepare_document(json.loads(line)))
                    count += 1
                    if len(batch) >= self.batch_size:
                        q.put(batch)
                        batch = []
                    if limit is not None and count >= limit:
                        break
            if batch:
                q.put(batch)
        finally:
            q.put(None)  # sentinel

    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build FAISS index from corpus using dense embeddings.

        Supports resuming: if interrupted, rerun with force=False to continue
        from the last checkpoint. Use force=True to rebuild from scratch.

        Args:
            force: Rebuild even if index exists.
            limit: Cap on total docs to index (useful for testing).
        """
        set_seed(self.seed)

        if self.index_exists and not force:
            resume_from = self._load_checkpoint()
            if resume_from == 0:
                # Index exists and complete — just load it
                self._load_index()
                return
            # Partial index — resume
            self._resume(resume_from=resume_from, limit=limit)
            return

        if self.index_exists and force:
            shutil.rmtree(self.index_path)

        self.index_path.mkdir(parents=True, exist_ok=True)
        self._build(skip=0, limit=limit)

    def _build(self, skip: int, limit: int | None) -> None:
        """Encode corpus and write FAISS index. Skips first `skip` lines (for resume)."""
        model = self._get_model()
        device = self._resolved_device()

        docid_path = self.index_path / "docid"
        docid_mode = "a" if skip > 0 else "w"

        # Load existing partial FAISS index if resuming
        if skip > 0 and (self.index_path / "index").exists():
            faiss_index: faiss.Index | None = faiss.read_index(str(self.index_path / "index"))
        else:
            faiss_index = None

        q: queue.Queue = queue.Queue(maxsize=4)
        producer = threading.Thread(
            target=self._producer, args=(q, skip, limit), daemon=True
        )
        producer.start()

        docid_file: TextIO | None = None
        indexed_count = skip

        try:
            docid_file = open(docid_path, docid_mode)

            with tqdm(
                total=limit if limit is not None else 1_445_495,
                initial=skip if limit else 0,
                desc=f"Encoding ({device})",
                unit="doc",
            ) as pbar:
                while True:
                    batch = q.get()
                    if batch is None:
                        break

                    embeddings = model.encode(
                        [d["contents"] for d in batch],
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ).astype(np.float32)

                    if faiss_index is None:
                        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])

                    faiss_index.add(np.ascontiguousarray(embeddings))  # type: ignore

                    for doc in batch:
                        docid_file.write(f"{doc['id']}\n")
                    docid_file.flush()

                    indexed_count += len(batch)
                    pbar.update(len(batch))

                    # Periodic checkpoint
                    if indexed_count % _CHECKPOINT_EVERY < self.batch_size:
                        faiss.write_index(faiss_index, str(self.index_path / "index"))
                        self._save_checkpoint(indexed_count)

        finally:
            if docid_file is not None and not docid_file.closed:
                docid_file.close()

        # Final save
        if faiss_index is not None:
            faiss.write_index(faiss_index, str(self.index_path / "index"))
            self._save_checkpoint(indexed_count)

    def _resume(self, resume_from: int, limit: int | None) -> None:
        """Resume encoding from `resume_from` docs into an existing partial index."""
        self._build(skip=resume_from, limit=limit)

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search and return top-k hits."""
        model = self._get_model()
        index, docids = self._load_index()
        q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
        scores, indices = index.search(q_emb, k)  # type: ignore
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
        all_scores, all_indices = index.search(q_embs, k)  # type: ignore
        return {
            qid: [
                {"doc_id": docids[idx], "score": float(all_scores[i][j])}
                for j, idx in enumerate(all_indices[i])
                if idx >= 0
            ]
            for i, qid in enumerate(qids)
        }
