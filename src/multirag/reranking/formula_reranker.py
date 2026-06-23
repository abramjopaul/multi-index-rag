"""Stage-2 formula-aware reranker using MaxSim over FastText tuple embeddings."""

import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from multirag.evaluation.metrics import _is_clearly_trivial

logger = logging.getLogger(__name__)


class FormulaMaxSimReranker:
    """Reranks a stage-1 text retrieval pool by blending text scores with
    formula-structural relevance.

    Formula relevance = MaxSim: for each non-trivial topic formula, take its best
    cosine similarity to any formula in the candidate post, then aggregate across
    topic formulas. Both score channels are min-max normalized per topic before
    blending: final = (1-alpha)*text_norm + alpha*formula_norm.

    Two embedding paths:
    - Topic formulas: subprocess-based LaTeX→MathML→tuples→FastText (query-time).
    - Candidate formulas: pre-computed MathML lookup from TSV files → tuples→FastText
      (no subprocess). Falls back to subprocess if a formula_id is missing from the
      lookup dict.

    Candidate scoring is parallelized across all available CPU cores using threads.
    The FastText model is shared across threads (read-only after load).
    """

    def __init__(
        self,
        embedding_dir: Path,
        representation: str = "slt",
        alpha: float = 0.5,
        aggregation: str = "max",
        n_candidates: int = 100,
    ):
        """
        Args:
            embedding_dir: Directory containing trained FastText models
                           (same layout as used by FormulaFAISSIndexerIVFScalarQuantizer).
            representation: Formula embedding type — "slt", "opt", or "slt_type".
            alpha: Blend weight in [0, 1]. 0 = pure text, 1 = pure formula.
            aggregation: How to collapse per-topic-formula MaxSim scores: "max", "mean", or "sum".
            n_candidates: Number of stage-1 candidates to rerank per topic.
        """
        if aggregation not in {"max", "mean", "sum"}:
            raise ValueError(f"aggregation must be max/mean/sum, got '{aggregation}'")
        if representation not in {"slt", "opt", "slt_type"}:
            raise ValueError(f"representation must be slt/opt/slt_type, got '{representation}'")

        self.embedding_dir = Path(embedding_dir)
        self.representation = representation
        self.alpha = alpha
        self.aggregation = aggregation
        self.n_candidates = n_candidates

        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._num_workers = os.cpu_count() or 1
        logger.info("FormulaMaxSimReranker: using %d worker threads", self._num_workers)

    # ------------------------------------------------------------------
    # Embedder lifecycle
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load FormulaFAISSIndexerIVFScalarQuantizer (embedding only, no FAISS index).

        Thread-safe via double-checked locking. The warm-up call ensures
        _query_model_manager is set before any worker thread uses it.
        """
        if self._embedder is not None:
            return self._embedder
        with self._embedder_lock:
            if self._embedder is None:
                from multirag.indexing.formula.faiss_scalar_quantizer import (
                    FormulaFAISSIndexerIVFScalarQuantizer,
                )
                tmp = tempfile.mkdtemp(prefix="multirag_reranker_")
                embedder = FormulaFAISSIndexerIVFScalarQuantizer(
                    index_path=tmp,
                    embedding_dir=str(self.embedding_dir),
                    representation=self.representation,
                )
                embedder.embed_formula("x")  # warm up: sets _query_model_manager before threads start
                self._embedder = embedder
        return self._embedder

    # ------------------------------------------------------------------
    # Per-formula embedding helpers
    # ------------------------------------------------------------------

    def _embed_from_latex(self, latex: str) -> np.ndarray | None:
        """Subprocess path: LaTeX → MathML (subprocess) → tuples → FastText vector."""
        vec = self._get_embedder().embed_formula(latex)
        return vec if (vec is not None and np.any(vec)) else None

    def _embed_from_mathml(self, mathml: str) -> np.ndarray | None:
        """TSV path: pre-computed MathML → tuples → FastText vector (no subprocess)."""
        vec = self._get_embedder().embed_formula_from_mathml(mathml)
        return vec if (vec is not None and np.any(vec)) else None

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ------------------------------------------------------------------
    # Formula-set embedding: two flavours
    # ------------------------------------------------------------------

    def _embed_topic_formula_set(self, latexes: list[str]) -> np.ndarray:
        """Embed topic formulas via subprocess (LaTeX→MathML each time).

        Returns array [K, 300], K <= len(latexes).
        """
        vecs = []
        for lx in latexes:
            if _is_clearly_trivial(lx):
                continue
            v = self._embed_from_latex(lx)
            if v is not None:
                vecs.append(self._l2_normalize(v))
        return np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 300), dtype=np.float32)

    def _embed_candidate_formula_set(
        self,
        fid_latex_pairs: list[tuple[str, str]],
        formula_mathml: dict[str, str],
    ) -> np.ndarray:
        """Embed candidate formulas using pre-computed MathML where available.

        For each (formula_id, latex) pair:
        - Hit in formula_mathml → TSV path (no subprocess).
        - Miss → subprocess fallback via LaTeX.

        Returns array [K, 300].
        """
        vecs = []
        for fid, lx in fid_latex_pairs:
            mathml = formula_mathml.get(fid)
            if mathml:
                v = self._embed_from_mathml(mathml)
            else:
                v = self._embed_from_latex(lx)
            if v is not None:
                vecs.append(self._l2_normalize(v))
        return np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 300), dtype=np.float32)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _formula_score(self, topic_vecs: np.ndarray, cand_vecs: np.ndarray) -> float:
        """MaxSim score between topic and candidate formula sets."""
        if len(topic_vecs) == 0 or len(cand_vecs) == 0:
            return 0.0
        sim = topic_vecs @ cand_vecs.T           # [T, C] — both L2-normalised
        per_topic_best = sim.max(axis=1)          # [T]
        if self.aggregation == "max":
            return float(per_topic_best.max())
        elif self.aggregation == "mean":
            return float(per_topic_best.mean())
        else:  # "sum"
            return float(per_topic_best.sum())

    @staticmethod
    def _minmax_normalize(scores: list[float]) -> list[float]:
        """Min-max normalize to [0, 1]. All-equal → 0.5 for every entry."""
        lo, hi = min(scores), max(scores)
        r = hi - lo
        if r == 0.0:
            return [0.5] * len(scores)
        return [(s - lo) / r for s in scores]

    def _score_candidate(
        self,
        doc_id: str,
        text_score: float,
        topic_vecs: np.ndarray,
        answer_formulas: dict[str, list[tuple[str, str]]],
        formula_mathml: dict[str, str],
    ) -> tuple[str, float, float]:
        """Score a single candidate. Designed to run concurrently in a thread pool."""
        fid_latex_pairs = answer_formulas.get(doc_id, [])
        cand_vecs = self._embed_candidate_formula_set(fid_latex_pairs, formula_mathml)
        return doc_id, text_score, self._formula_score(topic_vecs, cand_vecs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        run: dict[str, list[tuple[str, float]]],
        topics: list[dict[str, Any]],
        answer_formulas: dict[str, list[tuple[str, str]]],
        formula_mathml: dict[str, str],
    ) -> dict[str, list[tuple[str, float]]]:
        """Rerank each topic's stage-1 candidates.

        Args:
            run: Stage-1 run — {topic_id: [(doc_id, score), ...]} sorted by score desc.
            topics: Topics loaded from topics.jsonl (each dict has "topic_id", "formulas").
            answer_formulas: {answer_id: [(formula_id, latex), ...]} — non-trivial only.
            formula_mathml: {formula_id: pre_computed_mathml} from collection TSV shards.
                            Used for candidate embedding; missing ids fall back to subprocess.

        Returns:
            Reranked run in the same format as `run`.
        """
        topic_by_id = {t["topic_id"]: t for t in topics}
        output: dict[str, list[tuple[str, float]]] = {}

        self._get_embedder()  # force model load before spawning threads

        for topic_id, ranked_docs in tqdm(run.items(), desc="Reranking topics", unit="topic"):
            topic = topic_by_id.get(topic_id)
            if topic is None:
                logger.warning("Topic %s not found in topics.jsonl — keeping original order", topic_id)
                output[topic_id] = ranked_docs
                continue

            pool = ranked_docs[: self.n_candidates]
            tail = ranked_docs[self.n_candidates :]

            # Topic formulas: subprocess path (query-time LaTeX→MathML)
            topic_latexes = [f["latex"] for f in topic.get("formulas", [])]
            topic_vecs = self._embed_topic_formula_set(topic_latexes)

            if len(topic_vecs) == 0:
                logger.debug("Topic %s has no embeddable formulas — keeping original order", topic_id)
                output[topic_id] = ranked_docs
                continue

            # Candidate formulas: TSV MathML path (parallel across candidates)
            results: list[tuple[str, float, float]] = []
            n_workers = min(self._num_workers, len(pool))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self._score_candidate,
                        doc_id, text_score, topic_vecs, answer_formulas, formula_mathml,
                    ): doc_id
                    for doc_id, text_score in pool
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"  [{topic_id}]",
                    unit="doc",
                    leave=False,
                ):
                    results.append(future.result())

            doc_ids = [r[0] for r in results]
            text_scores = [r[1] for r in results]
            formula_scores = [r[2] for r in results]

            text_norm = self._minmax_normalize(text_scores)
            formula_norm = self._minmax_normalize(formula_scores)

            blended = [
                (doc_id, (1 - self.alpha) * t + self.alpha * f)
                for doc_id, t, f in zip(doc_ids, text_norm, formula_norm)
            ]
            blended.sort(key=lambda x: -x[1])

            output[topic_id] = blended + tail

        return output
