"""Stage-2 formula-aware reranker using MaxSim over FastText tuple embeddings."""

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from multirag.evaluation.metrics import _is_clearly_trivial

logger = logging.getLogger(__name__)


class FormulaMaxSimReranker:
    """Reranks a stage-1 text retrieval pool by blending text scores with
    formula-structural relevance.

    Formula relevance = MaxSim: for each non-trivial topic formula, take its best
    cosine similarity to any formula in the candidate post, then aggregate across
    topic formulas. Both score channels are min-max normalized per topic before
    blending: final = (1-alpha)*text_norm + alpha*formula_norm.
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
        self._embedder = None  # lazy-loaded

    def _get_embedder(self):
        """Lazy-load a FormulaFAISSIndexerIVFScalarQuantizer instance used only for embedding."""
        if self._embedder is not None:
            return self._embedder

        from multirag.indexing.formula.faiss_scalar_quantizer import (
            FormulaFAISSIndexerIVFScalarQuantizer,
        )

        # Use a throwaway temp dir for index_path — .index() is never called, so
        # no files are ever written there. We instantiate the object only to get
        # access to its lazy-loading embed_formula() method.
        tmp = tempfile.mkdtemp(prefix="multirag_reranker_")
        self._embedder = FormulaFAISSIndexerIVFScalarQuantizer(
            index_path=tmp,
            embedding_dir=str(self.embedding_dir),
            representation=self.representation,
        )
        return self._embedder

    def _embed(self, latex: str) -> np.ndarray | None:
        """Embed a LaTeX formula. Returns None if the result is a zero vector."""
        vec = self._get_embedder().embed_formula(latex)
        if vec is None or not np.any(vec):
            return None
        return vec

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _formula_score(
        self,
        topic_vecs: np.ndarray,
        cand_vecs: np.ndarray,
    ) -> float:
        """MaxSim score between topic and candidate formula sets."""
        if len(topic_vecs) == 0 or len(cand_vecs) == 0:
            return 0.0
        # sim[i, j] = cosine(topic_vecs[i], cand_vecs[j])  (both already L2-normalized)
        sim = topic_vecs @ cand_vecs.T  # [T, C]
        per_topic_best = sim.max(axis=1)  # [T]
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

    def _embed_formula_set(self, latexes: list[str]) -> np.ndarray:
        """Embed and L2-normalize a list of LaTeX formulas, dropping failures.

        Returns an array of shape [K, 300] where K <= len(latexes).
        """
        vecs = []
        for lx in latexes:
            if _is_clearly_trivial(lx):
                continue
            v = self._embed(lx)
            if v is not None:
                vecs.append(self._l2_normalize(v))
        return np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 300), dtype=np.float32)

    def rerank(
        self,
        run: dict[str, list[tuple[str, float]]],
        topics: list[dict[str, Any]],
        answer_formulas: dict[str, list[str]],
    ) -> dict[str, list[tuple[str, float]]]:
        """Rerank each topic's stage-1 candidates.

        Args:
            run: Stage-1 run — {topic_id: [(doc_id, score), ...]} sorted by score desc.
            topics: Topics loaded from topics.jsonl (each dict has "topic_id" and "formulas").
            answer_formulas: {answer_id: [non-trivial LaTeX, ...]} for all candidate posts.

        Returns:
            Reranked run in the same format as `run`.
        """
        topic_by_id = {t["topic_id"]: t for t in topics}
        output: dict[str, list[tuple[str, float]]] = {}

        for topic_id, ranked_docs in run.items():
            topic = topic_by_id.get(topic_id)
            if topic is None:
                logger.warning("Topic %s not found in topics.jsonl — keeping original order", topic_id)
                output[topic_id] = ranked_docs
                continue

            pool = ranked_docs[: self.n_candidates]
            tail = ranked_docs[self.n_candidates :]

            # Embed all non-trivial topic formulas
            topic_latexes = [f["latex"] for f in topic.get("formulas", [])]
            topic_vecs = self._embed_formula_set(topic_latexes)

            if len(topic_vecs) == 0:
                logger.debug("Topic %s has no embeddable formulas — keeping original order", topic_id)
                output[topic_id] = ranked_docs
                continue

            # Score each candidate
            text_scores: list[float] = []
            formula_scores: list[float] = []
            doc_ids: list[str] = []

            for doc_id, text_score in pool:
                cand_latexes = answer_formulas.get(doc_id, [])
                cand_vecs = self._embed_formula_set(cand_latexes)
                text_scores.append(text_score)
                formula_scores.append(self._formula_score(topic_vecs, cand_vecs))
                doc_ids.append(doc_id)

            text_norm = self._minmax_normalize(text_scores)
            formula_norm = self._minmax_normalize(formula_scores)

            blended = [
                (doc_id, (1 - self.alpha) * t + self.alpha * f)
                for doc_id, t, f in zip(doc_ids, text_norm, formula_norm)
            ]
            blended.sort(key=lambda x: -x[1])

            output[topic_id] = blended + tail

        return output
