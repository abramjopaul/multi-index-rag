# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Fused Formula FAISS indexer: combines L2-normalized SLT + OPT + SLT-TYPE
# embeddings into a single 300-dim vector per formula.

import csv
import json
import sys
import logging
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from tqdm import tqdm

from multirag.config.path_configs import ANSWERS_JSONL
from multirag.embedding.formula_model_manager import FastTextModelManager
from multirag.formula_search import (
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
    extract_tuples_from_mathml_direct,
)
from multirag.formula_search.encoder_maps import load_maps
from multirag.formula_search.latex_mml import LatexToMathML, LatexToMathMLPool, _get_optimal_workers
from multirag.formula_search.tuple_extraction import encode_tuples
from multirag.indexing.base import BaseIndexer

logger = logging.getLogger(__name__)

_TSV_MINI_BATCH = 1024

_REPRESENTATIONS = ["slt", "opt", "slt_type"]
_TREE_TYPE_MAP = {"slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE"}


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


class FormulaFAISSIndexerFused(BaseIndexer):
    """
    Fused formula indexer: combines SLT, OPT, and SLT-TYPE embeddings into
    one 300-dim vector per formula via L2-normalized sum, stored in a single
    IVFScalarQuantizer FAISS index.

    Normalization: each 300-dim representation vector is L2-normalized before
    summing, so no single representation can dominate due to scale differences.
    The combined vector is optionally L2-normalized again before storage.

    Memory: ~670 MB (single index, same compression as scalar_quantizer.py)
    """

    DIMENSION = 300
    NLIST = 16384
    NPROBE = 256
    QUANTIZER_TYPE = faiss.ScalarQuantizer.QT_fp16
    BYTES_PER_VECTOR = 8  # fp16 × 300 dims

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path = ANSWERS_JSONL,
        embedding_dir: Optional[str] = None,
        nprobe: int = 256,
        force_rebuild: bool = False,
        formula_tsv_base_dir: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Args:
            index_path: Directory to store/load FAISS index
            corpus_path: Path to answers.jsonl
            embedding_dir: Directory containing FastText models for all three representations
            nprobe: Number of clusters to search (higher = more accurate but slower)
            force_rebuild: Force rebuild even if index exists
            formula_tsv_base_dir: Base directory containing slt_representation_v3/ and
                                  opt_representation_v3/ subdirectories.
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.corpus_path = Path(corpus_path)
        self.embedding_dir = Path(embedding_dir) if embedding_dir else None
        self.formula_tsv_base_dir = Path(formula_tsv_base_dir) if formula_tsv_base_dir else None
        self.nprobe = nprobe
        self.force_rebuild = force_rebuild
        self.device = device

        self._index = None
        self.id_map: dict[int, tuple[str, str]] = {}

        # Lazily loaded model managers (keyed by representation name)
        self._model_managers: dict[str, FastTextModelManager] = {}
        self._tuple_tokenizers: dict[str, TupleTokenizer] = {}

        logger.info(
            f"Initialized FormulaFAISSIndexerFused: {self.index_path} "
            f"(nlist={self.NLIST}, nprobe={self.nprobe}, device={self._resolved_device()})"
        )

    def _resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if getattr(faiss, "StandardGpuResources", None) is not None:
            return "cuda"
        return "cpu"

    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer_id": raw_doc.get("id"),
            "parent_id": raw_doc.get("parent_id"),
            "formulas": raw_doc.get("formulas", []),
        }

    def get_representations(self) -> list[str]:
        return ["fused"]

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build the fused FAISS index.

        TSV path: streams slt_representation_v3/ and opt_representation_v3/ in
        lockstep, combining all three representations per formula.

        Bulk path: loads pre-computed embeddings from embedding_dir.
        """
        if force:
            self.force_rebuild = True

        index_file = self.index_path / "formula_index_fused_sq.faiss"

        # --- Streaming TSV path ---
        if self._slt_tsv_dir is not None and self._opt_tsv_dir is not None:
            if not self.force_rebuild:
                cp = self._load_tsv_checkpoint()
                shared_tsvs = self._shared_tsv_files()
                if shared_tsvs and set(cp.get("processed_tsv_files", [])) == {f.name for f in shared_tsvs}:
                    logger.info("Fused index already complete. Loading from disk...")
                    self._load_index_from_disk()
                    return

            if not self.embedding_dir or not self.embedding_dir.exists():
                raise ValueError(
                    f"Embedding directory not found: {self.embedding_dir}. "
                    "Generate embeddings first using formula_embedder.py"
                )

            logger.info("Building formula ID → metadata mapping from answers.jsonl...")
            fid_to_meta: dict[str, tuple[str, str]] = {}
            with open(self.corpus_path) as f:
                for line_idx, line in enumerate(tqdm(f, desc="Loading formula metadata")):
                    if limit is not None and line_idx >= limit:
                        break
                    try:
                        answer = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON at line {line_idx}")
                        continue
                    answer_id = answer.get("id")
                    for formula_obj in answer.get("formulas", []):
                        formula_id = str(formula_obj.get("formula_id"))
                        fid_to_meta[formula_id] = (answer_id, formula_id)

            logger.info(f"Loaded metadata for {len(fid_to_meta):,} formulas")

            self._setup_all_models()
            self._build_faiss_index_from_tsv(fid_to_meta)
            logger.info("Streaming TSV fused indexing complete")
            return

        # --- Bulk path ---
        if index_file.exists() and not self.force_rebuild:
            logger.info("Fused index already exists. Loading from disk...")
            self._load_index_from_disk()
            return

        logger.info("Building fused IVFScalarQuantizer index (bulk path)...")

        formulas = []
        with open(self.corpus_path) as f:
            for line_idx, line in enumerate(tqdm(f, desc="Loading formulas")):
                if limit is not None and line_idx >= limit:
                    break
                try:
                    answer = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_idx}")
                    continue
                answer_id = answer.get("id")
                for formula_obj in answer.get("formulas", []):
                    formulas.append({
                        "answer_id": answer_id,
                        "formula_id": formula_obj.get("formula_id"),
                        "latex": formula_obj.get("latex"),
                    })

        logger.info(f"Loaded {len(formulas)} formulas")

        if not self.embedding_dir or not self.embedding_dir.exists():
            raise ValueError(
                f"Embedding directory not found: {self.embedding_dir}. "
                "Generate embeddings first using formula_embedder.py"
            )

        self._setup_all_models()
        embeddings = self._load_fused_embeddings(formulas)

        if not embeddings:
            raise ValueError("No embeddings generated")

        self._build_faiss_index(embeddings, formulas)
        self._save_index_to_disk()
        logger.info("Fused indexing complete")

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search with a LaTeX query formula, returning top-k hits."""
        try:
            query_embedding = self._generate_query_embedding(query)
        except Exception as e:
            logger.warning(f"Failed to generate fused query embedding for '{query}': {e}")
            return []

        if self._index is None:
            self._load_index_from_disk()

        if self._index is None:
            logger.warning("Fused index not available")
            return []

        qvec = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self._index.search(qvec, k)  # type: ignore

        hits = []
        for rank, idx in enumerate(indices[0]):
            if idx in self.id_map:
                answer_id, formula_id = self.id_map[idx]
                hits.append({
                    "doc_id": answer_id,
                    "formula_id": formula_id,
                    "score": 1.0 / (1.0 + float(distances[0][rank])),
                    "representation": "fused",
                })
        return hits

    def batch_search(
        self,
        queries: list[tuple[str, str]],
        k: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        results = {}
        for qid, query in tqdm(queries, desc="Batch searching (fused)"):
            results[qid] = self.search(query, k)
        return results

    # -------------------------------------------------------------------------
    # Model setup
    # -------------------------------------------------------------------------

    def _setup_model_for_representation(self, rep: str) -> None:
        """Load FastText model and TupleTokenizer for a single representation."""
        if rep in self._model_managers:
            return

        if not self.embedding_dir or not self.embedding_dir.exists():
            raise ValueError(f"Embedding directory not found: {self.embedding_dir}")

        suffix = rep.lower().replace("-", "_")
        model_file = self.embedding_dir / suffix / f"fasttext_model_{suffix}.bin"
        metadata_file = self.embedding_dir / suffix / f"training_metadata_{suffix}.json"

        if not model_file.exists():
            raise FileNotFoundError(f"FastText model not found: {model_file}")

        logger.info(f"Loading FastText model for '{rep}': {model_file}")
        mm = FastTextModelManager(
            model_path=str(model_file),
            metadata_path=str(metadata_file),
            corpus_path="",
        )
        mm.load()

        metadata = mm.get_stats()
        embedding_type_name = metadata.get("embedding_type", "Both_Separated")
        tokenize_number = metadata.get("tokenize_number", True)

        embedding_type_map = {
            "Both_Separated": TupleTokenizationMode.Both_Separated,
            "Type": TupleTokenizationMode.Type,
        }
        embedding_type = embedding_type_map.get(embedding_type_name, TupleTokenizationMode.Both_Separated)

        encoder_maps_path = metadata.get("encoder_maps_path")
        if encoder_maps_path and Path(encoder_maps_path).exists():
            node_map, edge_map = load_maps(encoder_maps_path)
            node_id = max(node_map.values(), default=60000) + 1
            edge_id = max(edge_map.values(), default=500) + 1
            token_id_manager = TokenIDManager(node_id=node_id, edge_id=edge_id, node_map=node_map, edge_map=edge_map)
        else:
            logger.warning(f"Encoder maps not found for '{rep}'; token IDs will not match training vocabulary")
            token_id_manager = TokenIDManager()

        self._model_managers[rep] = mm
        self._tuple_tokenizers[rep] = TupleTokenizer(
            token_id_manager=token_id_manager,
            embedding_type=embedding_type,
            tokenize_number=tokenize_number,
        )

    def _setup_all_models(self) -> None:
        for rep in _REPRESENTATIONS:
            self._setup_model_for_representation(rep)

    # -------------------------------------------------------------------------
    # Fused embedding computation
    # -------------------------------------------------------------------------

    def _embedding_from_mathml(self, mathml: str, rep: str) -> Optional[np.ndarray]:
        """Extract tuples from MathML and return FastText vector, or None on failure."""
        tree_type = _TREE_TYPE_MAP[rep]
        try:
            tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore
            if not tuples:
                return None
            encoded = encode_tuples(tuples, self._tuple_tokenizers[rep])
            if not encoded:
                return None
            return np.array(self._model_managers[rep].get_sentence_vector(encoded), dtype=np.float32)
        except Exception:
            return None

    def _compute_fused_embedding(
        self,
        slt_mathml: str,
        opt_mathml: str,
    ) -> Optional[np.ndarray]:
        """Compute L2-normalized sum of SLT + OPT + SLT-TYPE embeddings.

        SLT-TYPE uses the same PMML MathML as SLT but with tree_type='SLT-TYPE'.
        Returns None if any representation fails to produce an embedding.
        """
        slt_vec = self._embedding_from_mathml(slt_mathml, "slt")
        opt_vec = self._embedding_from_mathml(opt_mathml, "opt")
        slt_type_vec = self._embedding_from_mathml(slt_mathml, "slt_type")

        if slt_vec is None or opt_vec is None or slt_type_vec is None:
            return None

        combined = _l2_normalize(slt_vec) + _l2_normalize(opt_vec) + _l2_normalize(slt_type_vec)
        return _l2_normalize(combined)

    # -------------------------------------------------------------------------
    # TSV path helpers
    # -------------------------------------------------------------------------

    @property
    def _slt_tsv_dir(self) -> Optional[Path]:
        if not self.formula_tsv_base_dir:
            return None
        p = self.formula_tsv_base_dir / "slt_representation_v3"
        return p if p.exists() else None

    @property
    def _opt_tsv_dir(self) -> Optional[Path]:
        if not self.formula_tsv_base_dir:
            return None
        p = self.formula_tsv_base_dir / "opt_representation_v3"
        return p if p.exists() else None

    def _shared_tsv_files(self) -> list[Path]:
        """Return TSV files present in BOTH slt and opt directories, sorted numerically."""
        slt_dir = self._slt_tsv_dir
        opt_dir = self._opt_tsv_dir
        if slt_dir is None or opt_dir is None:
            return []
        slt_names = {p.name for p in slt_dir.glob("*.tsv")}
        opt_names = {p.name for p in opt_dir.glob("*.tsv")}
        shared = slt_names & opt_names
        return sorted([slt_dir / name for name in shared], key=lambda p: int(p.stem))

    def _load_tsv_as_dict(self, tsv_file: Path) -> dict[str, str]:
        """Load a TSV file into {formula_id: mathml} dict."""
        csv.field_size_limit(sys.maxsize)
        result: dict[str, str] = {}
        with open(tsv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                fid = row.get("id", "").strip()
                mathml = row.get("formula", "").strip()
                if fid and mathml:
                    result[fid] = mathml
        return result

    def _tsv_checkpoint_path(self) -> Path:
        return self.index_path / "tsv_checkpoint_fused.json"

    def _load_tsv_checkpoint(self) -> dict:
        cp = self._tsv_checkpoint_path()
        if not cp.exists():
            return {"processed_tsv_files": [], "indexed_count": 0, "trained": False}
        with open(cp) as f:
            return json.load(f)

    def _save_tsv_checkpoint(self, processed_files: list[str], indexed_count: int, trained: bool) -> None:
        with open(self._tsv_checkpoint_path(), "w") as f:
            json.dump({"processed_tsv_files": processed_files, "indexed_count": indexed_count, "trained": trained}, f)

    def _collect_training_vectors(
        self, shared_tsv_files: list[Path], train_size: int
    ) -> np.ndarray:
        """Collect up to train_size fused embedding vectors for FAISS IVF training."""
        training_vectors: list[np.ndarray] = []

        for slt_tsv in tqdm(shared_tsv_files, desc="Collecting fused training vectors"):
            if len(training_vectors) >= train_size:
                break
            opt_tsv = self._opt_tsv_dir / slt_tsv.name  # type: ignore

            slt_dict = self._load_tsv_as_dict(slt_tsv)
            opt_dict = self._load_tsv_as_dict(opt_tsv)
            shared_ids = set(slt_dict.keys()) & set(opt_dict.keys())

            for fid in shared_ids:
                if len(training_vectors) >= train_size:
                    break
                vec = self._compute_fused_embedding(slt_dict[fid], opt_dict[fid])
                if vec is not None:
                    training_vectors.append(vec)

        if len(training_vectors) < train_size:
            logger.warning(
                f"Only collected {len(training_vectors)} training vectors "
                f"(requested {train_size})"
            )

        return np.array(training_vectors, dtype=np.float32)

    def _build_faiss_index_from_tsv(self, fid_to_meta: dict[str, tuple[str, str]]) -> None:
        """Stream paired SLT+OPT TSV files to build the fused FAISS index."""
        shared_tsv_files = self._shared_tsv_files()
        if not shared_tsv_files:
            raise ValueError(
                f"No shared TSV files found between {self._slt_tsv_dir} and {self._opt_tsv_dir}"
            )

        index_file = self.index_path / "formula_index_fused_sq.faiss"
        cp = self._load_tsv_checkpoint()
        processed_set: set[str] = set(cp.get("processed_tsv_files", []))
        indexed_count: int = cp.get("indexed_count", 0)
        already_trained: bool = cp.get("trained", False)

        # Phase 1: Train
        if not already_trained:
            train_size = min(
                max(len(fid_to_meta) // 10, self.NLIST * 40),
                len(fid_to_meta),
            )
            logger.info(f"Collecting {train_size:,} fused training vectors...")
            training_vectors = self._collect_training_vectors(shared_tsv_files, train_size)

            logger.info(f"Training FAISS index on {len(training_vectors):,} vectors...")
            quantizer = faiss.IndexFlat(self.DIMENSION)
            cpu_index = faiss.IndexIVFScalarQuantizer(
                quantizer, self.DIMENSION, self.NLIST, self.QUANTIZER_TYPE
            )

            if self._resolved_device() == "cuda":
                logger.info("Moving index to GPU for training...")
                res = getattr(faiss, "StandardGpuResources")()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # type: ignore
                gpu_index.train(training_vectors)
                self._index = faiss.index_gpu_to_cpu(gpu_index)  # type: ignore
                logger.info("Training complete; index moved back to CPU")
            else:
                cpu_index.train(training_vectors)  # type: ignore
                self._index = cpu_index

            self._index.nprobe = self.nprobe
            del training_vectors

            faiss.write_index(self._index, str(index_file))
            self._save_tsv_checkpoint(list(processed_set), indexed_count, trained=True)
            logger.info("Training complete; empty index saved.")
        else:
            logger.info(f"Resuming from checkpoint ({len(processed_set)} TSV files already done)...")
            self._index = faiss.read_index(str(index_file))
            self._index.nprobe = self.nprobe

            id_map_file = self.index_path / "id_map_fused_sq.json"
            if id_map_file.exists():
                with open(id_map_file) as f:
                    raw = json.load(f)
                    self.id_map = {int(k): tuple(v) for k, v in raw.items()}

        # Phase 2: Add vectors file by file
        remaining = [f for f in shared_tsv_files if f.name not in processed_set]
        logger.info(f"Adding vectors from {len(remaining)} remaining TSV file pairs...")

        failed_count = 0

        for slt_tsv in tqdm(remaining, desc="Indexing TSV pairs (fused)"):
            opt_tsv = self._opt_tsv_dir / slt_tsv.name  # type: ignore

            slt_dict = self._load_tsv_as_dict(slt_tsv)
            opt_dict = self._load_tsv_as_dict(opt_tsv)

            batch_vectors: list[np.ndarray] = []
            batch_metas: list[tuple[str, str]] = []

            for fid in slt_dict:
                if fid not in fid_to_meta or fid not in opt_dict:
                    continue

                vec = self._compute_fused_embedding(slt_dict[fid], opt_dict[fid])
                if vec is None:
                    failed_count += 1
                    continue

                batch_vectors.append(vec)
                batch_metas.append(fid_to_meta[fid])

                if len(batch_vectors) >= _TSV_MINI_BATCH:
                    base_idx = self._index.ntotal  # type: ignore
                    self._index.add(np.array(batch_vectors, dtype=np.float32))  # type: ignore
                    for i, meta in enumerate(batch_metas):
                        self.id_map[base_idx + i] = meta
                    indexed_count += len(batch_vectors)
                    batch_vectors = []
                    batch_metas = []

            if batch_vectors:
                base_idx = self._index.ntotal  # type: ignore
                self._index.add(np.array(batch_vectors, dtype=np.float32))  # type: ignore
                for i, meta in enumerate(batch_metas):
                    self.id_map[base_idx + i] = meta
                indexed_count += len(batch_vectors)

            processed_set.add(slt_tsv.name)
            self._save_index_to_disk()
            self._save_tsv_checkpoint(list(processed_set), indexed_count, trained=True)
            logger.info(f"Checkpoint: {slt_tsv.name} done — {indexed_count:,} vectors indexed total")

        logger.info(
            f"Fused streaming indexing complete: {indexed_count:,} vectors, "
            f"{failed_count} skipped (no valid fused embedding)"
        )

    # -------------------------------------------------------------------------
    # Bulk (non-TSV) helpers
    # -------------------------------------------------------------------------

    def _load_fused_embeddings(self, formulas: list[dict]) -> list[np.ndarray]:
        """Generate fused embeddings via latexmlmath subprocess (slow bulk path)."""
        embeddings: list[np.ndarray] = []
        failed_count = 0

        logger.info(f"Generating fused embeddings for {len(formulas)} formulas...")

        batch_size = 256
        # We need PMML (for SLT/SLT-TYPE) and CMML (for OPT) per formula.
        with (
            LatexToMathMLPool(num_workers=_get_optimal_workers(), mathml_type="pmml") as pmml_pool,
            LatexToMathMLPool(num_workers=_get_optimal_workers(), mathml_type="cmml") as cmml_pool,
        ):
            for batch_start in tqdm(range(0, len(formulas), batch_size), desc="Embedding (fused)"):
                batch = formulas[batch_start: batch_start + batch_size]
                batch_tex = [f.get("latex", "") for f in batch]

                pmml_results = pmml_pool.convert_batch(batch_tex)
                cmml_results = cmml_pool.convert_batch(batch_tex)

                for formula_data, pmml, cmml in zip(batch, pmml_results, cmml_results):
                    if not formula_data.get("latex", "") or not pmml or not cmml:
                        embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                        failed_count += 1
                        continue

                    vec = self._compute_fused_embedding(pmml, cmml)
                    if vec is None:
                        embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                        failed_count += 1
                    else:
                        embeddings.append(vec)

        if failed_count:
            logger.warning(f"Failed {failed_count}/{len(formulas)} fused embeddings; used zero vectors")

        return embeddings

    def _build_faiss_index(self, embeddings: list[np.ndarray], formulas: list[dict]) -> None:
        embeddings_array = np.array(embeddings, dtype=np.float32)

        train_size = min(max(len(embeddings) // 10, self.NLIST * 40), len(embeddings))
        training_indices = np.random.choice(len(embeddings), train_size, replace=False)
        training_vectors = embeddings_array[training_indices]

        logger.info(f"Training fused index on {train_size} vectors...")
        quantizer = faiss.IndexFlat(self.DIMENSION)
        cpu_index = faiss.IndexIVFScalarQuantizer(
            quantizer, self.DIMENSION, self.NLIST, self.QUANTIZER_TYPE
        )

        if self._resolved_device() == "cuda":
            res = getattr(faiss, "StandardGpuResources")()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # type: ignore
            gpu_index.train(training_vectors)
            self._index = faiss.index_gpu_to_cpu(gpu_index)  # type: ignore
        else:
            cpu_index.train(training_vectors)  # type: ignore
            self._index = cpu_index

        self._index.nprobe = self.nprobe
        logger.info(f"Adding {len(embeddings)} fused vectors...")
        self._index.add(embeddings_array)  # type: ignore

        for idx, formula_data in enumerate(formulas):
            self.id_map[idx] = (formula_data["answer_id"], formula_data["formula_id"])

        logger.info(f"Built fused index: {self._index.ntotal} vectors")

    # -------------------------------------------------------------------------
    # Disk I/O
    # -------------------------------------------------------------------------

    def _save_index_to_disk(self) -> None:
        if self._index is None:
            logger.warning("No fused index to save")
            return

        index_file = self.index_path / "formula_index_fused_sq.faiss"
        faiss.write_index(self._index, str(index_file))
        logger.info(f"Saved fused index to {index_file}")

        id_map_file = self.index_path / "id_map_fused_sq.json"
        with open(id_map_file, "w") as f:
            json.dump({str(k): v for k, v in self.id_map.items()}, f)
        logger.info(f"Saved fused ID map to {id_map_file}")

    def _load_index_from_disk(self) -> None:
        index_file = self.index_path / "formula_index_fused_sq.faiss"
        id_map_file = self.index_path / "id_map_fused_sq.json"

        if not index_file.exists():
            logger.warning(f"Fused index file not found: {index_file}")
            return
        if not id_map_file.exists():
            logger.warning(f"Fused ID map not found: {id_map_file}")
            return

        self._index = faiss.read_index(str(index_file))
        self._index.nprobe = self.nprobe

        with open(id_map_file) as f:
            raw = json.load(f)
            self.id_map = {int(k): tuple(v) for k, v in raw.items()}

        logger.info(f"Loaded fused index: {self._index.ntotal} vectors, nprobe={self.nprobe}")

    # -------------------------------------------------------------------------
    # Query embedding
    # -------------------------------------------------------------------------

    def _generate_query_embedding(self, query_latex: str) -> np.ndarray:
        """Generate fused embedding for a LaTeX query."""
        if not self._model_managers:
            self._setup_all_models()

        try:
            pmml = LatexToMathML.convert_to_mathml(query_latex)   # for SLT / SLT-TYPE
            cmml = LatexToMathML.convert_to_mathml2(query_latex)  # for OPT

            if not pmml or not cmml:
                return np.zeros(self.DIMENSION, dtype=np.float32)

            vec = self._compute_fused_embedding(pmml, cmml)
            return vec if vec is not None else np.zeros(self.DIMENSION, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error generating fused query embedding: {e}")
            return np.zeros(self.DIMENSION, dtype=np.float32)
