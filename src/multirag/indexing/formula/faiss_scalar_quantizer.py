# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Formula FAISS indexer using IVFScalarQuantizer compression.
# Suitable for memory-constrained scenarios with acceptable accuracy trade-off.

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
from multirag.formula_search.tuple_extraction import (
    encode_tuples,
)
from multirag.indexing.base import BaseIndexer

logger = logging.getLogger(__name__)

_TSV_MINI_BATCH = 1024


class FormulaFAISSIndexerIVFScalarQuantizer(BaseIndexer):
    """
    Formula indexer using FAISS IVFScalarQuantizer compression.

    Builds three parallel FAISS indices for SLT, OPT, and SLT_type representations.
    Uses Inverted File with Scalar Quantization (8-byte per vector compression).

    Memory: ~670 MB per representation (~2 GB total)
    Query time: 80-300ms depending on nprobe setting
    Accuracy: Quantized distances, ~1-2% loss vs IVFFlat

    Attributes:
        index_path: Path to store/load FAISS indices
        id_maps: Dict mapping representation → {idx: (answer_id, formula_id)}
        indices: Dict mapping representation → FAISS index
        embedder: Formula embedder for generating vectors
    """

    REPRESENTATIONS = ["slt", "opt", "slt_type"]
    DIMENSION = 300
    NLIST = 16384  # Number of clusters (4·√N and 16·√N. With √28M ≈ 5,300) ie 16384, 32768, or 65536.
    NPROBE = 256  # Number of clusters to search. Sweep upward (32 → 64 → 128 → 256
    QUANTIZER_TYPE = faiss.ScalarQuantizer.QT_fp16  # 8-bit or 16-bit quantization

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path = ANSWERS_JSONL,
        embedding_dir: Optional[str] = None,
        representation: str = "slt",
        quantizer_bits: str = "fp16",
        nprobe: int = 256,
        force_rebuild: bool = False,
        formula_tsv_base_dir: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize IVFScalarQuantizer formula indexer for a single representation.

        Args:
            index_path: Directory to store/load FAISS indices
            corpus_path: Path to answers.jsonl
            embedding_dir: Directory containing pre-computed embeddings
            representation: Which representation to index: 'slt', 'opt', or 'slt_type'.
                           Defaults to 'slt'
            quantizer_bits: Quantization type - 'fp16' (8 bytes) or '8bit' (4 bytes)
            nprobe: Number of clusters to search (higher = more accurate but slower)
            force_rebuild: Force rebuild even if indices exist
            formula_tsv_base_dir: Base directory containing slt_representation_v3/ and
                                  opt_representation_v3/ subdirectories. When set,
                                  pre-computed MathML is read directly from TSV files,
                                  bypassing all latexmlmath subprocess calls during indexing.
                                  Example: "data/raw/collection/formula"
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.corpus_path = Path(corpus_path)
        self.embedding_dir = Path(embedding_dir) if embedding_dir else None
        self.formula_tsv_base_dir = Path(formula_tsv_base_dir) if formula_tsv_base_dir else None
        self.nprobe = nprobe
        self.force_rebuild = force_rebuild
        self.device = device

        # Set quantizer type
        if quantizer_bits == "fp16":
            self.quantizer_type = faiss.ScalarQuantizer.QT_fp16
            self.bytes_per_vector = 8  # 16 bits × 300 dims / 8 / 300 ≈ 8 bytes
        elif quantizer_bits == "8bit":
            self.quantizer_type = faiss.ScalarQuantizer.QT_8bit
            self.bytes_per_vector = 4
        else:
            raise ValueError(f"Unknown quantizer_bits: {quantizer_bits}")

        # Validate and set representation
        if representation not in self.REPRESENTATIONS:
            raise ValueError(
                f"Invalid representation: {representation}. "
                f"Must be one of {self.REPRESENTATIONS}"
            )
        self.representation = representation

        # Initialize index storage for single representation
        self._index = None
        self.id_map = {}

        logger.info(
            f"Initialized IVFScalarQuantizer indexer: {self.index_path} "
            f"(representation={self.representation}, nlist={self.NLIST}, nprobe={self.nprobe}, "
            f"quantizer={quantizer_bits}, device={self._resolved_device()})"
        )

    def _resolved_device(self) -> str:
        """Return the device to use for FAISS GPU operations.

        Detects faiss-gpu by probing StandardGpuResources (only present in faiss-gpu,
        not faiss-cpu). Falls back to 'cpu' when faiss-gpu is not installed or when
        device is explicitly set to 'cpu'. Note: FAISS GPU is CUDA-only (no MPS).
        """
        if self.device != "auto":
            return self.device
        if getattr(faiss, "StandardGpuResources", None) is not None:
            return "cuda"
        return "cpu"

    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Convert raw answer document to indexer format."""
        return {
            "answer_id": raw_doc.get("id"),
            "parent_id": raw_doc.get("parent_id"),
            "formulas": raw_doc.get("formulas", []),
        }

    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build FAISS index for this representation.

        When formula_tsv_base_dir is set, uses a streaming TSV path that processes
        one TSV file at a time and checkpoints after each file, avoiding the ~70 GB
        peak memory of the bulk path.

        Args:
            force: If True, rebuild even if index exists
            limit: Maximum number of answers to load metadata for (for testing)
        """
        if force:
            self.force_rebuild = True

        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"

        # --- Streaming TSV path ---
        if self._tsv_dir is not None:
            if not self.force_rebuild:
                cp = self._load_tsv_checkpoint()
                all_tsv = sorted(self._tsv_dir.glob("*.tsv"), key=lambda p: int(p.stem))
                if all_tsv and set(cp.get("processed_tsv_files", [])) == {f.name for f in all_tsv}:
                    logger.info("Index already complete. Loading from disk...")
                    self._load_index_from_disk()
                    return

            if not self.embedding_dir or not self.embedding_dir.exists():
                raise ValueError(
                    f"Embedding directory not found: {self.embedding_dir}. "
                    "Please generate embeddings first using formula_embedder.py"
                )

            # Build formula_id → (answer_id, formula_id) mapping
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

            model_manager, tuple_tokenizer, tree_type = self._setup_model_and_tokenizer()
            self._build_faiss_index_from_tsv(fid_to_meta, model_manager, tuple_tokenizer, tree_type)
            logger.info("Streaming TSV indexing complete")
            return

        # --- Non-TSV (bulk) path ---
        if index_file.exists() and not self.force_rebuild:
            logger.info("Index already exists. Loading from disk...")
            self._load_index_from_disk()
            return

        logger.info(f"Building IVFScalarQuantizer index for {self.representation}...")

        formulas = []
        logger.info("Loading formulas from answers.jsonl...")
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
                    formula_id = formula_obj.get("formula_id")
                    latex = formula_obj.get("latex")
                    formulas.append(
                        {
                            "answer_id": answer_id,
                            "formula_id": formula_id,
                            "latex": latex,
                        }
                    )

        logger.info(f"Loaded {len(formulas)} formulas")

        if not self.embedding_dir or not self.embedding_dir.exists():
            raise ValueError(
                f"Embedding directory not found: {self.embedding_dir}. "
                "Please generate embeddings first using formula_embedder.py"
            )

        embeddings = self._load_embeddings(formulas)

        if embeddings is None or len(embeddings) == 0:
            raise ValueError(f"No embeddings found for {self.representation}")

        self._build_faiss_index(embeddings, formulas)
        self._save_index_to_disk()
        logger.info("Indexing complete")

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search index and return top-k hits for LaTeX query formula.

        Pipeline: LaTeX → tuples → encoded tokens → FastText embedding → FAISS search
        """
        try:
            query_embedding = self._generate_query_embedding(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding for '{query}': {e}")
            return []

        if self._index is None:
            self._load_index_from_disk()

        if self._index is None:
            logger.warning("Index not available")
            return []

        qvec = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self._index.search(qvec, k)  # type: ignore

        hits = []
        for rank, idx in enumerate(indices[0]):
            if idx in self.id_map:
                answer_id, formula_id = self.id_map[idx]
                dist = float(distances[0][rank])
                hits.append(
                    {
                        "doc_id": answer_id,
                        "formula_id": formula_id,
                        "score": 1.0 / (1.0 + dist),
                        "representation": self.representation,
                    }
                )

        return hits

    def batch_search(
        self,
        queries: list[tuple[str, str]],
        k: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch search across multiple queries."""
        results = {}
        for qid, query in tqdm(queries, desc=f"Batch searching ({self.representation})"):
            results[qid] = self.search(query, k)
        return results

    def embed_formula(self, latex: str) -> np.ndarray:
        """Embed a LaTeX formula string. FastText model is loaded lazily on first call."""
        return self._generate_query_embedding(latex)

    def get_representations(self) -> list[str]:
        """Return representation being indexed."""
        return [self.representation]

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @property
    def _tsv_dir(self) -> Optional[Path]:
        """Return the TSV directory for this representation, or None if not configured."""
        if not self.formula_tsv_base_dir:
            return None
        subdir = "opt_representation_v3" if self.representation == "opt" else "slt_representation_v3"
        tsv_dir = self.formula_tsv_base_dir / subdir
        return tsv_dir if tsv_dir.exists() else None

    def _setup_model_and_tokenizer(self) -> tuple[FastTextModelManager, TupleTokenizer, str]:
        """Load FastText model and build tokenizer for this representation.

        Returns:
            (model_manager, tuple_tokenizer, tree_type_str)
        """
        if not self.embedding_dir or not self.embedding_dir.exists():
            raise ValueError(f"Embedding directory not found: {self.embedding_dir}")

        tree_type_suffix = self.representation.lower().replace("-", "_")
        model_file = self.embedding_dir / tree_type_suffix / f"fasttext_model_{tree_type_suffix}.bin"
        metadata_file = self.embedding_dir / tree_type_suffix / f"training_metadata_{tree_type_suffix}.json"

        if not model_file.exists():
            raise FileNotFoundError(f"FastText model not found: {model_file}")

        logger.info(f"Loading FastText model for {self.representation}: {model_file}")
        model_manager = FastTextModelManager(
            model_path=str(model_file),
            metadata_path=str(metadata_file),
            corpus_path="",
        )
        model_manager.load()

        metadata = model_manager.get_stats()
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
            logger.warning("Encoder maps not found; token IDs will not match training vocabulary")
            token_id_manager = TokenIDManager()

        tuple_tokenizer = TupleTokenizer(
            token_id_manager=token_id_manager,
            embedding_type=embedding_type,
            tokenize_number=tokenize_number,
        )

        tree_type_mapping = {
            "slt": "SLT",
            "opt": "OPT",
            "slt_type": "SLT-TYPE",
        }
        tree_type_str = tree_type_mapping.get(self.representation, "SLT")

        return model_manager, tuple_tokenizer, tree_type_str

    # ---- Checkpoint helpers -------------------------------------------------

    def _tsv_checkpoint_path(self) -> Path:
        return self.index_path / f"tsv_checkpoint_{self.representation}.json"

    def _load_tsv_checkpoint(self) -> dict:
        cp = self._tsv_checkpoint_path()
        if not cp.exists():
            return {"processed_tsv_files": [], "indexed_count": 0, "trained": False}
        with open(cp) as f:
            return json.load(f)

    def _save_tsv_checkpoint(
        self, processed_files: list[str], indexed_count: int, trained: bool
    ) -> None:
        with open(self._tsv_checkpoint_path(), "w") as f:
            json.dump(
                {
                    "processed_tsv_files": processed_files,
                    "indexed_count": indexed_count,
                    "trained": trained,
                },
                f,
            )

    # ---- Streaming TSV indexing ----------------------------------------------

    def _collect_tsv_training_vectors(
        self,
        tsv_files: list[Path],
        model_manager: FastTextModelManager,
        tuple_tokenizer: TupleTokenizer,
        tree_type: str,
        train_size: int,
    ) -> np.ndarray:
        """Stream TSV files and collect up to train_size embedding vectors for FAISS training.

        Any TSV row with valid MathML is used — no fid_to_meta filtering needed here.
        """
        csv.field_size_limit(sys.maxsize)
        training_vectors: list[np.ndarray] = []

        for tsv_file in tqdm(tsv_files, desc=f"Collecting training vectors ({self.representation})"):
            if len(training_vectors) >= train_size:
                break
            with open(tsv_file, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    if len(training_vectors) >= train_size:
                        break
                    mathml = row.get("formula", "").strip()
                    if not mathml:
                        continue
                    try:
                        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore
                        if not tuples:
                            continue
                        encoded = encode_tuples(tuples, tuple_tokenizer)
                        if not encoded:
                            continue
                        vec = np.array(model_manager.get_sentence_vector(encoded), dtype=np.float32)
                        training_vectors.append(vec)
                    except Exception:
                        continue

        if len(training_vectors) < train_size:
            logger.warning(
                f"Only collected {len(training_vectors)} training vectors "
                f"(requested {train_size}); corpus may be smaller than expected."
            )

        return np.array(training_vectors, dtype=np.float32)

    def _build_faiss_index_from_tsv(
        self,
        fid_to_meta: dict[str, tuple[str, str]],
        model_manager: FastTextModelManager,
        tuple_tokenizer: TupleTokenizer,
        tree_type: str,
    ) -> None:
        """Stream TSV files to build FAISS index with per-file checkpoints.

        Peak memory: one mini-batch of embeddings in the add phase (~1.2 MB),
        plus training vectors (~3.4 GB, freed after train()).
        Checkpoints after every TSV file so indexing can resume if interrupted.
        """
        csv.field_size_limit(sys.maxsize)
        tsv_dir = self._tsv_dir
        if tsv_dir is None:
            raise ValueError(f"TSV directory not found under {self.formula_tsv_base_dir}")

        all_tsv_files = sorted(tsv_dir.glob("*.tsv"), key=lambda p: int(p.stem))

        cp = self._load_tsv_checkpoint()
        processed_set: set[str] = set(cp.get("processed_tsv_files", []))
        indexed_count: int = cp.get("indexed_count", 0)
        already_trained: bool = cp.get("trained", False)

        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"

        # Phase 1: Train (skip if checkpoint says already done)
        if not already_trained:
            train_size = min(
                max(len(fid_to_meta) // 10, self.NLIST * 40),
                len(fid_to_meta),
            )
            logger.info(f"Collecting {train_size:,} training vectors from TSV files...")
            training_vectors = self._collect_tsv_training_vectors(
                all_tsv_files, model_manager, tuple_tokenizer, tree_type, train_size
            )

            logger.info(f"Training FAISS index on {len(training_vectors):,} vectors...")
            quantizer = faiss.IndexFlat(self.DIMENSION)
            cpu_index = faiss.IndexIVFScalarQuantizer(
                quantizer, self.DIMENSION, self.NLIST, self.quantizer_type
            )

            if self._resolved_device() == "cuda":
                logger.info("Moving index to GPU for training...")
                res = getattr(faiss, "StandardGpuResources")()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # type: ignore[attr-defined]
                gpu_index.train(training_vectors)
                self._index = faiss.index_gpu_to_cpu(gpu_index)  # type: ignore[attr-defined]
                logger.info("Training complete; index moved back to CPU")
            else:
                cpu_index.train(training_vectors)  # type: ignore
                self._index = cpu_index

            self._index.nprobe = self.nprobe
            del training_vectors

            # Save trained (empty) index so resume can reload it
            faiss.write_index(self._index, str(index_file))
            self._save_tsv_checkpoint(list(processed_set), indexed_count, trained=True)
            logger.info("Training complete; empty index saved.")
        else:
            # Resume: reload the partially-built index
            logger.info(f"Resuming from checkpoint ({len(processed_set)} TSV files already done)...")
            self._index = faiss.read_index(str(index_file))
            self._index.nprobe = self.nprobe

            id_map_file = self.index_path / f"id_map_sq_{self.representation}.json"
            if id_map_file.exists():
                with open(id_map_file) as f:
                    raw = json.load(f)
                    self.id_map = {int(k): tuple(v) for k, v in raw.items()}

        # Phase 2: Add vectors TSV file by TSV file
        remaining = [f for f in all_tsv_files if f.name not in processed_set]
        logger.info(f"Adding vectors from {len(remaining)} remaining TSV files...")

        failed_count = 0

        for tsv_file in tqdm(remaining, desc=f"Indexing TSVs ({self.representation})"):
            batch_vectors: list[np.ndarray] = []
            batch_metas: list[tuple[str, str]] = []

            with open(tsv_file, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    fid = row.get("id", "")
                    if fid not in fid_to_meta:
                        continue
                    mathml = row.get("formula", "").strip()
                    if not mathml:
                        continue

                    try:
                        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore
                        if not tuples:
                            failed_count += 1
                            continue
                        encoded = encode_tuples(tuples, tuple_tokenizer)
                        if not encoded:
                            failed_count += 1
                            continue
                        vec = np.array(model_manager.get_sentence_vector(encoded), dtype=np.float32)
                        batch_vectors.append(vec)
                        batch_metas.append(fid_to_meta[fid])
                    except Exception as e:
                        logger.debug(f"Skipping formula_id={fid}: {e}")
                        failed_count += 1
                        continue

                    if len(batch_vectors) >= _TSV_MINI_BATCH:
                        base_idx = self._index.ntotal  # type: ignore
                        self._index.add(np.array(batch_vectors, dtype=np.float32))  # type: ignore
                        for i, meta in enumerate(batch_metas):
                            self.id_map[base_idx + i] = meta
                        indexed_count += len(batch_vectors)
                        batch_vectors = []
                        batch_metas = []

            # Flush remainder of this file
            if batch_vectors:
                base_idx = self._index.ntotal  # type: ignore
                self._index.add(np.array(batch_vectors, dtype=np.float32))  # type: ignore
                for i, meta in enumerate(batch_metas):
                    self.id_map[base_idx + i] = meta
                indexed_count += len(batch_vectors)

            # Checkpoint after each TSV file
            processed_set.add(tsv_file.name)
            self._save_index_to_disk()
            self._save_tsv_checkpoint(list(processed_set), indexed_count, trained=True)
            logger.info(
                f"Checkpoint: {tsv_file.name} done — {indexed_count:,} vectors indexed total"
            )

        logger.info(
            f"Streaming indexing complete: {indexed_count:,} vectors, "
            f"{failed_count} skipped (no valid tuples)"
        )

    # ---- Non-TSV (bulk) indexing --------------------------------------------

    def _build_faiss_index(
        self,
        embeddings: list[np.ndarray],
        formulas: list[dict[str, Any]],
    ) -> None:
        """Build FAISS IVFScalarQuantizer index from a pre-loaded embeddings list."""
        logger.info(f"Building index for {self.representation} ({len(embeddings)} vectors)")

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Sample for training (~10% or 2.8M, whichever is smaller).
        # Cap at len(embeddings) so np.random.choice never requests more samples than available.
        train_size = min(
            max(len(embeddings) // 10, self.NLIST * 40),
            len(embeddings),
        )
        training_indices = np.random.choice(len(embeddings), train_size, replace=False)
        training_vectors = embeddings_array[training_indices]

        logger.info(f"Training on {train_size} vectors...")

        quantizer = faiss.IndexFlat(self.DIMENSION)
        cpu_index = faiss.IndexIVFScalarQuantizer(
            quantizer, self.DIMENSION, self.NLIST, self.quantizer_type
        )

        # Train on GPU if available (10-50x faster for k-means with NLIST=16384)
        if self._resolved_device() == "cuda":
            logger.info("Moving index to GPU for training...")
            res = getattr(faiss, "StandardGpuResources")()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # type: ignore[attr-defined]
            gpu_index.train(training_vectors)
            self._index = faiss.index_gpu_to_cpu(gpu_index)  # type: ignore[attr-defined]
            logger.info("Training complete; index moved back to CPU")
        else:
            cpu_index.train(training_vectors)  # type: ignore
            self._index = cpu_index

        self._index.nprobe = self.nprobe

        # Add all vectors on CPU (28M × 300 × 4 bytes ≈ 33 GB — won't fit in VRAM)
        logger.info(f"Adding {len(embeddings)} vectors...")
        self._index.add(embeddings_array)  # type: ignore

        for idx, formula_data in enumerate(formulas):
            self.id_map[idx] = (
                formula_data["answer_id"],
                formula_data["formula_id"],
            )

        logger.info(
            f"Built {self.representation} index: {self._index.ntotal} vectors, "
            f"index size: {self._estimate_index_size(self._index) / 1e9:.2f} GB"
        )

    def _load_embeddings(self, formulas: list[dict]) -> list[np.ndarray]:
        """Generate embeddings for formulas via latexmlmath subprocess (slow path).

        Only used when formula_tsv_base_dir is not set.
        """
        model_manager, tuple_tokenizer, tree_type = self._setup_model_and_tokenizer()

        embeddings: list[np.ndarray] = []
        failed_count = 0
        mathml_type = "cmml" if self.representation == "opt" else "pmml"

        logger.info(f"Generating embeddings for {len(formulas)} formulas...")

        batch_size = 256
        with LatexToMathMLPool(num_workers=_get_optimal_workers(), mathml_type=mathml_type) as pool:
            for batch_start in tqdm(range(0, len(formulas), batch_size), desc=f"Embedding {self.representation}"):
                batch_formulas = formulas[batch_start : batch_start + batch_size]
                batch_tex = [formula_data.get("latex", "") for formula_data in batch_formulas]

                mathml_results = pool.convert_batch(batch_tex)

                for formula_data, mathml in zip(batch_formulas, mathml_results):
                    try:
                        if not formula_data.get("latex", ""):
                            embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                            continue

                        if not mathml:
                            embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                            failed_count += 1
                            continue

                        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore

                        if not tuples:
                            embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                            failed_count += 1
                            continue

                        encoded_sequence = encode_tuples(tuples, tuple_tokenizer)

                        if not encoded_sequence:
                            embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                            failed_count += 1
                            continue

                        embedding_list = model_manager.get_sentence_vector(encoded_sequence)
                        embedding = np.array(embedding_list, dtype=np.float32)
                        embeddings.append(embedding)

                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for formula: {e}")
                        embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                        failed_count += 1

        if failed_count > 0:
            logger.warning(
                f"Failed to generate {failed_count}/{len(formulas)} embeddings, "
                f"using zero vectors as fallback"
            )

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def _save_index_to_disk(self) -> None:
        """Save FAISS index and id_map to disk."""
        logger.info("Saving index to disk...")

        if self._index is None:
            logger.warning(f"No index to save for {self.representation}")
            return

        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"
        faiss.write_index(self._index, str(index_file))
        logger.info(f"Saved {self.representation} index to {index_file}")

        id_map_file = self.index_path / f"id_map_sq_{self.representation}.json"
        with open(id_map_file, "w") as f:
            json.dump(
                {str(k): v for k, v in self.id_map.items()},
                f,
            )
        logger.info(f"Saved ID map for {self.representation} to {id_map_file}")

    def _load_index_from_disk(self) -> None:
        """Load FAISS index and id_map from disk."""
        logger.info("Loading index from disk...")

        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"
        id_map_file = self.index_path / f"id_map_sq_{self.representation}.json"

        if not index_file.exists():
            logger.warning(f"Index file not found for {self.representation}: {index_file}")
            return

        if not id_map_file.exists():
            logger.warning(f"ID map file not found for {self.representation}: {id_map_file}")
            return

        self._index = faiss.read_index(str(index_file))
        self._index.nprobe = self.nprobe

        with open(id_map_file) as f:
            id_map_raw = json.load(f)
            self.id_map = {int(k): tuple(v) for k, v in id_map_raw.items()}

        logger.info(
            f"Loaded {self.representation} index: {self._index.ntotal} vectors, "
            f"nprobe={self.nprobe}"
        )

    def _generate_query_embedding(self, query_latex: str) -> np.ndarray:
        """Generate embedding for a query LaTeX formula."""
        if not hasattr(self, "_query_model_manager"):
            mm, tok, _ = self._setup_model_and_tokenizer()
            self._query_model_manager = mm
            self._query_tokenizer = tok

        tree_type_mapping = {
            "slt": "SLT",
            "opt": "OPT",
            "slt_type": "SLT-TYPE",
        }
        tree_type = tree_type_mapping.get(self.representation, "SLT")

        try:
            if self.representation == "opt":
                mathml = LatexToMathML.convert_to_mathml2(query_latex)  # CMML for OPT
            else:
                mathml = LatexToMathML.convert_to_mathml(query_latex)   # PMML for SLT/SLT-TYPE

            if not mathml:
                logger.debug(f"No MathML produced for query: {query_latex}")
                return np.zeros(self.DIMENSION, dtype=np.float32)

            tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore

            if not tuples:
                logger.debug(f"No tuples extracted from query: {query_latex}")
                return np.zeros(self.DIMENSION, dtype=np.float32)

            encoded_sequence = encode_tuples(tuples, self._query_tokenizer)

            if not encoded_sequence:
                logger.debug(f"Failed to encode tuples for query: {query_latex}")
                return np.zeros(self.DIMENSION, dtype=np.float32)

            embedding_list = self._query_model_manager.get_sentence_vector(encoded_sequence)
            return np.array(embedding_list, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error generating query embedding: {e}")
            return np.zeros(self.DIMENSION, dtype=np.float32)

    def _estimate_index_size(self, index: faiss.Index) -> int:
        """Estimate index size in bytes."""
        return index.ntotal * self.bytes_per_vector + 50_000_000
