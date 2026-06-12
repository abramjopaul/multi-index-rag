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
from multirag.formula_search.latex_mml import LatexToMathML, _get_optimal_workers
from multirag.formula_search.tuple_extraction import (
    encode_tuples,
)
from multirag.indexing.base import BaseIndexer

logger = logging.getLogger(__name__)


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
    NPROBE = 32  # Number of clusters to search. Sweep upward (32 → 64 → 128 → 256
    QUANTIZER_TYPE = faiss.ScalarQuantizer.QT_fp16  # 8-bit or 16-bit quantization

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path = ANSWERS_JSONL,
        embedding_dir: Optional[str] = None,
        representation: str = "slt",
        quantizer_bits: str = "fp16",
        nprobe: int = 64,
        force_rebuild: bool = False,
        formula_tsv_base_dir: Optional[str] = None,
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
            f"(representation={self.representation}, nlist={self.NLIST}, nprobe={self.nprobe}, quantizer={quantizer_bits})"
        )

    def prepare_document(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Convert raw answer document to indexer format.

        Args:
            raw_doc: Answer document from answers.jsonl

        Returns:
            Formatted document with extracted formulas
        """
        return {
            "answer_id": raw_doc.get("id"),
            "parent_id": raw_doc.get("parent_id"),
            "formulas": raw_doc.get("formulas", []),
        }

    def index(self, force: bool = False, limit: int | None = None) -> None:
        """Build FAISS index for this representation.

        Args:
            force: If True, rebuild even if index exists
            limit: Maximum number of answers to index (for testing)
        """
        if force:
            self.force_rebuild = True

        # Check if index already exists
        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"

        if index_file.exists() and not self.force_rebuild:
            logger.info("Index already exists. Loading from disk...")
            self._load_index_from_disk()
            return

        logger.info(f"Building IVFScalarQuantizer index for {self.representation}...")

        # Step 1: Load formulas
        formulas = []
        embeddings = None

        logger.info("Loading formulas from answers.jsonl...")
        with open(self.corpus_path) as f:
            for line_idx, line in enumerate(tqdm(f, desc="Loading formulas")):
                if limit and line_idx >= limit:
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

        # Step 2: Load pre-computed embeddings
        logger.info(f"Loading embeddings for {self.representation}...")
        if self.embedding_dir and self.embedding_dir.exists():
            embeddings = self._load_embeddings(formulas)
        else:
            raise ValueError(
                f"Embedding directory not found: {self.embedding_dir}. "
                "Please generate embeddings first using formula_embedder.py"
            )

        if embeddings is None or len(embeddings) == 0:
            raise ValueError(f"No embeddings found for {self.representation}")

        # Step 3: Build index
        self._build_faiss_index(embeddings, formulas)

        # Step 4: Save index to disk
        self._save_index_to_disk()
        logger.info("Indexing complete")

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search index and return top-k hits for LaTeX query formula.

        Pipeline: LaTeX → tuples → encoded tokens → FastText embedding → FAISS search

        Args:
            query: Query LaTeX formula string
            k: Number of results to return

        Returns:
            List of top-k hits with: doc_id, formula_id, score, representation
        """
        # Generate query embedding
        try:
            query_embedding = self._generate_query_embedding(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding for '{query}': {e}")
            return []

        # Load index if needed
        if self._index is None:
            self._load_index_from_disk()

        if self._index is None:
            logger.warning("Index not available")
            return []

        # Search
        qvec = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self._index.search(qvec, k)

        # Build results
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
        """Batch search across multiple queries.

        Args:
            queries: List of (qid, query_latex) tuples
            k: Number of results per query

        Returns:
            Dict mapping qid → list of top-k hits
        """
        results = {}
        for qid, query in tqdm(queries, desc=f"Batch searching ({self.representation})"):
            results[qid] = self.search(query, k)
        return results

    # def search_by_representation(
    #     self, query: str, k: int = 10, query_embedding: Optional[np.ndarray] = None
    # ) -> dict[str, list[dict[str, Any]]]:
    #     """Search index and return top-k results grouped by representation.

    #     Pipeline (if query_embedding not provided): LaTeX → embedding → search

    #     Args:
    #         query: Query LaTeX formula string (or identifier if query_embedding provided)
    #         k: Number of results
    #         query_embedding: Optional pre-computed query embedding vector. If None, will be generated from query

    #     Returns:
    #         Dict mapping representation name to list of hits
    #     """
    #     if self._index is None:
    #         self._load_index_from_disk()

    #     if self._index is None:
    #         logger.warning(f"Index for {self.representation} not available")
    #         return {self.representation: []}

    #     # Generate embedding if not provided
    #     if query_embedding is None:
    #         try:
    #             query_embedding = self._generate_query_embedding(query)
    #         except Exception as e:
    #             logger.warning(f"Failed to generate query embedding: {e}")
    #             return {self.representation: []}

    #     qvec = query_embedding.astype(np.float32).reshape(1, -1)

    #     # Search
    #     distances, indices = self._index.search(qvec, k)

    #     hits = []
    #     for rank, idx in enumerate(indices[0]):
    #         if idx in self.id_map:
    #             answer_id, formula_id = self.id_map[idx]
    #             hits.append(
    #                 {
    #                     "rank": rank + 1,
    #                     "answer_id": answer_id,
    #                     "formula_id": formula_id,
    #                     "distance": float(distances[0][rank]),
    #                     "representation": self.representation,
    #                 }
    #             )

    #     return {self.representation: hits}

    def get_representations(self) -> list[str]:
        """Return representation being indexed."""
        return [self.representation]

    # Private methods

    def _build_faiss_index(
        self,
        embeddings: list[np.ndarray],
        formulas: list[dict[str, Any]],
    ) -> None:
        """Build FAISS IVFScalarQuantizer index for this representation.

        Args:
            embeddings: List of embedding vectors
            formulas: List of formula metadata
        """
        logger.info(f"Building index for {self.representation} ({len(embeddings)} vectors)")

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Sample for training (~10% or 2.8M, whichever is smaller)
        # But ensure minimum NLIST * 40 samples for FAISS clustering requirement
        train_size = max(min(len(embeddings) // 10, 2_800_000), self.NLIST * 40)
        training_indices = np.random.choice(len(embeddings), train_size, replace=False)
        training_vectors = embeddings_array[training_indices]

        logger.info(f"Training on {train_size} vectors...")

        # Create IVFScalarQuantizer index
        # First create a flat index as the quantizer
        quantizer = faiss.IndexFlat(self.DIMENSION)
        # Then create the IVF index with scalar quantization
        # Constructor: (quantizer_index, dimension, nlist, quantizer_type)
        self._index = faiss.IndexIVFScalarQuantizer(
            quantizer, self.DIMENSION, self.NLIST, self.quantizer_type
        )
        self._index.nprobe = self.nprobe

        # Train
        self._index.train(training_vectors)

        # Add all vectors
        logger.info(f"Adding {len(embeddings)} vectors...")
        self._index.add(embeddings_array)

        # Build ID map
        for idx, formula_data in enumerate(formulas):
            self.id_map[idx] = (
                formula_data["answer_id"],
                formula_data["formula_id"],
            )

        logger.info(
            f"Built {self.representation} index: {self._index.ntotal} vectors, "
            f"index size: {self._estimate_index_size(self._index) / 1e9:.2f} GB"
        )

    def _load_embeddings(
        self, formulas: list[dict]
    ) -> list[np.ndarray]:
        """Generate embeddings for formulas using trained FastText model.
        
        Pipeline:
        1. Load trained FastText model for this representation
        2. Create TupleTokenizer with metadata from training
        3. For each formula (LaTeX):
           - Extract tuples: LaTeX → tuples
           - Encode tuples: tuples → Unicode tokens
           - Generate embedding: tokens → vector via FastText
        4. Return all embeddings
        
        Args:
            formulas: List of formula metadata with 'latex' field
            
        Returns:
            List of embedding vectors (np.ndarray of float32)
        """
        # Validate embedding_dir
        if not self.embedding_dir:
            raise ValueError("embedding_dir not configured")

        embedding_dir = Path(self.embedding_dir)

        # Setup model paths (organized in subdirectories: slt/, opt/, slt_type/)
        tree_type_suffix = self.representation.lower().replace("-", "_")
        model_file = embedding_dir / tree_type_suffix / f"fasttext_model_{tree_type_suffix}.bin"
        metadata_file = embedding_dir / tree_type_suffix / f"training_metadata_{tree_type_suffix}.json"

        # Verify model file exists
        if not model_file.exists():
            raise FileNotFoundError(f"FastText model not found: {model_file}")

        logger.info(f"Loading FastText model for {self.representation}: {model_file}")

        # Load model and metadata
        model_manager = FastTextModelManager(
            model_path=str(model_file),
            metadata_path=str(metadata_file),
            corpus_path="",  # Not needed for inference
        )
        model_manager.load()

        # Get tokenization configuration from metadata
        metadata = model_manager.get_stats()
        embedding_type_name = metadata.get("embedding_type", "Both_Separated")
        tokenize_number = metadata.get("tokenize_number", True)

        # Map embedding type name to enum
        embedding_type_map = {
            "Both_Separated": TupleTokenizationMode.Both_Separated,
            "Type": TupleTokenizationMode.Type,
        }
        embedding_type = embedding_type_map.get(
            embedding_type_name, TupleTokenizationMode.Both_Separated
        )

        # Create fresh TokenIDManager for tokenization
        token_id_manager = TokenIDManager()
        tuple_tokenizer = TupleTokenizer(
            token_id_manager=token_id_manager,
            embedding_type=embedding_type,
            tokenize_number=tokenize_number,
        )

        # Determine tree type for tuple extraction
        tree_type_mapping = {
            "slt": "SLT",
            "opt": "OPT",
            "slt_type": "SLT-TYPE",
        }
        tree_type_str = tree_type_mapping.get(self.representation, "SLT")
        tree_type = tree_type_str  # type: ignore

        # Fast path: use pre-computed MathML from TSV files (no subprocess)
        if self._tsv_dir is not None:
            logger.info(f"Using pre-computed MathML from {self._tsv_dir} (bypassing latexmlmath)")
            return self._load_embeddings_from_tsv(formulas, model_manager, tuple_tokenizer, tree_type)

        # Slow path: convert LaTeX → MathML via latexmlmath subprocess
        logger.warning(
            "formula_tsv_base_dir not set — falling back to latexmlmath subprocess. "
            "Set formula_tsv_base_dir='data/raw/collection/formula' to use pre-computed MathML."
        )
        embeddings: list[np.ndarray] = []
        failed_count = 0

        logger.info(f"Generating embeddings for {len(formulas)} formulas...")

        batch_size = 256
        for batch_start in tqdm(range(0, len(formulas), batch_size), desc=f"Embedding {self.representation}"):
            batch_formulas = formulas[batch_start : batch_start + batch_size]
            batch_tex = [formula_data.get("latex", "") for formula_data in batch_formulas]

            mathml_results = LatexToMathML.convert_batch2(batch_tex, num_workers=_get_optimal_workers())

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

    @property
    def _tsv_dir(self) -> Optional[Path]:
        """Return the TSV directory for this representation, or None if not configured."""
        if not self.formula_tsv_base_dir:
            return None
        subdir = "opt_representation_v3" if self.representation == "opt" else "slt_representation_v3"
        tsv_dir = self.formula_tsv_base_dir / subdir
        return tsv_dir if tsv_dir.exists() else None

    def _load_embeddings_from_tsv(
        self,
        formulas: list[dict],
        model_manager: FastTextModelManager,
        tuple_tokenizer: "TupleTokenizer",
        tree_type: str,
    ) -> list[np.ndarray]:
        """Generate embeddings by reading pre-computed MathML from TSV files.

        Streams through TSV files in numeric order and looks up each formula by id,
        calling extract_tuples_from_mathml_direct instead of latexmlmath.
        Only one TSV file (~56 MB) is held in memory at a time.

        Args:
            formulas: Formula metadata list with 'formula_id' and 'latex' fields
            model_manager: Loaded FastText model manager
            tuple_tokenizer: Configured TupleTokenizer
            tree_type: "SLT", "OPT", or "SLT-TYPE"

        Returns:
            List of 300-dim float32 embedding vectors (zero vector on failure)
        """
        csv.field_size_limit(sys.maxsize)
        tsv_dir = self._tsv_dir
        if tsv_dir is None:
            raise ValueError(f"TSV directory not found under {self.formula_tsv_base_dir}")

        # Build formula_id → position index for O(1) lookup
        fid_to_idx: dict[str, int] = {str(f["formula_id"]): i for i, f in enumerate(formulas)}

        embeddings: list[np.ndarray] = [np.zeros(self.DIMENSION, dtype=np.float32)] * len(formulas)
        failed_count = 0
        found_count = 0

        tsv_files = sorted(tsv_dir.glob("*.tsv"), key=lambda p: int(p.stem))
        logger.info(f"Streaming {len(tsv_files)} TSV files from {tsv_dir}")

        for tsv_file in tqdm(tsv_files, desc=f"TSV→embed ({self.representation})"):
            with open(tsv_file, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    fid = row["id"]
                    if fid not in fid_to_idx:
                        continue

                    idx = fid_to_idx[fid]
                    mathml = row["formula"].strip()

                    try:
                        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore
                        if not tuples:
                            failed_count += 1
                            continue

                        encoded_sequence = encode_tuples(tuples, tuple_tokenizer)
                        if not encoded_sequence:
                            failed_count += 1
                            continue

                        embedding_list = model_manager.get_sentence_vector(encoded_sequence)
                        embeddings[idx] = np.array(embedding_list, dtype=np.float32)
                        found_count += 1

                    except Exception as e:
                        logger.warning(f"Failed embedding for formula_id={fid}: {e}")
                        failed_count += 1

        logger.info(
            f"TSV embedding complete: {found_count} embedded, {failed_count} failed, "
            f"{len(formulas) - found_count - failed_count} not found in TSV"
        )
        return embeddings

    def _save_index_to_disk(self) -> None:
        """Save index and metadata to disk."""
        logger.info("Saving index to disk...")

        if self._index is None:
            logger.warning(f"No index to save for {self.representation}")
            return

        # Save FAISS index (use _sq suffix to distinguish from IVFFlat)
        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"
        faiss.write_index(self._index, str(index_file))
        logger.info(f"Saved {self.representation} index to {index_file}")

        # Save ID map
        id_map_file = self.index_path / f"id_map_sq_{self.representation}.json"
        with open(id_map_file, "w") as f:
            json.dump(
                {str(k): v for k, v in self.id_map.items()},
                f,
            )
        logger.info(f"Saved ID map for {self.representation} to {id_map_file}")

    def _load_index_from_disk(self) -> None:
        """Load index from disk."""
        logger.info("Loading index from disk...")

        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"
        id_map_file = self.index_path / f"id_map_sq_{self.representation}.json"

        if not index_file.exists():
            logger.warning(f"Index file not found for {self.representation}: {index_file}")
            return

        # Load FAISS index
        self._index = faiss.read_index(str(index_file))
        self._index.nprobe = self.nprobe

        # Load ID map
        with open(id_map_file) as f:
            id_map_raw = json.load(f)
            self.id_map = {int(k): tuple(v) for k, v in id_map_raw.items()}

        logger.info(
            f"Loaded {self.representation} index: {self._index.ntotal} vectors, "
            f"nprobe={self.nprobe}"
        )

    def _generate_query_embedding(self, query_latex: str) -> np.ndarray:
        """Generate embedding for a query LaTeX formula.

        Pipeline: LaTeX → tuples → encoded tokens → FastText embedding

        Args:
            query_latex: LaTeX formula string

        Returns:
            300-dim float32 embedding vector (zero vector on failure)
        """
        # Load model and tokenizer if not already cached
        if not hasattr(self, "_query_model_manager"):
            embedding_dir = Path(self.embedding_dir) if self.embedding_dir else None
            if not embedding_dir or not embedding_dir.exists():
                raise ValueError(f"Embedding directory not found: {embedding_dir}")

            tree_type_suffix = self.representation.lower().replace("-", "_")
            model_file = embedding_dir / tree_type_suffix / f"fasttext_model_{tree_type_suffix}.bin"
            metadata_file = embedding_dir / tree_type_suffix / f"training_metadata_{tree_type_suffix}.json"

            if not model_file.exists():
                raise FileNotFoundError(f"FastText model not found: {model_file}")

            # Load model and metadata
            self._query_model_manager = FastTextModelManager(
                model_path=str(model_file),
                metadata_path=str(metadata_file),
                corpus_path="",
            )
            self._query_model_manager.load()

            # Setup tokenizer
            metadata = self._query_model_manager.get_stats()
            embedding_type_name = metadata.get("embedding_type", "Both_Separated")
            tokenize_number = metadata.get("tokenize_number", True)

            embedding_type_map = {
                "Both_Separated": TupleTokenizationMode.Both_Separated,
                "Type": TupleTokenizationMode.Type,
            }
            embedding_type = embedding_type_map.get(
                embedding_type_name, TupleTokenizationMode.Both_Separated
            )

            token_id_manager = TokenIDManager()
            self._query_tokenizer = TupleTokenizer(
                token_id_manager=token_id_manager,
                embedding_type=embedding_type,
                tokenize_number=tokenize_number,
            )

        # Map representation to tree type
        tree_type_mapping = {
            "slt": "SLT",
            "opt": "OPT",
            "slt_type": "SLT-TYPE",
        }
        tree_type_str = tree_type_mapping.get(self.representation, "SLT")
        tree_type = tree_type_str  # type: ignore

        try:
            mathml = LatexToMathML.convert_to_mathml2(query_latex)

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
        # Vectors compressed to bytes_per_vector bytes + overhead
        return index.ntotal * self.bytes_per_vector + 50_000_000
