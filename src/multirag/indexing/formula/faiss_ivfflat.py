# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Formula FAISS indexer using IVFFlat quantization.
# Suitable for scenarios where accuracy is prioritized over memory.

import json
import logging
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from tqdm import tqdm

from multirag.config.path_configs import ANSWERS_JSONL
from multirag.embedding.formula_model_manager import FastTextModelManager
from multirag.formula_search import TokenIDManager, TupleTokenizationMode, TupleTokenizer
from multirag.formula_search.tuple_extraction import (
    encode_tuples,
    extract_tuples_from_latex_subprocess,
)
from multirag.indexing.base import BaseIndexer

logger = logging.getLogger(__name__)


class FormulaFAISSIndexerIVFFlat(BaseIndexer):
    """
    Formula indexer using FAISS IVFFlat quantization.

    Builds three parallel FAISS indices for SLT, OPT, and SLT_type representations.
    Uses Inverted File with flat (exact) distance computation.

    Memory: ~31.5 GB per representation (31.5 GB × 3 = ~94.5 GB total)
    Query time: 100-500ms depending on nprobe setting
    Accuracy: Exact distances, high recall

    Attributes:
        index_path: Path to store/load FAISS indices
        id_maps: Dict mapping representation → {idx: (answer_id, formula_id)}
        indices: Dict mapping representation → FAISS index
        embedder: Formula embedder for generating vectors
    """

    REPRESENTATIONS = ["slt", "opt", "slt_type"]
    DIMENSION = 300
    NLIST = 32  # Number of clusters (sqrt(28M) ≈ 5.3K, rounded to power of 2)
    NPROBE = 32  # Number of clusters to search (tune: 1-256)

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path = ANSWERS_JSONL, #type: ignore
        embedding_dir: Optional[str] = None,
        representation: str = "slt",
        nprobe: int = 64,
        force_rebuild: bool = False,
        device: str = "auto",
    ):
        """
        Initialize IVFFlat formula indexer for a single representation.

        Args:
            index_path: Directory to store/load FAISS indices
            corpus_path: Path to answers.jsonl
            embedding_dir: Directory containing pre-computed embeddings
            representation: Which representation to index: 'slt', 'opt', or 'slt_type'.
                           Defaults to 'slt'
            nprobe: Number of clusters to search (higher = more accurate but slower)
            force_rebuild: Force rebuild even if indices exist
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.corpus_path = Path(corpus_path)
        self.embedding_dir = Path(embedding_dir) if embedding_dir else None
        self.nprobe = nprobe
        self.force_rebuild = force_rebuild
        self.device = device

        # Validate and set representation
        if representation not in self.REPRESENTATIONS:
            raise ValueError(
                f"Invalid representation: {representation}. "
                f"Must be one of {self.REPRESENTATIONS}"
            )
        self.representation = representation

        # Initialize index storage for single representation
        self._faiss_index = None
        self.id_map = {}

        logger.info(
            f"Initialized IVFFlat indexer: {self.index_path} "
            f"(representation={self.representation}, nlist={self.NLIST}, nprobe={self.nprobe}, "
            f"device={self._resolved_device()})"
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
        """Build FAISS indices for all representations.

        Args:
            force: If True, rebuild even if indices exist
            limit: Maximum number of answers to index (for testing)
        """
        if force:
            self.force_rebuild = True

        # Check if index already exists
        index_file = self.index_path / f"formula_index_{self.representation}.faiss"
        if index_file.exists() and not self.force_rebuild:
            logger.info(f"Index for {self.representation} already exists. Loading from disk...")
            self._load_index()
            return

        logger.info(f"Building IVFFlat index for {self.representation}...")

        # Step 1: Load formulas and embeddings
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
        self._build_index(embeddings, formulas)

        # Step 4: Save index to disk
        self._save_index()
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
        if self._faiss_index is None:
            self._load_index()

        if self._faiss_index is None:
            logger.warning("Index not available")
            return []

        # Search
        qvec = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self._faiss_index.search(qvec, k) # type: ignore

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


    def get_representations(self) -> list[str]:
        """Return representation being indexed."""
        return [self.representation]

    # Private methods

    def _build_index(
        self, embeddings: list[np.ndarray], formulas: list[dict[str, Any]]
    ) -> None:
        """Build IVFFlat index for this representation.

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

        # Create IVFFlat index
        quantizer = faiss.IndexFlatL2(self.DIMENSION)
        cpu_index = faiss.IndexIVFFlat(quantizer, self.DIMENSION, self.NLIST)

        # Train on GPU if available
        if self._resolved_device() == "cuda":
            logger.info("Moving index to GPU for training...")
            res = getattr(faiss, "StandardGpuResources")()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # type: ignore[attr-defined]
            gpu_index.train(training_vectors)
            self._faiss_index = faiss.index_gpu_to_cpu(gpu_index)  # type: ignore[attr-defined]
            logger.info("Training complete; index moved back to CPU")
        else:
            cpu_index.train(training_vectors)  # type: ignore
            self._faiss_index = cpu_index

        self._faiss_index.nprobe = self.nprobe

        # Add all vectors on CPU (28M × 300 × 4 bytes ≈ 33 GB — won't fit in VRAM)
        logger.info(f"Adding {len(embeddings)} vectors...")
        self._faiss_index.add(embeddings_array)  # type: ignore

        # Build ID map
        for idx, formula_data in enumerate(formulas):
            self.id_map[idx] = (
                formula_data["answer_id"],
                formula_data["formula_id"],
            )

        logger.info(
            f"Built {self.representation} index: {self._faiss_index.ntotal} vectors, "
            f"index size: {self._estimate_index_size(self._faiss_index) / 1e9:.2f} GB"
        )

    def _load_embeddings(
        self, formulas: list[dict]
    ) -> list[np.ndarray]:
        """Generate embeddings for formulas using trained FastText model.
        
        Pipeline:
        1. Load trained FastText model for this representation
        2. Load encoder maps and create TupleTokenizer
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
        # (encoder_maps are for reference; we generate tokens fresh during inference)
        token_id_manager = TokenIDManager()
        tuple_tokenizer = TupleTokenizer(
            token_id_manager=token_id_manager,
            embedding_type=embedding_type,
            tokenize_number=tokenize_number,
        )
        
        # Determine tree type for tuple extraction (cast string to Literal)
        tree_type_mapping = {
            "slt": "SLT",
            "opt": "OPT",
            "slt_type": "SLT-TYPE",
        }
        tree_type_str = tree_type_mapping.get(self.representation, "SLT")
        tree_type = tree_type_str  # type: ignore
        
        embeddings = []
        failed_count = 0
        
        logger.info(f"Generating embeddings for {len(formulas)} formulas...")
        
        for formula_data in tqdm(formulas, desc=f"Embedding {self.representation}"):
            try:
                latex = formula_data.get("latex", "")
                if not latex:
                    # No LaTeX, use zero vector
                    embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                    continue
                
                # Step 1: LaTeX → Tuples
                tuples = extract_tuples_from_latex_subprocess(latex, tree_type=tree_type)  # type: ignore
                
                if not tuples:
                    # Tuple extraction failed, use zero vector
                    embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                    failed_count += 1
                    continue
                
                # Step 2: Tuples → Encoded tokens
                encoded_sequence = encode_tuples(tuples, tuple_tokenizer)
                
                if not encoded_sequence:
                    # Encoding failed, use zero vector
                    embeddings.append(np.zeros(self.DIMENSION, dtype=np.float32))
                    failed_count += 1
                    continue
                
                # Step 3: Encoded tokens → Embedding vector
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

    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        logger.info("Saving index to disk...")

        if self._faiss_index is None:
            logger.warning(f"No index to save for {self.representation}")
            return

        # Save FAISS index
        index_file = self.index_path / f"formula_index_{self.representation}.faiss"
        faiss.write_index(self._faiss_index, str(index_file))
        logger.info(f"Saved {self.representation} index to {index_file}")

        # Save ID map
        id_map_file = self.index_path / f"id_map_{self.representation}.json"
        with open(id_map_file, "w") as f:
            json.dump(
                {str(k): v for k, v in self.id_map.items()},
                f,
            )
        logger.info(f"Saved ID map for {self.representation} to {id_map_file}")

    def _load_index(self) -> None:
        """Load index from disk."""
        logger.info("Loading index from disk...")

        index_file = self.index_path / f"formula_index_{self.representation}.faiss"
        id_map_file = self.index_path / f"id_map_{self.representation}.json"

        if not index_file.exists():
            logger.warning(f"Index file not found for {self.representation}: {index_file}")
            return

        # Load FAISS index
        self._faiss_index = faiss.read_index(str(index_file))
        self._faiss_index.nprobe = self.nprobe

        # Load ID map
        with open(id_map_file) as f:
            id_map_raw = json.load(f)
            self.id_map = {int(k): tuple(v) for k, v in id_map_raw.items()}

        logger.info(
            f"Loaded {self.representation} index: {self._faiss_index.ntotal} vectors, "
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
            # Step 1: LaTeX → Tuples
            tuples = extract_tuples_from_latex_subprocess(query_latex, tree_type=tree_type)  # type: ignore

            if not tuples:
                logger.debug(f"No tuples extracted from query: {query_latex}")
                return np.zeros(self.DIMENSION, dtype=np.float32)

            # Step 2: Tuples → Encoded tokens
            encoded_sequence = encode_tuples(tuples, self._query_tokenizer)

            if not encoded_sequence:
                logger.debug(f"Failed to encode tuples for query: {query_latex}")
                return np.zeros(self.DIMENSION, dtype=np.float32)

            # Step 3: Encoded tokens → Embedding
            embedding_list = self._query_model_manager.get_sentence_vector(encoded_sequence)
            return np.array(embedding_list, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error generating query embedding: {e}")
            return np.zeros(self.DIMENSION, dtype=np.float32)

    @staticmethod
    def _estimate_index_size(index: faiss.Index) -> int:
        """Estimate index size in bytes."""
        # Rough estimate: 4 bytes per dimension per vector + overhead
        return index.ntotal * 300 * 4 + 50_000_000  # +50MB for overhead
