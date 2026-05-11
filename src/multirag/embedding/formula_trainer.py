# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Formula-to-Embedding conversion using FastText training.
# Trains FastText model on tokenized formula representations.

import json
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from multirag.config.path_configs import (
    FASTTEXT_MODEL_DIR,
    FORMULA_INDEX_DIR,
    LATEX_REPRESENTATION,
)
from multirag.formula_search import (
    OPTGenerator,
    SLTGenerator,
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
)
from multirag.formula_search.encoder_maps import save_maps
from multirag.utils.file_utils import (
    makedirs,
    path_exists,
    open_file,
    get_file_size,
    glob_files,
    delete_file,
    count_lines,
)

logger = logging.getLogger(__name__)


class TrainingProgressCallback(CallbackAny2Vec):
    """Callback to track FastText training progress with tqdm."""

    def __init__(self, epochs: int, corpus_file: str):
        """Initialize callback with epoch count and corpus file size."""
        self.epochs = epochs
        self.epoch = 0

        # Calculate corpus size (number of lines)
        with open_file(corpus_file, "r", encoding="utf-8") as f:
            self.corpus_size = sum(1 for _ in f)

        self.pbar = None

    def on_epoch_begin(self, model) -> None:
        """Called at start of each epoch."""
        self.epoch += 1
        total_words = (
            self.corpus_size * model.corpus_total_words
            if model.corpus_total_words
            else self.corpus_size
        )
        desc = f"Training (Epoch {self.epoch}/{self.epochs})"

        if self.pbar:
            self.pbar.close()

        self.pbar = tqdm(
            total=total_words, desc=desc, unit=" words", unit_scale=True, leave=True
        )

    def on_epoch_end(self, model) -> None:
        """Called at end of each epoch."""
        if self.pbar:
            self.pbar.update(self.pbar.total - self.pbar.n)  # Complete the bar
            self.pbar.close()

    def on_train_end(self, model) -> None:
        """Called when training finishes."""
        if self.pbar:
            self.pbar.close()

    def __getstate__(self) -> Dict:
        """Exclude pbar from pickling (tqdm with file handles can't be pickled)."""
        state = self.__dict__.copy()
        state["pbar"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        """Restore state, ensuring pbar is properly initialized."""
        self.__dict__.update(state)


class FormulaTrainer:
    """
    Trains FastText model on tokenized mathematical formulas using LineSentence format.

    Workflow:
    1. Load LaTeX formulas from TSV files in latex_representation_v3
    2. Generate tuples using SLT or OPT tree representations
    3. Encode tuples with TokenIDManager (saves encoder maps)
    4. Save encoded corpus in LineSentence format (whitespace-separated tokens)
    5. Train FastText model with gensim
    6. Store training metadata (files, formula count, tree type)
    """

    def __init__(
        self,
        tree_type: Literal["SLT", "OPT", "SLT-TYPE"] = "SLT",
        embedding_type: Optional[TupleTokenizationMode] = None,
        vector_size: int = 200,
        window: int = 5,
        min_n: int = 1,
        max_n: int = 100,
        negative: int = 5,
        output_dir: Optional[str] = None,
        use_process_pool: bool = True,
        num_workers: Optional[int] = 4,
        tokenize_number: Optional[bool] = None,
    ):
        """
        Initialize FormulaTrainer.

        Args:
            tree_type: "SLT" (Symbol Layout Tree), "OPT" (Operator Tree), or "SLT-TYPE" (SLT type-only)
            embedding_type: Node tokenization mode
                - None (auto): Both_Separated for SLT/OPT, Type for SLT-TYPE
                - Type: Extract only node types (for SLT-TYPE or custom)
                - Both_Separated: Extract type and value separately (for SLT/OPT)
            vector_size: FastText vector dimension (default: 200)
            window: Context window size (default: 5)
            min_n: Minimum n-gram size (default: 2)
            max_n: Maximum n-gram size (default: 100)
            negative: Number of negative samples (default: 5)
            output_dir: Output directory (default: data/formula-indexing)
            use_process_pool: Use process pool for parallel LaTeX to MathML conversion (default: True)
            num_workers: Number of worker processes (default: 4, None = auto-tune)
            tokenize_number: Whether to split numeric values into individual digits
                - None (auto): True for SLT, False for OPT/SLT-TYPE
                - SLT (True): Layout cares about digit positioning
                - OPT (False): Operations care about number identity
        """
        self.tree_type = tree_type.upper()
        if self.tree_type not in {"SLT", "OPT", "SLT-TYPE"}:
            raise ValueError(f"tree_type must be SLT, OPT, or SLT-TYPE, got {tree_type}")

        # Auto-configure embedding_type based on tree_type if not provided
        if embedding_type is None:
            if self.tree_type == "SLT-TYPE":
                self.embedding_type = TupleTokenizationMode.Type
            else:  # SLT or OPT
                self.embedding_type = TupleTokenizationMode.Both_Separated
        else:
            self.embedding_type = embedding_type
        
        self.vector_size = vector_size
        self.window = window
        self.min_n = min_n
        self.max_n = max_n
        self.negative = negative
        self.use_process_pool = use_process_pool

        # Set tokenize_number based on tree_type if not explicitly provided
        if tokenize_number is None:
            self.tokenize_number = self.tree_type == "SLT"  # True for SLT, False for OPT/SLT-TYPE
        else:
            self.tokenize_number = tokenize_number
        
        logger.info(
            f"Initialized with tree_type={self.tree_type}, embedding_type={self.embedding_type.name}, tokenize_number={self.tokenize_number}"
        )

        # Auto-tune num_workers based on CPU count if not provided
        if num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.num_workers = min(
                cpu_count, 8
            )  # Cap at 8 threads for I/O-bound operations
            logger.info(
                f"Auto-tuned num_workers to {self.num_workers} (CPU cores: {cpu_count})"
            )
        else:
            self.num_workers = num_workers

        # Setup output directories
        self.output_dir = FORMULA_INDEX_DIR
        makedirs(str(self.output_dir), exist_ok=True)

        # Artifact paths (use lowercase tree_type with hyphens replaced by underscores)
        tree_type_suffix = self.tree_type.lower().replace("-", "_")
        self.encoder_maps_path = self.output_dir / f"encoder_maps_{tree_type_suffix}.tsv"
        self.corpus_path = FASTTEXT_MODEL_DIR / f"corpus_{tree_type_suffix}.txt"  # LineSentence format
        self.model_path = FASTTEXT_MODEL_DIR / f"fasttext_model_{tree_type_suffix}.bin"
        self.metadata_path = FASTTEXT_MODEL_DIR / f"training_metadata_{tree_type_suffix}.json"
        self.checkpoint_path = FASTTEXT_MODEL_DIR / f"checkpoint_{tree_type_suffix}.json"  # Batch training checkpoint

        # Initialize generators and tokenizer
        self.slt_generator = SLTGenerator()
        self.opt_generator = OPTGenerator()

        # Token ID manager
        self.token_id_manager = TokenIDManager()
        self.tuple_tokenizer = TupleTokenizer(
            token_id_manager=self.token_id_manager,
            embedding_type=self.embedding_type,
            tokenize_number=self.tokenize_number,
        )

        self.model: Optional[FastText] = None
        self.training_stats: Dict = {}
        self.used_files: List[str] = []
        self.num_formulas_loaded = 0

    def load_latex_formulas(
        self,
        file_numbers: Optional[List[int]] = None,
        num_formulas: Optional[int] = None,
        start_file_number: Optional[int] = None,
        start_file_row_index: int = 0,
    ) -> pd.DataFrame:
        """
        Load LaTeX formulas from TSV files in latex_representation_v3 directory.

        Args:
            file_numbers: List of file numbers to load (e.g., [1, 2, 3]).
                         If None, loads all files.
            num_formulas: Maximum number of formulas to load across all files.
                         If None, loads everything.
            start_file_number: Resume from specific file number (for batch training).
                             If provided, starts from this file and skips file_numbers parameter.
            start_file_row_index: Row index within start_file_number to begin from (default: 0).

        Returns:
            DataFrame with loaded formulas

        Example:
            >>> trainer = FormulaTrainer(tree_type="SLT")
            >>> df = trainer.load_latex_formulas(file_numbers=[1, 2], num_formulas=5000)
            >>> # Resume from middle of file 4:
            >>> df = trainer.load_latex_formulas(start_file_number=4, start_file_row_index=150, num_formulas=10000)
        """
        latex_dir = LATEX_REPRESENTATION
        if not path_exists(str(latex_dir)):
            raise FileNotFoundError(
                f"LaTeX representation directory not found: {latex_dir}"
            )

        logger.info(f"Loading LaTeX formulas from {latex_dir}")

        # Determine which files to load
        if start_file_number is not None:
            # Resume from specific file number (for batch training)
            all_files = glob_files(str(latex_dir), "*.tsv")
            all_files = sorted(all_files, key=lambda x: int(Path(x).stem) if not str(x).startswith('gs://') else int(str(x).split('/')[-1].replace('.tsv', '')))
            file_nums = [int(Path(f).stem) if not str(f).startswith('gs://') else int(str(f).split('/')[-1].replace('.tsv', '')) for f in all_files]
            start_idx = file_nums.index(start_file_number) if start_file_number in file_nums else 0
            files = all_files[start_idx:]
        elif file_numbers is None:
            # Load all files
            files = glob_files(str(latex_dir), "*.tsv")
            files = sorted(files, key=lambda x: int(Path(x).stem) if not str(x).startswith('gs://') else int(str(x).split('/')[-1].replace('.tsv', '')))
        else:
            # Load specific files
            files = []
            for num in file_numbers:
                file_path = str(latex_dir / f"{num}.tsv")
                if not path_exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                else:
                    files.append(file_path)
            files = sorted(files, key=lambda x: int(Path(x).stem) if not str(x).startswith('gs://') else int(str(x).split('/')[-1].replace('.tsv', '')))

        if not files:
            raise ValueError(f"No files found to load")

        logger.info(f"Found {len(files)} files to load")

        all_formulas = []
        total_loaded = 0
        is_first_file = True

        with tqdm(
            total=num_formulas or sum(1 for f in files),
            desc="Loading LaTeX",
            unit="formula",
        ) as pbar:
            for file_path in files:
                if num_formulas and total_loaded >= num_formulas:
                    break

                try:
                    df = pd.read_csv(file_path, sep="\t", dtype=str)

                    # Skip rows if resuming from middle of file
                    if is_first_file and start_file_row_index > 0:
                        df = df.iloc[start_file_row_index:]
                        logger.info(f"Resuming from row {start_file_row_index} in {file_path.name}")
                        is_first_file = False
                    elif is_first_file:
                        is_first_file = False

                    # Limit if needed
                    if num_formulas:
                        remaining = num_formulas - total_loaded
                        if len(df) > remaining:
                            df = df.head(remaining)

                    all_formulas.append(df)
                    # Extract filename from both local Path and GCS string paths
                    filename = Path(file_path).name if not str(file_path).startswith('gs://') else str(file_path).split('/')[-1]
                    self.used_files.append(filename)
                    total_loaded += len(df)
                    pbar.update(len(df))

                    logger.info(f"Loaded {len(df)} formulas from {filename}")

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        if not all_formulas:
            raise ValueError(f"No formulas loaded from {latex_dir}")

        combined_df = pd.concat(all_formulas, ignore_index=True)
        self.num_formulas_loaded = len(combined_df)

        logger.info(f"Total formulas loaded: {self.num_formulas_loaded}")
        logger.info(f"Files used: {', '.join(self.used_files)}")

        return combined_df

    def generate_tuples(self, latex: str) -> List[str]:
        """
        Generate tuples from LaTeX formula using SLT or OPT.

        Note: SLT-TYPE also uses SLT tree generation, with Type-only tokenization.

        Args:
            latex: LaTeX formula string

        Returns:
            List of tab-separated tuples (or empty list if failed)
        """
        try:
            if self.tree_type == "OPT":
                tree = self.opt_generator.generate(latex)
            else:  # SLT or SLT-TYPE both use SLT trees
                tree = self.slt_generator.generate(latex)

            if tree is None:
                return []

            # Get tuples with window=2 and end-of-block marker
            tuples = tree.get_pairs(window=2, eob=True)
            return tuples if tuples else []

        except Exception as e:
            logger.debug(f"Error generating {self.tree_type} tuples from LaTeX: {e}")
            return []

    def encode_tuples(self, tuples: List[str]) -> str:
        """
        Encode a list of tuples into a whitespace-separated token string.

        Args:
            tuples: List of tab-separated tuples

        Returns:
            Whitespace-separated encoded tokens (one token per tuple)
        """
        encoded_tokens = []

        for tuple_str in tuples:
            encoded = self.tuple_tokenizer.tokenize_tuple(tuple_str)
            if encoded:
                encoded_tokens.append(encoded)

        return " ".join(encoded_tokens)  # Whitespace-separated for LineSentence

    def process_formulas_batch(
        self,
        formulas_df: pd.DataFrame,
        formula_column: str = "formula",
    ) -> List[str]:
        """
        Process batch of formulas: generate tuples and encode them.

        Uses parallel processing for tree generation if enabled.

        Args:
            formulas_df: DataFrame with formulas
            formula_column: Column name containing LaTeX strings

        Returns:
            List of encoded sequences (one per formula)
        """
        logger.info(
            f"Processing {len(formulas_df)} formulas ({self.tree_type} trees)..."
        )

        if self.use_process_pool:
            return self._process_formulas_batch_parallel(formulas_df, formula_column)
        else:
            return self._process_formulas_batch_sequential(formulas_df, formula_column)

    def _process_formulas_batch_sequential(
        self,
        formulas_df: pd.DataFrame,
        formula_column: str = "formula",
    ) -> List[str]:
        """Sequential processing (original method)."""
        encoded_sequences = []
        successful = 0
        failed = 0

        with tqdm(total=len(formulas_df), desc="Processing", unit="formula") as pbar:
            for _, row in formulas_df.iterrows():
                latex = row[formula_column]

                if pd.isna(latex):
                    pbar.update(1)
                    failed += 1
                    continue

                # Generate tuples
                tuples = self.generate_tuples(latex)

                if tuples:
                    # Encode tuples
                    encoded = self.encode_tuples(tuples)
                    if encoded:
                        encoded_sequences.append(encoded)
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1

                pbar.update(1)

        logger.info(f"Processed: {successful} successful, {failed} failed")
        return encoded_sequences

    def _process_formulas_batch_parallel(
        self,
        formulas_df: pd.DataFrame,
        formula_column: str = "formula",
    ) -> List[str]:
        """
        Parallel processing using thread pool for tree generation.

        Strategy: Use ThreadPoolExecutor to parallelize tree generation calls.
        Since generate_tuples() calls latexmlmath internally, this parallelizes
        the I/O-bound subprocess calls without bootstrapping issues.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        latex_formulas_with_idx = [
            (idx, row[formula_column])
            for idx, (_, row) in enumerate(formulas_df.iterrows())
            if not pd.isna(row[formula_column])
        ]

        if not latex_formulas_with_idx:
            logger.info("No valid formulas to process")
            return []

        # Use ThreadPoolExecutor to parallelize tree generation
        num_workers = self.num_workers
        logger.info(
            f"Phase 1: Generating {self.tree_type} trees in parallel ({num_workers} workers)..."
        )
        logger.info(
            f"  Sample LaTeX formulas: {[latex for _, latex in latex_formulas_with_idx[:3]]}"
        )

        results = {}  # idx -> encoded_sequence
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tree generation tasks
            future_to_idx = {
                executor.submit(self.generate_tuples, latex): idx
                for idx, latex in latex_formulas_with_idx
            }

            # Collect results as they complete
            with tqdm(
                total=len(future_to_idx), desc="Processing", unit="formula"
            ) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        tuples = future.result()
                        if tuples:
                            # Encode tuples
                            encoded = self.encode_tuples(tuples)
                            if encoded:
                                results[idx] = encoded
                                successful += 1
                            else:
                                failed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.debug(f"Error generating tuples for formula {idx}: {e}")
                        failed += 1

                    pbar.update(1)

        # Return results in original order
        encoded_sequences = [results[idx] for idx in sorted(results.keys())]
        logger.info(f"Phase 2: Completed. {successful} successful, {failed} failed")

        return encoded_sequences

    def save_corpus_and_maps(self, encoded_sequences: List[str]) -> None:
        """
        Save encoded sequences in LineSentence format and encoder maps.

        Args:
            encoded_sequences: List of whitespace-separated token strings
        """
        # Save corpus in LineSentence format
        logger.info(f"Saving {len(encoded_sequences)} sequences to {self.corpus_path}")
        makedirs(str(Path(str(self.corpus_path)).parent), exist_ok=True)

        with open_file(str(self.corpus_path), "w", encoding="utf-8") as f:
            for seq in encoded_sequences:
                f.write(seq + "\n")

        corpus_size_kb = get_file_size(str(self.corpus_path)) / 1024
        logger.info(f"Corpus saved: {corpus_size_kb:.1f} KB")

        # Save encoder maps
        logger.info(f"Saving encoder maps to {self.encoder_maps_path}")
        node_map, edge_map = self.token_id_manager.get_updates()
        save_maps(node_map, edge_map, str(self.encoder_maps_path))
        logger.info(f"Encoder maps: {len(node_map)} nodes, {len(edge_map)} edges")

    def train(
        self,
        file_numbers: Optional[List[int]] = None,
        num_formulas: Optional[int] = None,
        formula_column: str = "formula",
    ) -> FastText:
        """
        Complete training pipeline: load → generate tuples → encode → train FastText.

        Args:
            file_numbers: List of TSV file numbers to load (e.g., [1, 2, 3])
            num_formulas: Maximum number of formulas to load
            formula_column: Column name containing LaTeX strings

        Returns:
            Trained FastText model

        Example:
            >>> trainer = FormulaTrainer(tree_type="SLT", vector_size=200)
            >>> model = trainer.train(file_numbers=[1, 2], num_formulas=50000)
        """
        # Set all random seeds for reproducibility
        import random
        import numpy as np
        
        random.seed(42)
        np.random.seed(42)
        logger.info("=" * 70)
        logger.info(f"FASTTEXT TRAINING PIPELINE ({self.tree_type})")
        logger.info(f"Embedding type: {self.embedding_type.name}, Tokenize number: {self.tokenize_number}")
        logger.info("=" * 70)

        print(f"Number of cpus: {self.num_workers}")

        # Phase 1: Load formulas
        logger.info("\nPhase 1: Loading LaTeX Formulas")
        formulas_df = self.load_latex_formulas(
            file_numbers=file_numbers,
            num_formulas=num_formulas,
        )

        # Phase 2: Generate tuples and encode
        logger.info(f"\nPhase 2: Generating {self.tree_type} Tuples and Encoding")
        encoded_sequences = self.process_formulas_batch(formulas_df, formula_column)

        if not encoded_sequences:
            raise ValueError("No sequences to train on (all formulas failed)")

        # Phase 3: Save corpus and maps
        logger.info("\nPhase 3: Saving Corpus and Encoder Maps")
        self.save_corpus_and_maps(encoded_sequences)

        # Phase 4: Train FastText
        logger.info("\nPhase 4: Training FastText Model")
        logger.info(f"  Tree type: {self.tree_type}")
        logger.info(f"  Embedding type: {self.embedding_type.name}")
        logger.info(f"  Tokenize number: {self.tokenize_number}")
        logger.info(f"  Vector size: {self.vector_size}")
        logger.info(f"  Window: {self.window}")
        logger.info(f"  Min n-gram: {self.min_n}")
        logger.info(f"  Max n-gram: {self.max_n}")
        logger.info(f"  Negative: {self.negative}")

        # Train with progress callback
        epochs = 5
        callback = TrainingProgressCallback(
            epochs=epochs, corpus_file=str(self.corpus_path)
        )

        # Train with LineSentence corpus file
        self.model = FastText(
            corpus_file=str(self.corpus_path),
            vector_size=self.vector_size,
            window=self.window,
            min_n=self.min_n,
            max_n=self.max_n,
            negative=self.negative,
            sg=0,  # CBOW (0) or Skip-gram (1)
            seed=42,
            epochs=epochs,
            callbacks=[callback],  # Add progress callback
        )

        logger.info("Training complete")

        # Phase 5: Save model and metadata
        logger.info("\nPhase 5: Saving Model and Metadata")
        self._save_model()
        self._save_metadata(encoded_sequences)

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Model saved to: {self.model_path}")
        logger.info(f"Corpus saved to: {self.corpus_path}")
        logger.info(f"Encoder maps saved to: {self.encoder_maps_path}")
        logger.info(f"Metadata saved to: {self.metadata_path}")

        return self.model

    def _save_model(self) -> None:
        """Save trained FastText model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        makedirs(str(Path(str(self.model_path)).parent), exist_ok=True)
        self.model.save(str(self.model_path))
        logger.info(f"Model saved to: {self.model_path}")

    def _save_metadata(self, encoded_sequences: List[str]) -> None:
        """
        Save training metadata to model object and JSON file.

        Stores metadata in two ways:
        1. Attached to model object as model.meta (for easy access)
        2. Saved to JSON file as backup (for persistence after save/load)

        Metadata includes:
        - Files used
        - Number of formulas
        - Tree type (SLT/OPT)
        - Training parameters
        - Corpus stats
        - Processing configuration
        """
        self.training_stats = {
            "tree_type": self.tree_type,
            "embedding_type": self.embedding_type.name,
            "tokenize_number": self.tokenize_number,
            "files_used": self.used_files,
            "num_formulas_loaded": self.num_formulas_loaded,
            "num_sequences_encoded": len(encoded_sequences),
            "vector_size": self.vector_size,
            "window": self.window,
            "negative": self.negative,
            "model_path": str(self.model_path),
            "corpus_path": str(self.corpus_path),
            "encoder_maps_path": str(self.encoder_maps_path),
            "processing": {
                "use_process_pool": self.use_process_pool,
                "num_workers": self.num_workers if self.use_process_pool else None,
                "processing_method": (
                    "parallel" if self.use_process_pool else "sequential"
                ),
            },
        }

        # Attach metadata directly to model object for easy access
        if self.model is not None:
            try:
                self.model.meta = self.training_stats.copy()
                logger.info("Metadata attached to model.meta")
            except Exception as e:
                logger.warning(f"Could not attach metadata to model: {e}")

        # Also save to JSON file as backup (in case meta is lost during save/load)
        makedirs(str(Path(str(self.metadata_path)).parent), exist_ok=True)
        with open_file(str(self.metadata_path), "w") as f:
            json.dump(self.training_stats, f, indent=2)

        logger.info(f"Metadata saved to: {self.metadata_path}")

    def load_model(self, model_path: Optional[str] = None) -> FastText:
        """
        Load a pre-trained FastText model.

        Args:
            model_path: Path to model file. If None, uses default path for current tree_type.

        Returns:
            Loaded FastText model
        """
        model_path = model_path or str(self.model_path)

        if not path_exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = FastText.load(model_path)
        logger.info(f"Model loaded from: {model_path}")

        # Load metadata if it exists
        self._load_metadata()

        return self.model

    def _load_metadata(self) -> None:
        """
        Load training metadata from model or JSON file.

        Priority:
        1. Check if model has .meta attribute (direct attachment)
        2. Fall back to JSON file if available
        3. Log warning if neither exists
        """
        # Try to get metadata from model object first
        if self.model is not None and hasattr(self.model, "meta"):
            try:
                self.training_stats = self.model.meta
                logger.info("Metadata loaded from model.meta")
                return
            except Exception as e:
                logger.warning(f"Could not read model.meta: {e}")

        # Fall back to JSON file
        if path_exists(str(self.metadata_path)):
            try:
                with open_file(str(self.metadata_path), "r") as f:
                    self.training_stats = json.load(f)
                logger.info(f"Metadata loaded from: {self.metadata_path}")
            except Exception as e:
                logger.warning(f"Could not load metadata from JSON: {e}")

    def get_sentence_vector(self, encoded_sequence: str) -> list:
        """
        Get embedding vector for an encoded sequence.

        Args:
            encoded_sequence: Whitespace-separated encoded tokens

        Returns:
            List representing the embedding vector
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        vector = self.model.get_sentence_vector(encoded_sequence)
        return vector.tolist()

    def _load_checkpoint(self) -> Dict:
        """
        Load batch training checkpoint.

        Returns:
            Checkpoint dict with keys: completed_files, last_file_number, last_file_row_index,
            total_formulas_processed, batches_completed, total_formulas_available

        Example checkpoint structure:
            {
                "tree_type": "SLT",
                "completed_files": [1, 2, 3],
                "last_file_number": 4,
                "last_file_row_index": 150,
                "total_formulas_processed": 10150,
                "batches_completed": 2,
                "total_formulas_available": 100000
            }
        """
        if not path_exists(str(self.checkpoint_path)):
            logger.info(f"No checkpoint found, starting fresh: {self.checkpoint_path}")
            return {
                "tree_type": self.tree_type,
                "completed_files": [],
                "last_file_number": None,
                "last_file_row_index": 0,
                "total_formulas_processed": 0,
                "batches_completed": 0,
                "total_formulas_available": 0,
            }

        try:
            with open_file(str(self.checkpoint_path), "r") as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint loaded: {self.checkpoint_path}")
            logger.info(
                f"  Batches completed: {checkpoint.get('batches_completed', 0)}"
            )
            logger.info(
                f"  Formulas processed: {checkpoint.get('total_formulas_processed', 0)}"
            )
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {
                "tree_type": self.tree_type,
                "completed_files": [],
                "last_file_number": None,
                "last_file_row_index": 0,
                "total_formulas_processed": 0,
                "batches_completed": 0,
                "total_formulas_available": 0,
            }

    def _save_checkpoint(
        self,
        completed_files: List[int],
        last_file_number: Optional[int],
        last_file_row_index: int,
        total_formulas_processed: int,
        batches_completed: int,
        total_formulas_available: int,
    ) -> None:
        """
        Save batch training checkpoint.

        Args:
            completed_files: List of fully processed file numbers
            last_file_number: Current file number being processed
            last_file_row_index: Row index in current file
            total_formulas_processed: Total formulas processed across all batches
            batches_completed: Number of batches completed
            total_formulas_available: Total formulas available in dataset
        """
        checkpoint = {
            "tree_type": self.tree_type,
            "completed_files": completed_files,
            "last_file_number": last_file_number,
            "last_file_row_index": last_file_row_index,
            "total_formulas_processed": total_formulas_processed,
            "batches_completed": batches_completed,
            "total_formulas_available": total_formulas_available,
        }

        makedirs(str(Path(str(self.checkpoint_path)).parent), exist_ok=True)
        with open_file(str(self.checkpoint_path), "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved: {self.checkpoint_path}")
        logger.info(f"  Batches completed: {batches_completed}")
        logger.info(f"  Formulas processed: {total_formulas_processed}")

    def train_batch(
        self,
        batch_size: int = 30000,
        epochs: int = 5,
        formula_column: str = "formula",
    ) -> FastText:
        """
        Train FastText model on a batch of formulas with checkpoint resuming.

        Useful for training in Google Colab with limited compute. Each batch:
        1. Loads checkpoint to know where to resume from
        2. Loads next batch_size formulas
        3. Loads existing model (if available) or creates new one
        4. Trains on batch
        5. Saves model (overwrites) and updates checkpoint

        Args:
            batch_size: Number of formulas per batch (default: 30000)
            epochs: Training epochs per batch (default: 5)
            formula_column: Column name containing LaTeX strings

        Returns:
            Trained FastText model

        Example:
            >>> trainer = FormulaTrainer(tree_type="SLT")
            >>> # First batch
            >>> model = trainer.train_batch(batch_size=30000)
            >>> # Next batch (automatically resumes from checkpoint)
            >>> model = trainer.train_batch(batch_size=30000)
            >>> # If interrupted mid-file, next call resumes from exact row
        """
        # Set random seeds for reproducibility
        import random

        random.seed(42)
        np.random.seed(42)

        logger.info("=" * 70)
        logger.info("BATCH TRAINING ({self.tree_type})")
        logger.info("=" * 70)

        # Load checkpoint
        checkpoint = self._load_checkpoint()
        batches_completed = checkpoint["batches_completed"]
        total_formulas_processed = checkpoint["total_formulas_processed"]
        last_file_number = checkpoint["last_file_number"]
        last_file_row_index = checkpoint["last_file_row_index"]

        logger.info(f"\nBatch #{batches_completed + 1}")
        logger.info(f"  Previous batches: {batches_completed}")
        logger.info(f"  Total formulas processed: {total_formulas_processed}")

        # Load checkpoint to find total available formulas (on first batch)
        total_formulas_available = checkpoint.get("total_formulas_available", 0)
        if total_formulas_available == 0:
            # Count all available formulas
            latex_dir = LATEX_REPRESENTATION
            all_files = glob_files(str(latex_dir), "*.tsv")
            all_files = sorted(all_files, key=lambda x: int(Path(x).stem) if not str(x).startswith('gs://') else int(str(x).split('/')[-1].replace('.tsv', '')))
            total_formulas_available = sum(count_lines(f) for f in all_files)
            logger.info(f"Total formulas available: {total_formulas_available}")

        # Determine starting point
        start_file_number = last_file_number if last_file_number is not None else 1
        start_file_row_index = (
            last_file_row_index if last_file_number is not None else 0
        )

        # Phase 1: Load formulas
        logger.info(f"\nPhase 1: Loading {batch_size} formulas")
        formulas_df = self.load_latex_formulas(
            start_file_number=start_file_number,
            start_file_row_index=start_file_row_index,
            num_formulas=batch_size,
        )
        batch_size_actual = len(formulas_df)
        logger.info(f"Loaded {batch_size_actual} formulas")

        # Phase 2: Generate tuples and encode
        logger.info(f"\nPhase 2: Generating {self.tree_type} Tuples and Encoding")
        encoded_sequences = self.process_formulas_batch(formulas_df, formula_column)

        if not encoded_sequences:
            raise ValueError("No sequences to train on (all formulas failed)")

        # Phase 3: Save corpus (temporary, for this batch only)
        logger.info("\nPhase 3: Saving Batch Corpus")
        batch_corpus_path = (
            str(self.corpus_path).rsplit('.', 1)[0] + f"_batch_{batches_completed + 1}.txt"
        )
        makedirs(str(Path(batch_corpus_path).parent), exist_ok=True)

        with open_file(batch_corpus_path, "w", encoding="utf-8") as f:
            for seq in encoded_sequences:
                f.write(seq + "\n")

        corpus_size_kb = get_file_size(batch_corpus_path) / 1024
        logger.info(f"Batch corpus saved: {corpus_size_kb:.1f} KB")

        # Also update the main corpus file (append)
        logger.info(f"Appending to main corpus: {self.corpus_path}")
        with open_file(str(self.corpus_path), "a", encoding="utf-8") as f:
            for seq in encoded_sequences:
                f.write(seq + "\n")

        # Phase 4: Load or create model and train
        logger.info("\nPhase 4: Training FastText Model")
        logger.info(f"  Tree type: {self.tree_type}")
        logger.info(f"  Batch size: {batch_size_actual}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Vector size: {self.vector_size}")
        logger.info(f"  Window: {self.window}")

        # Load existing model if available (for incremental training)
        if path_exists(str(self.model_path)):
            logger.info(f"Loading existing model for incremental training: {self.model_path}")
            self.model = FastText.load(str(self.model_path))
        else:
            logger.info("Creating new FastText model")
            self.model = None

        # Train with progress callback
        callback = TrainingProgressCallback(epochs=epochs, corpus_file=str(batch_corpus_path))

        if self.model is None:
            # New model
            self.model = FastText(
                corpus_file=str(batch_corpus_path),
                vector_size=self.vector_size,
                window=self.window,
                min_n=self.min_n,
                max_n=self.max_n,
                negative=self.negative,
                sg=0,  # CBOW
                seed=42,
                epochs=epochs,
                callbacks=[callback],
            )
        else:
            # Incremental training on existing model
            self.model.train(
                corpus_file=str(batch_corpus_path),
                epochs=epochs,
                total_examples=self.model.corpus_count,
                callbacks=[callback],
            )

        logger.info("Training complete")

        # Phase 5: Save model and update checkpoint
        logger.info("\nPhase 5: Saving Model and Checkpoint")
        makedirs(str(Path(str(self.model_path)).parent), exist_ok=True)
        self.model.save(str(self.model_path))
        logger.info(f"Model saved to: {self.model_path}")

        # Save encoder maps (accumulate across batches)
        logger.info(f"Saving encoder maps to {self.encoder_maps_path}")
        node_map, edge_map = self.token_id_manager.get_updates()
        save_maps(node_map, edge_map, str(self.encoder_maps_path))
        logger.info(f"Encoder maps: {len(node_map)} nodes, {len(edge_map)} edges")

        # Update checkpoint: extract file numbers from used_files
        completed_files_from_batch = [
            int(f.replace(".tsv", "")) for f in self.used_files
        ]

        # Determine if last file was fully processed
        latex_dir = LATEX_REPRESENTATION
        if completed_files_from_batch:
            last_file_num = completed_files_from_batch[-1]
            last_file_path = str(latex_dir / f"{last_file_num}.tsv")
            last_file_total_rows = count_lines(last_file_path)

            # Check if we reached end of file or hit batch limit
            if len(formulas_df) < batch_size:
                # Reached end of file
                last_file_row_index = 0
                last_file_number = None
            else:
                # Hit batch limit, calculate row index in last file
                rows_in_completed_files = sum(
                    count_lines(str(latex_dir / f"{fn}.tsv"))
                    for fn in completed_files_from_batch[:-1]
                )
                last_file_row_index = batch_size_actual - rows_in_completed_files
                last_file_number = last_file_num

        # Update checkpoint
        all_completed_files = (
            checkpoint["completed_files"] + completed_files_from_batch
        )
        # Remove duplicates and sort
        all_completed_files = sorted(list(set(all_completed_files)))

        total_formulas_processed_new = total_formulas_processed + batch_size_actual
        batches_completed_new = batches_completed + 1

        self._save_checkpoint(
            completed_files=all_completed_files,
            last_file_number=last_file_number,
            last_file_row_index=last_file_row_index,
            total_formulas_processed=total_formulas_processed_new,
            batches_completed=batches_completed_new,
            total_formulas_available=total_formulas_available,
        )

        logger.info("\n" + "=" * 70)
        logger.info("BATCH TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Batch #{batches_completed_new} completed")
        logger.info(f"Total formulas processed: {total_formulas_processed_new}")
        logger.info(
            f"Progress: {total_formulas_processed_new / total_formulas_available * 100:.1f}%"
        )

        # Clean up batch corpus file
        try:
            delete_file(batch_corpus_path)
            logger.info(f"Temporary batch corpus deleted: {batch_corpus_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary corpus: {e}")

        return self.model

    def get_stats(self) -> Dict:
        """Get training statistics and metadata."""
        return self.training_stats.copy()
