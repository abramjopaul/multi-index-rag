# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Formula-to-Embedding conversion using FastText training.
# Trains FastText model on tokenized formula representations.

import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Literal
import pandas as pd
from gensim.models import FastText
from tqdm import tqdm

from multirag.config.path_configs import LATEX_REPRESENTATION
from multirag.formula_search import (
    SLTGenerator,
    OPTGenerator,
    TupleTokenizationMode,
    TupleTokenizer,
    TokenIDManager,
)
from multirag.formula_search.formula_tokenizer_pipeline import get_project_root
from multirag.formula_search.encoder_maps import save_maps, load_maps

logger = logging.getLogger(__name__)


def get_default_indexing_dir() -> Path:
    """Get default directory for formula indexing artifacts (data/formula-indexing)."""
    project_root = get_project_root()
    return project_root / "data" / "formula-indexing"


def get_fasttext_dir() -> Path:
    """Get directory for FastText models and training data."""
    return get_default_indexing_dir() / "fasttext"


# def get_latex_representation_dir() -> Path:
#     """Get directory for raw LaTeX representations."""
#     project_root = get_project_root()
#     return project_root / "data" / "raw" / "collection" / "formula" / "latex_representation_v3"


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
        tree_type: Literal["SLT", "OPT"] = "SLT",
        embedding_type: TupleTokenizationMode = TupleTokenizationMode.Both_Separated,
        vector_size: int = 200,
        window: int = 5,
        min_n: int = 1,
        max_n: int = 100,
        negative: int = 5,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize FormulaTrainer.
        
        Args:
            tree_type: "SLT" (Symbol Layout Tree) or "OPT" (Operator Tree)
            embedding_type: Node tokenization mode (default: Both_Separated)
            vector_size: FastText vector dimension (default: 200)
            window: Context window size (default: 5)
            min_n: Minimum n-gram size (default: 2)
            max_n: Maximum n-gram size (default: 100)
            negative: Number of negative samples (default: 5)
            output_dir: Output directory (default: data/formula-indexing)
        """
        self.tree_type = tree_type.upper()
        if self.tree_type not in {"SLT", "OPT"}:
            raise ValueError(f"tree_type must be SLT or OPT, got {tree_type}")
        
        self.embedding_type = embedding_type
        self.vector_size = vector_size
        self.window = window
        self.min_n = min_n
        self.max_n = max_n
        self.negative = negative
        
        # Setup output directories
        self.output_dir = Path(output_dir) if output_dir else get_fasttext_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Artifact paths
        self.encoder_maps_path = get_default_indexing_dir() / "encoder_maps.tsv"
        self.corpus_path = self.output_dir / "corpus.txt"  # LineSentence format
        self.model_path = self.output_dir / f"fasttext_model_{self.tree_type.lower()}.bin"
        self.metadata_path = self.output_dir / f"training_metadata_{self.tree_type.lower()}.json"
        
        # Initialize generators and tokenizer
        self.slt_generator = SLTGenerator()
        self.opt_generator = OPTGenerator()
        
        # Token ID manager
        self.token_id_manager = TokenIDManager()
        self.tuple_tokenizer = TupleTokenizer(
            token_id_manager=self.token_id_manager,
            embedding_type=embedding_type,
        )
        
        self.model: Optional[FastText] = None
        self.training_stats: Dict = {}
        self.used_files: List[str] = []
        self.num_formulas_loaded = 0
    
    def load_latex_formulas(
        self,
        file_numbers: Optional[List[int]] = None,
        num_formulas: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load LaTeX formulas from TSV files in latex_representation_v3 directory.
        
        Args:
            file_numbers: List of file numbers to load (e.g., [1, 2, 3]).
                         If None, loads all files.
            num_formulas: Maximum number of formulas to load across all files.
                         If None, loads everything.
        
        Returns:
            DataFrame with loaded formulas
        
        Example:
            >>> trainer = FormulaTrainer(tree_type="SLT")
            >>> df = trainer.load_latex_formulas(file_numbers=[1, 2], num_formulas=5000)
        """
        latex_dir = LATEX_REPRESENTATION
        if not latex_dir.exists():
            raise FileNotFoundError(f"LaTeX representation directory not found: {latex_dir}")
        
        logger.info(f"Loading LaTeX formulas from {latex_dir}")
        
        # Determine which files to load
        if file_numbers is None:
            # Load all files
            files = sorted(list(latex_dir.glob("*.tsv")), key=lambda x: int(x.stem))
        else:
            # Load specific files
            files = []
            for num in file_numbers:
                file_path = latex_dir / f"{num}.tsv"
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                else:
                    files.append(file_path)
            files = sorted(files, key=lambda x: int(x.stem))
        
        if not files:
            raise ValueError(f"No files found to load")
        
        logger.info(f"Found {len(files)} files to load")
        
        all_formulas = []
        total_loaded = 0
        
        with tqdm(total=num_formulas or sum(1 for f in files), desc="Loading LaTeX", unit="formula") as pbar:
            for file_path in files:
                if num_formulas and total_loaded >= num_formulas:
                    break
                
                try:
                    df = pd.read_csv(file_path, sep="\t", dtype=str)
                    
                    # Limit if needed
                    if num_formulas:
                        remaining = num_formulas - total_loaded
                        if len(df) > remaining:
                            df = df.head(remaining)
                    
                    all_formulas.append(df)
                    self.used_files.append(file_path.name)
                    total_loaded += len(df)
                    pbar.update(len(df))
                    
                    logger.info(f"Loaded {len(df)} formulas from {file_path.name}")
                    
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
        
        Args:
            latex: LaTeX formula string
        
        Returns:
            List of tab-separated tuples (or empty list if failed)
        """
        try:
            if self.tree_type == "SLT":
                tree = self.slt_generator.generate(latex)
            else:  # OPT
                tree = self.opt_generator.generate(latex)
            
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
        
        Args:
            formulas_df: DataFrame with formulas
            formula_column: Column name containing LaTeX strings
        
        Returns:
            List of encoded sequences (one per formula)
        """
        logger.info(f"Processing {len(formulas_df)} formulas ({self.tree_type} trees)...")
        
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
    
    def save_corpus_and_maps(self, encoded_sequences: List[str]) -> None:
        """
        Save encoded sequences in LineSentence format and encoder maps.
        
        Args:
            encoded_sequences: List of whitespace-separated token strings
        """
        # Save corpus in LineSentence format
        logger.info(f"Saving {len(encoded_sequences)} sequences to {self.corpus_path}")
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for seq in encoded_sequences:
                f.write(seq + "\n")
        
        corpus_size_kb = self.corpus_path.stat().st_size / 1024
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
        logger.info("="*70)
        logger.info(f"FASTTEXT TRAINING PIPELINE ({self.tree_type})")
        logger.info("="*70)
        
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
        logger.info(f"  Vector size: {self.vector_size}")
        logger.info(f"  Window: {self.window}")
        logger.info(f"  Min n-gram: {self.min_n}")
        logger.info(f"  Max n-gram: {self.max_n}")
        logger.info(f"  Negative: {self.negative}")
        
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
            epochs=5,  # Training epochs
        )
        
        logger.info("Training complete")
        
        # Phase 5: Save model and metadata
        logger.info("\nPhase 5: Saving Model and Metadata")
        self._save_model()
        self._save_metadata(encoded_sequences)
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Model saved to: {self.model_path}")
        logger.info(f"Corpus saved to: {self.corpus_path}")
        logger.info(f"Encoder maps saved to: {self.encoder_maps_path}")
        logger.info(f"Metadata saved to: {self.metadata_path}")
        
        return self.model
    
    def _save_model(self) -> None:
        """Save trained FastText model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
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
        """
        self.training_stats = {
            "tree_type": self.tree_type,
            "embedding_type": self.embedding_type.name,
            "files_used": self.used_files,
            "num_formulas_loaded": self.num_formulas_loaded,
            "num_sequences_encoded": len(encoded_sequences),
            "vector_size": self.vector_size,
            "window": self.window,
            "negative": self.negative,
            "model_path": str(self.model_path),
            "corpus_path": str(self.corpus_path),
            "encoder_maps_path": str(self.encoder_maps_path),
        }
        
        # Attach metadata directly to model object for easy access
        if self.model is not None:
            try:
                self.model.meta = self.training_stats.copy()
                logger.info("Metadata attached to model.meta")
            except Exception as e:
                logger.warning(f"Could not attach metadata to model: {e}")
        
        # Also save to JSON file as backup (in case meta is lost during save/load)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
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
        
        if not Path(model_path).exists():
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
        if self.model is not None and hasattr(self.model, 'meta'):
            try:
                self.training_stats = self.model.meta
                logger.info("Metadata loaded from model.meta")
                return
            except Exception as e:
                logger.warning(f"Could not read model.meta: {e}")
        
        # Fall back to JSON file
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
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
    
    def get_stats(self) -> Dict:
        """Get training statistics and metadata."""
        return self.training_stats.copy()

