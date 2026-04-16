# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Formula embedder: generates dense vectors for formulas using trained FastText model.

import logging
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
from gensim.models import FastText
from tqdm import tqdm

from multirag.formula_search import FormulaTokenizerPipeline, TupleTokenizationMode

logger = logging.getLogger(__name__)


class FormulaEmbedder:
    """
    Generates dense embeddings for formulas using trained FastText model.
    
    Workflow:
    1. Load trained FastText model
    2. Load formulas from TSV
    3. Tokenize each formula
    4. Generate embedding vectors
    5. Save embeddings and metadata
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_type: TupleTokenizationMode = TupleTokenizationMode.Both_Separated,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize FormulaEmbedder.
        
        Args:
            model_path: Path to trained FastText model. If None, uses default.
            embedding_type: Tokenization mode (default: Both_Separated)
            output_dir: Directory to save embeddings. If None, uses data/embeddings/
        """
        self.model_path = model_path or str(get_default_training_dir() / "fasttext_model.bin")
        self.output_dir = Path(output_dir) if output_dir else get_default_training_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_type = embedding_type
        self.embeddings_npy = self.output_dir / "formula_embeddings.npy"
        self.metadata_csv = self.output_dir / "formula_embeddings_metadata.csv"
        
        # Load model and tokenizer
        self.model: Optional[FastText] = None
        self.tokenizer = FormulaTokenizerPipeline(
            embedding_type=embedding_type,
            use_default_encoder_path=True,
        )
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained FastText model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading FastText model from {self.model_path}")
        self.model = FastText.load(self.model_path)
        logger.info(f"Model loaded. Vocabulary size: {len(self.model.wv)}")
    
    def embed_formula(self, latex: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single formula.
        
        Args:
            latex: LaTeX formula string
        
        Returns:
            Embedding vector (numpy array) or None on error
        """
        try:
            # Tokenize
            encoded = self.tokenizer.tokenize_formula(latex, tree_type="SLT")
            if not encoded:
                return None
            
            # Combine tuples
            combined = "".join(encoded)
            
            # Get vector
            vector = self.model.get_sentence_vector(combined)
            return vector
        
        except Exception as e:
            logger.warning(f"Error embedding formula: {e}")
            return None
    
    def embed_batch(
        self,
        formulas_df: pd.DataFrame,
        formula_column: str = "formula",
        formula_id_column: Optional[str] = None,
    ) -> tuple:
        """
        Generate embeddings for a batch of formulas.
        
        Args:
            formulas_df: DataFrame with formulas
            formula_column: Column name with formula strings
            formula_id_column: Column name with formula IDs (optional)
        
        Returns:
            Tuple of (embeddings_array, metadata_df)
            - embeddings_array: (N, vector_size) numpy array
            - metadata_df: DataFrame with formula IDs and status
        """
        logger.info(f"Generating embeddings for {len(formulas_df)} formulas...")
        
        embeddings = []
        formula_ids = []
        statuses = []
        
        with tqdm(total=len(formulas_df), desc="Embedding", unit="formula") as pbar:
            for idx, row in formulas_df.iterrows():
                formula_id = row.get(formula_id_column, idx) if formula_id_column else idx
                formula_text = row[formula_column]
                
                if pd.isna(formula_text):
                    statuses.append("skipped_nan")
                    pbar.update(1)
                    continue
                
                vector = self.embed_formula(formula_text)
                if vector is not None:
                    embeddings.append(vector)
                    formula_ids.append(formula_id)
                    statuses.append("success")
                else:
                    statuses.append("failed")
                
                pbar.update(1)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame({
            "formula_id": formula_ids,
            "status": statuses[:len(formula_ids)],
        })
        
        return embeddings_array, metadata_df
    
    def embed_all(
        self,
        formulas_tsv: str,
        formula_column: str = "formula",
        formula_id_column: Optional[str] = None,
        save: bool = True,
    ) -> tuple:
        """
        Embed all formulas from TSV file.
        
        Args:
            formulas_tsv: Path to TSV file with formulas
            formula_column: Column name with formula strings
            formula_id_column: Column name with formula IDs (optional)
            save: Whether to save embeddings to disk (default: True)
        
        Returns:
            Tuple of (embeddings_array, metadata_df)
        """
        logger.info(f"Loading formulas from {formulas_tsv}")
        df = pd.read_csv(formulas_tsv, sep="\t")
        logger.info(f"Loaded {len(df)} formulas")
        
        embeddings, metadata = self.embed_batch(
            df,
            formula_column=formula_column,
            formula_id_column=formula_id_column,
        )
        
        if save:
            self.save_embeddings(embeddings, metadata)
        
        return embeddings, metadata
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame,
        embeddings_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings: (N, vector_size) array
            metadata_df: DataFrame with formula info
            embeddings_path: Path to save embeddings.npy
            metadata_path: Path to save metadata.csv
        """
        embeddings_path = embeddings_path or str(self.embeddings_npy)
        metadata_path = metadata_path or str(self.metadata_csv)
        
        # Save embeddings
        Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        size_mb = Path(embeddings_path).stat().st_size / (1024 * 1024)
        logger.info(f"Embeddings saved to {embeddings_path} ({size_mb:.1f} MB)")
        
        # Save metadata
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_embeddings(
        self,
        embeddings_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> tuple:
        """
        Load saved embeddings and metadata.
        
        Args:
            embeddings_path: Path to embeddings.npy
            metadata_path: Path to metadata.csv
        
        Returns:
            Tuple of (embeddings_array, metadata_df)
        """
        embeddings_path = embeddings_path or str(self.embeddings_npy)
        metadata_path = metadata_path or str(self.metadata_csv)
        
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        
        logger.info(f"Loading metadata from {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings, metadata
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """Get statistics about embeddings."""
        return {
            "num_formulas": len(embeddings),
            "vector_size": embeddings.shape[1],
            "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
            "std_norm": float(np.linalg.norm(embeddings, axis=1).std()),
            "min_norm": float(np.linalg.norm(embeddings, axis=1).min()),
            "max_norm": float(np.linalg.norm(embeddings, axis=1).max()),
        }
