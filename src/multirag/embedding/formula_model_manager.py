# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# FastText Model Management for Formula Embeddings
# Handles model training, loading, saving, and vector generation

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from multirag.utils.file_utils import (
    count_lines,
    makedirs,
    open_file,
    path_exists,
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
            self.pbar.update(self.pbar.total - self.pbar.n)
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


class FastTextModelManager:
    """
    Manages FastText model lifecycle: training, loading, saving, metadata I/O, and vector generation.
    
    Separates model management from corpus preparation, improving code organization and readability.
    
    Attributes:
        model: FastText model instance (None until trained/loaded)
        training_stats: Dictionary with training metadata
        model_path: Path to saved model file
        metadata_path: Path to saved metadata file
        corpus_path: Path to corpus file used for training
    """

    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        corpus_path: str,
        vector_size: int = 300,
        window: int = 5,
        min_n: int = 3,
        max_n: int = 6,
        negative: int = 20,
        sg: int = 1,
        hs: int = 0,
        word_ngrams: int = 1,
        num_workers: int = 8,
    ):
        """
        Initialize FastTextModelManager.

        Args:
            model_path: Path where model will be saved/loaded
            metadata_path: Path where metadata will be saved/loaded
            corpus_path: Path to corpus file (LineSentence format)
            vector_size: FastText vector dimension (default: 300)
            window: Context window size (default: 5)
            min_n: Minimum n-gram size (default: 3)
            max_n: Maximum n-gram size (default: 6)
            negative: Number of negative samples (default: 20, 0 = hierarchical softmax)
            sg: Training algorithm: 1 = Skip-gram, 0 = CBOW (default: 1)
            hs: Use hierarchical softmax: 1 = yes, 0 = no (default: 0)
            word_ngrams: Number of word n-grams (default: 1)
            num_workers: Number of worker threads (default: 8)
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.corpus_path = Path(corpus_path)

        # Store hyperparameters
        self.vector_size = vector_size
        self.window = window
        self.min_n = min_n
        self.max_n = max_n
        self.negative = negative
        self.sg = sg
        self.hs = hs
        self.word_ngrams = word_ngrams
        self.num_workers = num_workers

        # Model state
        self.model: Optional[FastText] = None
        self.training_stats: Dict = {}

        logger.info(
            f"Initialized FastTextModelManager (vector_size={vector_size}, window={window})"
        )

    def train_from_corpus_file(
        self,
        epochs: int,
        num_sequences: Optional[int] = None,
        tree_type: str = "SLT",
        embedding_type_name: str = "Both_Separated",
        tokenize_number: bool = True,
        used_files: Optional[list] = None,
        num_formulas_loaded: int = 0,
        encoder_maps_path: Optional[str] = None,
    ) -> FastText:
        """
        Train FastText model from corpus file (memory-efficient streaming).

        Args:
            epochs: Number of training epochs
            num_sequences: Number of sequences in corpus (optional, for logging)
            tree_type: Type of tree representation (for metadata)
            embedding_type_name: Name of embedding type (for metadata)
            tokenize_number: Whether numbers are tokenized (for metadata)
            used_files: List of files used to create corpus (for metadata)
            num_formulas_loaded: Number of formulas loaded (for metadata)
            encoder_maps_path: Path to encoder maps file (for metadata)

        Returns:
            Trained FastText model
        """
        logger.info(
            f"Training FastText model from corpus: {self.corpus_path} ({epochs} epochs)"
        )

        if not path_exists(str(self.corpus_path)):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        # Create and train model
        self.model = FastText(
            corpus_file=str(self.corpus_path),
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,
            workers=self.num_workers,
            negative=self.negative,
            min_n=self.min_n,
            max_n=self.max_n,
            sg=self.sg,
            hs=self.hs,
            word_ngrams=self.word_ngrams,
            epochs=epochs,
            callbacks=[TrainingProgressCallback(epochs, str(self.corpus_path))],
        )

        # Save model and metadata
        self.save_model()
        self._save_metadata(
            epochs=epochs,
            num_sequences=num_sequences,
            tree_type=tree_type,
            embedding_type_name=embedding_type_name,
            tokenize_number=tokenize_number,
            used_files=used_files,
            num_formulas_loaded=num_formulas_loaded,
            encoder_maps_path=encoder_maps_path,
        )

        logger.info(f"Training complete. Model saved: {self.model_path}")

        return self.model

    def save_model(self) -> None:
        """Save trained FastText model."""
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        makedirs(str(self.model_path.parent), exist_ok=True)
        self.model.save(str(self.model_path))
        logger.info(f"Model saved: {self.model_path}")

    def load(self, model_path: Optional[str] = None) -> FastText:
        """
        Load a trained FastText model.

        Args:
            model_path: Path to model file (default: self.model_path)

        Returns:
            Loaded FastText model
        """
        if model_path is None:
            model_path = str(self.model_path)

        if not path_exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")
        self.model = FastText.load(model_path)
        self.load_metadata()

        return self.model

    def load_metadata(self) -> None:
        """Load training metadata."""
        if not path_exists(str(self.metadata_path)):
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            self.training_stats = {}
            return

        with open_file(str(self.metadata_path), "r", encoding="utf-8") as f:
            self.training_stats = json.load(f)

        logger.info(f"Metadata loaded: {self.metadata_path}")

    def _save_metadata(
        self,
        epochs: int,
        num_sequences: Optional[int] = None,
        tree_type: str = "SLT",
        embedding_type_name: str = "Both_Separated",
        tokenize_number: bool = True,
        used_files: Optional[list] = None,
        num_formulas_loaded: int = 0,
        encoder_maps_path: Optional[str] = None,
    ) -> None:
        """
        Create and save training metadata.

        Args:
            epochs: Number of training epochs
            num_sequences: Number of sequences in corpus (optional, will count if not provided)
            tree_type: Type of tree representation
            embedding_type_name: Name of embedding type
            tokenize_number: Whether numbers are tokenized
            used_files: List of files used to create corpus
            num_formulas_loaded: Number of formulas loaded
            encoder_maps_path: Path to encoder maps file
        """
        if num_sequences is None:
            num_sequences = count_lines(str(self.corpus_path))

        metadata = {
            "tree_type": tree_type,
            "embedding_type": embedding_type_name,
            "tokenize_number": tokenize_number,
            "vector_size": self.vector_size,
            "window": self.window,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "negative": self.negative,
            "sg": self.sg,
            "hs": self.hs,
            "word_ngrams": self.word_ngrams,
            "epochs": epochs,
            "num_workers": self.num_workers,
            "num_formulas_loaded": num_formulas_loaded,
            "num_formulas_encoded": num_sequences,
            "files_used": used_files or [],
            "corpus_path": str(self.corpus_path),
            "model_path": str(self.model_path),
            "encoder_maps_path": str(encoder_maps_path) if encoder_maps_path else None,
            "training_mode": "corpus_file_streaming",
        }

        makedirs(str(self.metadata_path.parent), exist_ok=True)
        with open_file(str(self.metadata_path), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.training_stats = metadata
        logger.info(f"Metadata saved: {self.metadata_path}")

    def get_sentence_vector(self, encoded_sequence: str) -> list:
        """
        Get vector representation for an encoded sequence.

        Args:
            encoded_sequence: Whitespace-separated encoded tokens

        Returns:
            Vector representation (list of floats, or zeros if model not loaded)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = encoded_sequence.split()
        vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]

        if not vectors:
            return np.zeros(self.vector_size).tolist()

        return np.mean(vectors, axis=0).tolist()

    def get_stats(self) -> Dict:
        """
        Get training statistics.

        Returns:
            Dictionary with training metadata
        """
        return self.training_stats
