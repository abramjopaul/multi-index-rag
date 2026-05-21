# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Golden Dataset Evaluator for Formula Embeddings
# Evaluates formula embedding models against multiple golden similarity datasets.

import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add src to path to import from multirag
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multirag.embedding.formula_trainer import (
    OPTGenerator,
    SLTGenerator,
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
)

logger = logging.getLogger(__name__)


class FormulaEmbeddingEvaluator:
    """
    Evaluates formula embedding models against a golden similarity dataset.

    Uses the same workflow as FormulaTrainer to generate embeddings:
    1. LaTeX → SLT/OPT tree → tuples → encode → tokens
    2. Get FastText vector for each token
    3. Mean pooling of vectors (like TangentCFT)
    4. Calculate cosine similarity between formula embeddings
    5. Compare with relevance judgments

    Workflow:
    - Load FastText model trained with FormulaTrainer
    - Load golden dataset with equation pairs and relevance scores
    - For each pair:
      * Generate tree representation (SLT/OPT)
      * Encode tuples to get space-separated token string
      * Retrieve FastText vectors and compute mean (TangentCFT style)
      * Calculate cosine similarity
    - Save results to consolidated TSV with model_type column

    Relevance Labels (from human annotations):
    - 3: Semantically equivalent (same mathematical meaning, only notation/structure changes)
    - 2: Closely related / transformed (same concept or derivable form, but not exact equivalence)
    - 1: Weakly related (similar appearance or same family/domain, meaning differs)
    - 0: Irrelevant / hard negative (structurally similar or visually deceptive but semantically wrong)
    """

    def __init__(
        self,
        tree_type: Literal["SLT", "OPT", "SLT-TYPE"] = "SLT",
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """
        Initialize FormulaEmbeddingEvaluator.

        Args:
            tree_type: Type of tree representation ("SLT", "OPT", or "SLT-TYPE")
            model_path: Path to FastText model file (required for evaluation)
            dataset_path: Path to golden dataset TSV file (required for evaluation)
            output_path: Path to save evaluation results (default: data/golden/golden_predictions.tsv)

        Raises:
            ValueError: If tree_type is not valid
            FileNotFoundError: If model_path or dataset_path do not exist
        """
        self.tree_type = tree_type.upper()
        if self.tree_type not in {"SLT", "OPT", "SLT-TYPE"}:
            raise ValueError(
                f"tree_type must be SLT, OPT, or SLT-TYPE, got {tree_type}"
            )

        # Validate paths
        self.model_path = Path(model_path) if model_path else None
        self.dataset_path = Path(dataset_path) if dataset_path else None

        if self.model_path and not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if self.dataset_path and not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Set output path
        if output_path:
            self.output_path = Path(output_path)
        else:
            output_dir = Path("data/golden")
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = output_dir / "golden_predictions.tsv"

        # Load FastText model
        self.model: Optional[FastText] = None
        if self.model_path:
            try:
                self.model = FastText.load(str(self.model_path))
                self.embedding_dim = self.model.wv.vector_size
                logger.info(
                    f"Loaded FastText model from {self.model_path} (dim={self.embedding_dim})"
                )
            except Exception as e:
                logger.error(f"Error loading FastText model: {e}")
                raise

        # Initialize generators
        self.slt_generator = SLTGenerator()
        self.opt_generator = OPTGenerator()

        # Setup TokenIDManager and TupleTokenizer
        self.token_id_manager = TokenIDManager()

        # Auto-configure embedding_type based on tree_type
        if self.tree_type == "SLT-TYPE":
            embedding_type = TupleTokenizationMode.Type
            tokenize_number = False
        else:  # SLT or OPT
            embedding_type = TupleTokenizationMode.Both_Separated
            tokenize_number = self.tree_type == "SLT"

        self.tuple_tokenizer = TupleTokenizer(
            token_id_manager=self.token_id_manager,
            embedding_type=embedding_type,
            tokenize_number=tokenize_number,
        )

        logger.info(
            f"Initialized FormulaEmbeddingEvaluator with tree_type={self.tree_type}, "
            f"embedding_type={embedding_type.name}, tokenize_number={tokenize_number}"
        )

    def generate_tuples(self, latex: str) -> List[str]:
        """
        Generate tuples from LaTeX formula using SLT or OPT.

        Note: SLT-TYPE also uses SLT tree generation, with Type-only tokenization.

        Args:
            latex: LaTeX formula string

        Returns:
            List of tab-separated tuples (empty list if parsing fails)
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
            logger.debug(
                f"Error generating {self.tree_type} tuples from LaTeX '{latex[:50]}...': {e}"
            )
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

        return " ".join(encoded_tokens)  # Whitespace-separated for processing

    def _get_formula_embedding(self, latex_str: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a formula using FastText vectors with mean pooling.

        This follows TangentCFT's approach (tangent_cft_module.py):
        1. Generate tuples from LaTeX
        2. Encode tuples to get space-separated token string
        3. For each token: get FastText vector from model.wv[token]
        4. Sum vectors and divide by count (mean pooling)
        5. Handle missing tokens gracefully

        Process:
        1. Generate tuples from LaTeX using tree representation
        2. Encode tuples to get space-separated token string
        3. Split tokens and retrieve FastText vectors for each token
        4. Handle missing vocab tokens gracefully
        5. Return mean pooled embedding

        Args:
            latex_str: LaTeX formula string

        Returns:
            Embedding as numpy array of shape (embedding_dim,), or None if:
            - LaTeX parsing fails
            - No valid tokens are found in vocabulary

        Note:
            - Tokens not in FastText vocabulary are skipped with a debug log
            - If all tokens are missing from vocab, returns None
        """
        if not self.model or not latex_str or not isinstance(latex_str, str):
            return None

        try:
            # Generate tuples
            tuples = self.generate_tuples(latex_str)
            if not tuples:
                logger.debug(f"No tuples generated for LaTeX: {latex_str[:50]}...")
                return None

            # Encode tuples to get space-separated tokens
            encoded_sequence = self.encode_tuples(tuples)
            if not encoded_sequence:
                logger.debug(f"No tokens encoded for LaTeX: {latex_str[:50]}...")
                return None

            # Split tokens
            tokens = encoded_sequence.split()
            if not tokens:
                return None

            # Retrieve FastText vectors for each token (TangentCFT style)
            vectors = []
            missing_count = 0

            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])
                else:
                    missing_count += 1

            # Log if significant tokens are missing
            if missing_count > 0 and len(tokens) > 0:
                logger.debug(
                    f"Missing {missing_count}/{len(tokens)} tokens from FastText vocab"
                )

            # Return None if all tokens are missing
            if not vectors:
                logger.debug(
                    f"All tokens missing from vocab for LaTeX: {latex_str[:50]}..."
                )
                return None

            # Return mean pooled embedding (like TangentCFT)
            return np.mean(vectors, axis=0)

        except Exception as e:
            logger.debug(f"Error computing formula embedding: {e}")
            return None

    def _calculate_similarity(
        self, vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]
    ) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            vec1: First embedding vector (or None)
            vec2: Second embedding vector (or None)

        Returns:
            Similarity score in [0, 1], or np.nan if either vector is None
        """
        if vec1 is None or vec2 is None:
            return np.nan

        try:
            # Reshape for sklearn cosine_similarity (expects 2D arrays)
            vec1_reshaped = vec1.reshape(1, -1)
            vec2_reshaped = vec2.reshape(1, -1)

            # Compute cosine similarity
            similarity = cosine_similarity(vec1_reshaped, vec2_reshaped)[0, 0]

            # Ensure float and within [0, 1]
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.debug(f"Error calculating similarity: {e}")
            return np.nan

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate formula embeddings against golden dataset.

        Loads the golden dataset, computes embeddings and similarities for all pairs,
        saves results to TSV, and logs summary statistics.

        Process:
        1. Load golden dataset from dataset_path
        2. For each row in the dataset:
           - Compute query formula embedding
           - Compute candidate formula embedding
           - Calculate cosine similarity
           - Record results with metadata
        3. Create DataFrame from results
        4. Save to output_path as TSV with headers
        5. Log summary statistics (count, similarity distribution, coverage)
        6. Return results DataFrame

        Returns:
            DataFrame with evaluation results containing columns:
            - dataset: Dataset name
            - query_equation: Query formula (LaTeX)
            - candidate_id: Candidate ID
            - candidate_equation: Candidate formula (LaTeX)
            - relevance: Human-provided relevance judgment (0, 1, 2, or 3)
            - model_type: Tree type used (SLT/OPT/SLT-TYPE)
            - predicted_similarity: Predicted similarity score [0, 1] or NaN

        Raises:
            FileNotFoundError: If dataset_path is not set or doesn't exist
            ValueError: If dataset is missing required columns
        """
        if not self.dataset_path or not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}. Set dataset_path during initialization."
            )

        logger.info(f"Loading golden dataset from {self.dataset_path}")

        # Load dataset
        try:
            df = pd.read_csv(self.dataset_path, sep="\t", dtype=str)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        # Validate required columns
        required_columns = [
            "dataset",
            "query_equation",
            "candidate_id",
            "candidate_equation",
            "relevance",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Dataset missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        logger.info(f"Dataset has {len(df)} rows")

        results = []

        # Evaluate each pair
        with tqdm(total=len(df), desc=f"Evaluating ({self.tree_type})", unit="pair") as pbar:
            for _, row in df.iterrows():
                try:
                    query_latex = str(row["query_equation"])
                    candidate_latex = str(row["candidate_equation"])
                    relevance = str(row["relevance"])
                    dataset_name = str(row["dataset"])
                    candidate_id = str(row["candidate_id"])

                    # Get embeddings
                    query_embedding = self._get_formula_embedding(query_latex)
                    candidate_embedding = self._get_formula_embedding(candidate_latex)

                    # Calculate similarity
                    predicted_similarity = self._calculate_similarity(
                        query_embedding, candidate_embedding
                    )

                    # Record result
                    result = {
                        "dataset": dataset_name,
                        "query_equation": query_latex,
                        "candidate_id": candidate_id,
                        "candidate_equation": candidate_latex,
                        "relevance": relevance,
                        "model_type": self.tree_type,
                        "predicted_similarity": predicted_similarity,
                    }
                    results.append(result)

                except Exception as e:
                    logger.warning(f"Error processing row: {e}")

                pbar.update(1)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Save to TSV - append if file already exists (for consolidated output)
        logger.info(f"Saving {len(results_df)} results to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we should append or create new
        file_exists = self.output_path.exists()

        if file_exists:
            # Append to existing file (consolidating all models)
            existing_df = pd.read_csv(self.output_path, sep="\t", dtype=str)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_csv(self.output_path, sep="\t", index=False)
            logger.info(
                f"Appended results to {self.output_path} (total rows: {len(combined_df)})"
            )
        else:
            # Create new file
            results_df.to_csv(self.output_path, sep="\t", index=False)
            logger.info(f"Results saved to {self.output_path}")

        # Log summary statistics
        total_pairs = len(results_df)
        valid_similarities = results_df["predicted_similarity"].dropna()
        coverage = (
            len(valid_similarities) / total_pairs * 100 if total_pairs > 0 else 0
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Summary ({self.tree_type})")
        logger.info(f"{'='*60}")
        logger.info(f"Total pairs: {total_pairs}")
        logger.info(
            f"Valid predictions: {len(valid_similarities)} ({coverage:.1f}% coverage)"
        )

        if len(valid_similarities) > 0:
            logger.info(f"Mean similarity: {valid_similarities.mean():.4f}")
            logger.info(f"Std similarity: {valid_similarities.std():.4f}")
            logger.info(f"Min similarity: {valid_similarities.min():.4f}")
            logger.info(f"Max similarity: {valid_similarities.max():.4f}")
            logger.info(f"Median similarity: {valid_similarities.median():.4f}")

            # Log distribution by relevance
            for relevance in sorted(results_df["relevance"].unique()):
                subset = results_df[results_df["relevance"] == relevance][
                    "predicted_similarity"
                ].dropna()
                if len(subset) > 0:
                    logger.info(
                        f"Relevance {relevance}: mean={subset.mean():.4f}, n={len(subset)}"
                    )

        logger.info(f"{'='*60}\n")

        return results_df


def main():
    """
    Demonstrate usage of FormulaEmbeddingEvaluator.

    This script evaluates formula embedding models across multiple curated datasets:
    - data/golden/datasets/chatgpt_golden_dataset.tsv
    - data/golden/datasets/claude_golden_dataset.tsv
    - data/golden/datasets/gemini_golden_dataset.tsv
    - etc.

    For each dataset, it:
    1. Evaluates all three tree types (SLT, OPT, SLT-TYPE)
    2. Saves consolidated results to data/golden/results/result_{dataset_name}.tsv
    3. Logs summary statistics and comparison across models

    Output format (per dataset):
    - result_chatgpt_golden_dataset.tsv: Contains all predictions for ChatGPT dataset
    - result_claude_golden_dataset.tsv: Contains all predictions for Claude dataset
    - etc.

    Each result file includes columns:
    - dataset, query_equation, candidate_id, candidate_equation, relevance
    - model_type (SLT/OPT/SLT-TYPE)
    - predicted_similarity (float in [0,1] or NaN if embedding failed)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Paths (relative to workspace root)
    datasets_dir = Path("data/golden/datasets")
    results_dir = Path("data/golden/results")
    model_dir = Path("data/formula-indexing")

    # Check if datasets directory exists
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        sys.exit(1)

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find all dataset TSV files
    dataset_files = sorted(datasets_dir.glob("*.tsv"))
    if not dataset_files:
        logger.error(f"No TSV files found in {datasets_dir}")
        sys.exit(1)

    logger.info(f"Found {len(dataset_files)} dataset(s) to evaluate")

    # Evaluate for each dataset
    tree_types = ["SLT", "OPT", "SLT-TYPE"]
    all_results_by_dataset = {}

    for dataset_path in dataset_files:
        dataset_name = dataset_path.stem  # e.g., "chatgpt_golden_dataset"
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*80}\n")

        # Clear output file at start for fresh consolidated results per dataset
        output_path = results_dir / f"result_{dataset_name}.tsv"
        if output_path.exists():
            output_path.unlink()
            logger.info(f"Cleared previous results from {output_path}")

        dataset_results = {}

        # Evaluate for each tree type
        for tree_type in tree_types:
            logger.info(f"\nEvaluating {tree_type}...")

            # Model path
            tree_type_suffix = tree_type.lower().replace("-", "_")
            model_path = (
                model_dir
                / tree_type_suffix
                / f"fasttext_model_{tree_type_suffix}.bin"
            )

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}. Skipping {tree_type}.")
                continue

            try:
                # Initialize evaluator
                evaluator = FormulaEmbeddingEvaluator(
                    tree_type=tree_type,
                    model_path=str(model_path),
                    dataset_path=str(dataset_path),
                    output_path=str(output_path),
                )

                # Run evaluation
                results_df = evaluator.evaluate()
                dataset_results[tree_type] = results_df

            except Exception as e:
                logger.error(f"Error evaluating {tree_type}: {e}", exc_info=True)

        all_results_by_dataset[dataset_name] = dataset_results

    # Summary across all datasets and models
    if all_results_by_dataset:
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY: Across All Datasets and Models")
        logger.info(f"{'='*80}\n")

        for dataset_name, dataset_results in all_results_by_dataset.items():
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  {'-'*70}")
            for tree_type, results_df in dataset_results.items():
                valid = results_df["predicted_similarity"].dropna()
                if len(valid) > 0:
                    logger.info(
                        f"  {tree_type:12s}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
                        f"coverage={len(valid)/len(results_df)*100:5.1f}%"
                    )
                else:
                    logger.warning(f"  {tree_type:12s}: No valid predictions")

        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {results_dir}")
        logger.info(
            f"Format: result_{{dataset_name}}.tsv with columns:"
        )
        logger.info(
            f"  - dataset, query_equation, candidate_id, candidate_equation"
        )
        logger.info(
            f"  - relevance (0/1/2/3), model_type (SLT/OPT/SLT-TYPE), predicted_similarity"
        )
        logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
