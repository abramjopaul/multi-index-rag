# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# High-level API for formula tokenization pipeline.
# Combines tree generation, tuple extraction, and token encoding.

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

from multirag.formula_search.encoder_maps import load_maps, save_maps
from multirag.formula_search.opt_generator import OPTGenerator
from multirag.formula_search.slt_generator import SLTGenerator
from multirag.formula_search.tuple_tokenizer import (TokenIDManager,
                                                     TupleTokenizationMode,
                                                     TupleTokenizer)

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory by finding the parent of src/ directory.

    Returns:
        Path to project root
    """
    # Start from the directory containing this file
    current = Path(__file__)
    # Go up: formula_search/ -> src/ -> multirag/
    while current != current.parent:
        if (current / "src").exists() and (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback: assume we're in the project already
    return Path.cwd()


def get_default_encoder_maps_path() -> str:
    """
    Get default path for encoder maps.

    Returns:
        Path string: <project_root>/data/embeddings/encoder_maps.tsv
    """
    project_root = get_project_root()
    encoder_dir = project_root / "data" / "embeddings"
    return str(encoder_dir / "encoder_maps.tsv")


class FormulaTokenizerPipeline:
    """
    High-level API for tokenizing mathematical formulas into encoded token sequences.

    Orchestrates:
    1. LaTeX formula → Symbol tree (SLT or OPT)
    2. Symbol tree → semantic tuples
    3. Semantic tuples → tokenized sequences

    Maintains persistent encoder maps (node/edge ID caches) across multiple formulas
    so consistent IDs are used during training and inference.
    """

    def __init__(
        self,
        embedding_type: TupleTokenizationMode = TupleTokenizationMode.Both_Separated,
        tokenize_all: bool = False,
        tokenize_number: bool = True,
        load_maps_from: Optional[str] = None,
        use_default_encoder_path: bool = False,
    ):
        """
        Initialize tokenizer pipeline.

        Args:
            embedding_type: Node tokenization mode (default: Both_Separated)
            tokenize_all: If True, tokenize each character in node values (default: False)
            tokenize_number: If True, split numbers into digits (default: True)
            load_maps_from: Path to TSV file with pre-existing encoder maps (default: None)
            use_default_encoder_path: If True and load_maps_from is None, try to load from
                default location (data/embeddings/encoder_maps.tsv) if it exists (default: False)

        Raises:
            FileNotFoundError: If load_maps_from path doesn't exist
            ValueError: If encoder maps file has invalid format
        """
        self.embedding_type = embedding_type
        self.encoder_maps_path = load_maps_from or (
            get_default_encoder_maps_path() if use_default_encoder_path else None
        )

        # Initialize token ID manager with optional pre-loaded maps
        if self.encoder_maps_path and Path(self.encoder_maps_path).exists():
            try:
                node_map, edge_map = load_maps(self.encoder_maps_path)
                # Calculate starting IDs from max existing IDs
                node_id = max(node_map.values()) + 1 if node_map else 60000
                edge_id = max(edge_map.values()) + 1 if edge_map else 500
                self.token_manager = TokenIDManager(
                    node_id=node_id,
                    edge_id=edge_id,
                    node_map=node_map,
                    edge_map=edge_map,
                )
                logger.info(f"Loaded encoder maps from {self.encoder_maps_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load encoder maps from {self.encoder_maps_path}: {e}"
                )
                self.token_manager = TokenIDManager()
        else:
            self.token_manager = TokenIDManager()

        # Initialize tree generators
        self.slt_generator = SLTGenerator()
        self.opt_generator = OPTGenerator()

        # Initialize tuple tokenizer
        self.tokenizer = TupleTokenizer(
            token_id_manager=self.token_manager,
            embedding_type=embedding_type,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number,
        )

    def tokenize_formula(
        self,
        latex_formula: str,
        tree_type: Literal["SLT", "OPT"] = "SLT",
        window: int = 2,
        include_eob: bool = True,
    ) -> Optional[List[str]]:
        """
        Tokenize a single LaTeX formula end-to-end.

        Args:
            latex_formula: LaTeX formula string (e.g., r"x^2 + y")
            tree_type: "SLT" for layout tree, "OPT" for operator tree (default: "SLT")
            window: Tuple window size for get_pairs() (default: 2)
            include_eob: Include end-of-baseline token in tuples (default: True)

        Returns:
            List of encoded tuple strings (each string contains chr(token_id) characters),
            or None if formula processing fails
        """
        try:
            # Step 1: Generate symbol tree
            if tree_type == "SLT":
                tree = self.slt_generator.generate(latex_formula)
            elif tree_type == "OPT":
                tree = self.opt_generator.generate(latex_formula)
            else:
                logger.error(f"Unknown tree type: {tree_type}")
                return None

            if tree is None:
                logger.warning(
                    f"Failed to generate {tree_type} for formula: {latex_formula}"
                )
                return None

            # Step 2: Extract tuples from tree
            tuples = tree.get_pairs(window=window, eob=include_eob)
            if not tuples:
                logger.warning(
                    f"No tuples extracted from {tree_type} for formula: {latex_formula}"
                )
                return None

            # Step 3: Tokenize tuples
            encoded = self.tokenizer.encode_batch(tuples)
            return encoded

        except Exception as e:
            logger.error(f"Error tokenizing formula '{latex_formula}': {e}")
            return None

    def tokenize_batch(
        self,
        formulas: Dict[str, str],
        tree_type: Literal["SLT", "OPT"] = "SLT",
        window: int = 2,
        include_eob: bool = True,
    ) -> Dict[str, Optional[List[str]]]:
        """
        Tokenize a batch of formulas, maintaining state across them.

        Args:
            formulas: Dictionary mapping formula_id -> latex_string
            tree_type: "SLT" or "OPT" (default: "SLT")
            window: Tuple window size (default: 2)
            include_eob: Include end-of-baseline token (default: True)

        Returns:
            Dictionary mapping formula_id -> encoded tuples (or None on failure)
        """
        results = {}
        for formula_id, latex in formulas.items():
            encoded = self.tokenize_formula(
                latex_formula=latex,
                tree_type=tree_type,
                window=window,
                include_eob=include_eob,
            )
            results[formula_id] = encoded

        return results

    def save_encoder_maps(self, filepath: Optional[str] = None) -> None:
        """
        Persist encoder maps to file for reuse in inference.

        Args:
            filepath: Path to save TSV file. If None, uses default location
                (data/embeddings/encoder_maps.tsv) from project root.

        Raises:
            IOError: If file cannot be written
        """
        target_path = filepath or get_default_encoder_maps_path()
        node_map, edge_map = self.token_manager.get_updates()
        save_maps(node_map, edge_map, target_path)
        logger.info(f"Saved encoder maps to {target_path}")
        print(f"Saved encoder maps to {target_path}")

    def load_encoder_maps(self, filepath: Optional[str] = None) -> None:
        """
        Load pre-existing encoder maps from file.

        Args:
            filepath: Path to TSV file created by save_encoder_maps(). If None,
                uses default location (data/embeddings/encoder_maps.tsv) from project root.

        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
        """
        target_path = filepath or get_default_encoder_maps_path()
        node_map, edge_map = load_maps(target_path)

        # Update manager with loaded maps
        node_id = max(node_map.values()) + 1 if node_map else 60000
        edge_id = max(edge_map.values()) + 1 if edge_map else 500
        self.token_manager = TokenIDManager(
            node_id=node_id,
            edge_id=edge_id,
            node_map=node_map,
            edge_map=edge_map,
        )

        # Recreate tokenizer with updated manager
        self.tokenizer = TupleTokenizer(
            token_id_manager=self.token_manager,
            embedding_type=self.embedding_type,
        )

        logger.info(f"Loaded encoder maps from {target_path}")

    def get_encoder_stats(self) -> Dict:
        """
        Get statistics about current encoder state.

        Returns:
            Dictionary with node/edge map sizes and ID counters
        """
        node_map, edge_map = self.token_manager.get_updates()
        return {
            "num_node_tokens": len(node_map),
            "num_edge_tokens": len(edge_map),
            "next_node_id": self.token_manager.node_id,
            "next_edge_id": self.token_manager.edge_id,
            "node_token_examples": dict(list(node_map.items())[:5]),
            "edge_token_examples": dict(list(edge_map.items())[:5]),
        }
