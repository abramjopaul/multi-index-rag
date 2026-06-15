#!/usr/bin/env python3
"""
Search FAISS IVFScalarQuantizer formula index with a LaTeX query.

Usage:
    python search_formula_sq_index.py "2q^2" --k 10 --representation opt
    python search_formula_sq_index.py "\\sqrt{x+y}" --k 5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from multirag.embedding.formula_model_manager import FastTextModelManager
from multirag.formula_search import (
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
    extract_tuples_from_mathml_direct,
    encode_tuples,
)
from multirag.formula_search.latex_mml import LatexToMathML

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FormulaIndexSearcher:
    """Search FAISS IVFScalarQuantizer formula index."""
    
    DIMENSION = 300
    REPRESENTATIONS = ["slt", "opt", "slt_type"]
    
    def __init__(
        self,
        index_path: str | Path,
        embedding_dir: str | Path,
        corpus_path: str | Path,
        representation: str = "opt",
        nprobe: int = 32,
    ):
        """
        Initialize searcher.
        
        Args:
            index_path: Path to FAISS index directory
            embedding_dir: Path to FastText models and encoder maps
            corpus_path: Path to answers.jsonl
            representation: Which representation to search: 'slt', 'opt', or 'slt_type'
            nprobe: Number of clusters to probe during search
        """
        self.index_path = Path(index_path)
        self.embedding_dir = Path(embedding_dir)
        self.corpus_path = Path(corpus_path)
        self.representation = representation.lower()
        self.nprobe = nprobe
        
        if self.representation not in self.REPRESENTATIONS:
            raise ValueError(f"Invalid representation: {self.representation}")
        
        # Load index
        self._index = None
        self.id_map = {}
        self._load_index()
        
        # Load model and tokenizer
        self.model_manager, self.tuple_tokenizer = self._setup_model_and_tokenizer()
        
        # Load formula lookup (formula_id -> latex)
        self._load_formula_lookup()
        
    def _load_index(self) -> None:
        """Load FAISS index from disk."""
        index_file = self.index_path / f"formula_index_sq_{self.representation}.faiss"
        id_map_file = self.index_path / f"id_map_sq_{self.representation}.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")
        
        logger.info(f"Loading FAISS index: {index_file}")
        self._index = faiss.read_index(str(index_file))
        self._index.nprobe = self.nprobe
        
        if id_map_file.exists():
            logger.info(f"Loading ID map: {id_map_file}")
            with open(id_map_file) as f:
                raw = json.load(f)
                self.id_map = {int(k): tuple(v) for k, v in raw.items()}
        
        logger.info(f"Index loaded: {self._index.ntotal} vectors")
    
    def _setup_model_and_tokenizer(self) -> tuple[FastTextModelManager, TupleTokenizer]:
        """Load FastText model and build tokenizer."""
        tree_type_suffix = self.representation.lower().replace("-", "_")
        model_file = self.embedding_dir / tree_type_suffix / f"fasttext_model_{tree_type_suffix}.bin"
        metadata_file = self.embedding_dir / tree_type_suffix / f"training_metadata_{tree_type_suffix}.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"FastText model not found: {model_file}")
        
        logger.info(f"Loading FastText model: {model_file}")
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
        
        token_id_manager = TokenIDManager()
        tuple_tokenizer = TupleTokenizer(
            token_id_manager=token_id_manager,
            embedding_type=embedding_type,
            tokenize_number=tokenize_number,
        )
        
        return model_manager, tuple_tokenizer
    
    def _load_formula_lookup(self) -> None:
        """Load formula_id -> latex mapping from corpus."""
        if not self.corpus_path.exists():
            logger.warning(f"Corpus not found: {self.corpus_path}, formulas won't be displayed")
            self.formula_lookup = {}
            return
        
        logger.info(f"Loading formula lookup from {self.corpus_path}")
        self.formula_lookup = {}
        
        with open(self.corpus_path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % 100000 == 0 and line_idx > 0:
                    logger.info(f"  Loaded {line_idx} documents...")
                
                try:
                    answer = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                for formula_obj in answer.get("formulas", []):
                    formula_id = formula_obj.get("formula_id")
                    latex = formula_obj.get("latex", "")
                    if formula_id:
                        self.formula_lookup[formula_id] = latex
        
        logger.info(f"Loaded {len(self.formula_lookup)} formula lookup entries")
    
    def _generate_query_embedding(self, query_latex: str) -> Optional[np.ndarray]:
        """Generate embedding for a LaTeX query formula."""
        tree_type_mapping = {
            "slt": "SLT",
            "opt": "OPT",
            "slt_type": "SLT-TYPE",
        }
        tree_type = tree_type_mapping.get(self.representation, "SLT")
        
        try:
            logger.info(f"Converting LaTeX to MathML: {query_latex}")
            mathml = LatexToMathML.convert_to_mathml2(query_latex)
            
            if not mathml:
                logger.warning(f"No MathML produced for query: {query_latex}")
                return None
            
            logger.info("Extracting tuples from MathML")
            tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore
            
            if not tuples:
                logger.warning(f"No tuples extracted from query: {query_latex}")
                return None
            
            logger.info(f"Encoding {len(tuples)} tuples")
            encoded_sequence = encode_tuples(tuples, self.tuple_tokenizer)
            
            if not encoded_sequence:
                logger.warning(f"Failed to encode tuples for query: {query_latex}")
                return None
            
            logger.info("Generating FastText embedding")
            embedding_list = self.model_manager.get_sentence_vector(encoded_sequence)
            return np.array(embedding_list, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}", exc_info=True)
            return None
    
    def search(self, query_latex: str, k: int = 10) -> list[dict]:
        """
        Search index with LaTeX query.
        
        Args:
            query_latex: LaTeX formula string
            k: Number of top results to return
        
        Returns:
            List of top-k hits with formula_id, post_id, formula (latex), and score
        """
        if self._index is None:
            raise RuntimeError("Index not loaded")
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query_latex)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding")
            return []
        
        # Search index
        logger.info(f"Searching index for top {k} results")
        qvec = query_embedding.reshape(1, -1)
        distances, indices = self._index.search(qvec, k)
        
        # Build results
        hits = []
        for rank, idx in enumerate(indices[0]):
            if idx in self.id_map:
                post_id, formula_id = self.id_map[idx]
                distance = float(distances[0][rank])
                score = 1.0 / (1.0 + distance)
                formula = self.formula_lookup.get(formula_id, "")
                
                hit = {
                    "rank": rank + 1,
                    "formula_id": formula_id,
                    "post_id": post_id,
                    "formula": formula,
                    "score": score,
                }
                hits.append(hit)
        
        return hits


def main():
    parser = argparse.ArgumentParser(
        description="Search FAISS IVFScalarQuantizer formula index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with OPT representation (default), get top 10
  python search_formula_sq_index.py "2q^2"
  
  # Search with SLT representation, get top 20
  python search_formula_sq_index.py "\\sqrt{x+y}" --representation slt -k 20
  
  # Increase search probe for higher accuracy (slower)
  python search_formula_sq_index.py "x^2+y^2" --nprobe 64
        """
    )
    
    parser.add_argument(
        "formula",
        type=str,
        help="LaTeX formula to search for",
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)",
    )
    
    parser.add_argument(
        "-r", "--representation",
        type=str,
        choices=["slt", "opt", "slt_type"],
        default="opt",
        help="Formula representation to search (default: opt)",
    )
    
    parser.add_argument(
        "--nprobe",
        type=int,
        default=32,
        help="Number of clusters to probe during search (default: 32, higher = more accurate but slower)",
    )
    
    parser.add_argument(
        "--index-path",
        type=str,
        default="/Users/abramjopaul/Documents/projects/multi-index-rag/data/indices/formula/sq_opt",
        help="Path to FAISS index directory",
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="/Users/abramjopaul/Documents/projects/multi-index-rag/data/formula-indexing",
        help="Path to FastText models and encoder maps",
    )
    
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="/Users/abramjopaul/Documents/projects/multi-index-rag/data/processed/collection/answers.jsonl",
        help="Path to answers.jsonl corpus (for formula lookup)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("FORMULA INDEX SEARCHER")
    logger.info("=" * 70)
    logger.info(f"Formula: {args.formula}")
    logger.info(f"Representation: {args.representation}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Nprobe: {args.nprobe}")
    logger.info("=" * 70)
    
    try:
        # Initialize searcher
        searcher = FormulaIndexSearcher(
            index_path=args.index_path,
            embedding_dir=args.embedding_dir,
            corpus_path=args.corpus_path,
            representation=args.representation,
            nprobe=args.nprobe,
        )
        
        # Search
        results = searcher.search(args.formula, k=args.top_k)
        
        # Display results
        logger.info("")
        logger.info("=" * 70)
        logger.info("SEARCH RESULTS")
        logger.info("=" * 70)
        
        if not results:
            logger.info("No results found")
        else:
            for hit in results:
                logger.info(f"Rank {hit['rank']}: {hit['formula_id']} | post_id={hit['post_id']} | formula={hit['formula']} | score={hit['score']:.4f}")
        
        logger.info("")
        logger.info("=" * 70)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
