import argparse
import logging
import time
from typing import Optional, List

import numpy
import numpy as np
from logging_config import configure_logging

from multirag.embedding.formula_embedder import FormulaEmbedder
from multirag.embedding.formula_trainer import FormulaTrainer
from multirag.embedding.formula_trainer_direct import FormulaTrainerDirect
from multirag.formula_search.opt_generator import OPTGenerator
from multirag.formula_search.slt_generator import SLTGenerator

# Configure logging (accepts CLI arg or defaults to INFO)
configure_logging()
logger = logging.getLogger(__name__)


def parse_file_numbers(file_numbers_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse file numbers from command-line argument.
    
    Supports:
    - None: load all files (default)
    - "1,2,3" → [1, 2, 3]
    - "1-101" → [1, 2, 3, ..., 101]
    - "1,5,10-15" → [1, 5, 10, 11, 12, 13, 14, 15]
    
    Args:
        file_numbers_str: String representation of file numbers
        
    Returns:
        List of file numbers or None to load all
    """
    if file_numbers_str is None or file_numbers_str.lower() == "all":
        return None
    
    file_numbers = []
    parts = file_numbers_str.split(",")
    
    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like "10-15"
            start, end = part.split("-")
            file_numbers.extend(range(int(start), int(end) + 1))
        else:
            # Single number
            file_numbers.append(int(part))
    
    return sorted(list(set(file_numbers)))  # Remove duplicates and sort


def main():
    """Main training function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train FastText models on formula representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SLT on files 1-5, all formulas
  python train_formula_models.py -t SLT -f "1-5"
  
  # Train OPT on files 1,2,3 with max 50000 formulas
  python train_formula_models.py -t OPT -f "1,2,3" -n 50000
  
  # Train SLT-TYPE on all files (no limits)
  python train_formula_models.py -t SLT-TYPE
  
  # Train SLT on specific files with max formulas
  python train_formula_models.py -t SLT -f "1-10,50-60" -n 100000
        """
    )
    
    parser.add_argument(
        "-t", "--tree-type",
        type=str,
        default="SLT",
        choices=["SLT", "OPT", "SLT-TYPE"],
        help="Tree representation type (default: SLT)"
    )
    
    parser.add_argument(
        "-f", "--file-numbers",
        type=str,
        default=None,
        help="""File numbers to train on. Supports:
                - None or 'all': all files (default)
                - '1,2,3': specific files
                - '1-101': range of files
                - '1,5,10-15': mixed format
                """
    )
    
    parser.add_argument(
        "-n", "--num-formulas",
        type=int,
        default=None,
        help="Maximum number of formulas to load (default: None = load all)"
    )
    
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=5000,
        help="Formulas per chunk when reading TSV (default: 5000)"
    )
    
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)"
    )
    
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=5,
        help="Training epochs (default: 5)"
    )
    
    parser.add_argument(
        "--vector-size",
        type=int,
        default=150,
        help="FastText vector dimension (default: 150)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for model artifacts (default: data/models). "
             "Use e.g. data/models/v2 for new model versions."
    )

    args = parser.parse_args()
    
    # Parse file numbers
    file_numbers = parse_file_numbers(args.file_numbers)
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("FORMULA TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Tree type: {args.tree_type}")
    logger.info(f"File numbers: {file_numbers if file_numbers else 'ALL'}")
    logger.info(f"Max formulas: {args.num_formulas if args.num_formulas else 'ALL'}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Vector size: {args.vector_size}")
    logger.info(f"Output dir: {args.output_dir if args.output_dir else 'default (data/models)'}")
    logger.info("=" * 70 + "\n")
    
    # Start training
    start_time = time.time()
    
    try:
        logger.info(f"Creating {args.tree_type} trainer...")
        trainer = FormulaTrainerDirect(
            tree_type=args.tree_type,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
        )
        
        ####Debugging: use FormulaTrainer for now to isolate issues with direct trainer
        # trainer = FormulaTrainer(
        #     tree_type=args.tree_type,
        #     num_workers=args.num_workers,
        #     vector_size=args.vector_size,

        # )
        
        logger.info("Starting training...")
        model = trainer.train(
            file_numbers=file_numbers,
            num_formulas=args.num_formulas,
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Formulas loaded: {trainer.num_formulas_loaded}")
        logger.info(f"Processing errors: {trainer.total_errors}")
        logger.info(f"Model saved: {trainer.model_path}")
        logger.info(f"Corpus saved: {trainer.corpus_path}")
        logger.info(f"Encoder maps: {trainer.encoder_maps_path}")
        logger.info("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
