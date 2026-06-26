import argparse
import logging
import time
from pathlib import Path

from logging_config import configure_logging
from multirag.config.path_configs import FORMULA_EMBEDDING_DIR
from multirag.embedding.formula_trainer_direct import FormulaTrainerDirect

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def main():
    """Train FastText model directly from corpus file (memory-efficient)."""
    parser = argparse.ArgumentParser(
        description="Train FastText models from existing corpus file (memory-efficient)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SLT from corpus with 5 epochs
  python train_from_corpus.py -c ./data/formula-indexing/slt/corpus_slt.txt -e 5
  
  # Train OPT from corpus with custom workers
  python train_from_corpus.py -c ./data/formula-indexing/opt/corpus_opt.txt -w 16 -e 10
  
  # Train SLT-TYPE from corpus
  python train_from_corpus.py -c ./data/formula-indexing/slt_type/corpus_slt_type.txt -t SLT-TYPE
        """
    )
    
    parser.add_argument(
        "-c", "--corpus-path",
        type=str,
        default=None,
        help="Path to corpus file (LineSentence format). "
             "Defaults to FORMULA_EMBEDDING_DIR/<tree_type>/corpus_<tree_type>.txt "
             "(resolved from path_configs.FORMULA_EMBEDDING_DIR)"
    )
    
    parser.add_argument(
        "-t", "--tree-type",
        type=str,
        default="SLT",
        choices=["SLT", "OPT", "SLT-TYPE"],
        help="Tree representation type (default: SLT)"
    )
    
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=5,
        help="Training epochs (default: 5)"
    )
    
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)"
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
        help="Output directory for model artifacts. "
             "Defaults to FORMULA_EMBEDDING_DIR (from path_configs.FORMULA_MODEL_VERSION). "
             "Override only to write to a different location than the config specifies."
    )

    args = parser.parse_args()

    # Resolve output dir: CLI arg > FORMULA_EMBEDDING_DIR from config
    output_dir = args.output_dir or str(FORMULA_EMBEDDING_DIR)

    # Resolve corpus path: CLI arg > <output_dir>/<tree_type_suffix>/corpus_<tree_type_suffix>.txt
    tree_type_suffix = args.tree_type.lower().replace("-", "_")
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
    else:
        corpus_path = Path(output_dir) / tree_type_suffix / f"corpus_{tree_type_suffix}.txt"

    if not corpus_path.exists():
        logger.error(f"✗ Corpus file not found: {corpus_path}")
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("CORPUS-BASED TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Corpus file: {corpus_path}")
    logger.info(f"Tree type: {args.tree_type}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"Vector size: {args.vector_size}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 70 + "\n")
    
    # Start training
    start_time = time.time()
    
    try:
        logger.info(f"Creating {args.tree_type} trainer...")
        trainer = FormulaTrainerDirect(
            tree_type=args.tree_type,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            epochs=args.epochs,
            output_dir=output_dir,
        )
        
        logger.info("Starting corpus-based training (memory-efficient streaming)...")
        model = trainer.train_from_corpus(
            corpus_path=str(corpus_path),
            epochs=args.epochs,
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Model saved: {trainer.model_path}")
        logger.info(f"Corpus used: {corpus_path}")
        logger.info("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
