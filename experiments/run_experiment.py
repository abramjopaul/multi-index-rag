#!/usr/bin/env python3
"""Main experiment runner script for ARQMath Task 1 retrieval evaluation.

This script orchestrates the complete evaluation pipeline:
1. Load YAML config file
2. Instantiate appropriate indexer (sparse, dense, formula)
3. Generate TREC run file
4. Evaluate metrics against qrels
5. Log results to Weights & Biases (W&B)
6. Print formatted report to console

Usage:
    poetry run python experiments/run_experiment.py configs/experiments/run_config.yaml
    poetry run python experiments/run_experiment.py configs/experiments/run_config.yaml --verbose
    poetry run python experiments/run_experiment.py configs/experiments/run_config.yaml --dry-run
"""

import os

# Must be set before any OpenMP or JVM library loads.
# Prevents SIGSEGV crash when PyTorch (libomp) and JVM coexist on macOS/ARM.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import wandb

# Add src to path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logging_config import configure_logging

from multirag.config import RunConfigManager
from multirag.config.path_configs import (
    ANSWERS_JSONL,
    DENSE_INDEX_PATH,
    FORMULA_DIR,
    FORMULA_FAISS_INDEX_DIR,
    FORMULA_INDEX_DIR,
    QREL_TASK1_2022_OFFICIAL,
    RUNS_DIR,
    SPARSE_INDEX_PATH,
    TOPICS_JSONL,
)
from multirag.evaluation.metrics import (
    evaluate_run,
    generate_run_file,
    print_evaluation_report,
)
# Configure logging (accepts CLI arg or defaults to INFO)
configure_logging()
logger = logging.getLogger(__name__)


def create_indexer(
    index_type: str | list[str],
    index_corpus_limit: int | None = None,
    force_rebuild: bool = False,
    **kwargs,
):
    """Factory function to create an indexer based on type.

    Args:
        index_type: Type of indexer ("sparse", "dense", "formula") or list for fusion.
        index_corpus_limit: Optional limit on number of documents to index.
        force_rebuild: Force rebuild of index (default: False, reuse existing).

    Returns:
        Instantiated and indexed BaseIndexer subclass.

    Raises:
        ValueError: If index_type is not supported.
        NotImplementedError: If index_type is not yet implemented.
    """
    # Handle list for future fusion systems
    if isinstance(index_type, list):
        raise NotImplementedError(
            f"Fusion systems (multiple index types) not yet implemented: {index_type}"
        )

    # Single index type
    if index_type == "sparse":
        from multirag.indexing.sparse import PyseriniSparseIndexer

        logger.info(f"Creating PyseriniSparseIndexer with path={SPARSE_INDEX_PATH}")
        indexer = PyseriniSparseIndexer(
            index_path=SPARSE_INDEX_PATH, #type: ignore
            corpus_path=ANSWERS_JSONL,  #type: ignore
        )
        logger.info(
            f"Indexing corpus (limit={index_corpus_limit}, force_rebuild={force_rebuild})..."
        )
        indexer.index(force=force_rebuild, limit=index_corpus_limit)
        logger.info("Indexing complete")
        return indexer

    elif index_type == "dense":
        from multirag.indexing.dense import PyseriniDenseIndexer

        logger.info(f"Creating PyseriniDenseIndexer with path={DENSE_INDEX_PATH}")
        indexer = PyseriniDenseIndexer(
            index_path=DENSE_INDEX_PATH,  # type: ignore
            corpus_path=ANSWERS_JSONL,  # type: ignore
        )
        logger.info(
            f"Indexing corpus (limit={index_corpus_limit}, force_rebuild={force_rebuild})..."
        )
        indexer.index(force=force_rebuild, limit=index_corpus_limit)
        logger.info("Indexing complete")
        return indexer

    elif index_type == "formula":
        from multirag.indexing.formula import FormulaFAISSIndexerIVFScalarQuantizer

        config = kwargs.get("config")
        if config is None:
            raise ValueError("create_indexer() requires 'config' kwarg for formula index_type")

        representation = config.formula_representation
        index_path = config.formula_index_path or str(
            FORMULA_FAISS_INDEX_DIR / f"sq_{representation}"
        )
        embedding_dir = config.formula_embedding_dir or str(FORMULA_INDEX_DIR)
        tsv_base_dir = config.formula_tsv_base_dir or str(FORMULA_DIR)

        logger.info(
            f"Creating FormulaFAISSIndexerIVFScalarQuantizer: "
            f"representation={representation}, index_path={index_path}"
        )
        indexer = FormulaFAISSIndexerIVFScalarQuantizer(
            index_path=index_path,
            corpus_path=ANSWERS_JSONL,  #type: ignore
            embedding_dir=embedding_dir,
            representation=representation,
            formula_tsv_base_dir=tsv_base_dir,
            force_rebuild=force_rebuild,
        )
        indexer.index(force=force_rebuild, limit=index_corpus_limit)
        return indexer

    else:
        raise ValueError(
            f"Unknown index_type: {index_type}. "
            f"Must be one of: 'sparse', 'dense', 'formula'"
        )


def main():
    """Main entry point for experiment runner."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run ARQMath Task 1 retrieval experiment with W&B logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python experiments/run_experiment.py configs/experiments/run_config.yaml
  poetry run python experiments/run_experiment.py configs/experiments/run_config.yaml --verbose
  poetry run python experiments/run_experiment.py configs/experiments/run_config.yaml --dry-run
        """,
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML config file (relative or absolute)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: skip W&B login and logging (useful for testing)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Convert config path to absolute path
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    try:
        # Load and validate config
        logger.info(f"Loading config from {config_path}")
        config = RunConfigManager.from_yaml(config_path)
        config.run_name = config.run_name.replace(" ", "_")
        config.run_name = f"{config.run_name}_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Config loaded successfully: {config.run_name}")

        # Initialize W&B (unless dry-run)
        if not args.dry_run:
            logger.info("Initializing Weights & Biases...")
            wandb.init(
                project="multi-index-rag",
                name=config.run_name,
                group=config.get_experiment_name(),
                tags=(
                    [config.index_type]
                    if isinstance(config.index_type, str)
                    else config.index_type
                ),
                config=config.model_dump(),
            )
            logger.info(f"W&B initialized: {wandb.run.url}")  #type: ignore
        else:
            logger.info("Dry-run mode: W&B logging disabled")

        # Create indexer
        logger.info(f"Creating indexer for type: {config.index_type}")
        indexer = create_indexer(
            index_type=config.index_type,
            index_corpus_limit=config.index_corpus_limit,
            force_rebuild=config.force_rebuild,
            config=config,
        )

        # Generate run file
        run_path = RUNS_DIR / f"{config.run_name}.tsv"
        logger.info(f"Generating run file at {run_path}")
        generate_run_file(
            indexer=indexer,
            topics_path=TOPICS_JSONL,
            output_path=run_path,
            run_name=config.run_name,
            k=config.num_hits,
        )
        logger.info(f"Run file generated: {run_path} ({run_path.stat().st_size} bytes)")  #type: ignore

        # Evaluate metrics
        logger.info("Evaluating run against qrels...")
        metrics_dict = evaluate_run(
            qrels_path=QREL_TASK1_2022_OFFICIAL,
            run_path=run_path,
        )
        logger.info("Evaluation complete")

        # Log to W&B (unless dry-run)
        if not args.dry_run:
            logger.info("Logging results to W&B...")

            # Create metrics table for W&B
            metrics_table = wandb.Table(columns=["Metric", "Value"])
            for metric_name, metric_value in sorted(metrics_dict.items()):
                metrics_table.add_data(metric_name, metric_value)

            # Log metrics as table
            wandb.log(
                {
                    "metrics_table": metrics_table,
                    "timestamp": datetime.now().isoformat(),
                    "run_file_path": str(run_path),
                    "config_file_path": str(config_path),
                    "topics_path": str(TOPICS_JSONL),
                    "qrels_path": str(QREL_TASK1_2022_OFFICIAL),
                }
            )

            # Upload files as artifacts
            logger.info("Uploading files as artifacts...")
            wandb.save(str(run_path), base_path=RUNS_DIR)
            wandb.save(str(config_path), base_path=config_path.parent)

            logger.info("W&B logging complete")

        # Print formatted report
        logger.info("Printing evaluation report...")
        print_evaluation_report(
            qrels_path=QREL_TASK1_2022_OFFICIAL,
            run_path=run_path,
            run_name=config.run_name,
        )

        # Summary
        logger.info("=" * 70)
        logger.info(f"Experiment completed successfully: {config.run_name}")
        logger.info(f"Run file: {run_path}")
        if not args.dry_run:
            logger.info(f"W&B URL: {wandb.run.url}")  #type: ignore
        logger.info("=" * 70)

        # Finalize W&B
        if not args.dry_run:
            wandb.finish()

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        if not args.dry_run:
            try:
                wandb.log({"error": str(e)})
                wandb.finish()
            except Exception:
                pass
        return 1

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        if not args.dry_run:
            try:
                wandb.log({"error": str(e)})
                wandb.finish()
            except Exception:
                pass
        return 1

    except NotImplementedError as e:
        logger.error(f"Not implemented: {e}")
        if not args.dry_run:
            try:
                wandb.log({"error": str(e)})
                wandb.finish()
            except Exception:
                pass
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if not args.dry_run:
            try:
                wandb.log({"error": str(e)})
                wandb.finish()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
