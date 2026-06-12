"""
Path configuration module using pathlib for consistent path management.
Intelligently switches between local and GCS paths based on environment.
"""

import os
from pathlib import Path

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================


def _is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        from google.colab import drive  # noqa

        return True
    except ImportError:
        return False


def _is_gcs_mode() -> bool:
    """
    Detect if should use GCS paths.
    - True if running in Colab (default)
    - Can be overridden by USE_GCS_PATHS environment variable
    """
    use_gcs = os.getenv("USE_GCS_PATHS", "").lower()

    if use_gcs in ("true", "1", "yes"):
        return True
    elif use_gcs in ("false", "0", "no"):
        return False
    else:
        # Default: use GCS if in Colab
        return _is_colab()


# Global flags
IN_COLAB = _is_colab()
USE_GCS = _is_gcs_mode()

# GCS bucket configuration
GCS_BUCKET = "multi-index-rag-bucket"


# ============================================================================
# PATH INITIALIZATION
# ============================================================================

if USE_GCS:
    # GCS paths (for Google Colab)
    from pathlib import PurePosixPath

    DATA_DIR = PurePosixPath(GCS_BUCKET) / "data"
    PROJECT_ROOT = PurePosixPath(GCS_BUCKET)
else:
    # Local paths (for local development)
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_RAW = DATA_DIR / "raw"

# Collection paths
COLLECTION_DIR = DATA_RAW / "collection"
POSTS_XML = COLLECTION_DIR / "Posts.V1.3.xml"

# Formula representation paths
FORMULA_DIR = COLLECTION_DIR / "formula"
LATEX_REPRESENTATION = FORMULA_DIR / "latex_representation_v3"
OPT_REPRESENTATION = FORMULA_DIR / "opt_representation_v3"
SLT_REPRESENTATION = FORMULA_DIR / "slt_representation_v3"

# Topics paths
TOPICS_DIR = DATA_RAW / "topics"
TOPICS_XML = TOPICS_DIR / "Topics_Task1_2022_V0.1.xml"
TOPICS_FORMULA_DIR = TOPICS_DIR / "formula"
TOPICS_FORMULAS_LATEX = TOPICS_FORMULA_DIR / "Topics_Formulas_Latex.V0.1.tsv"
TOPICS_FORMULAS_OPT = TOPICS_FORMULA_DIR / "Topics_Formulas_OPT.V0.1.tsv"
TOPICS_FORMULAS_SLT = TOPICS_FORMULA_DIR / "Topics_Formulas_SLT.V0.1.tsv"

# Qrels paths
QRELS_DIR = DATA_RAW / "qrels"
QREL_TASK1_2022_ADDITIONAL = QRELS_DIR / "qrel_task1_2022_additional.tsv"
QREL_TASK1_2022_ALL = QRELS_DIR / "qrel_task1_2022_all.tsv"
QREL_TASK1_2022_OFFICIAL = QRELS_DIR / "qrel_task1_2022_official.tsv"

# Experiments paths
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = DATA_DIR / "runs"

# Source code paths
SRC_DIR = PROJECT_ROOT / "src"
MULTIRAG_DIR = SRC_DIR / "multirag"
PREPROCESSING_DIR = MULTIRAG_DIR / "preprocessing"

# Processed output paths
COLLECTION_PROCESSED = DATA_PROCESSED / "collection"
ANSWERS_JSONL = COLLECTION_PROCESSED / "answers.jsonl"
QUESTIONS_JSONL = COLLECTION_PROCESSED / "questions.jsonl"
# INDEXABLE_CORPUS_JSONL = COLLECTION_PROCESSED / "indexable_corpus.jsonl"

TOPICS_PROCESSED = DATA_PROCESSED / "topics"
TOPICS_JSONL = TOPICS_PROCESSED / "topics.jsonl"

QRELS_PROCESSED = DATA_PROCESSED / "qrels"
QREL_2022_JSONL = QRELS_PROCESSED / "qrel_2022.jsonl"

# Index paths
INDEX_DIR = DATA_DIR / "indices"
SPARSE_INDEX_PATH = INDEX_DIR / "sparse_bm25"
DENSE_INDEX_PATH = INDEX_DIR / "dense"
FORMULA_FAISS_INDEX_DIR = INDEX_DIR / "formula"

# Formula embedding / FastText model paths
FORMULA_INDEX_DIR = DATA_DIR / "formula-indexing"
FASTTEXT_MODEL_DIR = FORMULA_INDEX_DIR / "fasttext"

# Formula preprocessing paths
FORMULA_CORPUS_CSV = DATA_PROCESSED / "formula_corpus.csv"
FORMULA_TSV = COLLECTION_PROCESSED / "formulas.tsv"

# Config paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_CONFIG_DIR = CONFIGS_DIR / "experiments"
PROMPTS_CONFIG_DIR = CONFIGS_DIR / "prompts"
FORMULA_CONFIG_PATH = CONFIGS_DIR / "formula_indexing.yaml"

# ============================================================================
# PATH CONFIGURATION INFO (for debugging)
# ============================================================================

import logging

logger = logging.getLogger(__name__)

# Log configuration info on import
_log_msg = f"Path configuration: "
if USE_GCS:
    _log_msg += f"GCS mode (bucket: {GCS_BUCKET})"
else:
    _log_msg += f"Local mode (root: {PROJECT_ROOT})"

_log_msg += f" | IN_COLAB: {IN_COLAB}"

try:
    logger.debug(_log_msg)
except:
    pass  # Logging not configured yet

# ============================================================================
# HOW TO CONFIGURE
# ============================================================================
# Local development (default):
#   USE_GCS_PATHS=0 python script.py
#   OR: export USE_GCS_PATHS=false
#
# Google Colab (auto-detected):
#   Automatically uses GCS when running in Colab
#   Override with: export USE_GCS_PATHS=false
#
# Manual GCS mode:
#   USE_GCS_PATHS=1 python script.py
# ============================================================================
