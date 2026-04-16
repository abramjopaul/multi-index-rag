"""
Path configuration module using pathlib for consistent path management.
All paths are relative to the project root.
"""

from pathlib import Path

# Project root is 4 levels up from this file (config -> multirag -> src -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data paths
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
RUNS_DIR = EXPERIMENTS_DIR / "runs"

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

# Formula index paths
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
