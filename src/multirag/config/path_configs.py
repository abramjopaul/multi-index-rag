"""Path configuration module using pathlib."""

from pathlib import Path

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
QREL_TASK2_2021_ALL = QRELS_DIR / "qrel_task2_2021_all.tsv"
QREL_TASK2_2022_OFFICIAL = QRELS_DIR / "qrel_task2_2022_official.tsv"

# Task 2 topics XML
TOPICS_TASK2_XML = DATA_RAW / "topics" / "Topics_Task2_2022_V0.1.xml"
TOPICS_TASK2_2021_XML = DATA_RAW / "topics" / "Topics_Task2_2021_V1.1.xml"

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

TOPICS_PROCESSED = DATA_PROCESSED / "topics"
TOPICS_JSONL = TOPICS_PROCESSED / "topics.jsonl"

QRELS_PROCESSED = DATA_PROCESSED / "qrels"
QREL_2022_JSONL = QRELS_PROCESSED / "qrel_2022.jsonl"

# Index paths
INDEX_DIR = DATA_DIR / "indices"
SPARSE_INDEX_PATH = INDEX_DIR / "sparse_bm25"
DENSE_INDEX_PATH = INDEX_DIR / "dense"
FORMULA_FAISS_INDEX_DIR = INDEX_DIR / "formula"

# Formula FAISS index sub-directories
ANSWER_FORMULA_INDEX_DIR = FORMULA_FAISS_INDEX_DIR / "answer"      # Task 1: answers.jsonl-based
COLLECTION_FORMULA_INDEX_DIR = FORMULA_FAISS_INDEX_DIR / "collection"  # Task 2: full-collection, versioned as collection/{v1,v2}/{repr}/

# FastText model directories (formerly data/formula-indexing/)
MODELS_DIR = DATA_DIR / "models"
FASTTEXT_MODELS_V1_DIR = MODELS_DIR / "v1"   # original models: n-grams, 300-dim
FASTTEXT_MODELS_V2_DIR = MODELS_DIR / "v2"   # retrained: no n-grams, 150-dim

# Active model version — change to "v2" to switch globally
FORMULA_MODEL_VERSION: str = "v1"
FORMULA_EMBEDDING_DIR = MODELS_DIR / FORMULA_MODEL_VERSION

# Formula preprocessing paths
FORMULA_CORPUS_CSV = DATA_PROCESSED / "formula_corpus.csv"
FORMULA_TSV = COLLECTION_PROCESSED / "formulas.tsv"

# Config paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_CONFIG_DIR = CONFIGS_DIR / "experiments"
PROMPTS_CONFIG_DIR = CONFIGS_DIR / "prompts"
TASK_C_CONFIG_DIR = CONFIGS_DIR / "task_c"
FORMULA_CONFIG_PATH = CONFIGS_DIR / "formula_indexing.yaml"

# Track C: generation + RAGAS evaluation outputs
TASK_C_RUNS_DIR = RUNS_DIR / "task_c"
