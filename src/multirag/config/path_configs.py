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

# Formula embedding / FastText model paths
FORMULA_INDEX_DIR = DATA_DIR / "formula-indexing"
FASTTEXT_MODEL_DIR = FORMULA_INDEX_DIR / "fasttext"
# Full-collection formula FAISS index (Task 2) — all post types, visual_id keyed
# Lives under data/indices/formula/ alongside other FAISS indices (not formula-indexing/)
COLLECTION_FORMULA_INDEX_DIR = FORMULA_FAISS_INDEX_DIR / "collection"

# Model version — change this single variable to switch between trained model versions.
# None  → original models at  data/formula-indexing/slt/, data/formula-indexing/opt/, ...
# "v2"  → retrained models at data/formula-indexing/v2/slt/, data/formula-indexing/v2/opt/, ...
FORMULA_MODEL_VERSION: str | None = None
FORMULA_EMBEDDING_DIR = FORMULA_INDEX_DIR / FORMULA_MODEL_VERSION if FORMULA_MODEL_VERSION else FORMULA_INDEX_DIR

# Formula preprocessing paths
FORMULA_CORPUS_CSV = DATA_PROCESSED / "formula_corpus.csv"
FORMULA_TSV = COLLECTION_PROCESSED / "formulas.tsv"

# Config paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_CONFIG_DIR = CONFIGS_DIR / "experiments"
PROMPTS_CONFIG_DIR = CONFIGS_DIR / "prompts"
FORMULA_CONFIG_PATH = CONFIGS_DIR / "formula_indexing.yaml"
