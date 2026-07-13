#!/usr/bin/env python3
"""
Build a full-collection FAISS formula index for Task 2 Formula Retrieval.

Differences from the Task 1 index:
  - Covers ALL post types (question, answer, comment), not just answers.jsonl.
  - Vectors are L2-normalised before insertion so FAISS L2 distances are directly
    interpretable as cosine distances (cosine = 1 - L2^2/2 for unit vectors).
  - Output goes to data/indices/formula/collection/{model_version}/{repr}/

The visual_id mapping (formula_instance_id -> visual_id) is NOT stored in the FAISS
id_map. It is loaded at retrieval time directly from the v3 TSV files.

Usage:
    python experiments/build_collection_formula_index.py --representation slt
    python experiments/build_collection_formula_index.py --representation opt
    python experiments/build_collection_formula_index.py --representation slt_type
    python experiments/build_collection_formula_index.py --representation slt --force
    python experiments/build_collection_formula_index.py --representation slt --limit 50000
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# MathML in TSV cells can exceed the default 131 KB csv field limit
csv.field_size_limit(sys.maxsize)

import faiss
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from logging_config import configure_logging
from multirag.config import Task2ConfigManager
from multirag.config.path_configs import (
    COLLECTION_FORMULA_INDEX_DIR,
    MODELS_DIR,
    OPT_REPRESENTATION,
    SLT_REPRESENTATION,
)
from multirag.formula_search import (
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
    encode_tuples,
    extract_tuples_from_mathml_direct,
)
from multirag.formula_search.encoder_maps import load_maps
from multirag.embedding.formula_model_manager import FastTextModelManager

configure_logging()
logger = logging.getLogger(__name__)

# FAISS index parameters — match FormulaFAISSIndexerIVFScalarQuantizer defaults
NLIST = 16384
NPROBE = 256
QUANTIZER_TYPE = faiss.ScalarQuantizer.QT_fp16
MINI_BATCH = 1024


# ---------------------------------------------------------------------------
# Model + tokenizer setup
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer(
    representation: str,
    embedding_dir: Path,
) -> tuple[FastTextModelManager, TupleTokenizer, str, int, np.ndarray]:
    """Load FastText model and build tokenizer. Returns (model, tokenizer, tree_type, dim, vocab_mean)."""
    suffix = representation.lower().replace("-", "_")
    model_file = embedding_dir / suffix / f"fasttext_model_{suffix}.bin"
    metadata_file = embedding_dir / suffix / f"training_metadata_{suffix}.json"

    if not model_file.exists():
        raise FileNotFoundError(f"FastText model not found: {model_file}")

    logger.info(f"Loading FastText model: {model_file}")
    mm = FastTextModelManager(
        model_path=str(model_file),
        metadata_path=str(metadata_file),
        corpus_path="",
    )
    mm.load()
    meta = mm.get_stats()

    dim: int = meta.get("vector_size", mm.model.vector_size)
    embedding_type_name: str = meta.get("embedding_type", "Both_Separated")
    tokenize_number: bool = meta.get("tokenize_number", True)

    embedding_type_map = {
        "Both_Separated": TupleTokenizationMode.Both_Separated,
        "Type": TupleTokenizationMode.Type,
    }
    embedding_type = embedding_type_map.get(embedding_type_name, TupleTokenizationMode.Both_Separated)

    encoder_maps_path = meta.get("encoder_maps_path")
    # Fall back to co-located file when stored path is stale (e.g. after directory rename)
    if not encoder_maps_path or not Path(encoder_maps_path).exists():
        candidate = embedding_dir / suffix / f"encoder_maps_{suffix}.tsv"
        if candidate.exists():
            encoder_maps_path = str(candidate)
    if encoder_maps_path and Path(encoder_maps_path).exists():
        node_map, edge_map = load_maps(encoder_maps_path)
        node_id = max(node_map.values(), default=60000) + 1
        edge_id = max(edge_map.values(), default=500) + 1
        tim = TokenIDManager(node_id=node_id, edge_id=edge_id, node_map=node_map, edge_map=edge_map)
    else:
        logger.warning("Encoder maps not found — token IDs will not match training vocabulary")
        tim = TokenIDManager()

    tokenizer = TupleTokenizer(
        token_id_manager=tim,
        embedding_type=embedding_type,
        tokenize_number=tokenize_number,
    )

    tree_type_map = {"slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE"}
    tree_type = tree_type_map.get(suffix, "SLT")

    vocab_mean = np.mean(mm.model.wv.vectors, axis=0).astype(np.float32)
    logger.info(f"Model loaded: dim={dim}, tree_type={tree_type}, embedding_type={embedding_type_name}, vocab_mean_norm={np.linalg.norm(vocab_mean):.4f}")
    return mm, tokenizer, tree_type, dim, vocab_mean


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_mathml(
    mathml: str,
    mm: FastTextModelManager,
    tokenizer: TupleTokenizer,
    tree_type: str,
    vocab_mean: np.ndarray,
) -> np.ndarray | None:
    """Embed MathML string as a mean-shifted, L2-normalised vector (cosine_ms metric).

    Returns a unit vector ready to insert into the FAISS L2 index, or None on failure.
    With unit vectors, FAISS squared-L2 distance d satisfies: cosine_sim = 1 - d/2.
    """
    try:
        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore[arg-type]
        if not tuples:
            return None
        encoded = encode_tuples(tuples, tokenizer)
        if not encoded:
            return None
        vec = np.array(mm.get_sentence_vector(encoded), dtype=np.float32)
        if not np.any(vec):
            return None
        vec = vec - vocab_mean          # mean-shift: removes bias toward common tokens
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return None
        return vec / norm               # unit vector → FAISS L2 ≡ cosine distance
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TSV helpers
# ---------------------------------------------------------------------------

def get_tsv_dir(representation: str) -> Path:
    suffix = representation.lower().replace("-", "_")
    if suffix in ("slt", "slt_type"):
        return SLT_REPRESENTATION
    return OPT_REPRESENTATION


def iter_tsv_files(tsv_dir: Path) -> list[Path]:
    files = sorted(tsv_dir.glob("*.tsv"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {tsv_dir}")
    return files


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _cp_path(index_dir: Path, representation: str) -> Path:
    return index_dir / f"build_checkpoint_{representation}.json"


def load_checkpoint(index_dir: Path, representation: str) -> dict:
    cp = _cp_path(index_dir, representation)
    if cp.exists():
        return json.loads(cp.read_text())
    return {"processed": [], "indexed_count": 0, "trained": False}


def save_checkpoint(index_dir: Path, representation: str, processed: list[str], indexed_count: int, trained: bool) -> None:
    _cp_path(index_dir, representation).write_text(
        json.dumps({"processed": processed, "indexed_count": indexed_count, "trained": trained})
    )


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_index(
    representation: str,
    embedding_dir: Path,
    tsv_dir: Path,
    index_dir: Path,
    force: bool = False,
    limit: int | None = None,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    index_file = index_dir / f"formula_index_sq_{representation}.faiss"
    id_map_file = index_dir / f"id_map_sq_{representation}.json"
    tsv_files = iter_tsv_files(tsv_dir)

    # Check if already complete
    cp = load_checkpoint(index_dir, representation)
    all_names = {f.name for f in tsv_files}
    if not force and cp.get("trained") and set(cp.get("processed", [])) == all_names:
        logger.info("Index already complete — skipping. Use --force to rebuild.")
        return

    # Load model
    mm, tokenizer, tree_type, dim, vocab_mean = setup_model_and_tokenizer(representation, embedding_dir)

    logger.info(f"Representation: {representation} | Dim: {dim} | TSV dir: {tsv_dir}")
    logger.info(f"Output index: {index_dir}")
    logger.info(f"Total TSV files: {len(tsv_files)}")

    id_map: dict[int, list] = {}
    if force:
        # Discard checkpoint — rebuild from scratch
        processed: list[str] = []
        indexed_count: int = 0
        already_trained: bool = False
    else:
        processed = cp.get("processed", [])
        indexed_count = cp.get("indexed_count", 0)
        already_trained = cp.get("trained", False)

    # ---- Phase 1: Train FAISS index ----------------------------------------
    if not already_trained:
        train_target = max(NLIST * 40, 655_360)  # at least 40 vectors/cluster
        logger.info(f"Collecting up to {train_target:,} training vectors...")
        training_vecs: list[np.ndarray] = []
        count = 0
        for tsv_file in tqdm(tsv_files, desc="Collecting training vectors"):
            if len(training_vecs) >= train_target:
                break
            with open(tsv_file, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    if limit is not None and count >= limit:
                        break
                    if len(training_vecs) >= train_target:
                        break
                    mathml = row.get("formula", "").strip()
                    if not mathml:
                        continue
                    vec = embed_mathml(mathml, mm, tokenizer, tree_type, vocab_mean)
                    if vec is not None:
                        training_vecs.append(vec)
                    count += 1

        if len(training_vecs) < NLIST:
            raise ValueError(f"Only {len(training_vecs)} training vectors — need at least {NLIST}")

        # embed_mathml already mean-shifts and normalises; faiss.normalize_L2 is a no-op
        # but kept to ensure exact unit length after float32 rounding
        training_arr = np.array(training_vecs, dtype=np.float32)
        faiss.normalize_L2(training_arr)
        del training_vecs

        logger.info(f"Training FAISS IVFScalarQuantizer (NLIST={NLIST}, dim={dim}) on {len(training_arr):,} vectors...")
        quantizer = faiss.IndexFlat(dim)
        index = faiss.IndexIVFScalarQuantizer(quantizer, dim, NLIST, QUANTIZER_TYPE)
        index.train(training_arr)
        index.nprobe = NPROBE
        del training_arr

        faiss.write_index(index, str(index_file))
        save_checkpoint(index_dir, representation, processed, indexed_count, trained=True)
        logger.info("Training complete — empty index saved.")
    else:
        logger.info(f"Resuming from checkpoint ({len(processed)} TSV files already done)...")
        index = faiss.read_index(str(index_file))
        index.nprobe = NPROBE
        if id_map_file.exists():
            raw = json.loads(id_map_file.read_text())
            id_map = {int(k): v for k, v in raw.items()}

    # ---- Phase 2: Add vectors TSV by TSV ------------------------------------
    remaining = [f for f in tsv_files if f.name not in set(processed)]
    logger.info(f"Adding vectors from {len(remaining)} remaining TSV files...")

    total_failed = 0
    total_count = 0

    for tsv_file in tqdm(remaining, desc=f"Indexing ({representation})"):
        batch_vecs: list[np.ndarray] = []
        batch_metas: list[list] = []

        with open(tsv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if limit is not None and total_count >= limit:
                    break
                formula_id = row.get("id", "").strip()
                post_id = row.get("post_id", "").strip()
                mathml = row.get("formula", "").strip()

                if not formula_id or not mathml:
                    continue

                vec = embed_mathml(mathml, mm, tokenizer, tree_type, vocab_mean)
                if vec is None:
                    total_failed += 1
                    continue

                batch_vecs.append(vec)
                batch_metas.append([post_id, formula_id])
                total_count += 1

                if len(batch_vecs) >= MINI_BATCH:
                    base = index.ntotal
                    index.add(np.array(batch_vecs, dtype=np.float32))
                    for i, meta in enumerate(batch_metas):
                        id_map[base + i] = meta
                    indexed_count += len(batch_vecs)
                    batch_vecs = []
                    batch_metas = []

        # Flush remainder of this file
        if batch_vecs:
            base = index.ntotal
            index.add(np.array(batch_vecs, dtype=np.float32))
            for i, meta in enumerate(batch_metas):
                id_map[base + i] = meta
            indexed_count += len(batch_vecs)

        # Checkpoint after each TSV file
        processed.append(tsv_file.name)
        faiss.write_index(index, str(index_file))
        id_map_file.write_text(json.dumps(id_map))
        save_checkpoint(index_dir, representation, processed, indexed_count, trained=True)
        logger.info(f"  {tsv_file.name}: {indexed_count:,} indexed total, {total_failed} failed")

        if limit is not None and total_count >= limit:
            logger.info(f"Reached limit of {limit} formulas — stopping.")
            break

    logger.info(
        f"Build complete: {indexed_count:,} vectors in index, "
        f"{total_failed} skipped (no valid tuples/norm)."
    )
    logger.info(f"Index: {index_file}")
    logger.info(f"ID map: {id_map_file}  ({len(id_map):,} entries)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full-collection FAISS formula index for Task 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/build_collection_formula_index.py --representation slt --model-version v1
  python experiments/build_collection_formula_index.py --representation opt --model-version v2
  python experiments/build_collection_formula_index.py --config configs/task2/step1_metric_cosine_ms.yaml
  python experiments/build_collection_formula_index.py --representation slt --model-version v1 --limit 100000
  python experiments/build_collection_formula_index.py --representation slt --model-version v1 --force
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Task 2 YAML config file. When provided, all other args are derived from it.",
    )
    parser.add_argument(
        "-r", "--representation",
        choices=["slt", "opt", "slt_type"],
        default=None,
        help="Which formula representation to build the index for (required without --config)",
    )
    parser.add_argument(
        "--model-version",
        choices=["v1", "v2"],
        default="v1",
        help="FastText model version: v1 (n-grams, 300-dim) | v2 (no n-grams, 150-dim) (default: v1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of formulas to index (for testing; default: all)",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default=None,
        help="FastText model directory override (default: data/models/{model-version})",
    )
    args = parser.parse_args()

    # Derive parameters — YAML config takes precedence over CLI flags
    if args.config:
        cfg = Task2ConfigManager.from_yaml(args.config)
        representations = cfg.get_representations()
        model_version = cfg.model_version
        embedding_dir = Path(cfg.embedding_dir) if cfg.embedding_dir else MODELS_DIR / model_version
        force = cfg.force_rebuild
        limit = cfg.index_limit
    else:
        if args.representation is None:
            parser.error("--representation is required when --config is not provided")
        representations = [args.representation]
        model_version = args.model_version
        embedding_dir = Path(args.embedding_dir) if args.embedding_dir else MODELS_DIR / model_version
        force = args.force
        limit = args.limit

    for representation in representations:
        tsv_dir = get_tsv_dir(representation)
        index_dir = COLLECTION_FORMULA_INDEX_DIR / model_version / representation

        logger.info("=" * 70)
        logger.info("TASK 2 COLLECTION FORMULA INDEX BUILD")
        logger.info("=" * 70)
        logger.info(f"Representation : {representation}")
        logger.info(f"Model version  : {model_version}")
        logger.info(f"Embedding dir  : {embedding_dir}")
        logger.info(f"TSV source     : {tsv_dir}")
        logger.info(f"Index output   : {index_dir}")
        logger.info(f"Limit          : {limit or 'all'}")
        logger.info(f"Force rebuild  : {force}")
        logger.info("=" * 70)

        build_index(
            representation=representation,
            embedding_dir=embedding_dir,
            tsv_dir=tsv_dir,
            index_dir=index_dir,
            force=force,
            limit=limit,
        )


if __name__ == "__main__":
    main()
