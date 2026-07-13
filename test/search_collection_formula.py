#!/usr/bin/env python3
"""
Search a formula in the Task 2 collection FAISS index and display top hits.

Analogous to test/retrieve_top_answers.py but for formula retrieval.
LaTeX for each hit is looked up from latex_representation_v3 (the canonical
formula source), regardless of which SLT/OPT index was searched.

Usage:
    python test/search_collection_formula.py "\\frac{x^2}{y}" --representation slt --model-version v1
    python test/search_collection_formula.py "\\sum_{i=1}^{n} x_i" --representation opt -k 20
    python test/search_collection_formula.py "\\sqrt{a^2+b^2}" --metric cosine
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))  # logging_config lives here

# MathML in TSV cells can exceed the default 131 KB csv field limit
csv.field_size_limit(sys.maxsize)

import faiss
import numpy as np
from tqdm import tqdm

from logging_config import configure_logging
from multirag.config.path_configs import (
    COLLECTION_FORMULA_INDEX_DIR,
    LATEX_REPRESENTATION,
    MODELS_DIR,
)
from multirag.embedding.formula_model_manager import FastTextModelManager
from multirag.formula_search import (
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
    encode_tuples,
    extract_tuples_from_mathml_direct,
)
from multirag.formula_search.encoder_maps import load_maps
from multirag.formula_search.latex_mml import LatexToMathML

configure_logging()
logger = logging.getLogger(__name__)

NPROBE = 256


# ---------------------------------------------------------------------------
# Model loading (shared with task2_formula_retrieval.py)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    representation: str,
    embedding_dir: Path,
) -> tuple[FastTextModelManager, TupleTokenizer, str, int, np.ndarray]:
    suffix = representation.lower().replace("-", "_")
    model_file = embedding_dir / suffix / f"fasttext_model_{suffix}.bin"
    metadata_file = embedding_dir / suffix / f"training_metadata_{suffix}.json"

    if not model_file.exists():
        raise FileNotFoundError(f"FastText model not found: {model_file}")

    mm = FastTextModelManager(
        model_path=str(model_file),
        metadata_path=str(metadata_file),
        corpus_path="",
    )
    mm.load()
    meta = mm.get_stats()

    dim: int = meta.get("vector_size", mm.model.vector_size)
    embedding_type_name = meta.get("embedding_type", "Both_Separated")
    tokenize_number = meta.get("tokenize_number", True)

    embedding_type_map = {
        "Both_Separated": TupleTokenizationMode.Both_Separated,
        "Type": TupleTokenizationMode.Type,
    }
    embedding_type = embedding_type_map.get(embedding_type_name, TupleTokenizationMode.Both_Separated)

    encoder_maps_path = meta.get("encoder_maps_path")
    # Fall back to co-located file when stored path is stale (e.g. after directory rename)
    if not encoder_maps_path or not Path(encoder_maps_path).exists():
        suffix = representation.lower().replace("-", "_")
        candidate = embedding_dir / suffix / f"encoder_maps_{suffix}.tsv"
        if candidate.exists():
            encoder_maps_path = str(candidate)
    if encoder_maps_path and Path(encoder_maps_path).exists():
        node_map, edge_map = load_maps(encoder_maps_path)
        node_id = max(node_map.values(), default=60000) + 1
        edge_id = max(edge_map.values(), default=500) + 1
        tim = TokenIDManager(node_id=node_id, edge_id=edge_id, node_map=node_map, edge_map=edge_map)
    else:
        logger.warning("Encoder maps not found; token IDs will not match training vocabulary")
        tim = TokenIDManager()

    tokenizer = TupleTokenizer(
        token_id_manager=tim,
        embedding_type=embedding_type,
        tokenize_number=tokenize_number,
    )

    tree_type_map = {"slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE"}
    tree_type = tree_type_map.get(suffix, "SLT")

    vocab_mean = np.mean(mm.model.wv.vectors, axis=0).astype(np.float32)
    logger.info(f"Model loaded: repr={representation}, dim={dim}")
    return mm, tokenizer, tree_type, dim, vocab_mean


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def embed_query(
    latex: str,
    mm: FastTextModelManager,
    tokenizer: TupleTokenizer,
    tree_type: str,
    vocab_mean: np.ndarray,
) -> np.ndarray | None:
    """Embed a LaTeX query as a mean-shifted, L2-normalised vector (cosine_ms).

    This must match the preprocessing applied by build_collection_formula_index.py.
    Score from FAISS squared-L2 distance d: cosine_sim = 1 - d/2.
    """
    try:
        if tree_type == "OPT":
            mathml = LatexToMathML.convert_to_mathml2(latex)
        else:
            mathml = LatexToMathML.convert_to_mathml(latex)

        if not mathml:
            return None

        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)  # type: ignore[arg-type]
        if not tuples:
            return None

        encoded = encode_tuples(tuples, tokenizer)
        if not encoded:
            return None

        vec = np.array(mm.get_sentence_vector(encoded), dtype=np.float32)
        if not np.any(vec):
            return None

        vec = vec - vocab_mean      # mean-shift (matches index build)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return None
        return vec / norm

    except Exception as e:
        logger.debug(f"Embedding failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FAISS index loading
# ---------------------------------------------------------------------------

def load_faiss_index(
    representation: str,
    model_version: str,
) -> tuple[faiss.Index, dict[int, list]]:
    index_dir = COLLECTION_FORMULA_INDEX_DIR / model_version / representation
    index_file = index_dir / f"formula_index_sq_{representation}.faiss"
    id_map_file = index_dir / f"id_map_sq_{representation}.json"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Collection FAISS index not found: {index_file}\n"
            f"Build it first:\n"
            f"  python experiments/build_collection_formula_index.py "
            f"--representation {representation} --model-version {model_version}"
        )

    logger.info(f"Loading FAISS index: {index_file}")
    index = faiss.read_index(str(index_file))
    index.nprobe = NPROBE
    logger.info(f"Index loaded: {index.ntotal:,} vectors")

    raw = json.loads(id_map_file.read_text())
    id_map: dict[int, list] = {int(k): v for k, v in raw.items()}
    return index, id_map


# ---------------------------------------------------------------------------
# LaTeX lookup from latex_representation_v3
# ---------------------------------------------------------------------------

def _build_file_index() -> list[tuple[int, Path]]:
    """Return [(first_formula_id, tsv_path), ...] sorted by first_formula_id.

    Formula IDs are monotonically increasing across files, so this lets us
    binary-search directly to the right file(s) instead of scanning all 100+.
    """
    tsv_files = sorted(LATEX_REPRESENTATION.glob("*.tsv"), key=lambda p: int(p.stem))
    index: list[tuple[int, Path]] = []
    for tsv_file in tsv_files:
        with open(tsv_file, newline="") as f:
            next(f)  # skip header
            first_line = next(f, "")
        first_id_str = first_line.split("\t", 1)[0].strip()
        if first_id_str.isdigit():
            index.append((int(first_id_str), tsv_file))
    return index


def build_formula_latex_map(formula_ids: set[str]) -> dict[str, str]:
    """Return {formula_id: latex, formula_id__visual_id: vid} for the requested ids.

    Uses the monotone formula_id ordering to binary-search to the right TSV
    file(s) — typically only 1-2 files scanned regardless of collection size.
    """
    import bisect
    from collections import defaultdict

    file_index = _build_file_index()
    if not file_index:
        logger.warning(f"No TSV files in {LATEX_REPRESENTATION}")
        return {}

    sorted_starts = [entry[0] for entry in file_index]

    # Group target formula_ids by which file they fall in
    file_to_ids: dict[int, set[str]] = defaultdict(set)
    for fid_str in formula_ids:
        if fid_str.isdigit():
            fid_int = int(fid_str)
            file_idx = bisect.bisect_right(sorted_starts, fid_int) - 1
            if file_idx >= 0:
                file_to_ids[file_idx].add(fid_str)

    # Scan only the relevant files
    latex_map: dict[str, str] = {}
    for file_idx, ids_to_find in file_to_ids.items():
        _, tsv_file = file_index[file_idx]
        remaining = set(ids_to_find)
        with open(tsv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                fid = row.get("id", "").strip()
                if fid in remaining:
                    latex_map[fid] = row.get("formula", "").strip()
                    latex_map[f"{fid}__visual_id"] = row.get("visual_id", "").strip()
                    remaining.discard(fid)
                    if not remaining:
                        break

    return latex_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a LaTeX formula in the Task 2 collection FAISS index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/search_collection_formula.py "\\frac{x^2}{y}" --representation slt
  python test/search_collection_formula.py "\\sum_{i=1}^{n} x_i" --representation opt -k 20
  python test/search_collection_formula.py "\\sqrt{a^2+b^2}" --metric cosine_ms
        """,
    )
    parser.add_argument(
        "query",
        type=str,
        help="LaTeX formula to search for (e.g. '\\\\frac{x^2}{y}')",
    )
    parser.add_argument(
        "-r", "--representation",
        choices=["slt", "opt", "slt_type"],
        default="slt",
        help="Formula representation to search (default: slt)",
    )
    parser.add_argument(
        "--model-version",
        choices=["v1", "v2"],
        default="v1",
        help="FastText model version (default: v1)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of results to display (default: 10)",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default=None,
        help="FastText model directory override (default: data/models/{model-version})",
    )
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else MODELS_DIR / args.model_version

    logger.info("=" * 60)
    logger.info("FORMULA SEARCH")
    logger.info("=" * 60)
    logger.info(f"Query      : {args.query}")
    logger.info(f"Repr       : {args.representation}")
    logger.info(f"Model ver  : {args.model_version}")
    logger.info(f"Top-K      : {args.top_k}")
    logger.info("=" * 60)

    # Load model
    mm, tokenizer, tree_type, dim, vocab_mean = load_model_and_tokenizer(
        args.representation, embedding_dir
    )

    # Embed query
    query_vec = embed_query(args.query, mm, tokenizer, tree_type, vocab_mean)
    if query_vec is None:
        print("ERROR: Could not embed query formula. Check that LaTeX is valid and latexmlmath is installed.")
        sys.exit(1)

    # Load index
    index, id_map = load_faiss_index(args.representation, args.model_version)

    # Search — fetch more than top_k to ensure we have enough after potential gaps
    fetch_k = min(args.top_k * 2, index.ntotal)
    distances, indices = index.search(query_vec.reshape(1, -1), fetch_k)

    # Build hits list
    hits: list[tuple[str, str, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        idx_int = int(idx)
        if idx_int < 0 or idx_int not in id_map:
            continue
        post_id, formula_id = id_map[idx_int]
        score = float(1.0 - dist / 2.0)  # cosine_ms: FAISS sq-L2 on unit sphere → 1 - d/2
        hits.append((post_id, formula_id, score))
        if len(hits) >= args.top_k:
            break

    # Look up LaTeX and visual_id from latex_representation_v3
    formula_ids = {fid for _, fid, _ in hits}
    latex_map = build_formula_latex_map(formula_ids)

    # Display results
    print(f"\nTop {len(hits)} results for: {args.query}\n")
    print(f"{'Rank':<5} {'Score':>8}  {'formula_id':<15} {'visual_id':<15} {'post_id':<12}  formula")
    print("-" * 100)
    for rank, (post_id, formula_id, score) in enumerate(hits, start=1):
        visual_id = latex_map.get(f"{formula_id}__visual_id", "—")
        latex = latex_map.get(formula_id, "")
        print(f"{rank:<5} {score:>8.4f}  {formula_id:<15} {visual_id:<15} {post_id:<12}  {latex[:80]}")
    print()


if __name__ == "__main__":
    main()
