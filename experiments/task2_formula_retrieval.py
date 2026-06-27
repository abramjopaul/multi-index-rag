#!/usr/bin/env python3
"""
ARQMath-3 Task 2 Formula Retrieval evaluation harness.

Pipeline (per topic):
  1. Parse Task 2 topics XML  → {topic_id, query_latex, source_post_id}
  2. Load visual_id map       → formula_instance_id → visual_id  (from v3 TSVs)
  3. Embed query formula      → normalised vector (optional mean-shift)
  4. FAISS retrieval          → top-N (formula_instance_id, score) pairs
  5. Exclude query post       → drop hits from same post_id or same visual_id
  6. Collapse visual_ids      → keep first occurrence of each visual_id (BEFORE truncate)
  7. Truncate to 1000
  8. Write TREC run           → "topic_id Q0 visual_id rank score run_name"
  9. Score with pytrec_eval   → nDCG'@10, MAP', P'@10

Ablations:
  --representation slt | opt | slt_type | slt,opt | slt,opt,slt_type  (RRF if comma-list)
  --metric cosine | cosine_ms | l2
  --n 5000  (retrieval depth before collapse)
  --rrf-k 60

Usage:
  # Smoke-test (5 topics, fast validation):
  python experiments/task2_formula_retrieval.py --representation slt --smoke-test

  # Full run, SLT with mean-shift cosine:
  python experiments/task2_formula_retrieval.py --representation slt --metric cosine_ms

  # RRF fusion (TangentCFT2 analogue):
  python experiments/task2_formula_retrieval.py --representation slt,opt --metric cosine_ms

  # Score against 2021 qrels (for tuning):
  python experiments/task2_formula_retrieval.py --representation slt \\
      --qrels data/raw/qrels/qrel_task2_2021_all.tsv
"""

import argparse
import json
import logging
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from logging_config import configure_logging
from multirag.config.path_configs import (
    COLLECTION_FORMULA_INDEX_DIR,
    FORMULA_EMBEDDING_DIR,
    OPT_REPRESENTATION,
    QREL_TASK2_2022_OFFICIAL,
    SLT_REPRESENTATION,
    TOPICS_TASK2_XML,
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
MAX_RESULTS = 1000


# ---------------------------------------------------------------------------
# Step 1 — Parse Task 2 topics XML
# ---------------------------------------------------------------------------

def parse_task2_topics(xml_path: Path) -> list[dict]:
    """Parse Topics_Task2_2022_V0.1.xml.

    Returns list of {topic_id, query_latex, source_formula_id, source_post_id}.
    source_post_id is not in the XML directly — set to None here; exclusion uses
    the visual_id of the query formula identified during retrieval.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    topics = []
    for topic_el in root.findall("Topic"):
        tid = topic_el.get("number", "").strip()
        latex = (topic_el.findtext("Latex") or "").strip()
        formula_id = (topic_el.findtext("Formula_Id") or "").strip()
        if tid and latex:
            topics.append({
                "topic_id": tid,
                "query_latex": latex,
                "source_formula_id": formula_id,
            })
    return topics


# ---------------------------------------------------------------------------
# Step 2 — Build visual_id map from v3 TSVs
# ---------------------------------------------------------------------------

def build_visual_id_map(representation: str) -> tuple[dict[str, str], dict[str, str]]:
    """Stream v3 TSV files and build two maps:
      formula_id_to_visual_id: str -> str
      formula_id_to_post_id:   str -> str

    Returns (fid_to_vid, fid_to_post).
    Memory: ~224 MB for 28M entries (both maps combined).
    """
    import csv, sys as _sys
    csv.field_size_limit(_sys.maxsize)  # MathML cells can exceed the 131 KB default

    suffix = representation.lower().replace("-", "_")
    tsv_dir = SLT_REPRESENTATION if suffix in ("slt", "slt_type") else OPT_REPRESENTATION

    fid_to_vid: dict[str, str] = {}
    fid_to_post: dict[str, str] = {}

    tsv_files = sorted(tsv_dir.glob("*.tsv"), key=lambda p: int(p.stem))
    logger.info(f"Building visual_id map from {len(tsv_files)} TSV files in {tsv_dir} ...")

    for tsv_file in tqdm(tsv_files, desc="Loading visual_id map"):
        with open(tsv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                fid = row.get("id", "").strip()
                vid = row.get("visual_id", "").strip()
                pid = row.get("post_id", "").strip()
                if fid:
                    fid_to_vid[fid] = vid
                    fid_to_post[fid] = pid

    logger.info(f"Visual_id map: {len(fid_to_vid):,} entries")
    return fid_to_vid, fid_to_post


# ---------------------------------------------------------------------------
# Step 3 — Model loading + query embedding
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    representation: str, embedding_dir: Path
) -> tuple[FastTextModelManager, TupleTokenizer, str, int, np.ndarray]:
    """Load FastText model, tokenizer, and compute vocab mean for mean-shift.

    Returns (model_manager, tokenizer, tree_type, dim, vocab_mean).
    """
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
    if encoder_maps_path and Path(encoder_maps_path).exists():
        node_map, edge_map = load_maps(encoder_maps_path)
        node_id = max(node_map.values(), default=60000) + 1
        edge_id = max(edge_map.values(), default=500) + 1
        tim = TokenIDManager(node_id=node_id, edge_id=edge_id, node_map=node_map, edge_map=edge_map)
    else:
        logger.warning("Encoder maps not found")
        tim = TokenIDManager()

    tokenizer = TupleTokenizer(
        token_id_manager=tim,
        embedding_type=embedding_type,
        tokenize_number=tokenize_number,
    )

    tree_type_map = {"slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE"}
    tree_type = tree_type_map.get(suffix, "SLT")

    vocab_mean = np.mean(mm.model.wv.vectors, axis=0).astype(np.float32)
    logger.info(
        f"Model loaded: repr={representation}, dim={dim}, "
        f"vocab_mean_norm={np.linalg.norm(vocab_mean):.4f}"
    )
    return mm, tokenizer, tree_type, dim, vocab_mean


def embed_query(
    latex: str,
    mm: FastTextModelManager,
    tokenizer: TupleTokenizer,
    tree_type: str,
    metric: str,
    vocab_mean: np.ndarray,
) -> np.ndarray | None:
    """Embed a LaTeX query formula and prepare the vector for the chosen metric."""
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

        if metric == "cosine_ms":
            vec = vec - vocab_mean

        if metric in ("cosine", "cosine_ms"):
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                return None
            vec = vec / norm

        # For l2: keep un-normalised vector (use raw L2 distance in embedding space)

        return vec

    except Exception as e:
        logger.debug(f"Embedding failed for '{latex[:40]}': {e}")
        return None


# ---------------------------------------------------------------------------
# Step 4 — FAISS index loading + retrieval
# ---------------------------------------------------------------------------

def load_faiss_index(representation: str) -> tuple[faiss.Index, dict[int, list]]:
    index_dir = COLLECTION_FORMULA_INDEX_DIR / representation
    index_file = index_dir / f"formula_index_sq_{representation}.faiss"
    id_map_file = index_dir / f"id_map_sq_{representation}.json"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Collection FAISS index not found: {index_file}\n"
            f"Run: python experiments/build_collection_formula_index.py --representation {representation}"
        )

    logger.info(f"Loading FAISS index: {index_file}")
    index = faiss.read_index(str(index_file))
    index.nprobe = NPROBE
    logger.info(f"Index loaded: {index.ntotal:,} vectors")

    logger.info(f"Loading id_map: {id_map_file}")
    raw = json.loads(id_map_file.read_text())
    id_map: dict[int, list] = {int(k): v for k, v in raw.items()}
    return index, id_map


def retrieve(
    query_vec: np.ndarray,
    faiss_index: faiss.Index,
    id_map: dict[int, list],
    n: int,
    metric: str,
) -> list[tuple[str, str, float]]:
    """Return up to n hits as [(post_id, formula_id, score)]."""
    distances, indices = faiss_index.search(query_vec.reshape(1, -1), n)
    hits = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx not in id_map:
            continue
        post_id, formula_id = id_map[idx]
        if metric in ("cosine", "cosine_ms"):
            score = float(1.0 - dist / 2.0)   # cosine = 1 - L2^2/2 for unit vectors
        else:  # l2
            score = float(-dist)               # higher (less negative) = more similar
        hits.append((post_id, formula_id, score))
    return hits


# ---------------------------------------------------------------------------
# Steps 5-6 — Exclude + collapse
# ---------------------------------------------------------------------------

def exclude_and_collapse(
    hits: list[tuple[str, str, float]],
    fid_to_vid: dict[str, str],
    fid_to_post: dict[str, str],
    source_post_id: str | None,
    query_visual_id: str | None,
    max_results: int = MAX_RESULTS,
) -> list[tuple[str, float]]:
    """Exclude query post / visual_id, collapse to distinct visual_ids.

    Returns [(visual_id, score)] in descending score order, truncated to max_results.
    COLLAPSE BEFORE TRUNCATE.
    """
    seen_vids: set[str] = set()
    if query_visual_id:
        seen_vids.add(query_visual_id)

    collapsed: list[tuple[str, float]] = []

    for post_id, formula_id, score in hits:
        # Exclude same-post hits
        if source_post_id and post_id == source_post_id:
            continue

        vid = fid_to_vid.get(formula_id)
        if not vid:
            continue

        if vid in seen_vids:
            continue

        seen_vids.add(vid)
        collapsed.append((vid, score))

        if len(collapsed) >= max_results:
            break

    return collapsed


# ---------------------------------------------------------------------------
# Step 7 — Write TREC run
# ---------------------------------------------------------------------------

def write_run(
    results: dict[str, list[tuple[str, float]]],
    output_path: Path,
    run_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for topic_id, hits in sorted(results.items()):
            for rank, (visual_id, score) in enumerate(hits, start=1):
                f.write(f"{topic_id}\tQ0\t{visual_id}\t{rank}\t{score:.6f}\t{run_name}\n")
    logger.info(f"Run written: {output_path}  ({sum(len(v) for v in results.values())} lines)")


# ---------------------------------------------------------------------------
# Step 8 — Score
# ---------------------------------------------------------------------------

def score_run(run_path: Path, qrels_path: Path) -> dict[str, float]:
    """Evaluate run with pytrec_eval (judged-only, relevance_level=2)."""
    from multirag.evaluation.metrics import evaluate_run
    return evaluate_run(
        qrels_path=str(qrels_path),
        run_path=str(run_path),
        relevance_level=2,
    )


# ---------------------------------------------------------------------------
# Single-representation retrieval
# ---------------------------------------------------------------------------

def run_single_representation(
    topics: list[dict],
    representation: str,
    metric: str,
    n: int,
    embedding_dir: Path,
) -> dict[str, list[tuple[str, float]]]:
    """Embed queries, retrieve, exclude, collapse for one representation.

    Returns {topic_id: [(visual_id, score), ...]} in descending score order.
    """
    suffix = representation.lower().replace("-", "_")

    logger.info(f"[{suffix}] Loading model and FAISS index...")
    mm, tokenizer, tree_type, dim, vocab_mean = load_model_and_tokenizer(suffix, embedding_dir)
    faiss_index, id_map = load_faiss_index(suffix)

    logger.info(f"[{suffix}] Building visual_id map...")
    fid_to_vid, fid_to_post = build_visual_id_map(suffix)

    results: dict[str, list[tuple[str, float]]] = {}

    for topic in tqdm(topics, desc=f"Retrieving [{suffix}/{metric}]"):
        tid = topic["topic_id"]

        query_vec = embed_query(
            topic["query_latex"], mm, tokenizer, tree_type, metric, vocab_mean
        )
        if query_vec is None:
            logger.warning(f"[{tid}] Failed to embed query: {topic['query_latex'][:50]}")
            results[tid] = []
            continue

        hits = retrieve(query_vec, faiss_index, id_map, n, metric)

        # Determine query visual_id for exclusion (look up from fid_to_vid if possible)
        query_vid = fid_to_vid.get(topic.get("source_formula_id", ""))

        collapsed = exclude_and_collapse(
            hits=hits,
            fid_to_vid=fid_to_vid,
            fid_to_post=fid_to_post,
            source_post_id=None,   # query is from a topic post, not in the collection
            query_visual_id=query_vid,
        )
        results[tid] = collapsed

    return results


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(
    ranked_lists: list[dict[str, list[tuple[str, float]]]],
    rrf_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Fuse multiple ranked lists per topic with RRF.

    Each entry in ranked_lists is {topic_id: [(visual_id, score), ...]}.
    """
    all_topics = set()
    for rl in ranked_lists:
        all_topics.update(rl.keys())

    fused: dict[str, list[tuple[str, float]]] = {}
    for tid in all_topics:
        scores: dict[str, float] = defaultdict(float)
        for rl in ranked_lists:
            for rank, (vid, _) in enumerate(rl.get(tid, []), start=1):
                scores[vid] += 1.0 / (rank + rrf_k)
        sorted_hits = sorted(scores.items(), key=lambda x: -x[1])
        fused[tid] = sorted_hits[:MAX_RESULTS]

    return fused


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARQMath-3 Task 2 Formula Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--representation", "-r",
        type=str,
        default="slt",
        help="Representation(s) to use. Single: slt|opt|slt_type. "
             "Comma-separated for RRF fusion: slt,opt  or  slt,opt,slt_type",
    )
    parser.add_argument(
        "--metric", "-m",
        choices=["cosine", "cosine_ms", "l2"],
        default="cosine_ms",
        help="Similarity metric (default: cosine_ms = mean-shift cosine)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="Retrieval depth per query before collapse (default: 5000)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k constant for fusion (default: 60)",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=str(TOPICS_TASK2_XML),
        help=f"Task 2 topics XML (default: {TOPICS_TASK2_XML})",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default=str(QREL_TASK2_2022_OFFICIAL),
        help=f"Qrels file for scoring (default: {QREL_TASK2_2022_OFFICIAL})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output run file path (default: data/runs/task2_<repr>_<metric>.tsv)",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default=None,
        help=f"FastText model directory (default: FORMULA_EMBEDDING_DIR = {FORMULA_EMBEDDING_DIR})",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on first 5 topics only for fast pipeline validation",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip pytrec_eval scoring step (write run only)",
    )
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else FORMULA_EMBEDDING_DIR
    representations = [r.strip() for r in args.representation.split(",")]
    run_name = f"task2_{'_'.join(representations)}_{args.metric}"

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "smoke" if args.smoke_test else "full"
        output_path = REPO_ROOT / "data" / "runs" / f"{run_name}_{suffix}.tsv"

    logger.info("=" * 70)
    logger.info("ARQMath-3 TASK 2 FORMULA RETRIEVAL")
    logger.info("=" * 70)
    logger.info(f"Representations : {representations}")
    logger.info(f"Metric          : {args.metric}")
    logger.info(f"N (depth)       : {args.n}")
    logger.info(f"RRF k           : {args.rrf_k}")
    logger.info(f"Embedding dir   : {embedding_dir}")
    logger.info(f"Topics          : {args.topics}")
    logger.info(f"Qrels           : {args.qrels}")
    logger.info(f"Output          : {output_path}")
    logger.info(f"Smoke-test      : {args.smoke_test}")
    logger.info("=" * 70)

    # Step 1: parse topics
    topics = parse_task2_topics(Path(args.topics))
    logger.info(f"Parsed {len(topics)} topics")

    if args.smoke_test:
        topics = topics[:5]
        logger.info(f"Smoke-test: using first {len(topics)} topics: {[t['topic_id'] for t in topics]}")

    # Steps 3-6: retrieve for each representation
    if len(representations) == 1:
        results = run_single_representation(
            topics=topics,
            representation=representations[0],
            metric=args.metric,
            n=args.n,
            embedding_dir=embedding_dir,
        )
    else:
        ranked_lists = []
        for repr_ in representations:
            rl = run_single_representation(
                topics=topics,
                representation=repr_,
                metric=args.metric,
                n=args.n,
                embedding_dir=embedding_dir,
            )
            ranked_lists.append(rl)
        results = rrf_fuse(ranked_lists, rrf_k=args.rrf_k)

    # Step 7: write run
    write_run(results, output_path, run_name)

    # Per-topic summary (always printed)
    print("\nPer-topic result counts:")
    for tid, hits in sorted(results.items()):
        print(f"  {tid}: {len(hits)} results")

    if args.smoke_test:
        print("\nTop-10 results for each topic:")
        for tid, hits in sorted(results.items()):
            print(f"\n  [{tid}] {next(t['query_latex'] for t in topics if t['topic_id'] == tid)[:60]}")
            for rank, (vid, score) in enumerate(hits[:10], 1):
                print(f"    {rank:>2}. visual_id={vid}  score={score:.4f}")

    # Step 8: score
    if not args.no_score:
        qrels_path = Path(args.qrels)
        if not qrels_path.exists():
            logger.warning(f"Qrels file not found: {qrels_path} — skipping scoring")
        else:
            logger.info(f"Scoring against {qrels_path.name} ...")
            metrics = score_run(output_path, qrels_path)
            print("\n" + "=" * 50)
            print(f"Task 2 Evaluation Results — {run_name}")
            print("=" * 50)
            for key in sorted(metrics):
                if any(k in key for k in ("ndcg", "map", "P_")):
                    print(f"  {key:<20}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
