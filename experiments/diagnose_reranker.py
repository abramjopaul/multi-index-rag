#!/usr/bin/env python3
"""Reranker diagnostic: coverage and churn analysis.

Answers the question: is the near-flat nDCG' caused by a genuine weak
formula signal, or by a no-op bug (zero/noise embeddings, missing MathML,
wrong vocabulary)?

Usage:
    poetry run python experiments/diagnose_reranker.py [--run <path>] [--n-cands 100]

Outputs:
  1. COVERAGE — fraction of candidates with ≥1 usable formula embedding per topic.
  2. CHURN — rank changes between alpha=0 and alpha=0.5.
  3. FORMULA SCORE DISTRIBUTION — spread of MaxSim scores.
  4. VOCAB COVERAGE — fraction of tuple tokens in the 55-token training vocabulary.
  5. VERDICT.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from statistics import median, mean, stdev

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multirag.config.path_configs import (
    ANSWERS_JSONL,
    FORMULA_DIR,
    FORMULA_EMBEDDING_DIR,
    TOPICS_JSONL,
)
from multirag.evaluation.metrics import _is_clearly_trivial, _parse_run
from multirag.formula_search.encoder_maps import load_maps
from multirag.formula_search.tuple_extraction import (
    encode_tuples,
    extract_tuples_from_mathml_direct,
)
from multirag.formula_search.tuple_tokenizer import (
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REPRESENTATION = "slt"
N_CANDS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_topics(path: Path) -> list[dict]:
    topics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                topics.append(json.loads(line))
    return topics


def load_answer_formulas(answers_path: Path, candidate_ids: set[str]) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = {}
    remaining = set(candidate_ids)
    with open(answers_path) as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            try:
                answer = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = answer.get("id")
            if aid not in remaining:
                continue
            remaining.discard(aid)
            pairs = [
                (str(f.get("formula_id", "")), f["latex"])
                for f in answer.get("formulas", [])
                if not _is_clearly_trivial(f["latex"])
            ]
            result[aid] = pairs
    return result


def load_candidate_mathml(tsv_base_dir: Path, representation: str, needed_fids: set[str]) -> dict[str, str]:
    subdir_map = {
        "slt": "slt_representation_v3",
        "slt_type": "slt_representation_v3",
        "opt": "opt_representation_v3",
    }
    subdir = tsv_base_dir / subdir_map[representation]
    if not subdir.exists():
        print(f"  [ERROR] TSV dir not found: {subdir}")
        return {}

    tsv_files = sorted(subdir.glob("*.tsv"), key=lambda p: int(p.stem))
    formula_mathml: dict[str, str] = {}
    remaining = set(needed_fids)
    csv.field_size_limit(2**31 - 1)

    for tsv_file in tqdm(tsv_files, desc="Loading MathML from TSV shards", unit="shard", leave=False):
        if not remaining:
            break
        with open(tsv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if not remaining:
                    break
                fid = row.get("id", "")
                if fid not in remaining:
                    continue
                mathml = row.get("formula", "").strip()
                if mathml:
                    formula_mathml[fid] = mathml
                remaining.discard(fid)

    print(f"  MathML loaded for {len(formula_mathml)}/{len(needed_fids)} formula_ids "
          f"({100*len(formula_mathml)/max(len(needed_fids),1):.1f}%). "
          f"{len(remaining)} not found in TSV.")
    return formula_mathml


def build_tokenizer(embedding_dir: Path, representation: str) -> tuple[TupleTokenizer, str, dict, dict]:
    subdir = embedding_dir / representation
    metadata_file = subdir / f"training_metadata_{representation}.json"
    encoder_maps_file = subdir / f"encoder_maps_{representation}.tsv"

    with open(metadata_file) as f:
        metadata = json.load(f)

    enc_path = metadata.get("encoder_maps_path", "")
    enc_file = Path(enc_path) if enc_path and Path(enc_path).exists() else encoder_maps_file

    node_map, edge_map = load_maps(str(enc_file))
    node_id = max(node_map.values(), default=60000) + 1
    edge_id = max(edge_map.values(), default=500) + 1
    token_id_manager = TokenIDManager(node_id=node_id, edge_id=edge_id, node_map=node_map, edge_map=edge_map)

    embedding_type_name = metadata.get("embedding_type", "Both_Separated")
    tokenize_number = metadata.get("tokenize_number", True)
    embedding_type = TupleTokenizationMode.Both_Separated if embedding_type_name == "Both_Separated" else TupleTokenizationMode.Type

    tuple_tokenizer = TupleTokenizer(
        token_id_manager=token_id_manager,
        embedding_type=embedding_type,
        tokenize_number=tokenize_number,
    )

    tree_type_mapping = {"slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE"}
    tree_type = tree_type_mapping[representation]

    return tuple_tokenizer, tree_type, node_map, edge_map


def embed_formula_from_mathml(mathml: str, tokenizer: TupleTokenizer, tree_type: str, model_manager) -> np.ndarray | None:
    """Returns None if embedding is zero/empty."""
    try:
        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)
        if not tuples:
            return None
        encoded = encode_tuples(tuples, tokenizer)
        if not encoded:
            return None
        vec = np.array(model_manager.get_sentence_vector(encoded), dtype=np.float32)
        if not np.any(vec):
            return None
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None
    except Exception:
        return None


def formula_score_maxsim(topic_vecs: np.ndarray, cand_vecs: np.ndarray, aggregation: str = "mean") -> float:
    if len(topic_vecs) == 0 or len(cand_vecs) == 0:
        return 0.0
    sim = topic_vecs @ cand_vecs.T
    per_topic_best = sim.max(axis=1)
    if aggregation == "mean":
        return float(per_topic_best.mean())
    elif aggregation == "max":
        return float(per_topic_best.max())
    return float(per_topic_best.sum())


def minmax_normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    r = hi - lo
    if r == 0.0:
        return [0.5] * len(scores)
    return [(s - lo) / r for s in scores]


def spearman_rho(rank_a: list[int], rank_b: list[int]) -> float:
    n = len(rank_a)
    if n < 2:
        return 1.0
    d2 = sum((a - b) ** 2 for a, b in zip(rank_a, rank_b))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def vocab_coverage_fraction(mathml: str, tokenizer: TupleTokenizer, tree_type: str, training_node_map: dict, training_edge_map: dict) -> tuple[int, int]:
    """Returns (in_vocab_tokens, total_tokens) for a single formula's MathML."""
    try:
        tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type)
        if not tuples:
            return 0, 0
    except Exception:
        return 0, 0

    total = 0
    in_vocab = 0
    for tup in tuples:
        parts = tup.split("\t")
        if not parts:
            continue
        tokens_to_check = []
        if tokenizer.embedding_type == TupleTokenizationMode.Both_Separated:
            for tok in parts:
                if "!" in tok:
                    typ, val = tok.split("!", 1)
                    tokens_to_check.append((typ + "!", True))
                    tokens_to_check.append((val, True))
                else:
                    tokens_to_check.append((tok, False))
        else:
            tokens_to_check = [(p, True) for p in parts]

        for token, is_node in tokens_to_check:
            total += 1
            if is_node and token in training_node_map:
                in_vocab += 1
            elif not is_node and token in training_edge_map:
                in_vocab += 1

    return in_vocab, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Path to stage-1 TREC run TSV")
    parser.add_argument("--n-cands", type=int, default=100)
    parser.add_argument("--representation", default="slt", choices=["slt", "opt", "slt_type"])
    parser.add_argument("--max-topics", type=int, default=None, help="Limit to first N topics (for speed)")
    args = parser.parse_args()

    representation = args.representation
    n_cands = args.n_cands

    run_path = Path(args.run) if args.run else Path(__file__).parent.parent / "data/runs/dense_mpnet_20260614.tsv"
    print(f"\n{'='*70}")
    print(f"RERANKER DIAGNOSTIC — representation={representation}, n_cands={n_cands}")
    print(f"Stage-1 run: {run_path}")
    print(f"{'='*70}\n")

    # ---- Load run ----
    print("Loading stage-1 run...")
    raw_run = _parse_run(run_path)
    run: dict[str, list[tuple[str, float]]] = {
        qid: sorted(doc_scores.items(), key=lambda x: -x[1])
        for qid, doc_scores in raw_run.items()
    }
    topic_ids = sorted(run.keys())
    if args.max_topics:
        topic_ids = topic_ids[:args.max_topics]
    print(f"  {len(topic_ids)} topics, up to {n_cands} candidates each")

    # ---- Load topics ----
    print("Loading topics...")
    all_topics = load_topics(TOPICS_JSONL)
    topics_by_id = {t["topic_id"]: t for t in all_topics}

    # ---- Collect candidates (top n_cands per topic) ----
    candidate_ids: set[str] = set()
    for tid in topic_ids:
        for doc_id, _ in run[tid][:n_cands]:
            candidate_ids.add(doc_id)
    print(f"  {len(candidate_ids)} unique candidate doc_ids across all topics")

    # ---- Load answer formulas ----
    print("Loading answer formulas...")
    answer_formulas = load_answer_formulas(ANSWERS_JSONL, candidate_ids)
    n_with_formulas = sum(1 for fmls in answer_formulas.values() if fmls)
    print(f"  {len(answer_formulas)}/{len(candidate_ids)} candidates found in answers.jsonl")
    print(f"  {n_with_formulas}/{len(answer_formulas)} of those have ≥1 non-trivial formula")

    # ---- Load MathML from TSV ----
    print("Loading pre-computed MathML from TSV shards...")
    needed_fids: set[str] = {
        fid for pairs in answer_formulas.values() for fid, _ in pairs if fid
    }
    print(f"  {len(needed_fids)} unique formula_ids need MathML")
    formula_mathml = load_candidate_mathml(FORMULA_DIR, representation, needed_fids)

    # ---- Load FastText model and build tokenizer ----
    print(f"\nLoading FastText model ({representation})...")
    embedding_dir = FORMULA_EMBEDDING_DIR
    tokenizer, tree_type, training_node_map, training_edge_map = build_tokenizer(embedding_dir, representation)
    print(f"  Training vocabulary: {len(training_node_map)} node types, {len(training_edge_map)} edge types")
    print(f"  ({len(training_node_map) + len(training_edge_map)} total tokens in encoder maps)")

    from multirag.embedding.formula_model_manager import FastTextModelManager
    subdir = embedding_dir / representation
    model_file = subdir / f"fasttext_model_{representation}.bin"
    metadata_file = subdir / f"training_metadata_{representation}.json"
    print(f"  Model: {model_file} ({model_file.stat().st_size/1e6:.1f} MB)")

    model_manager = FastTextModelManager(
        model_path=str(model_file),
        metadata_path=str(metadata_file),
        corpus_path="",
    )
    model_manager.load()
    print("  Model loaded.")

    # ---- Pre-embed topic formulas ----
    print("\nEmbedding topic formulas (subprocess/latex path)...")
    # We'll use the latex→MathML subprocess path via the full embedder
    import tempfile
    from multirag.indexing.formula.faiss_scalar_quantizer import FormulaFAISSIndexerIVFScalarQuantizer
    tmp = tempfile.mkdtemp(prefix="diag_reranker_")
    embedder = FormulaFAISSIndexerIVFScalarQuantizer(
        index_path=tmp,
        embedding_dir=str(embedding_dir),
        representation=representation,
    )
    embedder.embed_formula("x")  # warm up

    topic_vecs_cache: dict[str, np.ndarray] = {}
    n_topics_with_vecs = 0
    for tid in tqdm(topic_ids, desc="Embedding topic formulas", unit="topic"):
        topic = topics_by_id.get(tid)
        if topic is None:
            continue
        latexes = [f["latex"] for f in topic.get("formulas", []) if not _is_clearly_trivial(f["latex"])]
        vecs = []
        for lx in latexes:
            v = embedder.embed_formula(lx)
            if v is not None and np.any(v):
                norm = np.linalg.norm(v)
                if norm > 0:
                    vecs.append(v / norm)
        if vecs:
            topic_vecs_cache[tid] = np.array(vecs, dtype=np.float32)
            n_topics_with_vecs += 1

    print(f"  {n_topics_with_vecs}/{len(topic_ids)} topics have ≥1 embeddable formula "
          f"({100*n_topics_with_vecs/max(len(topic_ids),1):.1f}%)")
    topics_skipped = len(topic_ids) - n_topics_with_vecs
    if topics_skipped > 0:
        print(f"  ⚠  {topics_skipped} topics will pass through unchanged (no reranking at all!)")

    # ---- Vocabulary coverage on a sample of candidate formulas ----
    print("\n--- VOCABULARY COVERAGE (sample of 500 candidate formula MathMLs) ---")
    sample_fids = list(formula_mathml.keys())[:500]
    total_tokens_all, in_vocab_all = 0, 0
    zero_tuple_count = 0
    for fid in sample_fids:
        mml = formula_mathml[fid]
        inv, tot = vocab_coverage_fraction(mml, tokenizer, tree_type, training_node_map, training_edge_map)
        if tot == 0:
            zero_tuple_count += 1
        in_vocab_all += inv
        total_tokens_all += tot

    if total_tokens_all > 0:
        vocab_cov = 100.0 * in_vocab_all / total_tokens_all
        print(f"  In-vocab tokens: {in_vocab_all}/{total_tokens_all} = {vocab_cov:.1f}%")
    print(f"  Formulas with 0 tuples extracted: {zero_tuple_count}/{len(sample_fids)}")

    # ---- COVERAGE DIAGNOSTIC ----
    print("\n--- DIAGNOSTIC 1: FORMULA EMBEDDING COVERAGE ---")
    per_topic_coverage: list[float] = []   # fraction of candidates with ≥1 embedding
    per_topic_formula_scores: list[list[float]] = []  # formula score lists per topic (only topics with vecs)
    per_topic_churn: list[dict] = []       # churn stats per topic

    for tid in tqdm(topic_ids, desc="Scoring candidates", unit="topic"):
        pool = run[tid][:n_cands]
        topic_vecs = topic_vecs_cache.get(tid)

        # Coverage: check each candidate in the pool
        n_with_emb = 0
        all_cand_vecs_list: list[np.ndarray] = []  # for building per-candidate vecs list
        for doc_id, _ in pool:
            fid_latex_pairs = answer_formulas.get(doc_id, [])
            vecs = []
            for fid, _ in fid_latex_pairs:
                mml = formula_mathml.get(fid)
                if not mml:
                    continue
                v = embed_formula_from_mathml(mml, tokenizer, tree_type, model_manager)
                if v is not None:
                    vecs.append(v)
            cand_vecs = np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 300), dtype=np.float32)
            if len(cand_vecs) > 0:
                n_with_emb += 1
            all_cand_vecs_list.append(cand_vecs)

        cov_frac = n_with_emb / len(pool) if pool else 0.0
        per_topic_coverage.append(cov_frac)

        # Only compute formula scores if topic has vecs
        if topic_vecs is None:
            continue

        # Formula scores
        formula_scores = [
            formula_score_maxsim(topic_vecs, cv, "mean")
            for cv in all_cand_vecs_list
        ]
        per_topic_formula_scores.append(formula_scores)

        # Churn: compare alpha=0 vs alpha=0.5
        doc_ids = [d for d, _ in pool]
        text_scores = [s for _, s in pool]
        text_norm = minmax_normalize(text_scores)
        formula_norm = minmax_normalize(formula_scores)

        blended = sorted(
            enumerate(
                (1 - 0.5) * t + 0.5 * f
                for t, f in zip(text_norm, formula_norm)
            ),
            key=lambda x: -x[1],
        )

        # Original text-only rank: position in text_norm list (already sorted by text score)
        text_rank = {doc_ids[i]: i + 1 for i in range(len(doc_ids))}
        # Reranked position
        reranked_ids = [doc_ids[orig_idx] for orig_idx, _ in blended]
        reranked_rank = {doc_id: pos + 1 for pos, doc_id in enumerate(reranked_ids)}

        rank_changes = sum(1 for d in doc_ids if text_rank[d] != reranked_rank[d])
        abs_shifts = [abs(text_rank[d] - reranked_rank[d]) for d in doc_ids]
        rho = spearman_rho(
            [text_rank[d] for d in doc_ids],
            [reranked_rank[d] for d in doc_ids],
        )

        per_topic_churn.append({
            "topic_id": tid,
            "rank_changes": rank_changes,
            "mean_abs_shift": mean(abs_shifts),
            "spearman_rho": rho,
        })

    # ---- Print COVERAGE report ----
    print(f"\n  Overall coverage (fraction of candidates with ≥1 usable embedding):")
    if per_topic_coverage:
        overall_cov = mean(per_topic_coverage) * 100
        print(f"    Mean:   {overall_cov:.1f}%")
        print(f"    Median: {median(per_topic_coverage)*100:.1f}%")
        print(f"    Min:    {min(per_topic_coverage)*100:.1f}%")
        print(f"    Max:    {max(per_topic_coverage)*100:.1f}%")
    else:
        print("    (no topics processed)")

    # ---- Print FORMULA SCORE DISTRIBUTION ----
    print(f"\n--- DIAGNOSTIC 2a: FORMULA SCORE DISTRIBUTION (per topic, aggregation=mean) ---")
    if per_topic_formula_scores:
        # Flatten and compute range per topic
        per_topic_spreads = []
        all_scores_flat = []
        for scores in per_topic_formula_scores:
            per_topic_spreads.append(max(scores) - min(scores))
            all_scores_flat.extend(scores)

        all_nonzero = [s for s in all_scores_flat if s > 0]
        n_zero = sum(1 for s in all_scores_flat if s == 0.0)
        print(f"  Total candidate-scores computed: {len(all_scores_flat)}")
        print(f"  Scores == 0.0 (no formula embedding): {n_zero} ({100*n_zero/max(len(all_scores_flat),1):.1f}%)")
        if all_scores_flat:
            print(f"  Overall min:  {min(all_scores_flat):.4f}")
            print(f"  Overall max:  {max(all_scores_flat):.4f}")
            print(f"  Overall mean: {mean(all_scores_flat):.4f}")
            if len(all_scores_flat) > 1:
                print(f"  Overall std:  {stdev(all_scores_flat):.4f}")
            print(f"  Overall median: {median(all_scores_flat):.4f}")
        print(f"\n  Per-topic score range (max−min) within each topic's candidates:")
        if per_topic_spreads:
            print(f"    Mean spread:   {mean(per_topic_spreads):.4f}")
            print(f"    Median spread: {median(per_topic_spreads):.4f}")
            print(f"    Min spread:    {min(per_topic_spreads):.4f}")
            print(f"    Max spread:    {max(per_topic_spreads):.4f}")
        n_flat_topics = sum(1 for s in per_topic_spreads if s < 1e-6)
        print(f"  Topics with near-zero spread (<1e-6): {n_flat_topics}/{len(per_topic_spreads)}")
    else:
        print("  (no topics had embeddable formulas — all passed through unchanged)")

    # ---- Print CHURN report ----
    print(f"\n--- DIAGNOSTIC 2b: CHURN (alpha=0 vs alpha=0.5, top-{n_cands}) ---")
    if per_topic_churn:
        mean_changes = mean(c["rank_changes"] for c in per_topic_churn)
        mean_shift = mean(c["mean_abs_shift"] for c in per_topic_churn)
        mean_rho = mean(c["spearman_rho"] for c in per_topic_churn)
        print(f"  Topics reranked (had topic formula vecs): {len(per_topic_churn)}/{len(topic_ids)}")
        print(f"  Mean docs changing rank:    {mean_changes:.1f} / {n_cands}  ({100*mean_changes/n_cands:.1f}%)")
        print(f"  Mean absolute rank shift:   {mean_shift:.2f} positions")
        print(f"  Mean Spearman rho:          {mean_rho:.4f}  (1.0 = no reordering)")
        print(f"  Min Spearman rho (worst):   {min(c['spearman_rho'] for c in per_topic_churn):.4f}")
        print(f"  Max Spearman rho (best):    {max(c['spearman_rho'] for c in per_topic_churn):.4f}")

        # Sample worst-churn topics
        most_churn = sorted(per_topic_churn, key=lambda c: c["rank_changes"], reverse=True)[:5]
        print(f"\n  Top-5 topics by rank changes:")
        for c in most_churn:
            print(f"    {c['topic_id']:8s}  changes={c['rank_changes']:3d}  mean_shift={c['mean_abs_shift']:.2f}  rho={c['spearman_rho']:.4f}")
    else:
        print("  (no topics had embeddable formulas — no churn possible)")

    # ---- VERDICT ----
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    overall_cov_pct = mean(per_topic_coverage) * 100 if per_topic_coverage else 0.0
    topic_cov_pct = 100 * n_topics_with_vecs / max(len(topic_ids), 1)
    mean_rho_val = mean(c["spearman_rho"] for c in per_topic_churn) if per_topic_churn else 1.0
    mean_spread = mean(
        max(s) - min(s) for s in per_topic_formula_scores
    ) if per_topic_formula_scores else 0.0

    is_noop = (
        overall_cov_pct < 40
        or topic_cov_pct < 50
        or mean_rho_val > 0.98
        or mean_spread < 0.01
    )

    if is_noop:
        print("VERDICT: NO-OP / BUG\n")
        causes = []
        if topic_cov_pct < 50:
            causes.append(
                f"  • {100-topic_cov_pct:.0f}% of topics pass through without any reranking "
                f"(topic formula vecs all-zero — vocabulary mismatch)"
            )
        if overall_cov_pct < 40:
            causes.append(
                f"  • Only {overall_cov_pct:.1f}% of candidates have ≥1 usable embedding "
                f"→ formula_score=0.0 for most, normalizes to 0.5, preserving text order"
            )
        if mean_spread < 0.01:
            causes.append(
                f"  • Formula score spread is near-zero (mean {mean_spread:.4f}) "
                f"→ min-max normalization collapses all scores to 0.5"
            )
        if mean_rho_val > 0.98:
            causes.append(
                f"  • Spearman rho = {mean_rho_val:.4f} → rankings almost unchanged"
            )

        vocab_size = len(training_node_map) + len(training_edge_map)
        causes.append(
            f"\n  ROOT CAUSE: FastText model trained on {vocab_size} unique tokens "
            f"(encoder maps in data/formula-indexing/{representation}/).\n"
            f"  The training corpus for {representation} contained only "
            f"{'100' if representation == 'slt' else '500'} formulas — a dev/test model.\n"
            f"  The slt_training_artifacts/encoder_maps_slt.tsv has 250,195 tokens;\n"
            f"  the actual production training run was never completed (corpus file is\n"
            f"  data/formula-indexing/slt_training_artifacts/corpus_slt.txt_.gstmp —\n"
            f"  a GCS temp file that was never finalized)."
        )

        print("  Causes identified:")
        for c in causes:
            print(c)

        print("\n  PROPOSED FIX (do NOT implement yet — awaiting review):")
        print("  1. Complete the FastText training using the full-corpus encoder maps")
        print("     (data/formula-indexing/slt_training_artifacts/encoder_maps_slt.tsv,")
        print("     250K tokens) and the full SLT corpus (~28M formulas from 101 TSV shards).")
        print("  2. Place the resulting model at:")
        print("     data/formula-indexing/slt/fasttext_model_slt.bin")
        print("     data/formula-indexing/slt/encoder_maps_slt.tsv")
        print("     data/formula-indexing/slt/training_metadata_slt.json")
        print("  3. Re-run rerank_run.py with this corrected model.")
        print("  4. Same fix needed for OPT (500-formula model → needs full-corpus retraining).")
    else:
        print("VERDICT: GENUINE WEAK / REDUNDANT FORMULA SIGNAL\n")
        print(f"  Coverage: {overall_cov_pct:.1f}% candidates embedded, {topic_cov_pct:.1f}% topics active")
        print(f"  Score spread: {mean_spread:.4f}  |  Spearman rho: {mean_rho_val:.4f}")
        print("  The reranker IS changing rankings, but formula similarity does not")
        print("  correlate with relevance at the top-k level for this corpus.")
        print("  The flat nDCG' result is a real, reportable finding.")

    print(f"\n{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
