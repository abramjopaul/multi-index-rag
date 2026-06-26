#!/usr/bin/env python3
"""
Probe whether cosine similarity between formula embeddings is inflated.

Tests three categories of formula pairs sourced from topics.jsonl:
  - Very similar   : same structure, minor variation (should be high)
  - Subset match   : one formula contains the other as a sub-expression
  - Not matching   : formulas from completely different mathematical domains

Usage:
    python test/cosine_similarity_probe.py
    python test/cosine_similarity_probe.py --representation opt
    python test/cosine_similarity_probe.py --diagnose   # step-by-step pipeline dump
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

EMBEDDING_DIR = ROOT / "data/formula-indexing"

# ---------------------------------------------------------------------------
# Test pairs — (category, formula_a, formula_b, description)
# Formulas taken from topics.jsonl; surrounding $ stripped.
# ---------------------------------------------------------------------------

TEST_PAIRS = [
    # ---- Very similar -------------------------------------------------------
    (
        "very_similar",
        r"\|A\|_2",
        r"\|A\|_1",
        r"Matrix 2-norm vs 1-norm (subscript only)",
    ),
    (
        "very_similar",
        r"[x,x]=0",
        r"[x,y]=-[x,y]",
        r"Lie bracket anti-symmetry axioms",
    ),
    (
        "very_similar",
        r"\lambda(n)",
        r"\lambda(n)=\max\{\operatorname{ord}_n(a):\gcd(a,n)=1\}",
        r"Carmichael fn — bare symbol vs its definition",
    ),
    # ---- Subset match -------------------------------------------------------
    (
        "subset_match",
        r"[x,y]",
        r"[x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0",
        r"Lie bracket vs Jacobi identity (bracket is sub-expr)",
    ),
    (
        "subset_match",
        r"\frac{1}{n^s}",
        r"\zeta(s)=\sum_{n=1}^\infty\frac{1}{n^s}=\frac{1}{\Gamma(s)}\int_0^\infty \frac{x^{s-1}}{e^x-1}dx",
        r"1/n^s vs full Riemann-zeta definition",
    ),
    (
        "subset_match",
        r"\|A\|_2\leq \sqrt{\|A\|_1 \|A\|_{\infty}}",
        r"\frac{1}{\sqrt{n}}\|A\|_{\infty}\leq \|A\|_2\leq\sqrt{m}\|A\|_{\infty}",
        r"Norm bound — one side vs extended inequality chain",
    ),
    # ---- Not matching -------------------------------------------------------
    (
        "not_matching",
        r"z^n=w",
        r"a^t\equiv1\pmod n",
        r"Complex n-th root equation vs modular exponent congruence",
    ),
    (
        "not_matching",
        r"\|A\|_2\leq \sqrt{\|A\|_1 \|A\|_{\infty}}",
        r"w=se^{i{\phi}}",
        r"Matrix norm inequality vs complex polar form",
    ),
    (
        "not_matching",
        r"[x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0",
        r"B = \gamma + \sum_p \left\{ \log\left( 1 - \frac 1p\right) + \frac 1p\right\}",
        r"Jacobi identity vs Mertens constant definition",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_embedder(representation: str):
    from multirag.indexing.formula.faiss_scalar_quantizer import (
        FormulaFAISSIndexerIVFScalarQuantizer,
    )
    tmp = tempfile.mkdtemp(prefix="cosine_probe_")
    embedder = FormulaFAISSIndexerIVFScalarQuantizer(
        index_path=tmp,
        embedding_dir=str(EMBEDDING_DIR),
        representation=representation,
    )
    embedder.embed_formula("x")  # warm up
    return embedder


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def pipeline_steps(embedder, latex: str) -> dict:
    """Run the full pipeline and return every intermediate artifact."""
    from multirag.formula_search.latex_mml import LatexToMathML
    from multirag.formula_search import extract_tuples_from_mathml_direct, encode_tuples

    rep = embedder.representation
    tree_type = {"slt": "SLT", "opt": "OPT", "slt_type": "SLT-TYPE"}.get(rep, "SLT")

    if rep == "opt":
        mathml = LatexToMathML.convert_to_mathml2(latex)
    else:
        mathml = LatexToMathML.convert_to_mathml(latex)

    tuples = extract_tuples_from_mathml_direct(mathml, tree_type=tree_type) if mathml else []  # type: ignore[arg-type]
    encoded = encode_tuples(tuples, embedder._query_tokenizer) if tuples else ""

    model = embedder._query_model_manager
    tokens = encoded.split() if encoded else []
    token_vecs = [model.model.wv[t] for t in tokens if t in model.model.wv]

    raw_vec = embedder.embed_formula(latex)

    return {
        "mathml_len": len(mathml) if mathml else 0,
        "tuples": tuples,
        "n_tuples": len(tuples),
        "tokens": tokens,
        "n_tokens": len(tokens),
        "n_oov_tokens": len(tokens) - len(token_vecs),
        "token_vecs": token_vecs,
        "raw_vec": raw_vec,
        "raw_norm": float(np.linalg.norm(raw_vec)) if raw_vec is not None else 0.0,
    }


def compute_vocab_mean(embedder) -> np.ndarray:
    """Mean vector over the entire FastText vocabulary — the 'bias direction'."""
    wv = embedder._query_model_manager.model.wv
    # gensim stores vectors in wv.vectors (already a numpy array)
    return np.mean(wv.vectors, axis=0).astype(np.float32)


def embed_corrected(embedder, latex: str, vocab_mean: np.ndarray) -> np.ndarray | None:
    """
    Embed a formula then subtract the vocabulary mean before returning.

    Equivalent to averaging (token_vec - vocab_mean) across all tokens, which
    removes the shared bias direction that collapses all sentence vectors into
    a tight cone.  This is the 'All-but-the-Top' mean-shift step (Arora et al. 2017).
    """
    raw = embedder.embed_formula(latex)
    if raw is None or not np.any(raw):
        return None
    return raw - vocab_mean


def vocab_cone_stat(embedder) -> tuple[float, float]:
    """
    Sample 500 random vocabulary vectors and compute their mean pairwise cosine.
    A high value (~0.99) means all token embeddings point the same direction —
    which collapses any sentence average to the same point.
    """
    wv = embedder._query_model_manager.model.wv
    keys = list(wv.key_to_index.keys())
    sample = np.random.default_rng(42).choice(len(keys), size=min(500, len(keys)), replace=False)
    vecs = np.array([wv[keys[i]] for i in sample], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.where(norms > 0, norms, 1)
    gram = vecs_n @ vecs_n.T
    upper = gram[np.triu_indices(len(vecs_n), k=1)]
    return float(upper.mean()), float(upper.std())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probe formula cosine similarity distribution")
    parser.add_argument("--representation", choices=["slt", "opt", "slt_type"], default="slt")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print step-by-step pipeline details for every formula pair",
    )
    args = parser.parse_args()

    print(f"\nBuilding embedder (representation={args.representation}) …")
    embedder = build_embedder(args.representation)
    # Ensure _query_model_manager and _query_tokenizer exist
    embedder._generate_query_embedding("x")
    print("Model loaded.\n")

    # ------------------------------------------------------------------
    # 1. Vocabulary cone — are token embeddings themselves clustered?
    # ------------------------------------------------------------------
    print("=" * 90)
    print("SECTION 1 — Vocabulary cone (pairwise cosine of 500 random token embeddings)")
    print("=" * 90)
    mean_cos, std_cos = vocab_cone_stat(embedder)
    print(f"  Mean pairwise cosine: {mean_cos:.4f}  std: {std_cos:.4f}")
    if mean_cos > 0.90:
        print("  DIAGNOSIS: Token embeddings are tightly clustered in a cone.")
        print("             Any sentence average → the same direction → cosine ~1.0 always.")
    elif mean_cos > 0.70:
        print("  NOTE: Token embeddings are moderately clustered.")
    else:
        print("  OK: Token embeddings are spread out; collapse is not from vocabulary geometry.")

    # ------------------------------------------------------------------
    # 2. Per-pair similarity table
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SECTION 2 — Per-pair cosine similarity")
    print("=" * 90)
    print(f"{'Category':<15} {'Cosine':>7}  Description")
    print("-" * 90)

    categories = ["very_similar", "subset_match", "not_matching"]
    results = {c: [] for c in categories}

    for category, formula_a, formula_b, description in TEST_PAIRS:
        vec_a = embedder.embed_formula(formula_a)
        vec_b = embedder.embed_formula(formula_b)

        if vec_a is None or vec_b is None or (not np.any(vec_a)) or (not np.any(vec_b)):
            print(f"{'[SKIP]':<15} {'   N/A':>7}  {description}  (embedding failed)")
            continue

        sim = cosine(vec_a, vec_b)
        results[category].append(sim)
        flag = "  <-- HIGH?" if (category == "not_matching" and sim > 0.90) else ""
        print(f"{category:<15} {sim:>7.4f}  {description}{flag}")

    print()
    print(f"{'Category':<15} {'n':>4} {'mean':>8} {'min':>8} {'max':>8}")
    print("-" * 50)
    for cat in categories:
        arr = np.array(results[cat]) if results[cat] else np.array([])
        if len(arr) == 0:
            print(f"{cat:<15} {'0':>4}  (no results)")
        else:
            print(f"{cat:<15} {len(arr):>4} {arr.mean():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")

    nm = np.array(results["not_matching"])
    vs = np.array(results["very_similar"])
    print()
    if len(nm) and len(vs):
        gap = vs.mean() - nm.mean()
        print(f"Discrimination gap (very_similar − not_matching): {gap:+.4f}")
        if gap < 0.05:
            print("WARNING: near-zero gap — embeddings do not discriminate formula structure.")
        elif gap < 0.15:
            print("NOTE: moderate gap — similarity is compressed upward.")
        else:
            print("OK: healthy gap.")

    # ------------------------------------------------------------------
    # 3. Anisotropy correction (mean-shift / All-but-the-Top)
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SECTION 3 — Anisotropy correction: subtract vocabulary mean before cosine")
    print("  vocab_mean = mean of all FastText token vectors")
    print("  corrected_vec = raw_sentence_vec - vocab_mean   (then L2-normalised)")
    print("=" * 90)

    print("Computing vocabulary mean vector …")
    vocab_mean = compute_vocab_mean(embedder)
    mean_norm = float(np.linalg.norm(vocab_mean))
    print(f"  vocab_mean norm: {mean_norm:.4f}\n")

    corrected_results = {c: [] for c in categories}

    print(f"{'Category':<15} {'Raw':>7}  {'Corrected':>9}  {'Δ':>7}  Description")
    print("-" * 95)

    for category, formula_a, formula_b, description in TEST_PAIRS:
        raw_a = embedder.embed_formula(formula_a)
        raw_b = embedder.embed_formula(formula_b)

        if raw_a is None or raw_b is None or (not np.any(raw_a)) or (not np.any(raw_b)):
            print(f"{'[SKIP]':<15} {'N/A':>7}  {'N/A':>9}  {'N/A':>7}  {description}")
            continue

        raw_sim = cosine(raw_a, raw_b)

        corr_a = raw_a - vocab_mean
        corr_b = raw_b - vocab_mean
        corr_sim = cosine(corr_a, corr_b)

        corrected_results[category].append(corr_sim)
        delta = corr_sim - raw_sim
        print(f"{category:<15} {raw_sim:>7.4f}  {corr_sim:>9.4f}  {delta:>+7.4f}  {description}")

    print()
    print(f"{'Category':<15} {'n':>4} {'corr_mean':>10} {'corr_min':>10} {'corr_max':>10}")
    print("-" * 55)
    for cat in categories:
        arr = np.array(corrected_results[cat]) if corrected_results[cat] else np.array([])
        if len(arr) == 0:
            print(f"{cat:<15} {'0':>4}  (no results)")
        else:
            print(f"{cat:<15} {len(arr):>4} {arr.mean():>10.4f} {arr.min():>10.4f} {arr.max():>10.4f}")

    cnm = np.array(corrected_results["not_matching"])
    cvs = np.array(corrected_results["very_similar"])
    print()
    if len(cnm) and len(cvs):
        corr_gap = cvs.mean() - cnm.mean()
        raw_gap = vs.mean() - nm.mean() if len(nm) and len(vs) else float("nan")
        print(f"Raw discrimination gap:       {raw_gap:+.4f}")
        print(f"Corrected discrimination gap: {corr_gap:+.4f}")
        improvement = corr_gap - raw_gap
        print(f"Improvement from correction:  {improvement:+.4f}")
        if corr_gap < 0.05:
            print("DIAGNOSIS: Correction did not help — the model itself has no structural signal.")
            print("           Consider retraining FastText or switching to a different similarity metric.")
        elif corr_gap < 0.20:
            print("NOTE: Some improvement — partial signal recovered, but still compressed.")
            print("      Consider also removing the top-k principal components (full All-but-the-Top).")
        else:
            print("OK: Correction recovered meaningful discrimination.")

    # ------------------------------------------------------------------
    # 4. L2 distance as an alternative metric
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SECTION 4 — L2 distance on raw (un-normalised) vectors")
    print("  Cosine throws away magnitude.  If the cone is degenerate, L2 may still")
    print("  carry signal in the difference of norms across formula sizes/complexities.")
    print("=" * 90)

    l2_results = {c: [] for c in categories}

    print(f"{'Category':<15} {'L2 dist':>9}  {'norm_A':>7}  {'norm_B':>7}  Description")
    print("-" * 95)

    for category, formula_a, formula_b, description in TEST_PAIRS:
        raw_a = embedder.embed_formula(formula_a)
        raw_b = embedder.embed_formula(formula_b)

        if raw_a is None or raw_b is None or (not np.any(raw_a)) or (not np.any(raw_b)):
            print(f"{'[SKIP]':<15} {'N/A':>9}  {'N/A':>7}  {'N/A':>7}  {description}")
            continue

        l2 = float(np.linalg.norm(raw_a - raw_b))
        l2_results[category].append(l2)
        na = float(np.linalg.norm(raw_a))
        nb = float(np.linalg.norm(raw_b))
        print(f"{category:<15} {l2:>9.4f}  {na:>7.4f}  {nb:>7.4f}  {description}")

    print()
    print(f"{'Category':<15} {'n':>4} {'mean_L2':>9} {'min_L2':>9} {'max_L2':>9}")
    print("-" * 55)
    for cat in categories:
        arr = np.array(l2_results[cat]) if l2_results[cat] else np.array([])
        if len(arr) == 0:
            print(f"{cat:<15} {'0':>4}  (no results)")
        else:
            print(f"{cat:<15} {len(arr):>4} {arr.mean():>9.4f} {arr.min():>9.4f} {arr.max():>9.4f}")

    l2_nm = np.array(l2_results["not_matching"])
    l2_vs = np.array(l2_results["very_similar"])
    print()
    if len(l2_nm) and len(l2_vs):
        # For L2 distance: larger = less similar, so gap is not_matching.mean - very_similar.mean
        l2_gap = l2_nm.mean() - l2_vs.mean()
        print(f"L2 discrimination gap (not_matching.mean − very_similar.mean): {l2_gap:+.4f}")
        if l2_gap < 0.01:
            print("DIAGNOSIS: L2 distance also has no signal — the model is fully degenerate.")
        elif l2_gap < 0.10:
            print("NOTE: Weak L2 signal — some magnitude variation exists but it is small.")
            print("      L2 could marginally outperform cosine in the reranker.")
        else:
            print("OK: L2 distance separates categories — use it instead of cosine.")

    # ------------------------------------------------------------------
    # 5. Detailed pipeline diagnosis (optional)
    # ------------------------------------------------------------------
    if args.diagnose:
        print()
        print("=" * 90)
        print("SECTION 5 — Step-by-step pipeline for each formula pair")
        print("=" * 90)

        for category, formula_a, formula_b, description in TEST_PAIRS:
            print(f"\n[{category}] {description}")
            print(f"  A: {formula_a[:80]}")
            print(f"  B: {formula_b[:80]}")

            sa = pipeline_steps(embedder, formula_a)
            sb = pipeline_steps(embedder, formula_b)

            tuple_overlap = set(sa["tuples"]) & set(sb["tuples"])
            token_overlap = set(sa["tokens"]) & set(sb["tokens"])

            print(f"  {'':20s} {'Formula A':>20} {'Formula B':>20}")
            print(f"  {'MathML length':20s} {sa['mathml_len']:>20} {sb['mathml_len']:>20}")
            print(f"  {'# tuples':20s} {sa['n_tuples']:>20} {sb['n_tuples']:>20}")
            print(f"  {'tuple overlap':20s} {len(tuple_overlap):>20}")
            print(f"  {'# tokens':20s} {sa['n_tokens']:>20} {sb['n_tokens']:>20}")
            print(f"  {'# OOV tokens':20s} {sa['n_oov_tokens']:>20} {sb['n_oov_tokens']:>20}")
            print(f"  {'token overlap':20s} {len(token_overlap):>20}")
            print(f"  {'raw vec norm':20s} {sa['raw_norm']:>20.4f} {sb['raw_norm']:>20.4f}")

            # Cosine between individual token vecs within each formula
            def intra_cosine(vecs):
                if len(vecs) < 2:
                    return float("nan")
                arr = np.array(vecs, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr_n = arr / np.where(norms > 0, norms, 1)
                gram = arr_n @ arr_n.T
                upper = gram[np.triu_indices(len(arr_n), k=1)]
                return float(upper.mean())

            ica = intra_cosine(sa["token_vecs"])
            icb = intra_cosine(sb["token_vecs"])
            print(f"  {'intra-token cosine':20s} {ica:>20.4f} {icb:>20.4f}")

            if sa["raw_vec"] is not None and sb["raw_vec"] is not None:
                sim = cosine(sa["raw_vec"], sb["raw_vec"])
                print(f"  cosine similarity: {sim:.4f}")

            # Show first few tuples of each
            print(f"  Sample tuples A: {sa['tuples'][:3]}")
            print(f"  Sample tuples B: {sb['tuples'][:3]}")


if __name__ == "__main__":
    main()
