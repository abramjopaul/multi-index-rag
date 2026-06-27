#!/usr/bin/env python3
"""
Retrieval test: for each of 5 query formulas, rank 100 candidate formulas
by embedding similarity and measure whether the model surfaces the right ones.

Candidate pool is sampled from data/raw/collection/formula/latex_representation_v3
and then enriched with hand-crafted near-matches and distractors to ensure
the pool contains all three similarity categories per query.

Usage:
    python test/formula_retrieval_test.py
    python test/formula_retrieval_test.py --metric cosine   # default
    python test/formula_retrieval_test.py --metric l2
    python test/formula_retrieval_test.py --metric corrected  # mean-shift cosine
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

from multirag.config.path_configs import FORMULA_EMBEDDING_DIR

# ---------------------------------------------------------------------------
# Candidate pool — 100 formulas across 10 mathematical domains.
# Format: (formula_id, latex, domain)
# ---------------------------------------------------------------------------

CANDIDATES = [
    # ── Fractions ──────────────────────────────────────────────────────────
    ("F01", r"\frac{a}{b}",                         "fraction"),
    ("F02", r"\frac{p}{q}",                         "fraction"),
    ("F03", r"\frac{m}{n}",                         "fraction"),
    ("F04", r"\frac{1}{2}",                         "fraction"),
    ("F05", r"\frac{A}{B}",                         "fraction"),
    ("F06", r"\frac{x}{y}",                         "fraction"),
    ("F07", r"\frac{1}{n}",                         "fraction"),
    ("F08", r"\frac{a}{b+c}",                       "fraction"),
    ("F09", r"\frac{a+b}{c+d}",                     "fraction"),
    ("F10", r"\frac{1}{x^2}",                       "fraction"),

    # ── Powers / polynomials ────────────────────────────────────────────────
    ("P01", r"a^2 + b^2",                           "power"),
    ("P02", r"x^2 + y^2",                           "power"),
    ("P03", r"m^2 + n^2",                           "power"),
    ("P04", r"a^2 + b^2 = c^2",                     "power"),
    ("P05", r"a^2 + b^2 + c^2",                     "power"),
    ("P06", r"a^2",                                  "power"),
    ("P07", r"x^2",                                  "power"),
    ("P08", r"n^2",                                  "power"),
    ("P09", r"x^2 + 2xy + y^2",                     "power"),
    ("P10", r"(a+b)^2 = a^2 + 2ab + b^2",           "power"),

    # ── Series / sums ───────────────────────────────────────────────────────
    ("S01", r"\sum_{n=1}^{\infty} \frac{1}{n^2}",   "series"),
    ("S02", r"\sum_{n=1}^{\infty} \frac{1}{n^s}",   "series"),
    ("S03", r"\sum_{k=1}^{\infty} \frac{1}{k^2}",   "series"),
    ("S04", r"\sum_{n=1}^{N} a_n",                   "series"),
    ("S05", r"\sum_{n=0}^{\infty} x^n",              "series"),
    ("S06", r"\sum_{k=0}^{n} \binom{n}{k}",         "series"),
    ("S07", r"\frac{1}{n^2}",                        "series"),
    ("S08", r"\sum_{n=1}^{\infty} \frac{1}{n}",     "series"),
    ("S09", r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}", "series"),
    ("S10", r"\prod_{n=1}^{\infty} \left(1 - \frac{1}{p_n^2}\right)", "series"),

    # ── Trigonometry ────────────────────────────────────────────────────────
    ("T01", r"\sin(x)",                              "trig"),
    ("T02", r"\cos(x)",                              "trig"),
    ("T03", r"\tan(x)",                              "trig"),
    ("T04", r"\sin(\theta)",                         "trig"),
    ("T05", r"\sin(x) + \cos(x)",                   "trig"),
    ("T06", r"\sin^2(x) + \cos^2(x) = 1",           "trig"),
    ("T07", r"\sin(x) \cos(y)",                      "trig"),
    ("T08", r"\cos(2x) = 1 - 2\sin^2(x)",           "trig"),
    ("T09", r"\sin(x + y) = \sin x \cos y + \cos x \sin y", "trig"),
    ("T10", r"\tan(x) = \frac{\sin(x)}{\cos(x)}",   "trig"),

    # ── Exponential / complex ───────────────────────────────────────────────
    ("E01", r"e^{i\pi} + 1 = 0",                    "exponential"),
    ("E02", r"e^{i\pi} = -1",                        "exponential"),
    ("E03", r"e^{ix} = \cos(x) + i\sin(x)",         "exponential"),
    ("E04", r"e^{i\pi}",                             "exponential"),
    ("E05", r"e^{ix}",                               "exponential"),
    ("E06", r"e^x",                                  "exponential"),
    ("E07", r"e^{x+y} = e^x e^y",                   "exponential"),
    ("E08", r"\ln(e^x) = x",                        "exponential"),
    ("E09", r"e^{-x}",                               "exponential"),
    ("E10", r"e^{2\pi i} = 1",                       "exponential"),

    # ── Function application ─────────────────────────────────────────────────
    ("G01", r"f(x)",                                 "function"),
    ("G02", r"g(x)",                                 "function"),
    ("G03", r"f(a)",                                 "function"),
    ("G04", r"P(X)",                                 "function"),
    ("G05", r"F(k)",                                 "function"),
    ("G06", r"\Gamma(z)",                            "function"),
    ("G07", r"f(x) + g(x)",                         "function"),
    ("G08", r"f(g(x))",                              "function"),
    ("G09", r"f'(x)",                                "function"),
    ("G10", r"f(x) = x^2",                          "function"),

    # ── Number theory ────────────────────────────────────────────────────────
    ("N01", r"\gcd(a, b)",                           "number_theory"),
    ("N02", r"\gcd(m, n) = 1",                       "number_theory"),
    ("N03", r"\gcd(a, b) = \gcd(b, a \mod b)",       "number_theory"),
    ("N04", r"a \equiv b \pmod{n}",                  "number_theory"),
    ("N05", r"p \mid n",                             "number_theory"),
    ("N06", r"\phi(n)",                              "number_theory"),
    ("N07", r"\phi(mn) = \phi(m)\phi(n)",            "number_theory"),
    ("N08", r"a^{\phi(n)} \equiv 1 \pmod{n}",        "number_theory"),
    ("N09", r"n = p_1^{e_1} p_2^{e_2} \cdots p_k^{e_k}", "number_theory"),
    ("N10", r"\lfloor \sqrt{n} \rfloor",             "number_theory"),

    # ── Linear algebra / matrices ────────────────────────────────────────────
    ("M01", r"Ax = b",                               "linear_algebra"),
    ("M02", r"A^{-1}",                               "linear_algebra"),
    ("M03", r"\det(A)",                              "linear_algebra"),
    ("M04", r"\det(AB) = \det(A)\det(B)",            "linear_algebra"),
    ("M05", r"\|x\|_2",                              "linear_algebra"),
    ("M06", r"\|x\|_2^2 = x^T x",                   "linear_algebra"),
    ("M07", r"A^T A",                                "linear_algebra"),
    ("M08", r"\text{rank}(A)",                       "linear_algebra"),
    ("M09", r"A v = \lambda v",                      "linear_algebra"),
    ("M10", r"\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}", "linear_algebra"),

    # ── Limits / calculus ────────────────────────────────────────────────────
    ("C01", r"\lim_{x \to 0} \frac{\sin x}{x} = 1", "calculus"),
    ("C02", r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e", "calculus"),
    ("C03", r"\int_a^b f(x)\, dx",                   "calculus"),
    ("C04", r"\int_0^1 x^2\, dx = \frac{1}{3}",      "calculus"),
    ("C05", r"\frac{d}{dx} \sin(x) = \cos(x)",       "calculus"),
    ("C06", r"\frac{d}{dx} e^x = e^x",               "calculus"),
    ("C07", r"\int e^x dx = e^x + C",                "calculus"),
    ("C08", r"\lim_{x \to \infty} \frac{1}{x} = 0",  "calculus"),
    ("C09", r"\frac{\partial f}{\partial x}",         "calculus"),
    ("C10", r"\nabla f = 0",                          "calculus"),

    # ── Probability / statistics ─────────────────────────────────────────────
    ("R01", r"P(A \cup B) = P(A) + P(B) - P(A \cap B)", "probability"),
    ("R02", r"P(A \mid B) = \frac{P(A \cap B)}{P(B)}", "probability"),
    ("R03", r"E[X]",                                 "probability"),
    ("R04", r"E[X^2] - E[X]^2",                      "probability"),
    ("R05", r"\text{Var}(X) = E[X^2] - (E[X])^2",   "probability"),
    ("R06", r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}", "probability"),
    ("R07", r"\mu = E[X]",                           "probability"),
    ("R08", r"\sigma^2 = \text{Var}(X)",             "probability"),
    ("R09", r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}", "probability"),
    ("R10", r"\sum_{k=0}^{n} P(X=k) = 1",            "probability"),
]

# ---------------------------------------------------------------------------
# 5 Query formulas with labeled ground truth in the candidate pool
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "qid": "Q1",
        "latex": r"\frac{a}{b+c}",
        "description": "Simple fraction with sum in denominator",
        "expected": {
            "very_similar": ["F01","F02","F03","F05","F06","F08"],  # simple fractions
            "subset_match": ["F04","F07","F09","F10","T10"],        # fractions, subset
            "not_matching": ["P01","S01","T01","E01","N01","M01"],   # unrelated
        },
    },
    {
        "qid": "Q2",
        "latex": r"x^2 + y^2",
        "description": "Sum of squares",
        "expected": {
            "very_similar": ["P01","P02","P03"],                    # exact same structure
            "subset_match": ["P04","P05","P06","P07","P08","P09","P10"],  # extended or partial
            "not_matching": ["S01","T01","E01","N01","M01","C01","R01"],   # unrelated
        },
    },
    {
        "qid": "Q3",
        "latex": r"\sum_{n=1}^{\infty} \frac{1}{n^3}",
        "description": "Infinite series with 1/n^3",
        "expected": {
            "very_similar": ["S01","S02","S03","S08"],              # same series structure
            "subset_match": ["S04","S05","S07","S09"],              # related sums
            "not_matching": ["T01","E01","G01","N01","M01","C03","R01"],   # unrelated
        },
    },
    {
        "qid": "Q4",
        "latex": r"e^{i\theta} = \cos(\theta) + i\sin(\theta)",
        "description": "Euler's formula",
        "expected": {
            "very_similar": ["E01","E02","E03","E10"],              # Euler variants
            "subset_match": ["E04","E05","E06","T01","T02","T06"],  # component parts
            "not_matching": ["S01","P01","N01","M01","C03","R01"],   # unrelated
        },
    },
    {
        "qid": "Q5",
        "latex": r"\gcd(a, b) = 1",
        "description": "Coprimality condition",
        "expected": {
            "very_similar": ["N01","N02","N03"],                    # gcd variants
            "subset_match": ["N04","N05","N06","N08"],              # number theory related
            "not_matching": ["F01","P01","S01","T01","E01","M01","C01","R01"],  # unrelated
        },
    },
]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embedder():
    from multirag.indexing.formula.faiss_scalar_quantizer import (
        FormulaFAISSIndexerIVFScalarQuantizer,
    )
    tmp = tempfile.mkdtemp(prefix="retrieval_test_")
    emb = FormulaFAISSIndexerIVFScalarQuantizer(
        index_path=tmp,
        embedding_dir=str(FORMULA_EMBEDDING_DIR),
        representation="slt",
    )
    emb.embed_formula("x")  # warm up
    return emb


def l2_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def score(vec_q, vec_c, metric, vocab_mean=None):
    if metric == "cosine":
        return float(np.dot(l2_norm(vec_q), l2_norm(vec_c)))
    elif metric == "corrected":
        return float(np.dot(l2_norm(vec_q - vocab_mean), l2_norm(vec_c - vocab_mean)))
    else:  # l2
        return -float(np.linalg.norm(vec_q - vec_c))  # negative so higher = more similar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["cosine","l2","corrected"], default="cosine")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    print(f"\nLoading embedder (embedding_dir={FORMULA_EMBEDDING_DIR}) …")
    embedder = build_embedder()
    embedder._generate_query_embedding("x")  # ensure _query_model_manager exists
    print("Model loaded.\n")

    vocab_mean = None
    if args.metric == "corrected":
        wv = embedder._query_model_manager.model.wv
        vocab_mean = np.mean(wv.vectors, axis=0).astype(np.float32)
        print(f"Vocab mean norm: {np.linalg.norm(vocab_mean):.4f}\n")

    # Embed all candidates once
    print(f"Embedding {len(CANDIDATES)} candidates …")
    cand_vecs = {}
    failed = []
    for cid, latex, domain in CANDIDATES:
        v = embedder.embed_formula(latex)
        if v is not None and np.any(v):
            cand_vecs[cid] = v
        else:
            failed.append(cid)
    print(f"  Embedded: {len(cand_vecs)}  |  Failed (no MathML): {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed}")
    print()

    # Per-query retrieval
    for q in QUERIES:
        qid = q["qid"]
        q_vec = embedder.embed_formula(q["latex"])
        if q_vec is None or not np.any(q_vec):
            print(f"[{qid}] SKIP — embedding failed for query: {q['latex'][:60]}")
            continue

        # Score all candidates
        scored = []
        for cid, latex, domain in CANDIDATES:
            if cid not in cand_vecs:
                continue
            s = score(q_vec, cand_vecs[cid], args.metric, vocab_mean)
            # Look up expected category
            cat = "—"
            for label, ids in q["expected"].items():
                if cid in ids:
                    cat = label
                    break
            scored.append((s, cid, latex, domain, cat))

        scored.sort(reverse=True)

        print("=" * 90)
        print(f"[{qid}] Query: {q['latex'][:70]}")
        print(f"       {q['description']}")
        print("=" * 90)
        print(f"{'Rank':<5} {'Score':>7}  {'ID':<5} {'Expected category':<18}  Formula")
        print("-" * 90)

        for rank, (s, cid, latex, domain, cat) in enumerate(scored[:args.top_k], 1):
            marker = "✓" if cat == "very_similar" else ("~" if cat == "subset_match" else " ")
            print(f"{rank:<5} {s:>7.4f}  {cid:<5} {marker} {cat:<17}  {latex[:50]}")

        # Precision metrics
        top10_ids = {cid for _, cid, _, _, _ in scored[:10]}
        expected_vs = set(q["expected"]["very_similar"]) & set(cand_vecs.keys())
        expected_sub = set(q["expected"]["subset_match"]) & set(cand_vecs.keys())

        vs_in_top10 = len(expected_vs & top10_ids)
        sub_in_top10 = len(expected_sub & top10_ids)

        print()
        print(f"  very_similar  in top-10: {vs_in_top10}/{len(expected_vs)}")
        print(f"  subset_match  in top-10: {sub_in_top10}/{len(expected_sub)}")
        print()


if __name__ == "__main__":
    main()
