"""Shared constants for the results/rearranged/ layout (decided 2026-09-18).

Scope: RQ1, ARQMath-3 (2022) Task 1, the 15 reported retrieval configurations
(manifest §4e), scored on the 78 official topics against one pooled judge qrel.
Standalone on purpose -- no multirag imports -- so every script here runs with
plain Python + pytrec_eval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- sources (read-only) -----------------------------------------------------
SRC_RUNS_DIR = ROOT / "data" / "runs" / "task1"
SRC_AUDIT_DIR = ROOT / "results" / "per_topic"  # <config>_answers_audit.csv
SRC_TRACK_V_DIR = ROOT / "results" / "track_v"
OFFICIAL_QRELS = ROOT / "data" / "raw" / "qrels" / "qrel_task1_2022_official.tsv"

# --- outputs -----------------------------------------------------------------
OUT = ROOT / "results" / "rearranged"
OUT_RUNS = OUT / "runs_task1"
OUT_PER_RUN = OUT / "judge_qrels_per_run"
OUT_POOLED = OUT / "judge_qrels_pooled"
OUT_TRACK_V = OUT / "track_v"
OUT_METRICS = OUT / "metrics_judge_pool20"

POOLED_QREL_NAME = "qrel_task1_2022_official_judge_pool20.tsv"
PER_RUN_SUFFIX = ".judge_top20.qrel.tsv"

POOL_DEPTH = 20          # judge labels exist for every run's top 20, nothing deeper
CUTOFFS = (5, 10, 20)    # reported cutoffs (manifest §9)
RELEVANCE_LEVEL = 2      # H+M binarisation for P'

# (registry id, short label, config_name) -- the 15 reported configs, manifest §4e / §14
CONFIGS: list[tuple[str, str, str]] = [
    ("R1.1", "BM25", "bm25_baseline_20260618"),
    ("R1.2", "Dense", "dense_mpnet_20260614"),
    ("R2.1", "SLT", "formula_v1_slt_20260706"),
    ("R2.4", "OPT", "formula_v1_opt_20260706"),
    ("R2.7", "SLT-TYPE", "formula_v1_slt_type_20260707"),
    ("R3.1", "BM25 + dense", "fusion_bm25_dense"),
    ("R3.2", "dense + SLT", "fuse_dense_mpnet_slt_fanout_20260709_rrf"),
    ("R3.3", "dense + OPT", "fuse_dense_mpnet_opt_fanout_20260709_rrf"),
    ("R3.4", "dense + SLT-TYPE", "fuse_dense_mpnet_slt_type_fanout_20260709_rrf"),
    ("R3.5", "formula block (SLT+OPT+SLT-TYPE)", "fuse_slt_opt_slt_type_fanout_20260709_rrf"),
    ("R3.6", "blocked (BM25 + dense + formula block)", "fuse_bm25_dense_formula_fused_block_20260709_rrf"),
    ("R3.7", "flat 5-way", "fuse_full_5_way_20260709_rrf"),
    ("R3.8", "blocked, -BM25", "fuse_dense_formula_fused_block_drop_bm25_20260709_rrf"),
    ("R3.9", "flat, -BM25", "fuse_full_drop_bm25_20260709_rrf"),
    ("R3.10", "flat, -SLT-TYPE", "fuse_full_drop_slt_type_20260709_rrf"),
]
CONFIG_NAMES = [c for _, _, c in CONFIGS]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def topic_key(t: str) -> tuple:
    """A.301 -> ('A', 301) so topics sort numerically."""
    p, _, n = t.partition(".")
    return (p, int(n)) if n.isdigit() else (p, n)


def doc_key(d: str) -> tuple:
    return (0, int(d)) if d.isdigit() else (1, d)


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    q: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                q.setdefault(p[0], {})[p[2]] = int(p[3])
    return q


def read_run(path: Path) -> dict[str, dict[str, float]]:
    """Same parsing as multirag.evaluation.metrics._parse_run (5- or 6-column TREC)."""
    run: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            if p[1].upper() == "Q0":
                qid, doc, score = p[0], p[2], float(p[4])
            else:
                qid, doc, score = p[0], p[1], float(p[3])
            run.setdefault(qid, {})[doc] = score
    return run


def ranking(run: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Score-descending, stable on file order -- identical to rescore_run._ranking_from_run,
    which is how the top 20 that the judge labelled was chosen."""
    return {q: [d for d, _ in sorted(docs.items(), key=lambda x: -x[1])] for q, docs in run.items()}


# --- added for the RQ1 extras (R6, human_only, Task 2, R7), 2026-09-18 -----------------
QRELS_DIR = ROOT / "data" / "raw" / "qrels"
OFFICIAL_QRELS_T2 = QRELS_DIR / "qrel_task2_2022_official.tsv"
SRC_R6_RUN = ROOT / "data" / "runs" / "arqmath-2022-runs" / "baseline_task1_tf_idf_terrier.tsv"
SRC_T2_OURS = ROOT / "data" / "runs" / "task2" / "a3_slt_opt_slt_type_rrf_k60_n1000_2022_20260902.tsv"
SRC_T2_CFT = ROOT / "data" / "runs" / "task2" / "DPRL-Task2-CFT-auto-math-A.tsv"
SLT_REPRESENTATION_DIR = ROOT / "data" / "raw" / "collection" / "formula" / "slt_representation_v3"


def judged_only(run: dict[str, dict[str, float]], qrel: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Prime variant: unjudged docs removed -- same as multirag.evaluation.metrics._to_judged_only."""
    return {q: {d: s for d, s in docs.items() if d in qrel.get(q, {})} for q, docs in run.items()}


def hole_rate_per_topic(rank: dict[str, list[str]], qrel: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    """Per-topic share of the top-k with no qrel entry -- as multirag.eval.hole_rate, before averaging."""
    out = {}
    for q, docs in rank.items():
        top = docs[:k]
        if top:
            judged = qrel.get(q, {})
            out[q] = sum(1 for d in top if d not in judged) / len(top)
    return out


def write_checksums(folder: Path) -> None:
    """SHA256SUMS over every file in `folder` (recursively), written last, never overwritten."""
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    with open(folder / "SHA256SUMS", "x") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(folder)}\n")
