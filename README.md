# Retrieval-Augmented Generation with Text–Symbolic Formula Indexing for Scientific Papers

> Master's Thesis — Web and Data Science, Universität Koblenz  
> Author: Abram Jopaul | Supervisor: Prof. Dr. Ralf Lämmel | Co-Supervisor: Susanne Göbel

---

## Project Overview

This project implements and evaluates a **hybrid multi-index Retrieval-Augmented Generation (RAG) system** designed for scientific question answering over mathematical content. Standard RAG pipelines treat documents as plain text, discarding the structure and semantics of mathematical formulas during indexing. This system addresses that gap by building **three parallel retrieval indices** — sparse text, dense text, and structure-aware symbolic formula — and fusing their results before passing context to an LLM.

The benchmark is **ARQMath Task 1** (Answer Retrieval for Math Questions), using questions from Mathematics StackExchange that combine natural language and LaTeX formulas.

---

## Research Questions

**RQ1:** How do sparse, dense, and structure-aware symbolic retrieval representations differ in retrieval quality on scientific questions, and how do those differences affect factuality and mathematical faithfulness in generated answers?

**RQ2:** To what extent does an integrated multi-index RAG system (combining all three representations) improve answer quality and mathematical faithfulness over a text-only RAG baseline?

---

## System Architecture

The pipeline has two phases:

### Preprocessing (Offline)
```
Dataset (ARQMath Task 1)
        │
        ▼
  Preprocessing
  ┌─────────────────────────────────────────────┐
  │  Textual Data  →  Sparse Index (BM25/SPLADE) │
  │                →  Dense Index (bi-encoder)   │
  │  Formula Data  →  Symbolic Formula Index     │
  │                   (Tangent-CFT / GCL)        │
  └─────────────────────────────────────────────┘
```

### Runtime (Query)
```
User Query
    │
    ▼
Query Decomposition (text vs. symbolic parts)
    │
    ├──► Sparse Text Retrieval  ─┐
    ├──► Dense Text Retrieval   ─┼──► Rank Fusion (RRF / CombSUM)
    └──► Formula Retrieval      ─┘
                                  │
                                  ▼
                           Top-K Chunks (Context)
                                  │
                                  ▼
                          LLM (RAG Generation)
                                  │
                                  ▼
                        Response + Evaluation (RAGAS)
```

---

## Components & Key Design Decisions

### 1. Sparse Text Index
- **Method:** BM25 (baseline) or SPLADE (learned sparse)
- **Purpose:** Exact term matching, strong on rare tokens and identifiers
- **Library/Tool:** Apache Lucene / Elasticsearch / Pyserini

### 2. Dense Text Index
- **Method:** Bi-encoder (DPR-style) or late-interaction (ColBERT)
- **Purpose:** Semantic similarity, handles paraphrase and linguistic variation
- **Negative sampling:** Hard negatives (ANCE-style dynamic mining)
- **ANN search:** FAISS or similar

### 3. Symbolic Formula Index
- **Input formats:** LaTeX / MathML
- **Canonical representations:**
  - **SLT (Symbol Layout Tree):** captures 2D spatial arrangement of symbols
  - **OT (Operator Tree):** captures hierarchical operator-operand semantics
- **Candidate systems (TBD):**
  - **Tangent-CFT:** tuple embeddings from SLT/OPT paths via fastText
  - **Approach0:** tree-based path retrieval, subexpression matching
  - **Graph Contrastive Learning (GCL):** formula-level embeddings via graph augmentation
- **Note:** Tangent-CFT + Approach0 are often combined (complementary strengths: semantic vs. structural/partial matching)

### 4. Rank Fusion
- **Methods:** Reciprocal Rank Fusion (RRF), CombSUM, CombMNZ
- **Input:** Ranked lists from all three indices
- **Output:** Single unified ranked list of answer-post candidates

### 5. LLM & Prompting
- Fixed prompt template used across **all** retrieval configurations (no agentic loops, no prompt rewriting)
- Formulas are passed to the LLM in **human-readable LaTeX** (SLT/OT representations are internal to retrieval only)
- LLM performs explanation and synthesis — not formal proof or symbolic derivation
- Improvements in output quality are attributed to **retrieval context quality**, not LLM capability

---

## Dataset

**ARQMath Task 1 (2020 edition)**
- Source: Mathematics StackExchange Q&A threads
- ~100 query topics combining natural language + mathematical expressions
- Corpus: Answer posts to be retrieved
- Relevance judgments (qrels): 4-level graded scale — `high`, `medium`, `low`, `non-relevant`
- Official reference: https://www.cs.rit.edu/~dprl/ARQMath/2020/Task1-answers.html

---

## Evaluation

### Retrieval Metrics (ARQMath qrels)
| Metric | Role |
|---|---|
| **nDCG@k** | Primary — graded relevance, judged docs only (ARQMath protocol) |
| MAP | Secondary — average precision across recall levels |
| Recall@k | Secondary — coverage of relevant documents |
| Precision@k | Secondary — early precision |

Comparisons are run across: individual indices (sparse, dense, symbolic) and all hybrid fusion combinations.

### Generation Metrics (RAGAS framework)
| Metric | Measures |
|---|---|
| **Faithfulness** | Is the response fully supported by retrieved context? (penalizes hallucination) |
| **Answer Relevance** | Does the answer address the question? |
| **Context Precision** | What proportion of retrieved context is relevant? |
| **Context Recall** | Does retrieved context cover the required information? |

---

## Hypotheses

**H1:** Compared to a text-only baseline, dense + sparse + structure-aware symbolic retrieval each achieve higher retrieval effectiveness (nDCG, MAP, Recall) on ARQMath Task 1. Structure-aware retrieval is expected to yield the strongest downstream generation gains.

**H2:** The integrated multi-index RAG system outperforms the text-only RAG baseline on RAGAS metrics (Faithfulness, Answer Relevance), with differences attributed to retrieval quality (same prompt template and LLM across all conditions).

---

## References

Key prior work this project builds on:

| Reference | Relevance |
|---|---|
| Robertson et al. (2009) — BM25 | Sparse retrieval baseline |
| Karpukhin et al. (2020) — DPR | Dense retrieval architecture |
| Khattab & Zaharia (2020) — ColBERT | Late-interaction dense retrieval |
| Formal et al. (2021) — SPLADE | Learned sparse retrieval |
| Mansouri et al. (2019) — Tangent-CFT | Formula embedding model |
| Zhong & Zanibbi (2019) — Approach0 | Structural formula retrieval |
| Wang & Chen (2024) — GCL for Math IR | Graph contrastive formula embeddings |
| Sojka et al. (2018) — MIaS | Math-aware retrieval system (Lucene-based) |
| Cormack et al. (2009) — RRF | Rank fusion method |
| Es et al. (2024) — RAGAS | RAG evaluation framework |
| Zanibbi et al. (2020) — ARQMath | Benchmark dataset and task definition |

---

## Project Status

This repository corresponds to an ongoing Master's thesis. Implementation is in progress.

### Quick Start

**Requirements:** Java (for Pyserini/Anserini), Python 3.12+

```bash
poetry install
poetry run python experiments/run_sparse.py
```
