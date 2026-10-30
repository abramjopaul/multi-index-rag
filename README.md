# Retrieval-Augmented Generation with Text–Symbolic Formula Indexing

**MSc thesis, Web and Data Science, Universität Koblenz** · Abram Jopaul
Examiner: Prof. Dr. Ralf Lämmel · Supervisor: Susanne Göbel

📄 **[Read the thesis](thesis/thesis-report.pdf)**

A mathematical question is asked in words and formulas together, and a retriever has to read both.
A formula enters an ordinary text index as the characters it was typed with — rename the variables
and the index sees an unrelated string. What the index never holds is the arrangement itself, and
that is where a formula's meaning lies.

Indexing that structure is what formula search does. Whether it also helps when *answers* are what
you retrieve is a separate question, because an answer is relevant when it resolves the question,
not when its formulas match. This thesis builds the system that answers it, and measures it on
community mathematical question answering (Math Stack Exchange, via ARQMath-3). Every claim here is
a claim about that collection.

## What it asks

- **RQ1** — Does adding structure-aware formula indexes to text-only retrieval improve answer
  retrieval, and does the way those indexes are combined govern how well the result works?
- **RQ2** — Does the best multi-index retriever's context make generated answers more factually
  correct, how faithful are those answers to it, and do both rise with retrieval quality?
- **RQ3** — Do the RQ2 answers hold as the generator grows from 1.5 to 7 billion parameters?

## What it found

**Formula evidence does not improve answer retrieval.** Dense retrieval alone reaches **0.557**
nDCG′@10 over the 78 official topics. Adding one formula index leaves it behind — 0.508 with OPT,
0.499 with SLT, 0.412 with SLT-TYPE. Adding all three, each voting separately, reaches **0.322**:
further back than one.

**How the rankings are merged decides how far behind.** Collapsing the three formula rankings into a
single vote before fusing ("blocked") reaches **0.551** against **0.374** for letting each vote on
its own — ahead in all six measured cells. It draws level with text-only retrieval and nothing built
here passes it. That arrangement earns its place by keeping formula evidence from doing harm, not by
supplying what the text rankings lack. Removing the weakest index improves flat fusion outright,
0.374 → 0.416.

**The index is not the problem.** It performs comparably with a published system on formula search.
The diagnostic shows why it does not carry over: a topic's formulas are contained in the relevant
answers more than the non-relevant ones on **70 of 78** topics — but among the best-matching
answers, where a ranking is actually decided, on only **30**. Formula matching finds the right
neighbourhood and cannot order inside it.

**Retrieval still reaches the generated answer, and does not write much of it.** A retrieved context
beats no context on factual correctness (0.206 against 0.170). But against the strongest text-only
retrieval the two measures move in opposite directions: more faithful (0.189 vs 0.171), less correct
(0.206 vs 0.241) — the generator treats formula-similar posts as pertinent and draws on them, and
the answer is not there. Both measures do rise with retrieval quality across the fifteen
configurations (Kendall's τ = 0.829 for faithfulness, 0.581 for correctness, over 105 pairs), and
the effect thins as the generator grows. The highest faithfulness anywhere is **0.225**: in every
configuration at every size, most of what an answer says comes from the generator.

| | Verdict |
|---|---|
| H1.1 one formula index helps · H1.2 all three help · H1.4 every index earns its place | not supported |
| H1.3 merge arrangement matters | **supported** |
| H2.1 context beats no context · H2.3 / H2.4 both measures rise with retrieval quality | **supported** |
| H2.2 beats text-only on both measures · H3.1 survives a larger generator | not supported |

## How it's built

Three retrieval channels, fused: **BM25** (Pyserini), a **dense** index
(`multi-qa-mpnet-base-dot-v1` + FAISS), and **three structure-aware formula indexes** — SLT, OPT and
SLT-TYPE, built on adapted TangentCFT path-tuple embeddings. Rankings merge by **reciprocal rank
fusion** (k = 60) in ten arrangements, giving fifteen reported configurations. Each supplies context
to **Qwen2.5-Instruct** (1.5B / 3B / 7B) under one fixed prompt at temperature 0.

## How it's evaluated

Post-hoc systems retrieve documents the ARQMath assessors never judged, which makes naive scores
meaningless: standard nDCG@10 reads 0.029–0.095 for the formula-only configurations where prime
(judged-only) scoring reads 0.218–0.283. The thesis handles this with a **validated LLM relevance
judge** (`gemini-3.5-flash-lite`, Kendall's τ = 0.80 / 0.87 / 0.84 against the human assessors
across the three ARQMath editions) and **one pooled qrel** shared by all fifteen configurations, so
the ideal ranking is identical for each.

Reported throughout: **nDCG′ and P′ at @5, @10 and @20**, with `ndcg'@10` primary. No significance
testing is run; claims are ordinal or rest on consistency across every measured cell. Chapter 4 of
the thesis sets the protocol out in full.

## Repository layout

| Path | Contents |
|---|---|
| `src/multirag/` | The library: indexing, formula representation, embedding, generation, evaluation |
| `experiments/` | Run drivers, grouped by phase (see below) |
| `experiments/rearrange/` | **The final evaluation base** — 78 topics, the pooled judge qrel, and the reported analyses |
| `configs/experiments/` | Hand-written input configs |
| `configs/task1`, `task2`, `task_c` | Exported run records, one per completed run |
| `thesis/report-tex/` | LaTeX sources |
| `test/` | `test_hole_rate.py` is a unit test; the rest are diagnostic probes |

The `experiments/` subdirectories are named by phase: **track A** validated the formula index on
formula search · **track B** retrieval scoring and calibration · **track C** generation ·
**track V** judge validation · **rearrange** the final evaluation base that every reported number
comes from.

## Reproducing

The ARQMath-3 collection (~31 GB) is not redistributed here — request it from the organisers.
Chapters 4 to 6 of the thesis document the pipeline, its settings and the order the stages run in.

```bash
poetry install
```

Pyserini needs a working Java installation. Building the thesis LaTeX additionally expects a file at
`thesis/report-tex/images/abram_jopaul_signature.png`, which is not published; any blank PNG at that
path lets the build run.

## Citation

```bibtex
@mastersthesis{jopaul2026formularag,
  author = {Jopaul, Abram},
  title  = {Retrieval-Augmented Generation with Text--Symbolic Formula Indexing},
  school = {Universität Koblenz},
  year   = {2026}
}
```
