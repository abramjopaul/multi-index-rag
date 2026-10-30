# Track C v3 — re-implementation brief

> **Purpose.** Re-run Track C under a *fair* prompt, with Qwen2.5-**1.5B**-Instruct as the workhorse
> generator, then ladder to 3B and 7B on a reduced set. Replaces the v1/v2 ladder, which is
> confounded (manifest O-F, §15 O-1) and cannot support Chapter 6.
>
> **This file is written to be handed to a coding agent verbatim.** Paths, config keys and CLI flags
> are the real ones in `multi-index-rag` @ `feature/dense-multi-thread`.

---

## 0. Why this re-run exists

The first ladder compared a floor running prompt **v1** (open-book, own knowledge, never refuses)
against retrieval conditions running **v2** (context only, outside knowledge forbidden, refusals
scored zero). Retrieval was handicapped twice, so "retrieval does not help" was never a finding about
retrieval. Fix: **one template for every condition**, context block present or absent.

**What the three generator sizes buy.** Run on the same conditions under the same prompt, they show
whether Chapter 6's verdicts hold at more than one scale. The old data suggests they may not: 7B with
no retrieval reached 0.361 FC-P, above every 3B retrieval condition, which hints that the marginal
value of retrieved context falls as the generator knows more. If 1.5B shows retrieval helping and 7B
shows it helping little, that is a crossover worth reporting.

> ⚠ **This does not become a hypothesis.** Göbel ruled out a generator-capacity hypothesis — the
> hypotheses stay on the retrieval side, per RQ1 and RQ2. So the ladder is a **robustness check
> reported without a verdict** (manifest §6.6): it shows whether the RQ2 answer is stable across
> generator scale, and rules on nothing. The interpretation goes in the Discussion as an observation
> about the limits of what this pipeline can show. An earlier version of this brief proposed it as
> "H2.4"; that is superseded — H2.4 is now the context-depth hypothesis.

What the ladder *is* for: it makes 1.5B a defensible choice of workhorse rather than a cost compromise,
and it is the only evidence that the Chapter 6 verdicts are not an artefact of one small generator.

---

## 1. Decisions to make before writing code

Three, all cheap, all blocking.

### 1.1 Prompt v3 — the exact template

Add this to `src/multirag/generation/prompt_template.py`, after `PromptTemplateV2`. It follows the
existing frozen pattern. **Copy it verbatim** — the SHA256 below is the frozen hash of these exact
bytes, and the smoke test asserts it.

```python
class PromptTemplateV3(PromptTemplate):
    """Version 3 — frozen 2026-09-13. Do not modify; create v4 instead.

    The fair retrieval template, used for EVERY Track C v3 condition including
    the no-retrieval floor. Retrieved context is an addition to the model's own
    knowledge, not a replacement for it: there is no instruction to answer only
    from context, and no refusal instruction. The context block disappears
    entirely when `contexts` is empty (as in v1), so the floor and the retrieval
    conditions render from byte-identical template strings and therefore share
    one prompt_sha256. That shared hash is what closes the v1/v2 asymmetry
    (manifest O-F): under v2 the retrieval conditions were forbidden prior
    knowledge while the v1 floor was not, so the comparison was unfair by
    construction.
    """

    version = "v3"
    refusal_phrase = None  # no refusal is instructed; see refusal detection note

    @property
    def _system_template(self) -> str:
        return (
            "You are a mathematics question-answering assistant. "
            "Answer the question as accurately and completely as you can, and write "
            "all mathematics in LaTeX. Retrieved context may be provided below; when "
            "it is, use it wherever it helps, and prefer it over your own recollection "
            "where the two conflict. You may also draw on your own knowledge, whether "
            "or not retrieved context is provided. Give the reasoning needed to "
            "justify the answer, without restating the question or padding."
        )

    @property
    def _user_template(self) -> str:
        return (
            "{context_block}"
            "Question:\n"
            "{question}\n"
            "\n"
            "Answer:"
        )

    def render(self, question: str, contexts: list[str]) -> list[dict]:
        if contexts:
            context_block = (
                "Retrieved context:\n"
                + "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
                + "\n\n"
            )
        else:
            context_block = ""

        user_content = self._user_template.format(
            question=question,
            context_block=context_block,
        )
        return [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": user_content},
        ]
```

Register it:

```python
_REGISTRY: dict[str, PromptTemplate] = {
    "v1": PromptTemplateV1(),
    "v2": PromptTemplateV2(),
    "v3": PromptTemplateV3(),
}
```

**Frozen hash — assert this, do not recompute-and-accept:**

```
prompt_sha256(v3) = 45bdd2f257d0808a9392b3a1daba96fc37f8c72f19df2c346e56d8088e79b769
```

#### What the model actually sees

**A retrieval condition** (k = 2 shown; identical shape at k = 1, 5, 10, 20):

```
--- system ---
You are a mathematics question-answering assistant. Answer the question as accurately and
completely as you can, and write all mathematics in LaTeX. Retrieved context may be provided
below; when it is, use it wherever it helps, and prefer it over your own recollection where the
two conflict. You may also draw on your own knowledge, whether or not retrieved context is
provided. Give the reasoning needed to justify the answer, without restating the question or
padding.
--- user ---
Retrieved context:
[1] Consider the Fourier series of $x^2$ on $[-\pi,\pi]$ ...

[2] Euler's original argument uses the product expansion of $\sin x$ ...

Question:
Prove that $\sum_{n=1}^{\infty} 1/n^2 = \pi^2/6$.

Answer:
```

**The floor** (`--no-rag`, `contexts == []`) — same system string, context block gone:

```
--- system ---
You are a mathematics question-answering assistant. Answer the question as accurately and
completely as you can, and write all mathematics in LaTeX. Retrieved context may be provided
below; when it is, use it wherever it helps, and prefer it over your own recollection where the
two conflict. You may also draw on your own knowledge, whether or not retrieved context is
provided. Give the reasoning needed to justify the answer, without restating the question or
padding.
--- user ---
Question:
Prove that $\sum_{n=1}^{\infty} 1/n^2 = \pi^2/6$.

Answer:
```

#### Why each clause is there

| Clause | Why it is in, and what breaks without it |
|---|---|
| "Retrieved context **may** be provided below; when it is…" | Conditional phrasing is what lets one system string serve both the floor and the retrieval conditions. An unconditional "use the retrieved context below" would refer to nothing in the floor case and invites the model to announce that no context was given — a new asymmetry in place of the old one. |
| "use it wherever it helps, and **prefer it over your own recollection where the two conflict**" | Without a preference rule the model can simply ignore the passages and answer from priors, which would collapse every condition onto the floor and make the whole experiment uninformative. This is the clause that gives retrieval a channel to change the answer — i.e. the thing RQ2 measures. It is applied identically to every condition, so it biases no comparison. |
| "You may **also** draw on your own knowledge" | The repair itself. This is what v2 forbade and v1 permitted, and that difference is the confound. |
| "write all mathematics in LaTeX" | Matches v1/v2 and the surface form the retrieved posts use; keeps the RAGAS claim decomposition comparable across templates. |
| "Give the reasoning needed to justify the answer, **without restating the question or padding**" | Verbosity control. "Answer completely" plus a 1024-token ceiling is how truncation returns — 57/100 BM25 topics hit the old 512 ceiling mid-derivation. The brevity bound plus smoke checks C2/C3 is the guard. |
| **No** refusal instruction, `refusal_phrase = None` | A refusal instruction is why the old ladder scored zeros against a floor that never refuses. Coverage should now be ≈ 1.0 for every condition, which is what collapses the coverage / selective / composite triple into one number per measure. |
| Numbered passages `[1] [2] …` | A deliberate improvement over v1/v2's bare `\n\n` join: at k = 20 an undelimited concatenation of twenty posts is a wall of text. **Record it** — it means v2-versus-v3 differences have two causes (instruction *and* delimiters), so do not attribute them to the instruction alone. No hypothesis rests on a v2-versus-v3 comparison, so this costs nothing. If you want the instruction change isolated, drop the numbering and use `"\n\n".join(contexts)` — one line, and the hash above changes. |

#### Refusal detection under v3

`refusal_phrase` is `None`, so exact-match detection does not apply. Keep refusal detection as a
**diagnostic only**, via a small regex over the answer (`cannot answer`, `not enough information`,
`no context (was )?provided`, `unable to determine`). A detected refusal is **reported, never scored
zero** — under v3 a refusal is spontaneous model behaviour, not an instructed response, and smoke
check D7 flags it if it exceeds 5%.

#### ⚠ The code change that will silently undo all of this

`experiments/track_c/generate_and_score.py` documents its `--prompt-template-version` flag as:

> `--no-rag` defaults to `'v1'` (open-book) unless set here

So a floor run launched without an explicit override gets **v1**, its `prompt_sha256` differs from the
retrieval runs', and O-F is quietly back with no error raised. Fix it properly rather than relying on
remembering the flag: when the resolved config's `prompt_template_version` is `v3`, `--no-rag` must
use **v3** as well (v3 is context-optional by construction, so the v1 fallback exists only for v2's
sake). Then smoke check **A3** asserts floor-hash == retrieval-hash as the backstop.

> **Do not delete or modify v1 or v2.** v2 remains a legitimate *strict-grounding* side condition —
> it is the only setting in which faithfulness means "grounded in the supplied context" — and the
> existing v2 runs stay in W&B as the historical record.

### 1.2 Hold the reference construction constant — change one thing at a time

The code is the authority: `--max-reference-answers 5`, High-labelled first, topped up with Medium
where a topic has fewer than five High, topics with neither skipped for reference-requiring metrics
(**skipped, never scored zero**).

**Do not change this in the same re-run as the prompt.** The purpose here is to remove the prompt
confound; changing the reference at the same time would make the new numbers incomparable with the
old ones and leave two variables moving at once.

> ⚠ **Documentation discrepancy to fix separately.** Three sources disagree on the reference: the
> code and manifest R8 say *5, topped up with Medium*; the manifest Layer 6d says *up to N High*;
> project notes say *up to 10, High only*. Align the prose to the code. It changes no numbers, but it
> is the kind of thing a reviewer catches.

### 1.3 Pin the model revisions

`configs/task_c/generation_eval.yaml` carries `revision: "main"` with a TODO. Close it now — resolve
and pin the commit hash for all three generators. It costs one API call each and it is a genuine
reproducibility hole.

| Role | Model | Pin |
|---|---|---|
| Workhorse | `Qwen/Qwen2.5-1.5B-Instruct` | resolve + pin |
| Ladder mid | `Qwen/Qwen2.5-3B-Instruct` | resolve + pin |
| Ladder top | `Qwen/Qwen2.5-7B-Instruct` | resolve + pin |

---

## 2. How many configurations does 1.5B sweep? — 15, the ones the report keeps

**All 15 reported configurations, and the condition is about *reporting*, not running: the 15 are one
experiment, not 15 findings.**

The review's objection was never "too many runs". It was *"we need one coherent story and support for
it, not as many experiments as you managed to run, all with different assumptions."* A uniform sweep of
every reported configuration through one fixed generator, one fixed prompt, one fixed *k* is a single
experiment with a single assumption set. That is the opposite of what she criticised.

It is also the only way to test **H2.2** and **H2.3** honestly. The old §5.7 tested "better retrieval →
better answers" across five systems spanning **0.028 nDCG′@10** and had to concede that "the reversal is
as consistent with noise as with an inverse relationship" — which is not a test, it is an admission. The
15 configurations span from the formula block near the bottom of the retrieval range to the blocked
fusion at the top: a far wider spread, and 15 points instead of 5. That turns both hypotheses from an
eyeball comparison into an association with a coefficient, and it answers the separate worry that the
gaps are too small — you stop leaning on a 0.023 gap and start leaning on the slope across the range.

> **Why 15 and not 21** (decided 2026-09-14). The heuristic and filter-off variants were cut from the
> report: fanout with the triviality filter on is now stated as *the method* (manifest §4.3), so the
> formula channels are simply SLT, OPT and SLT-TYPE. Six run files therefore leave the narrative — the
> three `*_heuristic_20260707` and the three `*_no_filter_20260709`. Keeping them as unexplained points
> in a scatter would reintroduce configurations the report does not otherwise discuss, which is exactly
> the noise being cut. The sweep matches the report.

**The reporting discipline that makes it defensible:**

- Chapter 6 gets **one table of the 7 ladder conditions** (§4.2) and **one scatter over all 15**, with
  faithfulness on one *y* axis and FC-precision on the other, Spearman ρ on each. Two exhibits.
- Five of the 15 are named in the ladder table; **the other ten appear only as points**, never as
  paragraphs.
- **Never rank the 15 and never name a "best config for generation".** R4's resolving power forbids it:
  differences of a few points under a filled channel are not results. Say this once, in §6.1.
- 1.5B only. The sweep is affordable exactly where it is most useful, and the 3B/7B ladder exists to
  test robustness, not to repeat the sweep.

### The 15 configurations

Run files live in **`data/runs/task1/`** (note: the old configs record
`trec_file: data/runs/track_b/…`, which is a stale path — fix it).

| # | `config_name` | Run file stem | Role in the sweep |
|---|---|---|---|
| 1 | `bm25` | `bm25_baseline_20260618` | text, sparse — **ladder** |
| 2 | `dense_mpnet` | `dense_mpnet_20260614` | text, dense; the bar — **ladder** |
| 3 | `formula_v1_slt` | `formula_v1_slt_20260706` | formula, SLT fanout + filter |
| 4 | `formula_v1_opt` | `formula_v1_opt_20260706` | formula, OPT fanout + filter |
| 5 | `formula_v1_slt_type` | `formula_v1_slt_type_20260707` | formula, TYPE fanout + filter |
| 6 | `fusion_bm25_dense` | `fusion_bm25_dense` | text-only fusion — **ladder** |
| 7 | `fuse_dense_slt` | `fuse_dense_mpnet_slt_fanout_20260709_rrf` | dense + one formula rep |
| 8 | `fuse_dense_opt` | `fuse_dense_mpnet_opt_fanout_20260709_rrf` | dense + one formula rep |
| 9 | `fuse_dense_slt_type` | `fuse_dense_mpnet_slt_type_fanout_20260709_rrf` | dense + one formula rep |
| 10 | `formula_block` | `fuse_slt_opt_slt_type_fanout_20260709_rrf` | formula block alone; floor of retrieval quality — **ladder** |
| 11 | `blocked` | `fuse_bm25_dense_formula_fused_block_20260709_rrf` | the proposed system — **ladder** |
| 12 | `flat_5way` | `fuse_full_5_way_20260709_rrf` | flat fusion |
| 13 | `blocked_drop_bm25` | `fuse_dense_formula_fused_block_drop_bm25_20260709_rrf` | leave-one-out |
| 14 | `flat_drop_bm25` | `fuse_full_drop_bm25_20260709_rrf` | leave-one-out |
| 15 | `flat_drop_slt_type` | `fuse_full_drop_slt_type_20260709_rrf` | leave-one-out |

Plus, on every generator: `no_rag` (floor, `k=0`) and `oracle_retrieval_ceiling`
(`--oracle-mode disjoint`). Keep `oracle_reference_ceiling` (`--oracle-mode reference`) only if you
still want the instrument ceiling; it is not needed by any hypothesis.

---

## 3. Fixed settings

Create **`configs/task_c/generation_eval_v3_1_5b.yaml`** by copying `generation_eval.yaml` and
changing only what is listed. Then `_3b` and `_7b` variants differing **only** in `generator.model`
and `revision`.

```yaml
generator:
  model: "Qwen/Qwen2.5-1.5B-Instruct"
  revision: "<pinned commit>"        # §1.3
  dtype: "bfloat16"
  quantization: null
  backend: "hf"                      # vLLM still broken: vllm 0.25.0 wants transformers v5,
                                     # 4.57.6 is installed and needed by sentence-transformers
  decoding:
    temperature: 0.0
    top_p: 1.0
    max_new_tokens: 1024             # keep — 512 truncated 57/100 bm25 topics mid-derivation
    seed: 42

prompt_template_version: "v3"        # one template for floor AND retrieval — the whole point

judge:
  model: "gemini-3.5-flash-lite"
  api_key_env: "GEMINI_API_KEY"
  temperature: 0.0
  max_concurrency: 5
  cache_dir: ".cache/track_c_judge"
  metrics:
    - "faithfulness"
    - "factual_correctness_precision"
    - "factual_correctness_recall"

wandb:
  project: "one-last-run"
  group: "Track C : Generation v3"
  job_type: "track-c-generation-v3"
  tags: ["C", "generation", "arqmath-3", "v3", "qwen1.5b"]
```

**Held constant across every run in every phase:** 87 topics (`qrel_task1_2022_all`, sha256
`2b5565f9…`), temperature 0, seed 42, `max_new_tokens` 1024, `max_reference_answers` 5, judge
`gemini-3.5-flash-lite` @ T=0, prompt v3. Only the generator, the context source and *k* vary — and
each varies in exactly one phase.

> **Do not introduce batched generation partway through.** `HFGenerator` is sequential. If you batch
> for speed, padding and numerics can shift greedy outputs, so a batched run is not comparable with a
> sequential one. Either keep sequential throughout, or batch **every** run in **every** phase at the
> same batch size and re-run anything already done. Sequential is the safe default: at 1.5B on the L4
> a run is a few minutes.

---

## 4. Phase order

### Phase 0 — v3 template + smoke gate (blocking)

1. Implement `PromptTemplateV3` (§1.1, verbatim); register it in `_REGISTRY`; fix the `--no-rag`
   template default. Unit-test: `render(q, [])` omits the context block, `render(q, ctxs)` includes it
   with `[1] [2] …` numbering, the two renders carry the **same** `sha256`, that hash equals
   `45bdd2f2…`, and `refusal_phrase is None`.
2. Run **one** config end to end: `dense_mpnet`, k=5, 1.5B.
3. Run the smoke test of §5. **Do not launch anything else until it returns PASS.**
4. Also run the floor (`--no-rag`) once and assert its `prompt_sha256` is **byte-identical** to the
   `dense_mpnet` run's. This single assertion is what closes O-F — if the hashes differ, v3 is not
   being used for both and the whole re-run is void.

### Phase 1 — the 15-config sweep, 1.5B, k=5

All 15 + floor + retrieval ceiling = 17 runs. Smoke-test each; a WARN is inspected before the next
launch, a FAIL halts the phase.

### Phase 2 — capacity ladder, 3B then 7B, k=5

The **ladder set** — 5 retrieval conditions plus floor and ceiling, chosen to span the retrieval
range and to map onto hypotheses:

| Condition | Why it is in the ladder |
|---|---|
| `no_rag` (k=0) | same-prompt floor — the comparison H2.1 needs |
| `formula_block` | bottom of the retrieval range |
| `bm25` | weak text channel |
| `dense_mpnet` | strongest single channel; the bar H1.3 was read against |
| `fusion_bm25_dense` | text-only fusion |
| `blocked` | the proposed system — top of the range |
| `oracle_retrieval_ceiling` | what perfect retrieval is worth at this capacity |

7 runs per generator, 14 total. This feeds **§6.6, the robustness check** — no hypothesis rules on it —
and re-tests the generator-capacity reversal (dense and blocked swapping places between 3B and 7B),
which is the one observation from the old §5.7 that survives O-F untouched.

> 7B bf16 is ~15 GB on a 24 GB L4 — it fits, and it already ran. Expect roughly 4–5× the wall-clock
> of 1.5B per run.

### Phase 3 — depth sweep, 1.5B, k ∈ {1, 5, 10, 20}

Two systems only: `dense_mpnet` and `blocked`. They differ by exactly one component, which is what
makes the sweep a clean isolation rather than a parameter search. 8 cells, 2 already done at k=5 in
Phase 1 → **6 new runs**. Tests **H2.3**.

Frame it in the write-up as a sensitivity curve, not a search for the best *k*, or it reads as
selection on the test set.

### Phase 4 — analysis

1. `ndcg'@10` (judge channel, from `results/track_b/*_metrics.csv`) joined over all 15 configurations
   to **faithfulness** and to **FC-precision** → one scatter, two *y* axes, Spearman ρ on each and no
   *p*-value. **This is the H2.2 + H2.3 exhibit**, and the pair is the mechanism: faithfulness rising
   while FC-precision does not means the context *was* used and still didn't help; both flat means it
   was ignored.
2. Depth curve, FC-precision with faithfulness beside it as the mediator → **H2.4**.
3. Floor against the blocked multi-index, same prompt, all 87 topics → **H2.1**.
4. Ladder table across 1.5B / 3B / 7B → **§6.6 robustness, no verdict**.

---

## 5. The smoke test — `experiments/track_c/smoke_test_generation.py`

Follow the convention of `experiments/track_v/smoke_test_judge.py`. Takes a run directory (or W&B run
name), emits `results/track_c/<run_name>_smoke.md` plus JSON, prints a one-line verdict, and logs the
verdict to the W&B run summary. Exit code 0 = PASS, 1 = WARN, 2 = FAIL.

### A · Provenance and the O-F guard

| # | Check | FAIL if |
|---|---|---|
| A1 | `prompt_template_version == "v3"` and `prompt_sha256 == 45bdd2f257d0808a9392b3a1daba96fc37f8c72f19df2c346e56d8088e79b769` | mismatch |
| A2 | `prompt_sha256` is identical across **all** rows of the run | any variation |
| A3 | `prompt_sha256` equals the floor run's hash (once the floor exists) | mismatch — **O-F is back** |
| A4 | `temperature == 0.0`, `seed == 42`, `max_new_tokens == 1024` | any mismatch |
| A5 | `qrels_sha256 == 2b5565f9…`; `max_reference_answers == 5` | mismatch |
| A6 | `generator_model` and pinned `revision` recorded; `git_commit` recorded | missing |
| A7 | `wandb.config` equals the local manifest (the repo's existing reproducibility spine) | divergence |

### B · Coverage and integrity

| # | Check | Threshold |
|---|---|---|
| B1 | Row count == expected topics (87, or `--max-topics`) | FAIL on mismatch |
| B2 | `topic_id` set == qrels topic set; no duplicates | FAIL |
| B3 | Every row has non-empty answer text | FAIL on any empty |
| B4 | Context count per topic == k for RAG runs; == 0 for `no_rag` | WARN if any RAG topic has < k **and** the run file has ≥ k available; FAIL if a RAG topic has 0 |
| B5 | Contexts are non-empty strings and de-duplicated within a topic | WARN |

### C · Truncation — `finish_reason` and token ceiling

This is where the old ladder was silently wrong; treat it as first-class.

| # | Check | Threshold |
|---|---|---|
| C1 | `finish_reason` distribution over `{stop, length, …}` reported explicitly | — |
| C2 | Share with `finish_reason == "length"` (hit the ceiling) | WARN > 5%, **FAIL > 15%** |
| C3 | Completion-token histogram: p50, p95, max | WARN if p95 > 0.95 × `max_new_tokens` |
| C4 | Metrics reported **separately** for truncated vs complete rows | WARN if truncated rows' mean faithfulness is > 0.05 below complete rows' — truncation is depressing the metric for reasons unrelated to retrieval |
| C5 | No answer ends mid-LaTeX (unbalanced `$`, `\(`, `\[`, or an open `\begin{}`) | WARN with count and examples |

If C2 fails, the fix is a higher ceiling — not a re-scored metric. Raise it, re-run **every** condition
at the new ceiling, and say so.

### D · Metric sanity

| # | Check | Threshold |
|---|---|---|
| D1 | All three metrics within [0, 1] | FAIL on any out-of-range |
| D2 | NaN / null only where legitimately skipped (no reference available) | FAIL on unexplained nulls |
| D3 | Reference-skipped topics are **null, not 0.0** | FAIL — this is a known failure mode and it silently deflates every mean |
| D4 | `faithfulness` is null for `no_rag` (undefined without a context) | FAIL if it is 0.0 |
| D5 | Share of rows at exactly 0.0 or exactly 1.0, per metric | WARN > 20% — suggests claim extraction is failing, not that the data is extreme |
| D6 | If `answer_relevance` is enabled: count of exact 0.000 | WARN on any — this is the RAGAS refusal flag that faked a regression in C0.3 |
| D7 | Refusal detection (kept as a diagnostic under v3, which has no refusal instruction) | WARN > 5% — the model is refusing spontaneously and coverage must then be reported |
| D8 | Judge cache hit rate; count of judge API errors and retries | FAIL on any unretried error |
| D9 | Per-metric mean, sd, and n reported | — |

### E · Plausibility against the existing 3B v2 runs

Not a pass/fail on the numbers themselves — a check that nothing is absurd.

| # | Check | Threshold |
|---|---|---|
| E1 | FC-precision within a plausible band for a 1.5B model | WARN if > 0.8 or < 0.05 |
| E2 | Delta against the stored 3B v2 run for the same config, printed for eyeballing | — |
| E3 | Coverage ≈ 1.0 under v3 (no refusal instruction) | WARN if < 0.95 |
| E4 | Faithfulness under v3 **lower** than under v2 for the same config | expected, not a bug — v3 permits ungrounded-but-correct content. Note it; do not "fix" it |

> **Do not read E4 as a regression.** Under v3, faithfulness measures how much of the answer happens
> to be grounded in the supplied context, and the model is now free to answer from its own knowledge.
> FC-precision is the primary RQ2 outcome; faithfulness becomes descriptive. Say this once in §6.1.

---

## 6. Logging — match the existing shape

Per run, exactly as the current pipeline does:

- **Run name:** `C-<config_name>-k<k>-<UTC timestamp>` (existing convention), with the generator size
  in the tags, e.g. `qwen1.5b`.
- **Group:** `"Track C : Generation v3"`. Leave `"Track C : Generation (corrected)"` untouched — it
  is the historical record of the superseded ladder and Chapter 6 will cite it as such.
- **`job_type`:** `"track-c-generation-v3"`.
- **Logged config:** every key in §3 plus `config_sha256`, `prompt_sha256`, `qrels_sha256`,
  `git_commit`, `git_dirty`, library versions — `wandb.config` **must equal** the local manifest.
- **Artifacts:** `generations.jsonl` (prompt, contexts, answer, `finish_reason`, token counts) and the
  `per_topic` table, as now.
- **Local canonical outputs:** `results/track_c/<run_name>_metrics.csv`,
  `results/per_topic/<run_name>_answers_audit.csv`, `<run_name>_smoke.{md,json}`. W&B is a mirror,
  never a source of truth; `wandb.mode: disabled` must run the whole pipeline offline.

---

## 7. VM and cost

`multi-index-rag-gpu`, `g2-standard-4`, NVIDIA L4 24 GB, `europe-west3-b`.

- 1.5B bf16 ≈ 3 GB, 3B ≈ 6 GB, 7B ≈ 15 GB — all fit; load one generator per process and free it.
- Sequential `hf` generation, 87 topics × up to 1024 new tokens: a few minutes per run at 1.5B.
  Phase 1 (17 runs) is a couple of hours; Phase 2's 7B half dominates the total.
- Judge calls: ~37 runs × 87 topics × 3 metrics. The cache key must cover
  `(answer, context, reference, metric, judge_model, judge_temperature)` so re-runs and the smoke
  tests are nearly free. Keep `max_concurrency: 5`.
- **Stop the VM between phases** — `gcloud compute instances stop multi-index-rag-gpu` keeps the
  persistent disk; resume with `start`. Phase boundaries are natural pause points.
- Run each phase under `tmux`/`nohup` with a per-run log in `logs/track_c_v3/`, so a dropped SSH
  session does not kill a sweep.

---

## 8. Definition of done

- [ ] `PromptTemplateV3` frozen, unit-tested, SHA256 recorded
- [ ] Model revisions pinned for all three generators
- [ ] Floor and a retrieval run share one `prompt_sha256` (**O-F closed**)
- [ ] Phase 1: 17 runs at 1.5B, every one smoke-PASS
- [ ] Phase 2: 14 runs across 3B and 7B, every one smoke-PASS
- [ ] Phase 3: 6 new depth runs at 1.5B
- [ ] H2.1–H2.4 re-decided against the decision rules in manifest §1.3, and the verdicts written back
      into the manifest. **Faithfulness is reported as an attribution measure, not a quality score** —
      a low value on a correct answer means the model already knew it
- [ ] §6.6 written as robustness with no verdict, and saying so in its opening sentence
- [ ] Manifest §15 O-1 closed; C5, C-7, C-8 re-ruled; O-E and O-F resolved
- [ ] Reference-construction discrepancy (§1.2) fixed in the prose

---

## 9. Open questions for Göbel

1. ~~Is the capacity ladder a hypothesis?~~ **Answered: no.** Robustness check, no verdict (§6.6).
2. **The 15-config sweep** — confirm she accepts it as *one* correlation experiment with two exhibits,
   given her "less is more" instruction. The framing in §2 is the argument to put to her.
3. **1.5B as the primary generator** — the thesis title and framing currently assume 3B. If 1.5B
   becomes the workhorse, §6.1 has to say why, and the capacity ladder is that reason.
