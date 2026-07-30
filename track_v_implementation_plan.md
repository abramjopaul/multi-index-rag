# V-First Implementation Plan — Judge Validation, on Scalable Batch Infra

> **Build now:** the LLM-judge infrastructure and the **V (judge validation)** experiment only.
> **Do not build yet:** Track B re-scoring, Track C ladder, ablations. But the API layer, batch
> mode, caching, and schemas built now **must be sized and shaped so those land on top without
> rework.** This document is scoped to V; the "Forward-compatibility contract" section states
> exactly what the infra must support so nothing has to be rebuilt later.
>
> Companion docs (unchanged, still valid): `track_c_infra_step1_spec.md` (base infra + Step 1),
> `track_bc_roadmap_v2.md` (full roadmap). This plan is the concrete build order for V.

---

## Why V first

The hole problem is accepted. The response is a validated LLM relevance judge that fills
unjudged (topic, answer) pairs so metrics can be recomputed with fewer holes. **Nothing
downstream is trustworthy until the judge is validated**, so V is the gate and the methodology
pillar. It is also the single most expensive labelling job in the whole thesis (see cost model),
so getting its batch/cache infra right is what makes everything after it cheap.

Execution order overall: **build judge infra → V → (Track B re-score ∥ rest of Track C)**. This
doc covers the first two.

---

## Cost model (this is why the infra is shaped the way it is)

The judge is a **labeller, not a metrics engine.** pytrec_eval still computes nDCG′/mAP′/recall′.
The judge only produces 0–3 labels for holes. So API cost = number of distinct (topic, answer)
pairs labelled — nothing else.

- **V (this build):** label the human-judged pairs across **ARQMath-1 + 2 + 3** to compare judge
  vs human. ≈ 226 topics × ≈ 450 judged answers ≈ **~100K pairs.** This is the big spend, and it
  is unavoidable — it *is* the validation.
- **Track B re-score (later):** fill only the **top-~20** holes per config (shallow, because
  hole-sensitive metrics are @5/@10/P@10; deep metrics are insensitive to sparse filling and are
  reported human-only + hole rate). Union across configs dedupes heavily → **~6K pairs**, most of
  which are a **cache hit from V**. Nearly free on top of V.
- **Track C (later):** references + per-topic generation metrics; small relative to V.

Consequences the infra must honour from day one:
- **Batch mode** (Gemini Batch API, ~50% cost, high throughput, no latency need) — a ~100K job is
  impractical on synchronous free-tier limits.
- **Aggressive caching keyed on (topic_id, answer_id, prompt_sha256, judge_model)** so V's ~100K
  labels are reused verbatim by Track B and Track C. Do the big spend once.
- **Resumability** — a ~100K batch will span multiple submissions; never re-label a completed pair.

---

## Repo conventions

- Source: `src/multirag/eval/`
- Configs: `configs/eval/`
- Experiments: `experiments/track_v/`
- Results: `results/track_v/`
- Poetry; type hints; ruff + black clean.
- Judge stack: **`google-genai` (Google AI Studio, API key)** + the OpenAI-compat fallback for the
  instructor safety-settings bug. **Not** Vertex AI, **not** uv (this is why we implement rather
  than adopt `castorini/umbrela` — see roadmap doc's build-vs-adopt note).

---

# Part 1 — Judge infrastructure (build now, sized for everything)

## 1.1 `configs/eval/judge.yaml`

```yaml
judge:
  provider: google                 # google | openai_compat
  model: gemini-2.0-flash          # exact version string, never a floating alias
  temperature: 0.0
  max_tokens: 1024                 # a 0-3 label + short reasoning; keep tight for cost
  top_p: 1.0
  use_openai_compat_endpoint: false
  openai_compat_base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

execution:
  mode: batch                      # batch | sync   (batch is the default and the whole point)
  batch_poll_seconds: 60
  batch_max_in_flight: 4           # concurrent batch jobs
  sync_max_concurrency: 4          # only used when mode: sync (smoke tests)
  max_retries: 5
  retry_backoff_seconds: 2.0
  request_timeout_seconds: 120

cache:
  dir: .cache/judge
  # cache key = sha256(topic_id | answer_id | prompt_sha256 | judge_model)

prompt:
  template_path: src/multirag/eval/prompts/arqmath_relevance_v1.yaml

wandb:
  enabled: true
  entity: abramjopaul-abram        # same account as the Track B runs
  project: one-last-run            # same project as Track B — keep everything in one place
  group: V                         # NEW group for the validation experiment
  job_type: judge-validation       # distinguishes V runs from future B / C runs
  mode: online                     # online | offline | disabled
  tags: ["V", "judge", "arqmath-1-2-3"]
  # run name convention: "V-<judge_model>-<prompt_version>-<UTC-yyyymmdd-HHMM>"
```

## 1.2 `src/multirag/eval/judge_client.py` — the scalable core

`JudgeClient` with **two execution backends behind one interface**, so callers never change when
scale changes:

- `submit_batch(pairs: list[JudgePair]) -> BatchHandle` — writes a Gemini Batch job (JSONL input),
  returns a handle.
- `poll_batch(handle) -> BatchStatus` / `fetch_batch(handle) -> list[RelevanceJudgement]`.
- `judge_sync(pairs)` — synchronous path, **only** for smoke tests and tiny runs.

Requirements:
- **Cache-first**: before submitting, drop any pair already in cache. After fetch, write results
  to cache immediately. This is what makes V's spend reusable by B and C.
- **Batch chunking**: split large inputs into provider-sized batch jobs; respect
  `batch_max_in_flight`; checkpoint submitted/completed chunk IDs to disk so a crash resumes.
- **OpenAI-compat fallback**: when `use_openai_compat_endpoint: true`, route through the Gemini
  OpenAI-compatible base URL (works around instructor#1658 safety-settings error). Embeddings are
  unaffected and not needed by the relevance judge at all.
- **Cost/usage log**: record pairs submitted, cache hits, tokens, and an estimated cost per run
  into the manifest.
- No model strings or endpoints hardcoded outside `judge.yaml`.

> **Forward-compat requirement:** `JudgeClient` is the *only* module that talks to the API. Track B
> and Track C call it through `submit_batch` / `fetch_batch`. If any future step needs the API, it
> goes through this client — do not add a second API path anywhere.

## 1.3 `src/multirag/eval/prompts/arqmath_relevance_v1.yaml`

The relevance-judge prompt. **UMBRELA/Bing DNA structure, ARQMath semantics.**
- Follow the UMBRELA prompt pattern (descriptive → narrative → aspects, few-shot, forced single
  integer 0–3 output; cite Upadhyay et al. 2024).
- The relevance *definition* MUST be ARQMath's, not Bing's web wording: judge the **usefulness of
  the answer for the math question**, as if by an expert (a math professor) helping the asker,
  with the H(3)/M(2)/L(1)/N(0) level descriptions taken from the **ARQMath assessment guidelines**.
- Pass question + answer LaTeX through verbatim in the same human-readable form the RAG prompt
  uses. Do not strip/normalise formulas.
- Store as a versioned template; its **sha256** goes in every judgement and the manifest.

## 1.4 `src/multirag/eval/schema.py` (extend existing)

```python
@dataclass
class JudgePair:
    topic_id: str
    answer_id: str
    question: str          # topic title + body (LaTeX verbatim)
    answer_text: str       # answer post body (LaTeX verbatim)

@dataclass
class RelevanceJudgement:
    topic_id: str
    answer_id: str
    label: int | None      # 0|1|2|3 ; None if parse failed
    parse_ok: bool
    raw_response: str
    prompt_sha256: str
    judge_model: str
    source: str            # 'batch' | 'sync' | 'cache'
    timestamp: str         # UTC ISO-8601

@dataclass
class PairedLabel:
    """The side-by-side row: human label and judge label for the SAME (topic, answer).
    This is the raw material κ and τ are computed from, and the primary human-readable
    audit artifact. One row per judged (topic, answer) pair."""
    topic_id: str
    answer_id: str
    human_label: int           # 0|1|2|3  (from qrels)
    judge_label: int | None    # 0|1|2|3  (None if parse failed)
    human_bin: int             # H+M -> 1, L+N -> 0  (relevance_level=2)
    judge_bin: int | None
    agree_graded: bool | None  # human_label == judge_label
    agree_binary: bool | None  # human_bin  == judge_bin
    delta: int | None          # judge_label - human_label  (signed error; None if parse failed)
    topic_type: str | None     # computation | concept | proof (if available)
    multi_approach: bool        # flagged topics (e.g. A.302)
    question: str              # for eyeballing disagreements
    answer_text: str
    judge_raw_response: str    # so a disagreement can be inspected without a re-run
```

Round-trip through JSONL losslessly.

## 1.5 `src/multirag/eval/manifest.py` (reuse from base infra)

Every result file gets a manifest capturing: `ragas`/`google-genai`/Python versions, judge model
string, judge temperature, prompt sha256, execution mode (batch/sync), git commit + dirty flag,
config sha256, UTC timestamp, and the usage/cost log. (Reproducibility claim depends on the exact
judge + prompt being on record.)

## 1.6 `experiments/track_v/smoke_test_judge.py`

Push **one** hand-labelled pair through `judge_sync`. Assert: label ∈ {0,1,2,3}, `parse_ok=True`,
cache written, manifest written. Then push the **same** pair through `submit_batch`/`fetch_batch`
and assert the batch path returns an identical label and a cache hit. Run before any real spend —
this proves both backends and the cache work.

## 1.7 `src/multirag/eval/wandb_logger.py` — one logging entry point

W&B logging lives in **one** module, mirroring how the API lives in one `JudgeClient`. Track B's
runs already log to `entity=abramjopaul-abram, project=one-last-run` as individual runs; V logs to
the **same project** under a **new `group="V"`** so it sits alongside Track B without polluting it,
and so future Track B re-scoring / Track C runs can drop into their own groups later with no new
plumbing.

`ExperimentLogger` (thin wrapper over `wandb`):

- `start_run(cfg, run_name, extra_config) -> Run` — calls `wandb.init(entity, project, group,
  job_type, tags, name=run_name, config=<full resolved config>)`. **Everything in the manifest also
  goes into `wandb.config`**: judge model, judge temperature, prompt sha256, execution mode
  (batch/sync), `ragas`/`google-genai`/Python versions, git commit + dirty flag, config sha256.
  W&B config and the on-disk manifest must not drift — write the manifest, then pass the same dict
  to `wandb.config`.
- `log_metrics(dict)` — scalar summary metrics (κ, weighted κ, τ, ρ, parse-rate, cache-hit rate,
  cost).
- `log_table(name, df)` — for per-topic and per-segment breakdowns and the raw judgements sample.
- `log_confusion_matrix(y_true, y_pred, labels=[0,1,2,3])` — log **both** a `wandb.Table` (so the
  raw counts are queryable) and `wandb.plot.confusion_matrix` (for the UI). Do the same for the
  binary H+M / L+N version.
- `log_artifact(path, type)` — attach `agreement.jsonl`, the `.md` report, and the manifest as a
  W&B **artifact** so the exact validation output is versioned next to the run.
- `finish()` — always in a `finally`, so a crashed batch job still closes its run.

Rules:
- **No `wandb` calls anywhere outside this module.** Experiment scripts call `ExperimentLogger`,
  never `wandb.*` directly — same discipline as the single API client.
- Respect `wandb.mode` from config (`online`/`offline`/`disabled`); `disabled` must let the whole
  pipeline run offline (CI, no-network) with zero code changes.
- W&B is **logging only** — never a source of truth. Results still land in `results/track_v/`; W&B
  mirrors them. If W&B is down, the experiment must still complete and write local files.

---

# Part 2 — V: validate the judge

## 2.1 `experiments/track_v/build_validation_set.py`

- Load human-judged (topic, answer, label) triples from **ARQMath-1 (2020), ARQMath-2 (2021),
  ARQMath-3 (2022)** Task 1 qrels.
- Resolve every answer_id in the 2010–2018 collection (all three share it); drop/report any that
  don't resolve.
  - Qrel Files - data/raw/qrels/qrel_task1_2020_all
            - data/raw/qrels/qrel_task1_2021_all.tsv
            - data/raw/qrels/qrel_task1_2022_all.tsv
- Emit `JudgePair`s + a gold-label table. Report per-year and combined counts (expect ~226 topics,
  ~100K pairs). Include the H+M vs L+N binarisation (relevance_level=2) alongside graded labels.
- **Test/dry-run flags (required, so the pipeline is runnable at small scale before the ~100K
  spend):**
  - `--qrel-file PATH` (repeatable) — restrict to specific qrel file(s); default = all three years.
    Must accept a **single** file so one year, or one custom qrel, can be run alone.
  - `--max-topics N` — cap to the first N topics (deterministic order; log which were kept).
  - `--max-pairs-per-topic M` — cap judged answers per topic.
  - `--topic-ids A.302,A.301,...` — an explicit allowlist for targeted debugging.
  These compose (e.g. one qrel file + 5 topics + 10 pairs each ≈ 50 pairs = a full end-to-end dry
  run for a few cents). The full run is simply the flags left at their defaults.

## 2.2 `experiments/track_v/run_judge_over_validation.py`

- Feed the full validation set through `JudgeClient` in **batch mode**.
- Persist `RelevanceJudgement` JSONL + manifest. Report parse-success rate and cache-hit rate.
- Must be **resumable**: re-running after a partial completion submits only the unlabelled
  remainder (cache + checkpoint).
- **W&B**: open the V run here via `ExperimentLogger.start_run` (group `V`, job_type
  `judge-validation`). As batch chunks complete, `log_metrics` running totals — pairs labelled,
  cache-hit rate, parse-success rate, tokens, and estimated cost — so a long (~100K) job is
  observable in the dashboard while it runs, not only at the end. Keep the same run open through
  the agreement step so labelling and results live on one run.

## 2.3 `experiments/track_v/build_paired_labels.py` — human vs judge, side by side

Join the human qrels (from 2.1) with the judge labels (from 2.2) on `(topic_id, answer_id)` into
one `PairedLabel` row per pair. **This file is the raw material for the agreement stats and the
primary human-readable audit artifact** — every κ/τ number is derived from it, and every
disagreement can be inspected here without re-running the judge.

- Write `results/track_v/paired_labels.jsonl` (all fields) **and** `paired_labels.csv` (a compact
  view: topic_id, answer_id, human_label, judge_label, human_bin, judge_bin, delta, agree_graded,
  agree_binary) for eyeballing in a spreadsheet.
- Every human-judged pair must appear exactly once; assert no pair is dropped or duplicated in the
  join. Rows where `parse_ok=False` keep `judge_label=None` and are counted separately (never
  silently treated as a disagreement or an agreement).
- Emit a one-line summary: n pairs, graded-agreement %, binary-agreement %, parse-fail count.
- **W&B**: `log_table("paired_labels_sample", ...)` with a sample (all disagreements + a random
  slice of agreements), and attach the full `paired_labels.jsonl`/`.csv` as an artifact.

## 2.4 `experiments/track_v/agreement_analysis.py` — the two numbers

- **Label agreement vs human qrels** (computed from `paired_labels.jsonl`): Cohen's κ **and**
  linear-weighted κ (scale is ordinal); 4×4 confusion matrix. Compute on graded (0–3) **and** on
  the H+M/L+N binary split (the binary split is what feeds nDCG′). Report parse-fail count
  separately from disagreements.
- **System-ranking preservation**: score the **ARQMath participant runs** (available for all three
  years) under human qrels vs under judge qrels; compute **Kendall's τ** and Spearman's ρ between
  the two system orderings. This is the load-bearing number — ranking preservation is what
  licenses using the judge to compare *our* configs.
- **Segment** agreement by topic type (computation / concept / proof) and flag multi-approach
  topics (e.g. A.302) where principled disagreement is expected — read the `topic_type` /
  `multi_approach` fields straight off the paired rows.

## 2.5 `experiments/track_v/report.py`

- Compare math-domain κ/τ against UMBRELA's published TREC-DL numbers (κ ≈ 0.3–0.5, τ ≈ 0.8–0.9)
  as a reference point; state that math is harder and **τ, not κ, is decisive**.
- Emit verdict: `JUDGE_VALIDATED` / `JUDGE_MARGINAL` / `JUDGE_REJECTED`, with the full κ/τ table,
  confusion matrix, and per-segment breakdown.
- Write `results/track_v/agreement.jsonl` + a human-readable `.md` that is close to
  drop-in for the methodology chapter.
- **W&B**: `log_metrics` the headline scalars (κ, weighted κ, τ, ρ on graded + binary, parse-rate,
  final cost); `log_confusion_matrix` for graded (4×4) and binary; `log_table` for the per-segment
  breakdown (computation / concept / proof + multi-approach flag) and a sample of raw judgements;
  `log_artifact` for `agreement.jsonl`, the `.md` report, and the manifest. Set the verdict as a
  W&B summary field (`wandb.run.summary["verdict"] = ...`) so `JUDGE_VALIDATED / MARGINAL /
  REJECTED` is visible in the runs table at a glance. Then `finish()`.
- **Gate**: if τ shows system ranking is not preserved, hole-filling is not defensible — stop and
  reconsider before building Track B/C.

---

# Forward-compatibility contract (what V's infra MUST support, though we don't build it now)

The infra above is accepted **only if** the following land later with **no rework** to the API
layer, cache, or schemas:

1. **Track B re-scoring reuses V's cache.** Track B will pool the **top-~20** holes per config
   across ~15 configs (B1, B3, B4, B7, B8.1–8.3, B9, B10–B15), dedupe, and label only the
   unjudged remainder (~6K pairs, mostly cache hits from V). → `JudgeClient` cache keyed on
   (topic, answer, prompt_sha256, judge_model) must make this a lookup, not a re-spend. Shallow
   fill only; deep metrics (@100/@1000) stay human-only + hole rate.
2. **Batch mode scales to the ~100K V job and back down to the ~6K B job** with the same code path
   — chunking + in-flight cap + checkpointing already in `JudgeClient`.
3. **Hole rate is a free, no-API computation** (count unjudged docs at a cutoff). A small util
   `hole_rate(ranking, qrels, k)` should exist so B can report hole rate @10/@100/@1000 without
   touching the API. Add this util now (it's trivial and V's report can sanity-use it).
4. **The same `JudgeClient` serves RAGAS-adjacent needs in Track C** — Track C's RAGAS metrics use
   the RAGAS judge, but any bespoke 0–3 labelling (e.g. reference-quality checks) reuses this
   client and cache.
5. **Metrics computation stays in pytrec_eval**, entirely decoupled from labelling. Filling holes =
   merging judge labels into a qrels file, then running the existing evaluation unchanged. Do not
   couple the judge to metric code.
6. **W&B groups extend to B and C for free.** `ExperimentLogger` takes `group`/`job_type` from
   config, so Track B re-scoring logs under `group="B-rescore"` and Track C under `group="C"` in the
   **same `one-last-run` project** — no logging code changes, and V / B / C are filterable side by
   side next to the existing ungrouped Track B runs. (Do not build B/C logging now; just don't
   foreclose it — which the config-driven group already guarantees.)

Do **not** implement 1–5 now. Just don't foreclose them: single API client, durable cache, clean
schemas, batch-first.

---

# Constraints

- Judge temperature **0** everywhere.
- Batch mode is the default; sync is smoke-tests only.
- Every result file gets a manifest (incl. prompt sha256 + execution mode).
- W&B logs to `entity=abramjopaul-abram, project=one-last-run, group=V` (same project as Track B).
  `wandb.config` mirrors the manifest exactly. `wandb` is called only from `ExperimentLogger`, and
  `mode: disabled` must let the pipeline run fully offline. W&B mirrors results; local files remain
  the source of truth.
- The relevance-judge prompt uses **ARQMath** relevance definitions, not generic web-relevance.
- Do not touch retrieval, the generator, or the held-out ARQMath-3 test qrels in V.
- We implement the judge on our own stack; we reuse UMBRELA's **prompt structure + validation
  protocol + published numbers** and cite it — we do not take it as a dependency.

# Acceptance criteria (V)

1. `smoke_test_judge.py` passes on both the sync and batch paths and shows a cache hit on the
   second call; manifest written.
2. `JudgeClient` runs a real Gemini **batch** job end-to-end, is resumable after interruption, and
   never re-labels a cached pair.
3. Validation set builds from all three ARQMath years with per-year + combined counts, graded and
   binary labels.
4. Agreement report emits weighted κ (graded + binary), 4×4 confusion matrix, and Kendall's τ / ρ
   over participant runs, segmented by topic type, with a written `JUDGE_VALIDATED /
   MARGINAL / REJECTED` verdict.
5. Usage/cost (pairs submitted, cache hits, tokens, est. cost) is logged to the manifest.
6. `hole_rate(ranking, qrels, k)` util exists and is unit-tested (no API calls).
7. Re-running V is idempotent — full cache hit, no new spend.
8. A V run appears in W&B under `project=one-last-run, group=V`, with: `wandb.config` matching the
   manifest, running cost/parse/cache metrics logged during the batch job, final κ/weighted-κ/τ/ρ
   scalars, graded + binary confusion matrices, the per-segment table, `agreement.jsonl` + report +
   manifest attached as artifacts, and `verdict` set as a run-summary field.
9. `wandb.mode: disabled` runs the full V pipeline offline with no code changes and all local files
   still written.
10. `paired_labels.jsonl` + `paired_labels.csv` are produced with exactly one row per human-judged
    pair (join asserted lossless), human and judge labels side by side, parse-fails counted
    separately — and κ/τ are computed from this file, not from a separate path.
11. The pipeline runs **end-to-end at small scale** from a **single** qrel file with `--max-topics`
    / `--max-pairs-per-topic` (e.g. ~50 pairs), producing paired labels, agreement numbers, and a
    W&B run — proving the whole chain before the ~100K spend. The full run is the same command with
    the limit flags omitted.
