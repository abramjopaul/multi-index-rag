"""RAGAS evaluation wrapper for Track C.

Wraps RAGAS 0.2.x to:
  - Use Gemini free API as the LLM judge (rate-limited)
  - Use sentence-transformers for embedding-based metrics (SemanticSimilarity)
  - Run non-LLM metrics (ROUGE-L, BLEU) without any API calls
  - Skip metrics whose required inputs are absent (empty contexts, missing reference)
  - Return per-sample metric dicts with NaN for skipped metrics

Expected RAGAS version: >= 0.2.0
  pip install ragas>=0.2.0 langchain-google-genai>=1.0 langchain-huggingface>=0.1
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import types
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _patch_ragas_vertexai_shim() -> None:
    """Work around ragas.llms.base unconditionally importing a removed module.

    ragas/llms/base.py does `from langchain_community.chat_models.vertexai import
    ChatVertexAI` at import time, but langchain-community dropped that submodule
    (Vertex AI support moved to the separate langchain-google-vertexai package).
    This breaks `import ragas` entirely, even though this project only ever uses
    the 'gemini' backend. See https://github.com/vibrantlabsai/ragas/issues/2745.
    """
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
        return  # real module is present (e.g. langchain-community adds it back)
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(module_name)

    class ChatVertexAI:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ChatVertexAI is unavailable: this is a stub installed by "
                "ragas_eval.py because langchain-community no longer ships "
                "langchain_community.chat_models.vertexai. This project only "
                "uses the 'gemini' backend."
            )

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = stub


_patch_ragas_vertexai_shim()

# Metrics that require an LLM judge call
_LLM_METRICS = frozenset({"answer_relevance", "answer_correctness", "faithfulness",
                           "context_precision", "context_recall"})
# Metrics that require a reference (ground_truth)
_NEEDS_REFERENCE = frozenset({"answer_correctness", "semantic_similarity",
                               "rouge_l", "bleu", "context_precision", "context_recall"})
# Metrics that require non-empty contexts
_NEEDS_CONTEXT = frozenset({"faithfulness", "context_precision", "context_recall"})


@dataclass
class RagasSample:
    topic_id: str
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str | None = None
    ground_truth_answer_id: str | None = None
    ground_truth_score: int | None = None


class RagasEvaluator:
    """Runs RAGAS metrics using Gemini as LLM judge and sentence-transformers for embeddings.

    Args:
        judge_config: RagasJudgeConfig (backend, model, api_key_env, rpm_limit, etc.)
    """

    def __init__(self, judge_config):
        self._config = judge_config
        self._metrics = list(judge_config.metrics)
        self._rpm_limit = judge_config.rpm_limit
        self._llm = None
        self._embeddings = None
        self._initialized = False
        # Populated by _run_ragas_subset() whenever "faithfulness" runs; read via
        # get_faithfulness_claim_counts() immediately after evaluate() returns
        # (each evaluate() call resets and overwrites these).
        self.last_faithfulness_traces = None
        self.last_faithfulness_topic_ids = None

    def _lazy_init(self) -> None:
        """Initialize LLM and embedding wrappers on first use."""
        if self._initialized:
            return

        needs_llm = any(m in _LLM_METRICS for m in self._metrics)
        if needs_llm:
            if self._config.backend == "gemini":
                self._llm = self._build_gemini_llm()
            else:
                raise NotImplementedError(
                    f"RAGAS judge backend {self._config.backend!r} not yet implemented. "
                    "Use 'gemini'."
                )

        self._embeddings = self._build_embeddings()
        self._initialized = True

    def _build_gemini_llm(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper

        api_key = os.environ.get(self._config.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Gemini API key not found. Set the {self._config.api_key_env!r} "
                "environment variable."
            )
        # Rate limiting: InMemoryRateLimiter if available, else we handle manually
        try:
            from langchain_core.rate_limiters import InMemoryRateLimiter
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self._rpm_limit / 60.0,
                check_every_n_seconds=1,
            )
            chat_model = ChatGoogleGenerativeAI(
                model=self._config.model,
                google_api_key=api_key,
                rate_limiter=rate_limiter,
                seed=self._config.seed,
            )
        except (ImportError, TypeError):
            # Older langchain: no rate_limiter kwarg; fall back to manual sleep
            chat_model = ChatGoogleGenerativeAI(
                model=self._config.model,
                google_api_key=api_key,
                seed=self._config.seed,
            )
            self._manual_rate_limit = True
        else:
            self._manual_rate_limit = False

        logger.info(
            f"Gemini judge ready: model={self._config.model}, "
            f"rpm_limit={self._rpm_limit}, seed={self._config.seed}"
        )
        # bypass_n=True: Gemini rejects candidateCount>1 ("Multiple candidates is
        # not enabled for this model") for flash-lite models, so metrics like
        # ResponseRelevancy (strictness=3) must issue 3 separate n=1 calls instead
        # of one call requesting 3 candidates.
        wrapper = LangchainLLMWrapper(chat_model, bypass_n=True)
        # Force temperature=0 on every real judge call. Without this the judge
        # was never actually pinned despite the generator being frozen: ragas's
        # own get_temperature(n) returns 0.01 for n=1 calls (answer_correctness,
        # faithfulness, context_precision, context_recall) and 0.3 for
        # answer_relevance's n=3 strictness calls, and it MUTATES
        # chat_model.temperature in place before every real call — so passing
        # temperature=0 to ChatGoogleGenerativeAI's constructor would be a
        # silent no-op. This override is unconditional (not config-gated)
        # because there's no legitimate reason for a reproducibility-focused
        # judge to run above temperature 0. Note: Gemini documents seed as
        # best-effort reproducibility, not a hard guarantee — some residual
        # provider-side nondeterminism is expected even with both of these set.
        wrapper.get_temperature = lambda n: 0.0
        return wrapper

    def _build_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        logger.info(f"Loading embedding model: {self._config.embedding_model}")
        hf_embeddings = HuggingFaceEmbeddings(model_name=self._config.embedding_model)
        return LangchainEmbeddingsWrapper(hf_embeddings)

    def _build_metric_objects(self, metric_names: list[str]) -> list:
        """Instantiate RAGAS metric objects for the given names."""
        import ragas.metrics as rm

        # Mapping from config name to RAGAS class name (handles 0.2.x renames)
        _name_map = {
            "answer_relevance": ["ResponseRelevancy", "AnswerRelevancy"],
            "answer_correctness": ["AnswerCorrectness"],
            "semantic_similarity": ["SemanticSimilarity", "AnswerSimilarity"],
            "rouge_l": ["RougeScore"],
            "bleu": ["BleuScore"],
            "faithfulness": ["Faithfulness"],
            "context_precision": ["ContextPrecision", "LLMContextPrecisionWithReference"],
            "context_recall": ["ContextRecall"],
        }

        objs = []
        for name in metric_names:
            candidates = _name_map.get(name, [name])
            cls = None
            for cname in candidates:
                cls = getattr(rm, cname, None)
                if cls is not None:
                    break
            if cls is None:
                logger.warning(
                    f"RAGAS metric {name!r} not found in ragas.metrics "
                    f"(tried: {candidates}). Skipping."
                )
                continue

            # Instantiate with appropriate kwargs
            try:
                if name in _LLM_METRICS and name not in ("semantic_similarity",):
                    if name in ("answer_relevance", "answer_correctness"):
                        obj = cls(llm=self._llm, embeddings=self._embeddings)
                    else:
                        obj = cls(llm=self._llm)
                elif name == "semantic_similarity":
                    obj = cls(embeddings=self._embeddings)
                else:
                    obj = cls()
            except TypeError:
                # Fallback: try with no args (some versions use global config)
                obj = cls()

            objs.append((name, obj))
        return objs

    def evaluate(self, samples: list[RagasSample]) -> list[dict]:
        """Evaluate all samples and return per-sample metric dicts.

        Metrics are skipped (set to NaN) when required inputs are absent:
          - reference-requiring metrics: skipped when ground_truth is None
          - context-requiring metrics: skipped when contexts is empty
        """
        self._lazy_init()
        self.last_faithfulness_traces = None
        self.last_faithfulness_topic_ids = None

        # Partition metrics by what they need
        active = [m for m in self._metrics]
        ref_free = [m for m in active if m not in _NEEDS_REFERENCE and m not in _NEEDS_CONTEXT]
        ref_req = [m for m in active if m in _NEEDS_REFERENCE and m not in _NEEDS_CONTEXT]
        # context-requiring metrics split further: faithfulness needs context
        # only, while context_precision/context_recall need context AND
        # reference. LLMContextPrecisionWithReference does not NaN or error on
        # an empty-string reference — it silently produces a real, near-zero
        # score (denominator guarded by +1e-10) — so samples without
        # ground_truth must be excluded from that group, not just NaN-filled
        # downstream.
        ctx_only = [m for m in active if m in _NEEDS_CONTEXT and m not in _NEEDS_REFERENCE]
        ctx_and_ref = [m for m in active if m in _NEEDS_CONTEXT and m in _NEEDS_REFERENCE]

        # Base result: all NaN
        results: list[dict] = [
            {m: float("nan") for m in active} for _ in samples
        ]

        # Run reference-free metrics (answer_relevance) on all samples
        if ref_free:
            logger.info(f"Running reference-free metrics: {ref_free}")
            rf_results = self._run_ragas_subset(samples, ref_free)
            for i, r in enumerate(rf_results):
                results[i].update(r)

        # Run reference-requiring metrics on samples that have ground_truth
        if ref_req:
            has_ref_idx = [i for i, s in enumerate(samples) if s.ground_truth is not None]
            if has_ref_idx:
                ref_samples = [samples[i] for i in has_ref_idx]
                logger.info(
                    f"Running reference metrics: {ref_req} on "
                    f"{len(ref_samples)}/{len(samples)} samples with ground truth"
                )
                rr_results = self._run_ragas_subset(ref_samples, ref_req)
                for local_i, global_i in enumerate(has_ref_idx):
                    results[global_i].update(rr_results[local_i])
            else:
                logger.warning("No samples have ground_truth; skipping reference metrics.")

        # Run context-only metrics (faithfulness) on samples with non-empty contexts
        if ctx_only:
            has_ctx_idx = [i for i, s in enumerate(samples) if s.contexts]
            if has_ctx_idx:
                ctx_samples = [samples[i] for i in has_ctx_idx]
                logger.info(
                    f"Running context-only metrics: {ctx_only} on "
                    f"{len(ctx_samples)}/{len(samples)} samples with context"
                )
                co_results = self._run_ragas_subset(ctx_samples, ctx_only)
                for local_i, global_i in enumerate(has_ctx_idx):
                    results[global_i].update(co_results[local_i])
            else:
                logger.info("No samples have context; skipping context-only metrics.")

        # Run context+reference metrics (context_precision, context_recall) only
        # on samples that have BOTH — see comment above on why empty-reference
        # samples must be excluded rather than left for downstream NaN-filtering.
        if ctx_and_ref:
            has_both_idx = [
                i for i, s in enumerate(samples) if s.contexts and s.ground_truth is not None
            ]
            n_dropped = sum(1 for s in samples if s.contexts and s.ground_truth is None)
            if n_dropped:
                logger.warning(
                    f"Context+reference metrics {ctx_and_ref}: {n_dropped} sample(s) have "
                    "context but no ground_truth — skipped (NaN), not run with reference=''."
                )
            if has_both_idx:
                both_samples = [samples[i] for i in has_both_idx]
                logger.info(
                    f"Running context+reference metrics: {ctx_and_ref} on "
                    f"{len(both_samples)}/{len(samples)} samples with context and ground truth"
                )
                cr_results = self._run_ragas_subset(both_samples, ctx_and_ref)
                for local_i, global_i in enumerate(has_both_idx):
                    results[global_i].update(cr_results[local_i])
            else:
                logger.info(
                    "No samples have both context and ground_truth; "
                    "skipping context+reference metrics."
                )

        return results

    def _run_ragas_subset(
        self, samples: list[RagasSample], metric_names: list[str]
    ) -> list[dict]:
        """Run a subset of metrics on a subset of samples via ragas.evaluate()."""
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.run_config import RunConfig

        metric_objs_with_names = self._build_metric_objects(metric_names)
        if not metric_objs_with_names:
            return [{} for _ in samples]

        metric_objs = [obj for _, obj in metric_objs_with_names]
        built_names = [name for name, _ in metric_objs_with_names]

        data = {
            "user_input": [s.question for s in samples],
            "response": [s.answer for s in samples],
            "retrieved_contexts": [s.contexts for s in samples],
            "reference": [s.ground_truth or "" for s in samples],
        }
        dataset = Dataset.from_dict(data)

        if getattr(self, "_manual_rate_limit", False):
            # Crude manual rate limiting: sleep between calls if needed
            logger.info(
                f"Manual rate limiting at {self._rpm_limit} RPM "
                "(InMemoryRateLimiter not available)"
            )

        result = ragas_evaluate(
            dataset=dataset,
            metrics=metric_objs,
            llm=self._llm,
            embeddings=self._embeddings,
            # 300s was too short for the context_precision+context_recall batch:
            # context_precision alone can issue up to 5 Gemini calls/topic (one per
            # retrieved chunk), so 20 topics x up to 6 calls/topic at rpm_limit=10
            # (~1 call/6s) is up to ~12 min worst case. Confirmed via wandb run
            # 2fn5h7dp/output.log: dozens of "Job[N]: TimeoutError()" clustered
            # right at the old 300s mark, only in this batch (context_recall and
            # faithfulness, with far fewer calls/topic, never timed out). 900s
            # gives headroom above the ~720s worst case.
            run_config=RunConfig(timeout=900),
            raise_exceptions=False,
        )

        if "faithfulness" in metric_names:
            # result.traces is a documented public field (EvaluationResult),
            # in dataset row order — verified: per-row chain groups are
            # created synchronously before any async metric execution starts.
            self.last_faithfulness_traces = result.traces
            self.last_faithfulness_topic_ids = [s.topic_id for s in samples]

        # Convert RAGAS result to list of per-sample dicts.
        # ragas.evaluate() keys metrics with a `mode` attribute (e.g. RougeScore's
        # fmeasure/precision/recall) as "{name}(mode={mode})" instead of bare
        # `name` — replicate that here or the lookup silently misses and returns NaN.
        result_df = result.to_pandas()
        ragas_names = [
            f"{obj.name}(mode={obj.mode})" if hasattr(obj, "mode") else obj.name
            for obj in metric_objs
        ]
        per_sample = []
        for _, row in result_df.iterrows():
            d = {}
            for name, ragas_name in zip(built_names, ragas_names):
                val = row.get(ragas_name, float("nan"))
                d[name] = float(val) if val is not None and not (
                    isinstance(val, float) and math.isnan(val)
                ) else float("nan")
            per_sample.append(d)

        return per_sample

    def get_faithfulness_claim_counts(self) -> dict[str, int | None] | None:
        """Return {topic_id: claim_count} from the most recent evaluate() call,
        or None if faithfulness wasn't run that call (not requested, or no
        sample had context). Must be read immediately after evaluate() —
        each evaluate() call resets and overwrites these traces.

        Claim count = how many statements Faithfulness's statement-generation
        step decomposed the answer into for that topic. Diagnostic for
        distinguishing claim-decomposition drift (denominator changes between
        repeats) from entailment-verdict drift (same claims, different
        supported/not-supported calls).
        """
        if self.last_faithfulness_traces is None:
            return None
        counts: dict[str, int | None] = {}
        for topic_id, trace in zip(self.last_faithfulness_topic_ids, self.last_faithfulness_traces):
            try:
                statements = trace["faithfulness"]["statement_generator_prompt"]["output"].statements
                counts[topic_id] = len(statements)
            except (KeyError, TypeError, AttributeError):
                counts[topic_id] = None
        return counts

    def aggregate(self, per_sample: list[dict]) -> dict[str, float]:
        """Compute mean and std per metric (NaN samples excluded)."""
        if not per_sample:
            return {}

        all_metrics = list(per_sample[0].keys())
        agg: dict[str, float] = {}

        for metric in all_metrics:
            vals = [r[metric] for r in per_sample if not math.isnan(r.get(metric, float("nan")))]
            if not vals:
                agg[f"{metric}_mean"] = float("nan")
                agg[f"{metric}_std"] = float("nan")
                agg[f"headroom_{metric}"] = float("nan")
                continue
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = variance ** 0.5
            agg[f"{metric}_mean"] = round(mean, 4)
            agg[f"{metric}_std"] = round(std, 4)
            agg[f"headroom_{metric}"] = round(1.0 - mean, 4)

        return agg
