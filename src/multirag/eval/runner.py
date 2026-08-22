"""Evaluation runner for Track C: scores generations.jsonl with the ragas
metric registry, cache-first and concurrent (asyncio, not a Gemini batch
job -- ragas's Faithfulness/FactualCorrectness are internally multi-step and
don't map onto one upfront batch submission the way Track B's single-shot
relevance judging did; see the Track C plan for the full rationale).

Writes scores_per_sample.jsonl, summary.json, manifest.json, report.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from multirag.eval.generate import GenerationRecord
from multirag.eval.metrics_registry import BoundMetric
from multirag.eval.references import ReferenceResult, select_references

logger = logging.getLogger(__name__)


def _is_genuine_model_output_failure(e: Exception) -> bool:
    """True only if the LLM actually responded but its output failed
    structured-output validation -- never true for a transient transport/
    server error.

    instructor's InstructorRetryException is raised whenever retries are
    exhausted, REGARDLESS of why each attempt failed -- a naive
    isinstance(e, InstructorRetryException) check (the guard judge_client.py
    uses for Track V/B) conflates "the model's output never validated" with
    "the API kept returning 500s/503s and every attempt errored before a
    response even came back." Confirmed live: a Gemini 500 INTERNAL error
    got misclassified as a validation failure and cached, permanently
    poisoning that (topic, metric) pair. InstructorRetryException carries
    `failed_attempts: list[FailedAttempt]`, each with the real underlying
    exception for that attempt -- only treat this as cacheable if EVERY
    attempt's underlying exception was truly a validation error.
    """
    import pydantic
    from instructor.core import InstructorRetryException

    if isinstance(e, pydantic.ValidationError):
        return True
    if isinstance(e, InstructorRetryException):
        failed = getattr(e, "failed_attempts", None) or []
        if not failed:
            return False
        return all(isinstance(a.exception, pydantic.ValidationError) for a in failed)
    return False


# ---------------------------------------------------------------------------
# Cache (separate from Track B/V's .cache/judge/ -- different schema)
# ---------------------------------------------------------------------------


def _cache_key(
    topic_id: str,
    metric_name: str,
    question: str,
    response: str,
    contexts: list[str],
    reference: str | None,
    judge_model: str,
    requires_context: bool,
    requires_reference: bool,
) -> str:
    """Only fold in contexts/reference when the metric actually depends on
    them -- e.g. faithfulness never reads `reference`, so a reference-set
    change (e.g. max_reference_answers) must not invalidate its cache.
    """
    raw = "|".join(
        [
            topic_id,
            metric_name,
            question,
            response,
            "\n".join(contexts) if requires_context else "",
            (reference or "") if requires_reference else "",
            judge_model,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ScoreCache:
    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def put(self, key: str, data: dict) -> None:
        with open(self._path(key), "w") as f:
            json.dump(data, f)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoreResult:
    topic_id: str
    metric_name: str
    value: float | None
    parse_ok: bool
    source: str  # "cache" | "live"


async def _score_one(
    bound: BoundMetric,
    topic_id: str,
    question: str,
    response: str,
    contexts: list[str],
    reference: str | None,
    judge_model: str,
    cache: ScoreCache,
    semaphore: asyncio.Semaphore,
    usage: dict,
) -> ScoreResult:
    key = _cache_key(
        topic_id,
        bound.spec.name,
        question,
        response,
        contexts,
        reference,
        judge_model,
        requires_context=bound.spec.requires_context,
        requires_reference=bound.spec.requires_reference,
    )
    cached = cache.get(key)
    if cached is not None:
        usage["cache_hits"] += 1
        return ScoreResult(
            topic_id=topic_id,
            metric_name=bound.spec.name,
            value=cached["value"],
            parse_ok=cached["parse_ok"],
            source="cache",
        )

    async with semaphore:
        try:
            if bound.spec.name == "faithfulness":
                result = await bound.instance.ascore(
                    user_input=question, response=response, retrieved_contexts=contexts
                )
            else:  # factual_correctness_precision / factual_correctness_recall
                result = await bound.instance.ascore(response=response, reference=reference)
            value = float(result.value)
            parse_ok = True
        except Exception as e:  # noqa: BLE001 -- any judge/parse failure -> parse_ok=False
            logger.warning(f"judge call failed topic={topic_id} metric={bound.spec.name}: {e}")
            value = None
            parse_ok = False
            # Only cache a genuine "the model's output failed validation"
            # outcome. A transient error (network blip, 500/503, rate limit)
            # must never be cached, or it poisons this (topic, metric) pair's
            # score forever -- every future run would silently replay the
            # failure instead of retrying once the service recovers.
            if not _is_genuine_model_output_failure(e):
                return ScoreResult(
                    topic_id=topic_id,
                    metric_name=bound.spec.name,
                    value=None,
                    parse_ok=False,
                    source="live",
                )
        else:
            usage["new_calls"] += 1

    cache.put(key, {"value": value, "parse_ok": parse_ok})
    return ScoreResult(
        topic_id=topic_id, metric_name=bound.spec.name, value=value, parse_ok=parse_ok, source="live"
    )


def _claims_cache_key(topic_id: str, response: str, judge_model: str) -> str:
    # "claims:" prefix keeps this in a disjoint hash space from _cache_key's
    # metric-score entries -- both live in the same cache dir, must never collide.
    raw = "|".join(["claims", topic_id, response, judge_model])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def count_claims(metric_instance, response: str) -> int:
    """Number of atomic claims ragas decomposes a response into -- reuses
    FactualCorrectness's own (semi-private) decomposition step directly, so
    "verbosity" is measured the same way the metric itself sees it. Not
    exposed through the public ascore()/MetricResult interface in this ragas
    version, hence calling the internal method directly.
    """
    claims = await metric_instance._decompose_claims(response)
    return len(claims)


async def _count_claims_one(
    metric_instance,
    topic_id: str,
    response: str,
    judge_model: str,
    cache: ScoreCache,
    semaphore: asyncio.Semaphore,
    usage: dict,
) -> tuple[str, int | None]:
    key = _claims_cache_key(topic_id, response, judge_model)
    cached = cache.get(key)
    if cached is not None:
        usage["cache_hits"] += 1
        return topic_id, cached["n_claims"]

    async with semaphore:
        try:
            n = await count_claims(metric_instance, response)
        except Exception as e:  # noqa: BLE001 -- same non-caching rule as _score_one
            logger.warning(f"claim count failed topic={topic_id}: {e}")
            if not _is_genuine_model_output_failure(e):
                return topic_id, None
            n = None
        else:
            usage["new_calls"] += 1

    cache.put(key, {"n_claims": n})
    return topic_id, n


async def score_all(
    generations: list[GenerationRecord],
    metrics: list[BoundMetric],
    qrels: dict[str, dict[str, int]],
    answer_lookup: dict[str, tuple[str, int]],
    max_reference_answers: int,
    token_counter,
    judge_model: str,
    cache_dir: Path,
    max_concurrency: int,
) -> tuple[list[ScoreResult], dict[str, ReferenceResult], dict, dict[str, int | None]]:
    """Score every (topic, applicable metric) pair. Reference-requiring
    metrics are skipped (not scored, not an error) for topics whose
    ReferenceResult.reference_text is None; context-requiring metrics are
    skipped for topics with empty contexts (e.g. --no-rag runs).

    Also counts claims-per-answer (see count_claims()) for every generation,
    reusing whichever factual_correctness_* BoundMetric is present in
    `metrics` -- independent of context/reference availability, since claim
    decomposition only looks at the response text.
    """
    cache = ScoreCache(cache_dir)
    semaphore = asyncio.Semaphore(max_concurrency)
    usage = {"cache_hits": 0, "new_calls": 0}

    claims_metric = next(
        (b for b in metrics if b.spec.name.startswith("factual_correctness")), None
    )

    ref_results: dict[str, ReferenceResult] = {}
    tasks = []
    claims_tasks = []
    for gen in generations:
        ref = select_references(
            gen.topic_id, qrels, answer_lookup, max_reference_answers, token_counter
        )
        ref_results[gen.topic_id] = ref

        for bound in metrics:
            if bound.spec.requires_context and not gen.contexts:
                continue
            if bound.spec.requires_reference and ref.reference_text is None:
                continue
            tasks.append(
                _score_one(
                    bound,
                    gen.topic_id,
                    gen.question,
                    gen.response,
                    gen.contexts,
                    ref.reference_text,
                    judge_model,
                    cache,
                    semaphore,
                    usage,
                )
            )

        if claims_metric is not None:
            claims_tasks.append(
                _count_claims_one(
                    claims_metric.instance,
                    gen.topic_id,
                    gen.response,
                    judge_model,
                    cache,
                    semaphore,
                    usage,
                )
            )

    logger.info(f"Scoring {len(tasks)} (topic, metric) pairs (max_concurrency={max_concurrency})...")
    results = list(await asyncio.gather(*tasks))
    n_claims = dict(await asyncio.gather(*claims_tasks)) if claims_tasks else {}
    logger.info(f"Scoring complete: {usage['cache_hits']} cache hit, {usage['new_calls']} new")
    return results, ref_results, usage, n_claims


# ---------------------------------------------------------------------------
# Summary / manifest / report
# ---------------------------------------------------------------------------


def build_summary(
    generations: list[GenerationRecord],
    scores: list[ScoreResult],
    ref_results: dict[str, ReferenceResult],
    usage: dict,
    n_claims: dict[str, int | None] | None = None,
) -> dict:
    by_key = {(s.topic_id, s.metric_name): s for s in scores}
    metric_names = sorted({s.metric_name for s in scores})

    n_topics = len(generations)
    n_refusals = sum(1 for g in generations if g.is_refusal)
    refusal_rate = round(n_refusals / n_topics, 4) if n_topics else 0.0
    n_fallback_reference = sum(1 for r in ref_results.values() if r.fallback_used)
    n_excluded_no_reference = sum(1 for r in ref_results.values() if r.reference_text is None)

    summary: dict = {
        "n_topics": n_topics,
        "n_refusals": n_refusals,
        "refusal_rate": refusal_rate,
        "n_fallback_reference_topics": n_fallback_reference,
        "n_excluded_no_reference": n_excluded_no_reference,
        "usage": usage,
    }

    if n_claims:
        claim_vals = [v for v in n_claims.values() if v is not None]
        summary["mean_n_claims"] = (
            round(sum(claim_vals) / len(claim_vals), 2) if claim_vals else None
        )
        summary["n_claims_scored"] = len(claim_vals)

    parse_failures = 0
    for metric_name in metric_names:
        all_vals, answering_vals = [], []
        for g in generations:
            s = by_key.get((g.topic_id, metric_name))
            if s is None:
                continue
            if not s.parse_ok:
                parse_failures += 1
                continue
            all_vals.append(s.value)
            if not g.is_refusal:
                answering_vals.append(s.value)
        summary[f"{metric_name}_mean_all"] = (
            round(sum(all_vals) / len(all_vals), 4) if all_vals else None
        )
        summary[f"{metric_name}_mean_answering"] = (
            round(sum(answering_vals) / len(answering_vals), 4) if answering_vals else None
        )
        summary[f"{metric_name}_n_scored"] = len(all_vals)

    summary["parse_failures"] = parse_failures
    return summary


def _git_commit() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def _package_version(name: str) -> str | None:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def build_manifest(
    config,
    prompt_sha256: str,
    config_sha256: str,
    qrels_path: str,
    qrels_sha256: str,
    k: int,
    max_reference_answers: int,
    trec_file: str | None,
    config_name: str,
    prompt_template_version: str | None = None,
) -> dict:
    commit, dirty = _git_commit()
    return {
        "config_name": config_name,
        "trec_file": trec_file,
        "k": k,
        "max_reference_answers": max_reference_answers,
        "qrels_path": qrels_path,
        "qrels_sha256": qrels_sha256,
        "generator_model": config.generator.model,
        "generator_backend": config.generator.backend,
        "max_new_tokens": config.generator.decoding.max_new_tokens,
        "seed": config.generator.decoding.seed,
        # The actual template used (--prompt-template-version override, or
        # the --no-rag -> "v1" default), NOT config.prompt_template_version
        # -- those differ whenever either kicks in, and reporting the wrong
        # one here would make the manifest lie about what actually ran.
        "prompt_template_version": prompt_template_version or config.prompt_template_version,
        "prompt_sha256": prompt_sha256,
        "judge_model": config.judge.model,
        "judge_temperature": config.judge.temperature,
        "judge_metrics": config.judge.metrics,
        "config_sha256": config_sha256,
        "python_version": sys.version.split()[0],
        "ragas_version": _package_version("ragas"),
        "google_genai_version": _package_version("google-genai"),
        "instructor_version": _package_version("instructor"),
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_report(
    manifest: dict, summary: dict, generations: list[GenerationRecord]
) -> str:
    lines = [
        "# Track C: Generation Evaluation Report",
        "",
        f"Generated: {manifest['timestamp_utc']}",
        "",
        "## Run",
        "",
        f"- config: `{manifest['config_name']}` (trec_file: `{manifest['trec_file']}`)",
        f"- k: {manifest['k']}  |  max_reference_answers: {manifest['max_reference_answers']}",
        f"- generator: {manifest['generator_model']} (backend={manifest['generator_backend']})",
        f"- judge: {manifest['judge_model']} (temperature={manifest['judge_temperature']})",
        f"- qrels: `{manifest['qrels_path']}`",
        "",
        "## Summary",
        "",
        f"- n_topics: {summary['n_topics']}",
        f"- refusal_rate: {summary['refusal_rate']} ({summary['n_refusals']}/{summary['n_topics']})",
        f"- n_fallback_reference_topics: {summary['n_fallback_reference_topics']}",
        f"- n_excluded_no_reference: {summary['n_excluded_no_reference']}",
        f"- parse_failures: {summary['parse_failures']}",
        f"- usage: {summary['usage']}",
    ]
    if "mean_n_claims" in summary:
        lines.append(
            f"- mean_n_claims: {summary['mean_n_claims']} (n={summary['n_claims_scored']})"
        )
    lines += [
        "",
        "## Metrics (answering-only / all-topics)",
        "",
        "| metric | mean (answering) | mean (all) | n scored |",
        "|---|---|---|---|",
    ]
    metric_names = sorted(
        k[: -len("_mean_all")] for k in summary if k.endswith("_mean_all")
    )
    for m in metric_names:
        lines.append(
            f"| {m} | {summary.get(f'{m}_mean_answering')} | {summary.get(f'{m}_mean_all')} "
            f"| {summary.get(f'{m}_n_scored')} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    scores: list[ScoreResult],
    ref_results: dict[str, ReferenceResult],
    manifest: dict,
    summary: dict,
    generations: list[GenerationRecord],
    n_claims: dict[str, int | None] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_claims = n_claims or {}

    scores_path = output_dir / "scores_per_sample.jsonl"
    by_topic: dict[str, dict] = {}
    for s in scores:
        by_topic.setdefault(s.topic_id, {})[s.metric_name] = s.value if s.parse_ok else None
    with open(scores_path, "w") as f:
        for g in generations:
            row = {
                "topic_id": g.topic_id,
                "is_refusal": g.is_refusal,
                "finish_reason": g.finish_reason,
                "completion_tokens": g.completion_tokens,
                "n_contexts": len(g.contexts),
                "response": g.response,
                "n_claims": n_claims.get(g.topic_id),
                **by_topic.get(g.topic_id, {}),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(build_report(manifest, summary, generations))

    logger.info(f"Outputs written to {output_dir}")
    return {
        "scores_per_sample": scores_path,
        "manifest": manifest_path,
        "summary": summary_path,
        "report": report_path,
    }
