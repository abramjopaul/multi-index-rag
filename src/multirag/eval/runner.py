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
) -> str:
    raw = "|".join(
        [
            topic_id,
            metric_name,
            question,
            response,
            "\n".join(contexts),
            reference or "",
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
    key = _cache_key(topic_id, bound.spec.name, question, response, contexts, reference, judge_model)
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
        else:
            usage["new_calls"] += 1

    cache.put(key, {"value": value, "parse_ok": parse_ok})
    return ScoreResult(
        topic_id=topic_id, metric_name=bound.spec.name, value=value, parse_ok=parse_ok, source="live"
    )


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
) -> tuple[list[ScoreResult], dict[str, ReferenceResult], dict]:
    """Score every (topic, applicable metric) pair. Reference-requiring
    metrics are skipped (not scored, not an error) for topics whose
    ReferenceResult.reference_text is None; context-requiring metrics are
    skipped for topics with empty contexts (e.g. --no-rag runs).
    """
    cache = ScoreCache(cache_dir)
    semaphore = asyncio.Semaphore(max_concurrency)
    usage = {"cache_hits": 0, "new_calls": 0}

    ref_results: dict[str, ReferenceResult] = {}
    tasks = []
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

    logger.info(f"Scoring {len(tasks)} (topic, metric) pairs (max_concurrency={max_concurrency})...")
    results = list(await asyncio.gather(*tasks))
    logger.info(f"Scoring complete: {usage['cache_hits']} cache hit, {usage['new_calls']} new")
    return results, ref_results, usage


# ---------------------------------------------------------------------------
# Summary / manifest / report
# ---------------------------------------------------------------------------


def build_summary(
    generations: list[GenerationRecord],
    scores: list[ScoreResult],
    ref_results: dict[str, ReferenceResult],
    usage: dict,
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
        "prompt_template_version": config.prompt_template_version,
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
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

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
