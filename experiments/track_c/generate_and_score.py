#!/usr/bin/env python3
"""Track C: generate answers with a local model + retrieved context, score
with ragas (Faithfulness, FactualCorrectness precision/recall), log to W&B.

Every ARQMath topic gets a top-k context from a Track B TREC run file (or no
context at all with --no-rag), an answer from a local generator, and a score
against a human-qrels-derived reference (union of up to
--max-reference-answers High-labelled answers). See
track_c_generation_spec.md for the full design.

Usage:
    # RAG run, one config, 5-topic smoke test:
    poetry run python experiments/track_c/generate_and_score.py \\
        --trec-file data/runs/track_b/bm25_baseline_20260618.tsv \\
        --config-name bm25 --max-topics 5

    # No-RAG baseline:
    poetry run python experiments/track_c/generate_and_score.py \\
        --no-rag --config-name no_rag --max-topics 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.generation_config import (  # noqa: E402
    ContextSourceConfig,
    TrackCConfigManager,
)
from multirag.config.path_configs import (  # noqa: E402
    ANSWERS_JSONL,
    QREL_TASK1_2022_ALL,
    RESULTS_TRACK_C_DIR,
    TASK_C_CONFIG_DIR,
    TOPICS_JSONL,
)
from multirag.eval.generate import run_generation, write_generations  # noqa: E402
from multirag.eval.manifest import sha256_file  # noqa: E402
from multirag.eval.metrics_registry import build_judge_llm, build_metrics  # noqa: E402
from multirag.eval.runner import build_manifest, build_summary, score_all, write_outputs  # noqa: E402
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402
from multirag.generation.context_source import build_context_source  # noqa: E402
from multirag.generation.generator import build_generator  # noqa: E402
from multirag.generation.prompt_template import get_template  # noqa: E402
from multirag.generation.sample_builder import (  # noqa: E402
    build_answer_lookup,
    load_qrels,
    load_topics,
)

configure_logging(level="INFO")
logger = logging.getLogger(__name__)


def _api_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(f"API key not found. Set the {env_var!r} environment variable.")
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description="Track C: generate + score one config")
    parser.add_argument("--trec-file", default=None, help="A Track B run file")
    parser.add_argument(
        "--config-name", required=True, help='e.g. "bm25" | "dense" | "no_rag"'
    )
    parser.add_argument("--k", type=int, default=5, help="Retrieved docs fed as context")
    parser.add_argument(
        "--max-reference-answers", type=int, default=5, help="Cap on High-qrels answers in the reference; NOT swept"
    )
    parser.add_argument("--qrels", default=str(QREL_TASK1_2022_ALL))
    parser.add_argument("--topics-path", default=str(TOPICS_JSONL))
    parser.add_argument("--answers-path", default=str(ANSWERS_JSONL))
    parser.add_argument("--max-topics", type=int, default=None, help="Dry-run cap")
    parser.add_argument(
        "--topic-ids", default=None, help="Comma-separated allowlist, e.g. A.301,A.302"
    )
    parser.add_argument(
        "--no-rag", action="store_true", help="Ignore --trec-file, generate with zero context"
    )
    parser.add_argument(
        "--oracle-mode",
        choices=["reference", "disjoint"],
        default=None,
        help="Ceiling experiments: ignore --trec-file, build context directly from qrels. "
        "'reference' = context is the exact answer set used as the reference (instrument "
        "ceiling). 'disjoint' = context is the next --k High-labelled answers ranked after "
        "the reference (retrieval ceiling); topics without enough surplus get empty context.",
    )
    parser.add_argument(
        "--prompt-template-version",
        default=None,
        help="Override config's prompt_template_version. --no-rag defaults to 'v1' "
        "(open-book) unless set here -- 'v2' is RAG-only (instructed to refuse without "
        "context) and would score near-100%% refusals if used with empty context.",
    )
    parser.add_argument("--config", default=str(TASK_C_CONFIG_DIR / "generation_eval.yaml"))
    parser.add_argument("--output-dir", default=str(RESULTS_TRACK_C_DIR))
    args = parser.parse_args()

    n_modes = sum([bool(args.trec_file), args.no_rag, bool(args.oracle_mode)])
    if n_modes == 0:
        parser.error("One of --trec-file, --no-rag, or --oracle-mode is required")
    if n_modes > 1:
        parser.error("--trec-file, --no-rag, and --oracle-mode are mutually exclusive")

    if args.oracle_mode == "reference":
        effective_k = args.max_reference_answers
    elif args.oracle_mode == "disjoint":
        effective_k = args.k
    else:
        effective_k = 0 if args.no_rag else args.k

    config = TrackCConfigManager.from_yaml(args.config)

    # --- Load data ---
    qrels = load_qrels(Path(args.qrels))
    answer_lookup = build_answer_lookup(Path(args.answers_path))
    topics = load_topics(Path(args.topics_path), None)
    if args.topic_ids:
        allowlist = set(args.topic_ids.split(","))
        topics = [t for t in topics if t["topic_id"] in allowlist]
    if args.max_topics is not None:
        topics = topics[: args.max_topics]
    if not topics:
        logger.error("No topics selected (check --topic-ids / --max-topics / --topics-path)")
        return 1
    logger.info(f"{len(topics)} topics selected")

    # --- Context source (reused as-is: multirag.generation.context_source) ---
    if args.no_rag:
        context_source_config = ContextSourceConfig(type="no_rag")
    elif args.oracle_mode:
        context_source_config = ContextSourceConfig(
            type="oracle", oracle_mode=args.oracle_mode, k=args.k
        )
    else:
        context_source_config = ContextSourceConfig(
            type="run_file", run_path=args.trec_file, k=args.k
        )
    context_source = build_context_source(
        context_source_config,
        answer_lookup=answer_lookup,
        qrels=qrels,
        max_reference_answers=args.max_reference_answers,
    )
    if args.prompt_template_version:
        template_version = args.prompt_template_version
    elif args.no_rag:
        template_version = "v1"
    else:
        template_version = config.prompt_template_version
    template = get_template(template_version)

    # --- Generation (reused as-is: multirag.generation.generator, additive
    # generate_batch_with_metadata) ---
    generator = build_generator(config.generator)
    try:
        records = run_generation(
            topics=topics,
            context_source=context_source,
            template=template,
            generator=generator,
            config_name=args.config_name,
            k=effective_k,
            generator_model=config.generator.model,
            seed=config.generator.decoding.seed,
            max_new_tokens=config.generator.decoding.max_new_tokens,
        )
    finally:
        generator.unload()

    n_refusals = sum(1 for r in records if r.is_refusal)
    logger.info(f"Generation done: {len(records)} answers, {n_refusals} refusal(s)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.config_name}_k{effective_k}_{timestamp}"
    write_generations(records, output_dir / "generations.jsonl")

    # --- Scoring (fresh: multirag.eval.{metrics_registry,references,runner}) ---
    api_key = _api_key(config.judge.api_key_env)
    llm = build_judge_llm(config.judge.model, api_key, config.judge.temperature)
    metrics = build_metrics(config.judge.metrics, llm)

    from google import genai

    genai_client = genai.Client(api_key=api_key)

    def token_counter(text: str) -> int:
        return genai_client.models.count_tokens(
            model=config.judge.model, contents=text
        ).total_tokens

    scores, ref_results, usage, n_claims = asyncio.run(
        score_all(
            generations=records,
            metrics=metrics,
            qrels=qrels,
            answer_lookup=answer_lookup,
            max_reference_answers=args.max_reference_answers,
            token_counter=token_counter,
            judge_model=config.judge.model,
            cache_dir=Path(config.judge.cache_dir),
            max_concurrency=config.judge.max_concurrency,
        )
    )

    summary = build_summary(records, scores, ref_results, usage, n_claims)
    manifest = build_manifest(
        config=config,
        prompt_sha256=template.sha256,
        config_sha256=sha256_file(args.config),
        qrels_path=args.qrels,
        qrels_sha256=sha256_file(args.qrels),
        k=effective_k,
        max_reference_answers=args.max_reference_answers,
        trec_file=args.trec_file,
        config_name=args.config_name,
        prompt_template_version=template_version,
    )
    outputs = write_outputs(output_dir, scores, ref_results, manifest, summary, records, n_claims)

    print(f"\n{'=' * 70}")
    print(f"Track C: {args.config_name} (k={effective_k})")
    print(f"{'=' * 70}")
    print(f"n_topics: {summary['n_topics']}  refusal_rate: {summary['refusal_rate']}")
    print(f"usage: {usage}")
    if "mean_n_claims" in summary:
        print(f"mean_n_claims: {summary['mean_n_claims']} (n={summary['n_claims_scored']})")
    for name in config.judge.metrics:
        print(
            f"  {name:<32} answering={summary.get(f'{name}_mean_answering')}  "
            f"all={summary.get(f'{name}_mean_all')}"
        )
    print(f"\nWritten to {output_dir}")
    print(f"{'=' * 70}\n")

    # --- W&B ---
    exp_logger = ExperimentLogger(config.wandb)
    run_name = f"C-{args.config_name}-k{effective_k}-{timestamp}"[:128]
    exp_logger.start_run(
        config=manifest,
        run_name=run_name,
        extra_config={
            "max_reference_answers": args.max_reference_answers,
            "no_rag": args.no_rag,
        },
    )
    try:
        by_topic: dict[str, dict] = {}
        for s in scores:
            by_topic.setdefault(s.topic_id, {})[s.metric_name] = s.value if s.parse_ok else None

        # One row per topic, every topic (no flagged-only filtering), every
        # column together -- per-topic metrics, reference-construction detail,
        # and refusal/truncation detail all live on the same row so nothing
        # needs cross-referencing across tables in the UI.
        exp_logger.log_table(
            "per_topic",
            [
                {
                    "topic_id": r.topic_id,
                    **by_topic.get(r.topic_id, {}),
                    "is_refusal": r.is_refusal,
                    "finish_reason": r.finish_reason,
                    "completion_tokens": r.completion_tokens,
                    "max_new_tokens": r.max_new_tokens,
                    "hit_ceiling": r.hit_ceiling,
                    "response": r.response,
                    "n_claims": n_claims.get(r.topic_id),
                    "n_contexts": len(r.contexts),
                    "n_high_available": ref_results[r.topic_id].n_high_available,
                    "n_high_used": ref_results[r.topic_id].n_high_used,
                    "fallback_used": ref_results[r.topic_id].fallback_used,
                    "reference_char_len": ref_results[r.topic_id].reference_char_len,
                    "reference_token_len": ref_results[r.topic_id].reference_token_len,
                }
                for r in records
            ],
        )

        # One row per metric: mean (answering-only / all-topics) + n scored.
        exp_logger.log_table(
            "metrics_summary",
            [
                {
                    "metric": name,
                    "mean_answering": summary.get(f"{name}_mean_answering"),
                    "mean_all": summary.get(f"{name}_mean_all"),
                    "n_scored": summary.get(f"{name}_n_scored"),
                }
                for name in config.judge.metrics
            ],
        )

        numeric_summary = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        numeric_summary["usage_cache_hits"] = usage["cache_hits"]
        numeric_summary["usage_new_calls"] = usage["new_calls"]
        exp_logger.log_metrics(numeric_summary)

        exp_logger.save_file(output_dir / "generations.jsonl")
        for path in outputs.values():
            exp_logger.save_file(path)
        if args.trec_file:
            exp_logger.save_file(args.trec_file)
        exp_logger.save_file(args.config)
    finally:
        exp_logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
