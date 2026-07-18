#!/usr/bin/env python3
"""C0.1 No-RAG headroom gate for ARQMath Track C.

Generates answers to ARQMath Task 1 topics using a local GPU model with NO
retrieved context, then scores with RAGAS (Gemini judge + embedding metrics).
Reports per-model headroom to inform generator model selection.

Usage:
    # Dry run (no GPU, no API — prints prompts and exits):
    poetry run python experiments/run_c0_1.py configs/task_c/c0_1_qwen2.5_3b.yaml --dry-run

    # Smoke test (5 topics, non-LLM metrics only — fast):
    poetry run python experiments/run_c0_1.py configs/task_c/c0_1_qwen2.5_3b.yaml \\
        --n-topics 5 --skip-llm-metrics

    # Smoke test (5 topics, all metrics including Gemini):
    poetry run python experiments/run_c0_1.py configs/task_c/c0_1_qwen2.5_3b.yaml \\
        --n-topics 5

    # Full run (all topics):
    poetry run python experiments/run_c0_1.py configs/task_c/c0_1_qwen2.5_7b.yaml

    # Headroom comparison report across all completed runs:
    poetry run python experiments/run_c0_1.py --report data/runs/task_c/
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logging_config import configure_logging

from multirag.config.generation_config import GenerationRunConfigManager
from multirag.config.path_configs import TASK_C_RUNS_DIR
from multirag.generation.context_source import build_context_source
from multirag.generation.prompt_template import get_template
from multirag.generation.ragas_eval import RagasEvaluator
from multirag.generation.sample_builder import build_generation_samples, load_topics

configure_logging(level="INFO")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Headroom report
# ---------------------------------------------------------------------------

def print_headroom_report(aggregates: list[dict]) -> str:
    """Print and return a formatted headroom comparison table."""
    if not aggregates:
        return ""

    # Gather all metric names
    metric_keys = sorted({
        k.replace("_mean", "")
        for agg in aggregates
        for k in agg
        if k.endswith("_mean") and not k.startswith("headroom_")
    })

    header = f"{'Model':<40}" + "".join(f"{m:>18}" for m in metric_keys) + "".join(
        f"{'headroom(' + m + ')':>22}" for m in metric_keys
    )
    lines = ["\n" + "=" * len(header), "C0.1 HEADROOM REPORT", "=" * len(header), header]

    for agg in aggregates:
        row = f"{agg.get('run_name', '?'):<40}"
        for m in metric_keys:
            v = agg.get(f"{m}_mean", float("nan"))
            row += f"{v:>18.4f}" if not math.isnan(v) else f"{'N/A':>18}"
        for m in metric_keys:
            v = agg.get(f"headroom_{m}", float("nan"))
            row += f"{v:>22.4f}" if not math.isnan(v) else f"{'N/A':>22}"
        lines.append(row)

    lines.append("=" * len(header))

    # Decision
    ar_key = "headroom_answer_relevance"
    best = None
    best_headroom = -1.0
    for agg in aggregates:
        hr = agg.get(ar_key, float("nan"))
        mean = agg.get("answer_relevance_mean", float("nan"))
        if not math.isnan(hr) and not math.isnan(mean) and mean <= 0.80 and hr > best_headroom:
            best_headroom = hr
            best = agg

    if best:
        lines.append(
            f"\nSELECTED MODEL: {best['run_name']} "
            f"(AR={best.get('answer_relevance_mean', float('nan')):.4f}, "
            f"headroom={best_headroom:.4f})"
        )
        lines.append("Proceed to C0.2: freeze this model's config.")
    else:
        lines.append(
            "\nWARNING: All models have answer_relevance > 0.80 or data is missing."
        )
        lines.append("Track C may be inconclusive. Consider math-specific metrics.")

    report = "\n".join(lines)
    print(report)
    return report


def run_report(runs_dir: Path) -> None:
    """Load all *_aggregate.json files from runs_dir and print headroom table."""
    agg_files = sorted(runs_dir.glob("*_aggregate.json"))
    if not agg_files:
        logger.error(f"No aggregate JSON files found in {runs_dir}")
        sys.exit(1)

    aggregates = []
    for f in agg_files:
        with open(f) as fh:
            aggregates.append(json.load(fh))
    print_headroom_report(aggregates)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> int:
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    logger.info(f"Loading config from {config_path}")
    config = GenerationRunConfigManager.from_yaml(config_path)

    # CLI overrides
    n_topics: int | None = args.n_topics if args.n_topics is not None else config.n_topics

    # Stamp run name with date
    run_name = config.run_name + "_" + datetime.now().strftime("%Y%m%d")

    logger.info(f"Run: {run_name} | Phase: {config.phase} | n_topics: {n_topics or 'all'}")

    if args.dry_run:
        topics = load_topics(Path(config.topics_path), n_topics)
        _print_dry_run(config, topics)
        return 0

    # W&B init
    wandb.init(
        project="one-last-run",
        name=run_name,
        group="track C",
        tags=[config.phase, config.generator.model.split("/")[-1]],
        config=config.model_dump(),
    )

    samples, template, n_topics_loaded, n_with_gt = build_generation_samples(config, n_topics)

    # === EVALUATION PHASE ===
    logger.info("=== RAGAS EVALUATION PHASE ===")

    ragas_config = config.ragas
    if args.skip_llm_metrics:
        from copy import deepcopy
        ragas_config = deepcopy(ragas_config)
        non_llm = [m for m in ragas_config.metrics
                   if m in ("semantic_similarity", "rouge_l", "bleu")]
        ragas_config.metrics = non_llm
        logger.info(f"--skip-llm-metrics: only running {non_llm}")

    evaluator = RagasEvaluator(ragas_config)
    per_sample_results = evaluator.evaluate(samples)
    aggregate = evaluator.aggregate(per_sample_results)
    aggregate["run_name"] = run_name
    aggregate["phase"] = config.phase
    aggregate["n_topics"] = n_topics_loaded
    aggregate["n_topics_with_ground_truth"] = n_with_gt

    # === WRITE RESULTS ===
    TASK_C_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = TASK_C_RUNS_DIR / f"{run_name}.jsonl"
    agg_path = TASK_C_RUNS_DIR / f"{run_name}_aggregate.json"

    with open(jsonl_path, "w") as f:
        # Line 0: metadata header
        meta = {
            "_meta": True,
            "run_name": run_name,
            "phase": config.phase,
            "config": config.model_dump(),
            "template_sha256": template.sha256,
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # Lines 1..N: per-topic results
        for sample, metrics in zip(samples, per_sample_results):
            row = {
                "topic_id": sample.topic_id,
                "answer": sample.answer,
                "contexts": sample.contexts,
                "ground_truth": sample.ground_truth,
                "ground_truth_answer_id": sample.ground_truth_answer_id,
                "ground_truth_score": sample.ground_truth_score,
                **metrics,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Per-topic results written to {jsonl_path}")

    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    logger.info(f"Aggregate results written to {agg_path}")

    # W&B logging
    numeric_aggregate = {k: v for k, v in aggregate.items() if isinstance(v, (int, float))}
    wandb.log({
        **numeric_aggregate,
        "results_path": str(jsonl_path),
        "template_sha256": template.sha256,
    })

    metrics_table = wandb.Table(
        columns=["Metric", "Value"],
        data=[[k, v] for k, v in sorted(numeric_aggregate.items())],
    )
    per_topic_table = wandb.Table(
        columns=["topic_id", "answer", "ground_truth", *sorted(per_sample_results[0].keys())]
        if per_sample_results else ["topic_id", "answer", "ground_truth"],
        data=[
            [sample.topic_id, sample.answer, sample.ground_truth,
             *[metrics[k] for k in sorted(metrics.keys())]]
            for sample, metrics in zip(samples, per_sample_results)
        ],
    )
    wandb.log({
        "metrics_table": metrics_table,
        "per_topic_table": per_topic_table,
    })
    wandb.finish()

    # Print headroom report for this run
    print_headroom_report([aggregate])

    logger.info(f"Run complete: {run_name}")
    logger.info(f"JSONL: {jsonl_path}")
    logger.info(f"Aggregate: {agg_path}")
    return 0


def _print_dry_run(config, topics: list[dict]) -> None:
    """Print sample prompts and ground_truth preview without any API or GPU calls."""
    template = get_template(config.prompt_template_version)
    context_source = build_context_source(config.context_source)

    print(f"\n=== DRY RUN: {config.run_name} ===")
    print(f"Template SHA256: {template.sha256}")
    print(f"Topics: {len(topics)} (showing first 3)")
    print()

    for topic in topics[:3]:
        contexts = context_source.get_contexts(topic)
        question = topic["title"] + "\n\n" + topic["question"]
        messages = template.render(question, contexts)
        print(f"--- Topic: {topic['topic_id']} ---")
        for msg in messages:
            print(f"[{msg['role'].upper()}]\n{msg['content']}\n")
        print()

    print("Dry run complete. No GPU or API calls made.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="C0.1 No-RAG headroom gate for ARQMath Track C",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to YAML config file (configs/task_c/*.yaml)",
    )
    parser.add_argument(
        "--n-topics", type=int, default=None,
        help="Limit to first N topics (overrides config n_topics; useful for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sample prompts and exit without GPU or API calls",
    )
    parser.add_argument(
        "--skip-llm-metrics", action="store_true",
        help="Skip Gemini-based metrics; only run non-LLM metrics (fast, no API cost)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--report", nargs="?", const=str(TASK_C_RUNS_DIR), metavar="RUNS_DIR",
        help="Print headroom comparison table from *_aggregate.json files in RUNS_DIR "
             "(default: data/runs/task_c/). Cannot be combined with config_path.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.report is not None:
        run_report(Path(args.report))
        return 0

    if not args.config_path:
        parser.print_help()
        return 1

    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
