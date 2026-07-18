#!/usr/bin/env python3
"""C0.3 RAGAS judge reliability gate for ARQMath Track C.

Generates answers ONCE for a fixed set of topics under a frozen condition,
then re-runs the RAGAS judge N times against that same fixed sample set.
Variance across repeats isolates judge noise from condition noise, giving
the floor later significance tests between conditions must clear to be
considered a real effect (not just judge jitter).

Deterministic metrics (rouge_l, semantic_similarity — proven exactly-zero
variance under repeated judging) run once regardless of --n-repeats; only
the LLM-judged metrics (answer_relevance, answer_correctness, faithfulness,
context_precision, context_recall) are actually repeated.

Usage:
    # Structural dry run (free metrics only, cheap — validates the mechanism):
    poetry run python experiments/run_c0_3.py configs/task_c/c0_2_frozen.yaml \\
        --n-topics 5 --n-repeats 2 --skip-llm-metrics

    # No-RAG judge reliability (Gemini judge, 20 topics, 10 repeats):
    poetry run python experiments/run_c0_3.py configs/task_c/c0_2_frozen.yaml \\
        --n-topics 20 --n-repeats 10

    # Real-retrieval judge reliability, including faithfulness/context_precision/
    # context_recall (undefined under no-RAG):
    poetry run python experiments/run_c0_3.py configs/task_c/c0_2_block_k5.yaml \\
        --n-topics 20 --n-repeats 10
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logging_config import configure_logging

from multirag.config.generation_config import GenerationRunConfigManager
from multirag.config.path_configs import TASK_C_RUNS_DIR
from multirag.generation.ragas_eval import RagasEvaluator
from multirag.generation.sample_builder import build_generation_samples

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

# Proven exactly-zero variance under repeated judging (embedding cosine /
# string overlap, no LLM call involved) — repeating these is pure waste.
DETERMINISTIC_METRICS = frozenset({"rouge_l", "semantic_similarity"})


def _metric_names(aggregate: dict) -> list[str]:
    return sorted(k[:-len("_mean")] for k in aggregate if k.endswith("_mean"))


def compute_overall_noise_floor(
    run_level_aggregates: list[dict],
) -> dict[str, dict[str, float]]:
    """Std of the run-level aggregate mean across repeats, per metric.

    Answers: "how much does the reported condition score itself swing from
    just re-running the judge" — the number later significance tests need
    as their floor.
    """
    if not run_level_aggregates:
        return {}
    metrics = _metric_names(run_level_aggregates[0])

    noise_floor: dict[str, dict[str, float]] = {}
    for m in metrics:
        vals = [
            ra[f"{m}_mean"] for ra in run_level_aggregates
            if not math.isnan(ra.get(f"{m}_mean", float("nan")))
        ]
        if not vals:
            noise_floor[m] = {"repeat_mean": float("nan"), "repeat_std": float("nan"), "cv": float("nan")}
            continue
        repeat_mean = statistics.mean(vals)
        repeat_std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        cv = (repeat_std / repeat_mean) if repeat_mean else float("nan")
        noise_floor[m] = {
            "repeat_mean": round(repeat_mean, 4),
            "repeat_std": round(repeat_std, 4),
            "cv": round(cv, 4),
        }
    return noise_floor


def compute_mean_per_topic_std(
    per_topic_noise: dict[str, dict], metric_names: list[str]
) -> dict[str, float]:
    """Mean, across topics, of each topic's own std-across-repeats.

    This is what a paired per-topic significance test (e.g. Wilcoxon) will
    actually see, so it matters more downstream than the run-level repeat_std.
    """
    result: dict[str, float] = {}
    for m in metric_names:
        vals = [
            stats[f"{m}_std"] for stats in per_topic_noise.values()
            if not math.isnan(stats.get(f"{m}_std", float("nan")))
        ]
        result[m] = round(statistics.mean(vals), 4) if vals else float("nan")
    return result


def check_per_topic_std_consistency(
    overall_noise_floor: dict[str, dict[str, float]],
    mean_per_topic_std: dict[str, float],
    raw_matrix_path: Path,
) -> None:
    """Hard stop if repeat_std > 0 but per_topic_std == 0 for the same metric.

    These two numbers are computed from the same underlying (topic x repeat)
    matrix via two different reductions (aggregate-then-std-across-repeats vs
    std-within-topic-then-mean-across-topics). It is mathematically possible
    for one to be exactly 0 while the other is not ONLY if per_topic_std's
    per-topic std values were computed over fewer than 2 non-NaN points for
    every topic (e.g. NaN-dropout leaves each topic with a single usable
    repeat) while the run-level means across repeats still vary — this is a
    real, explainable data condition (see the raw matrix), not a code defect,
    but it must never be reported to a user without being surfaced loudly.
    """
    for m, nf in overall_noise_floor.items():
        repeat_std = nf.get("repeat_std", float("nan"))
        pts = mean_per_topic_std.get(m, float("nan"))
        if math.isnan(repeat_std) or math.isnan(pts):
            continue
        if repeat_std > 1e-6 and pts == 0.0:
            raise RuntimeError(
                f"INCONSISTENCY CHECK FAILED for metric '{m}': repeat_std="
                f"{repeat_std} (run-level mean varies across repeats) but "
                f"per_topic_std=0.0 (every topic's own std-across-repeats is "
                f"exactly zero). Inspect the raw (topic x repeat) matrix at "
                f"{raw_matrix_path} to determine whether this is per-repeat "
                "NaN dropout (each topic has <2 usable repeats) shifting "
                "which topics contribute to the run-level mean, or a genuine "
                "bug in the aggregation. STOPPING before writing the final "
                "report."
            )


def check_context_precision_not_saturated(
    all_repeats: list[list[dict]], context_snapshot_path: Path
) -> None:
    """Hard stop if context_precision == 1.0 for every (topic, repeat).

    Real top-k retrieval should include non-relevant passages; a saturated
    1.0 means context is being filtered to relevance somewhere (e.g. built
    from qrels instead of the raw run file) — the noise measurement would be
    a ceiling artifact, not real judge noise.
    """
    values = [
        r["context_precision"] for repeat in all_repeats for r in repeat
        if "context_precision" in r and not math.isnan(r["context_precision"])
    ]
    if not values:
        return
    if all(v == 1.0 for v in values):
        raise RuntimeError(
            f"SANITY CHECK FAILED: context_precision == 1.0 for all {len(values)} "
            "(topic, repeat) pairs. Real top-k retrieval should include non-relevant "
            "passages; a saturated 1.0 means context is being filtered to relevance "
            "somewhere (e.g. built from qrels instead of the raw TREC run file). "
            f"STOPPING before writing the final report. Context snapshot for "
            f"debugging: {context_snapshot_path}"
        )


def build_metric_rows(
    stoch_names: list[str],
    det_names: list[str],
    overall_noise_floor: dict[str, dict[str, float]],
    mean_per_topic_std: dict[str, float],
    det_aggregate: dict[str, float],
    n_repeats: int,
) -> list[dict]:
    rows = []
    for m in sorted(stoch_names):
        nf = overall_noise_floor.get(
            m, {"repeat_mean": float("nan"), "repeat_std": float("nan"), "cv": float("nan")}
        )
        rows.append({
            "metric": m,
            "repeat_mean": nf["repeat_mean"],
            "repeat_std": nf["repeat_std"],
            "per_topic_std": mean_per_topic_std.get(m, float("nan")),
            "cv": nf["cv"],
            "n_repeats_used": n_repeats,
            "deterministic": False,
        })
    for m in sorted(det_names):
        rows.append({
            "metric": m,
            "repeat_mean": det_aggregate.get(f"{m}_mean", float("nan")),
            "repeat_std": 0.0,
            "per_topic_std": 0.0,
            "cv": 0.0,
            "n_repeats_used": 1,
            "deterministic": True,
        })
    return rows


def verdict_for_cv(cv: float) -> str:
    """CV bucket thresholds: <3% / 3-10% / >10%. (The 3%-5% boundary is left
    unstated upstream; resolved here as <3% / 3-10% / >10%, not <3% / 5-10%.)
    """
    if math.isnan(cv):
        return "N/A (insufficient data)"
    if cv < 0.03:
        return "single judge pass is fine"
    if cv <= 0.10:
        return "average 3 passes per topic in C1"
    return "flag as unreliable"


def print_reliability_report(condition_name: str, n_repeats: int, metric_rows: list[dict]) -> str:
    """Print and return the C0.3 judge-reliability summary + verdicts."""
    header = f"{'Metric':<25}{'repeat_mean':>15}{'repeat_std':>15}{'per_topic_std':>15}{'CV':>10}{'n_reps':>8}"
    lines = [
        "\n" + "=" * len(header),
        "C0.3 RAGAS JUDGE RELIABILITY",
        f"Condition: {condition_name}  |  N repeats: {n_repeats}",
        "=" * len(header),
        header,
    ]
    for row in sorted(metric_rows, key=lambda r: r["metric"]):
        lines.append(
            f"{row['metric']:<25}{row['repeat_mean']:>15.4f}{row['repeat_std']:>15.4f}"
            f"{row['per_topic_std']:>15.4f}{row['cv']:>10.4f}{row['n_repeats_used']:>8}"
        )
    lines.append("=" * len(header))
    lines.append(
        "\nrepeat_std is the judge noise floor for the run-level mean; per_topic_std is "
        "what a paired per-topic significance test will actually see.\n"
    )
    for row in sorted(metric_rows, key=lambda r: r["metric"]):
        if row["deterministic"]:
            lines.append(f"{row['metric']:<22} deterministic (n_repeats=1) -> single judge pass is fine")
        else:
            lines.append(
                f"{row['metric']:<22} CV={row['cv']:.4f} ({row['cv'] * 100:.2f}%) "
                f"-> {verdict_for_cv(row['cv'])}"
            )

    report = "\n".join(lines)
    print(report)
    return report


def run_reliability(args: argparse.Namespace) -> int:
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    logger.info(f"Loading config from {config_path}")
    config = GenerationRunConfigManager.from_yaml(config_path)

    n_topics: int | None = args.n_topics if args.n_topics is not None else config.n_topics
    n_repeats = args.n_repeats

    run_name = config.run_name + "_c03_reliability_" + datetime.now().strftime("%Y%m%d")
    logger.info(
        f"Run: {run_name} | Phase: C0.3 | n_topics: {n_topics or 'all'} | n_repeats: {n_repeats} "
        f"| require_ground_truth: {config.require_ground_truth}"
    )

    wandb.init(
        project="one-last-run",
        name=run_name,
        group="track C",
        tags=["C0.3", config.generator.model.split("/")[-1]],
        config={**config.model_dump(), "n_repeats": n_repeats},
    )

    # Generation runs exactly once: temp=0 + fixed seed already makes it
    # deterministic, and only the judge step is what C0.3 measures. Topic
    # selection has no randomness either way (pure file-order filter+truncate)
    # — there is nothing to random-seed here.
    samples, template, n_topics_loaded, n_with_gt = build_generation_samples(
        config, n_topics, require_ground_truth=config.require_ground_truth
    )
    logger.info(f"Fixed sample set: {n_topics_loaded} topics ({n_with_gt} with ground truth)")

    TASK_C_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    context_snapshot_path = TASK_C_RUNS_DIR / f"{run_name}_context_snapshot.json"
    with open(context_snapshot_path, "w") as f:
        json.dump(
            [{"topic_id": s.topic_id, "contexts": s.contexts} for s in samples],
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"Context snapshot written to {context_snapshot_path} ({len(samples)} topics)")

    ragas_config = config.ragas
    if args.skip_llm_metrics:
        ragas_config = deepcopy(ragas_config)
        non_llm = [m for m in ragas_config.metrics if m in ("semantic_similarity", "rouge_l", "bleu")]
        ragas_config.metrics = non_llm
        logger.info(f"--skip-llm-metrics: only running {non_llm}")

    configured = list(ragas_config.metrics)
    det_names = [m for m in configured if m in DETERMINISTIC_METRICS]
    stoch_names = [m for m in configured if m not in DETERMINISTIC_METRICS]

    det_config = deepcopy(ragas_config)
    det_config.metrics = det_names
    stoch_config = deepcopy(ragas_config)
    stoch_config.metrics = stoch_names

    det_evaluator = RagasEvaluator(det_config) if det_names else None
    stoch_evaluator = RagasEvaluator(stoch_config) if stoch_names else None

    det_per_sample: list[dict] | None = None
    if det_evaluator is not None:
        logger.info(
            f"Deterministic track ({det_names}): running ONCE, n_repeats forced to 1 "
            "(proven exactly-zero variance — repeating is pure waste)"
        )
        det_per_sample = det_evaluator.evaluate(samples)

    all_repeats: list[list[dict]] = []
    faithfulness_claim_counts: list[dict[str, int | None]] = []

    if stoch_evaluator is not None:
        logger.info(
            f"=== JUDGE RELIABILITY: {n_repeats} repeats on {n_topics_loaded} fixed topics: "
            f"{stoch_names} ==="
        )
        for i in range(n_repeats):
            logger.info(f"--- Repeat {i + 1}/{n_repeats} ---")
            per_sample_results = stoch_evaluator.evaluate(samples)
            all_repeats.append(per_sample_results)
            if "faithfulness" in stoch_names:
                faithfulness_claim_counts.append(stoch_evaluator.get_faithfulness_claim_counts() or {})

    if "context_precision" in stoch_names:
        check_context_precision_not_saturated(all_repeats, context_snapshot_path)

    # Raw (topic x repeat) matrix, long format — written unconditionally,
    # before any aggregation, so it survives even if a later sanity check
    # aborts the run. Lets the exact per-repeat pattern behind any aggregate
    # number be inspected directly (e.g. distinguishing genuine judge noise
    # from per-repeat missing-data dropout).
    raw_matrix_path = TASK_C_RUNS_DIR / f"{run_name}_raw_scores.csv"
    with open(raw_matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic_id", "repeat", "metric", "value"])
        for idx, sample in enumerate(samples):
            for r in range(n_repeats):
                for m in stoch_names:
                    val = all_repeats[r][idx].get(m, float("nan"))
                    writer.writerow([sample.topic_id, r + 1, m, val])
    logger.info(f"Raw (topic x repeat) score matrix written to {raw_matrix_path}")

    # Per-topic judge noise: reuse RagasEvaluator.aggregate() unmodified by
    # transposing the data — for a fixed topic, its N repeat results become
    # the "samples" aggregate() averages over, so mean/std are computed
    # ACROSS REPEATS instead of across topics.
    per_topic_noise: dict[str, dict] = {}
    if stoch_evaluator is not None:
        for idx, sample in enumerate(samples):
            repeats_for_topic = [all_repeats[r][idx] for r in range(n_repeats)]
            per_topic_noise[sample.topic_id] = stoch_evaluator.aggregate(repeats_for_topic)

    # Overall noise floor: aggregate() across topics within each repeat (one
    # run-level score per repeat), then std of those run-level scores across
    # repeats.
    run_level_aggregates = (
        [stoch_evaluator.aggregate(all_repeats[r]) for r in range(n_repeats)]
        if stoch_evaluator is not None else []
    )
    overall_noise_floor = compute_overall_noise_floor(run_level_aggregates)
    mean_per_topic_std = compute_mean_per_topic_std(per_topic_noise, stoch_names)
    check_per_topic_std_consistency(overall_noise_floor, mean_per_topic_std, raw_matrix_path)
    det_aggregate = det_evaluator.aggregate(det_per_sample) if det_evaluator is not None else {}

    metric_rows = build_metric_rows(
        stoch_names, det_names, overall_noise_floor, mean_per_topic_std, det_aggregate, n_repeats
    )

    # === WRITE RESULTS ===
    metrics_csv_path = TASK_C_RUNS_DIR / f"{run_name}_metrics.csv"
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "repeat_mean", "repeat_std", "per_topic_std", "cv", "n_repeats_used", "deterministic"]
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    logger.info(f"Metrics CSV written to {metrics_csv_path}")

    claims_csv_path = None
    if faithfulness_claim_counts:
        claims_csv_path = TASK_C_RUNS_DIR / f"{run_name}_faithfulness_claims.csv"
        with open(claims_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["topic_id", "repeat", "claim_count"])
            for repeat_idx, counts in enumerate(faithfulness_claim_counts):
                for topic_id, count in counts.items():
                    writer.writerow([topic_id, repeat_idx + 1, count if count is not None else ""])
        logger.info(f"Faithfulness claim-count sidecar written to {claims_csv_path}")

    report_path = TASK_C_RUNS_DIR / f"{run_name}.json"
    with open(report_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "phase": "C0.3",
            "condition_config": str(config_path),
            "template_sha256": template.sha256,
            "n_topics": n_topics_loaded,
            "n_topics_with_ground_truth": n_with_gt,
            "n_repeats": n_repeats,
            "deterministic_metrics": det_names,
            "stochastic_metrics": stoch_names,
            "overall_noise_floor": overall_noise_floor,
            "mean_per_topic_std": mean_per_topic_std,
            "per_topic_noise": per_topic_noise,
            "run_level_aggregates": run_level_aggregates,
            "det_aggregate": det_aggregate,
            "context_snapshot_path": str(context_snapshot_path),
            "metrics_csv_path": str(metrics_csv_path),
            "raw_scores_csv_path": str(raw_matrix_path),
            "faithfulness_claims_csv_path": str(claims_csv_path) if claims_csv_path else None,
        }, f, indent=2)
    logger.info(f"Reliability report written to {report_path}")

    # W&B logging
    flat_noise_floor = {
        f"noise_floor.{m}.{k}": v
        for m, stats in overall_noise_floor.items()
        for k, v in stats.items()
    }
    wandb.log({
        **flat_noise_floor,
        "report_path": str(report_path),
        "metrics_csv_path": str(metrics_csv_path),
        "context_snapshot_path": str(context_snapshot_path),
        "raw_scores_csv_path": str(raw_matrix_path),
    })

    metrics_table = wandb.Table(
        columns=["metric", "repeat_mean", "repeat_std", "per_topic_std", "cv", "n_repeats_used", "deterministic"],
        data=[
            [r["metric"], r["repeat_mean"], r["repeat_std"], r["per_topic_std"], r["cv"], r["n_repeats_used"], r["deterministic"]]
            for r in metric_rows
        ],
    )
    wandb_log_payload = {"metrics_table": metrics_table}

    if per_topic_noise:
        metric_names_for_table = _metric_names(next(iter(per_topic_noise.values())))
        per_topic_noise_table = wandb.Table(
            columns=["topic_id", *[f"{m}_std" for m in metric_names_for_table]],
            data=[
                [tid, *[stats.get(f"{m}_std", float("nan")) for m in metric_names_for_table]]
                for tid, stats in per_topic_noise.items()
            ],
        )
        wandb_log_payload["per_topic_noise_table"] = per_topic_noise_table

    if claims_csv_path:
        wandb_log_payload["faithfulness_claims_csv_path"] = str(claims_csv_path)

    wandb.log(wandb_log_payload)

    # The *_path keys logged above are plain strings pointing at local disk —
    # visible in the run config/summary, but not fetchable from wandb itself.
    # wandb.save() actually uploads the file content so the run is
    # self-contained (inspectable from the dashboard / another machine).
    for path in (report_path, metrics_csv_path, context_snapshot_path, raw_matrix_path, claims_csv_path):
        if path is not None:
            wandb.save(str(path), policy="now")

    wandb.finish()

    print_reliability_report(config.run_name, n_repeats, metric_rows)

    logger.info(f"Run complete: {run_name}")
    logger.info(f"Report: {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="C0.3 RAGAS judge reliability gate for ARQMath Track C",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config_path",
        help="Path to a Track C YAML config, e.g. configs/task_c/c0_2_frozen.yaml or "
             "configs/task_c/c0_2_block_k5.yaml",
    )
    parser.add_argument(
        "--n-topics", type=int, default=None,
        help="Fixed topic subset size (overrides config n_topics; keep small — this "
             "multiplies judge API calls by --n-repeats)",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=10,
        help="Number of times to re-run the judge on the fixed sample set (default: 10). "
             "Deterministic metrics (rouge_l, semantic_similarity) always run once regardless.",
    )
    parser.add_argument(
        "--skip-llm-metrics", action="store_true",
        help="Skip Gemini-based metrics; only run non-LLM metrics. Use for a cheap "
             "structural dry run of the repeat/aggregation mechanism before spending "
             "real judge API calls (non-LLM metric noise should come out ~0).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return run_reliability(args)


if __name__ == "__main__":
    sys.exit(main())
