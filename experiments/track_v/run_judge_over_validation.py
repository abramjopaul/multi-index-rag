#!/usr/bin/env python3
"""Track V: feed the validation set through the judge (batch mode by default).

Resumable: re-running after a partial completion (crash, interrupted poll
loop, ...) fetches any batch chunks left pending in the checkpoint first,
then submits only the still-unlabelled remainder -- cache + checkpoint mean
nothing already labelled is ever re-spent.

Opens the V W&B run here (group="V", job_type="judge-validation") and logs
running totals (pairs labelled, cache-hit rate, parse-success rate, tokens,
estimated cost) as batch chunks complete, so a long (~100K pair) job is
observable while it runs. The run is left open for build_paired_labels.py /
agreement_analysis.py / report.py to continue logging into (see report.py,
which calls finish()).

Usage:
    poetry run python experiments/track_v/run_judge_over_validation.py
    poetry run python experiments/track_v/run_judge_over_validation.py --mode sync  # tiny/dry runs only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.judge_config import JudgeConfigManager  # noqa: E402
from multirag.config.path_configs import (
    EVAL_CONFIG_DIR,  # noqa: E402
    RESULTS_TRACK_V_DIR,
)
from multirag.eval.judge_client import BatchHandle, JudgeClient  # noqa: E402
from multirag.eval.manifest import build_manifest, sha256_file  # noqa: E402
from multirag.eval.schema import JudgePair, RelevanceJudgement  # noqa: E402
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)


def _load_pairs(validation_set_path: Path) -> list[JudgePair]:
    pairs = []
    with open(validation_set_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.append(
                JudgePair(
                    topic_id=row["topic_id"],
                    answer_id=row["answer_id"],
                    question=row["question"],
                    answer_text=row["answer_text"],
                )
            )
    return pairs


def _poll_and_fetch(
    client: JudgeClient,
    handles: list[BatchHandle],
    poll_seconds: int,
    exp_logger: ExperimentLogger,
    running_totals: dict,
) -> list[RelevanceJudgement]:
    results: list[RelevanceJudgement] = []
    pending = list(handles)
    while pending:
        still_pending = []
        for handle in pending:
            status = client.poll_batch(handle)
            if status.done:
                logger.info(f"Batch job {handle.job_name} done: state={status.state}")
                fetched = client.fetch_batch(handle)
                results.extend(fetched)
                running_totals["pairs_labelled"] += len(fetched)
                running_totals["parse_ok"] += sum(1 for j in fetched if j.parse_ok)
                exp_logger.log_metrics(
                    {
                        "pairs_labelled": running_totals["pairs_labelled"],
                        "parse_success_rate": (
                            running_totals["parse_ok"]
                            / running_totals["pairs_labelled"]
                            if running_totals["pairs_labelled"]
                            else 0.0
                        ),
                        "cache_hits": client.usage["cache_hits"],
                        "tokens": client.usage["tokens"],
                        "estimated_cost_usd": client.usage["estimated_cost_usd"],
                    }
                )
            else:
                logger.info(
                    f"Batch job {handle.job_name} state={status.state}, still running"
                )
                still_pending.append(handle)
        pending = still_pending
        if pending:
            time.sleep(poll_seconds)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the judge over the Track V validation set"
    )
    parser.add_argument(
        "--config",
        default=str(EVAL_CONFIG_DIR / "judge.yaml"),
        help="Path to judge.yaml",
    )
    parser.add_argument(
        "--validation-set",
        default=str(RESULTS_TRACK_V_DIR / "validation_set.jsonl"),
        help="Path to build_validation_set.py's output",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "sync"],
        default=None,
        help="Override execution.mode from config (sync is for tiny dry runs only)",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_TRACK_V_DIR / "judgements.jsonl")
    )
    args = parser.parse_args()

    config = JudgeConfigManager.from_yaml(args.config)
    mode = args.mode or config.execution.mode

    client = JudgeClient(config)
    pairs = _load_pairs(Path(args.validation_set))
    logger.info(f"Loaded {len(pairs)} pairs from {args.validation_set}")

    cached, remaining = client.partition_cached(pairs)
    logger.info(f"Cache: {len(cached)} hit, {len(remaining)} need labelling")

    # Resume the SAME W&B run across repeated invocations of this script (a large
    # (~39K-pair) job spans multiple invocations due to batch_max_in_flight -- without
    # this, every invocation would open a new run and fragment the V run across dozens
    # of separate W&B entries instead of one continuous one).
    existing_run_id = None
    existing_run_name = None
    output_path_check = Path(args.output)
    if output_path_check.exists():
        with open(output_path_check) as f:
            first_line = f.readline()
            if first_line.strip():
                candidate = json.loads(first_line)
                if candidate.get("_meta"):
                    existing_run_id = candidate.get("wandb_run_id")
                    existing_run_name = candidate.get("run_name")

    exp_logger = ExperimentLogger(config.wandb)
    config_sha256 = sha256_file(args.config)
    manifest = build_manifest(
        config=config,
        execution_mode=mode,
        prompt_sha256=client.prompt_sha256,
        config_sha256=config_sha256,
        usage=client.usage,
    )
    run_name = existing_run_name or (
        f"V-{config.judge.model}-{config.prompt.template_path.split('/')[-1]}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    )
    exp_logger.start_run(config=manifest, run_name=run_name, run_id=existing_run_id)

    running_totals = {
        "pairs_labelled": len(cached),
        "parse_ok": sum(1 for j in cached if j.parse_ok),
    }

    try:
        all_judgements: list[RelevanceJudgement] = list(cached)

        # Resume: fetch anything left pending in the checkpoint from a prior
        # interrupted run before submitting anything new.
        pending_handles = client.pending_batches()
        if pending_handles:
            logger.info(f"Resuming {len(pending_handles)} pending batch chunk(s)")
            all_judgements.extend(
                _poll_and_fetch(
                    client,
                    pending_handles,
                    config.execution.batch_poll_seconds,
                    exp_logger,
                    running_totals,
                )
            )

        if mode == "sync":
            # judge_sync re-partitions internally; passing the full `pairs`
            # list keeps a single source of truth for what's cached.
            fresh = client.judge_sync(pairs)
            # Avoid double-counting pairs already accounted for above.
            already = {(j.topic_id, j.answer_id) for j in all_judgements}
            all_judgements.extend(
                j for j in fresh if (j.topic_id, j.answer_id) not in already
            )
        else:
            handles = client.submit_batch(pairs)
            logger.info(f"Submitted {len(handles)} new batch chunk(s)")
            all_judgements.extend(
                _poll_and_fetch(
                    client,
                    handles,
                    config.execution.batch_poll_seconds,
                    exp_logger,
                    running_totals,
                )
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            manifest["usage"] = client.usage
            # wandb_run_id lets build_paired_labels.py / agreement_analysis.py /
            # report.py resume logging into this SAME run instead of opening a
            # new one -- see ExperimentLogger.start_run's run_id parameter.
            manifest["wandb_run_id"] = exp_logger.run_id
            f.write(
                json.dumps({"_meta": True, "run_name": run_name, **manifest}) + "\n"
            )
            for j in all_judgements:
                f.write(json.dumps(j.to_dict(), ensure_ascii=False) + "\n")

        parse_rate = (
            sum(1 for j in all_judgements if j.parse_ok) / len(all_judgements)
            if all_judgements
            else 0.0
        )
        cache_hit_rate = client.usage["cache_hits"] / len(pairs) if pairs else 0.0

        print(f"\n{'=' * 60}")
        print("Track V judge run")
        print(f"{'=' * 60}")
        print(f"  Total pairs:        {len(pairs)}")
        print(f"  Labelled this run:  {len(all_judgements)}")
        print(f"  Parse success rate: {parse_rate:.4f}")
        print(f"  Cache hit rate:     {cache_hit_rate:.4f}")
        print(f"  Tokens:             {client.usage['tokens']}")
        print(f"  Est. cost (USD):    {client.usage['estimated_cost_usd']:.4f}")
        print(f"Written to {output_path}")
        print(f"{'=' * 60}\n")

        exp_logger.log_metrics(
            {
                "final_pairs_labelled": len(all_judgements),
                "final_parse_success_rate": parse_rate,
                "final_cache_hit_rate": cache_hit_rate,
                "final_tokens": client.usage["tokens"],
                "final_estimated_cost_usd": client.usage["estimated_cost_usd"],
            }
        )
    except Exception:
        exp_logger.finish()
        raise

    # The run is intentionally left open (no finish() here) so
    # build_paired_labels.py / agreement_analysis.py can resume it via
    # wandb_run_id above -- only report.py calls finish(), in its own
    # try/finally, once the full V pipeline has logged everything.
    return 0


if __name__ == "__main__":
    sys.exit(main())
