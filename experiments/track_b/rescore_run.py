#!/usr/bin/env python3
"""Track B: re-score retrieval configs with hole rate + 3 channels.

Every ARQMath qrels file only judges the original competition's pooled
candidates -- a retrieval config built after that competition surfaces new
candidates with NO qrel entry at all ("holes"). This script measures the
size of that gap and whether filling it with the Track-V-validated judge
changes the picture, for every TREC run file ("config") in a directory:

1. Hole rate @5/10/20/100/1000 (free, no API -- multirag.eval.hole_rate).
2. Three channels, all at @5/10/20 (shallow fill depth, matches the
   original Track B plan's "top-~20 holes per config"):
   - human_only:     existing evaluate_run() behaviour, unchanged, full
                      cutoff range (5/10/20/100/1000).
   - llm_plus_human: holes within the top-20 get a judge label, merged into
                      a copy of the human qrels; everything else unchanged.
   - llm_only:       every top-20 doc gets a judge label (human label
                      ignored). Cache-first via the same JudgeClient V used
                      means docs already human-judged were already judged
                      by V -- free; only genuine holes cost anything, and
                      channels 2/3 share that one judging pass.

Usage:
    # Every TREC file in data/runs/track_b/:
    poetry run python experiments/track_b/rescore_run.py

    # One config only:
    poetry run python experiments/track_b/rescore_run.py \\
        --trec-file data/runs/track_b/bm25_baseline_20260618.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.judge_config import JudgeConfigManager  # noqa: E402
from multirag.config.path_configs import (
    ANSWERS_JSONL,  # noqa: E402
    EVAL_CONFIG_DIR,
    QREL_TASK1_2022_ALL,
    RESULTS_TRACK_B_DIR,
    TOPICS_JSONL,
    TRACK_B_RUNS_DIR,
)
from multirag.eval.hole_rate import hole_rate  # noqa: E402
from multirag.eval.judge_client import JudgeClient  # noqa: E402
from multirag.eval.manifest import build_manifest, sha256_file  # noqa: E402
from multirag.eval.schema import JudgePair, RelevanceJudgement  # noqa: E402
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402
from multirag.evaluation.metrics import _parse_qrels, _parse_run, evaluate  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

DEFAULT_HOLE_RATE_CUTOFFS = [5, 10, 20, 100, 1000]
DEFAULT_FILL_DEPTH = 20
CHANNEL_CUTOFF_KEYS = {
    "5",
    "10",
    "20",
}  # metric suffixes kept for llm_plus_human/llm_only


# ---------------------------------------------------------------------------
# Local data helpers (2022-scoped, mirrors build_validation_set.py's pattern
# without importing it -- consistent with build_paired_labels.py not
# importing from build_validation_set.py either)
# ---------------------------------------------------------------------------


def _load_topics(path: Path) -> dict[str, dict]:
    topics: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                topics[row["topic_id"]] = row
    return topics


def _load_answer_texts(path: Path, needed_ids: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in needed_ids:
                texts[row["id"]] = row.get("body_text", "")
    return texts


def _question_text(topic: dict) -> str:
    return f"{topic.get('title', '')}\n\n{topic.get('question', '')}".strip()


# ---------------------------------------------------------------------------
# Ranking / hole helpers
# ---------------------------------------------------------------------------


def _ranking_from_run(run: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Score-sorted doc list per topic -- _parse_run returns an unordered
    dict[doc_id -> score], not a ranking.
    """
    return {
        qid: [doc_id for doc_id, _ in sorted(docs.items(), key=lambda x: -x[1])]
        for qid, docs in run.items()
    }


def _find_holes(
    ranking: dict[str, list[str]], qrels: dict[str, dict[str, int]], depth: int
) -> list[tuple[str, str]]:
    holes = []
    for topic_id, docs in ranking.items():
        judged = qrels.get(topic_id, {})
        for doc_id in docs[:depth]:
            if doc_id not in judged:
                holes.append((topic_id, doc_id))
    return holes


def _restrict_run_to_depth(
    run: dict[str, dict[str, float]], ranking: dict[str, list[str]], depth: int
) -> dict[str, dict[str, float]]:
    """Restrict a run dict to the top-`depth` docs per topic (by rank), so
    llm_plus_human/llm_only scoring never sees deeper, unfilled ranks.
    """
    restricted: dict[str, dict[str, float]] = {}
    for qid, docs in ranking.items():
        top = set(docs[:depth])
        restricted[qid] = {
            doc_id: score for doc_id, score in run[qid].items() if doc_id in top
        }
    return restricted


def _filter_channel_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep only @5/@10/@20 (standard + prime) plus cutoff-independent
    map/bpref -- llm_plus_human/llm_only were only filled to the shallow
    depth, so @100/@1000 on them would be misleading, never reported.

    Keys look like "ndcg@100" (standard) or "ndcg'@100" (prime) -- the
    cutoff is always the part after the last '@', regardless of the
    apostrophe's position, so split on '@' from the right.
    """
    out = {}
    for k, v in metrics.items():
        if "@" not in k or k.rsplit("@", 1)[1] in CHANNEL_CUTOFF_KEYS:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Judging (cache-first; shared JudgeClient across all configs in this run)
# ---------------------------------------------------------------------------


def _judge_pairs(
    client: JudgeClient, pairs: list[JudgePair], mode: str, poll_seconds: int
) -> list[RelevanceJudgement]:
    if not pairs:
        return []
    cached, remaining = client.partition_cached(pairs)
    logger.info(
        f"  judging {len(pairs)} pairs: {len(cached)} cache hit, {len(remaining)} new"
    )
    if not remaining:
        return cached
    if mode == "sync":
        # Pass only `remaining` -- judge_sync() re-partitions internally, and
        # passing the full `pairs` here would find the same cache hits again,
        # double-counting client.usage["cache_hits"].
        return cached + client.judge_sync(remaining)

    handles = client.submit_batch(remaining)  # re-partitions internally (no-op: all uncached)
    fetched: list[RelevanceJudgement] = []
    pending = list(handles)
    while pending:
        still_pending = []
        for h in pending:
            status = client.poll_batch(h)
            if status.done:
                logger.info(f"  batch job {h.job_name} done: state={status.state}")
                fetched.extend(client.fetch_batch(h))
            else:
                still_pending.append(h)
        pending = still_pending
        if pending:
            time.sleep(poll_seconds)
    return cached + fetched


# ---------------------------------------------------------------------------
# Per-config processing
# ---------------------------------------------------------------------------


def process_config(
    trec_file: Path,
    qrels: dict[str, dict[str, int]],
    client: JudgeClient,
    topics: dict[str, dict],
    mode: str,
    poll_seconds: int,
    hole_rate_cutoffs: list[int],
    fill_depth: int,
) -> dict:
    config_name = trec_file.stem
    logger.info(f"=== {config_name} ===")

    run = _parse_run(trec_file)
    ranking = _ranking_from_run(run)

    # --- Hole rate (free) ---
    hole_rates = {str(k): hole_rate(ranking, qrels, k) for k in hole_rate_cutoffs}
    logger.info(f"  hole_rate: {hole_rates}")

    # --- Channel 1: human only (unchanged, full range) ---
    human_only = evaluate(qrels, run)

    # --- Holes within the shallow fill depth ---
    holes = _find_holes(ranking, qrels, fill_depth)
    n_slots = sum(min(len(d), fill_depth) for d in ranking.values())
    logger.info(f"  {len(holes)} holes in top-{fill_depth} (of {n_slots} slots)")

    hole_answer_ids = {doc_id for _, doc_id in holes}
    hole_answer_texts = _load_answer_texts(ANSWERS_JSONL, hole_answer_ids)
    hole_pairs = [
        JudgePair(
            topic_id=topic_id,
            answer_id=doc_id,
            question=_question_text(topics[topic_id]),
            answer_text=hole_answer_texts[doc_id],
        )
        for topic_id, doc_id in holes
        if topic_id in topics and doc_id in hole_answer_texts
    ]
    hole_judgements = _judge_pairs(client, hole_pairs, mode, poll_seconds)
    hole_labels = {
        (j.topic_id, j.answer_id): j.label for j in hole_judgements if j.parse_ok
    }

    # --- Channel 2: LLM + Human (shallow fill only) ---
    llm_plus_human_qrels = {qid: dict(docs) for qid, docs in qrels.items()}
    for (topic_id, doc_id), label in hole_labels.items():
        llm_plus_human_qrels.setdefault(topic_id, {})[doc_id] = label
    run_shallow = _restrict_run_to_depth(run, ranking, fill_depth)
    llm_plus_human = _filter_channel_metrics(
        evaluate(llm_plus_human_qrels, run_shallow)
    )

    # --- Channel 3: LLM only (every top-fill_depth doc; cache-first) ---
    topn_ids = {doc_id for docs in ranking.values() for doc_id in docs[:fill_depth]}
    topn_answer_texts = _load_answer_texts(ANSWERS_JSONL, topn_ids)
    topn_pairs = [
        JudgePair(
            topic_id=topic_id,
            answer_id=doc_id,
            question=_question_text(topics[topic_id]),
            answer_text=topn_answer_texts[doc_id],
        )
        for topic_id, docs in ranking.items()
        if topic_id in topics
        for doc_id in docs[:fill_depth]
        if doc_id in topn_answer_texts
    ]
    topn_judgements = _judge_pairs(client, topn_pairs, mode, poll_seconds)
    llm_only_labels = {
        (j.topic_id, j.answer_id): j.label for j in topn_judgements if j.parse_ok
    }
    llm_only_qrels: dict[str, dict[str, int]] = {}
    for (topic_id, doc_id), label in llm_only_labels.items():
        if label is not None:
            llm_only_qrels.setdefault(topic_id, {})[doc_id] = label
    llm_only = _filter_channel_metrics(evaluate(llm_only_qrels, run_shallow))

    # --- Per-answer audit rows ---
    audit_rows = []
    for topic_id, docs in ranking.items():
        judged = qrels.get(topic_id, {})
        for rank, doc_id in enumerate(docs[:fill_depth], start=1):
            audit_rows.append(
                {
                    "config_name": config_name,
                    "topic_id": topic_id,
                    "answer_id": doc_id,
                    "rank": rank,
                    "retrieval_score": run[topic_id][doc_id],
                    "human_label": judged.get(doc_id),
                    "llm_label": llm_only_labels.get((topic_id, doc_id)),
                    "is_hole": doc_id not in judged,
                }
            )

    return {
        "config_name": config_name,
        "trec_file": str(trec_file),
        "hole_rate": hole_rates,
        "channels": {
            "human_only": human_only,
            "llm_plus_human": llm_plus_human,
            "llm_only": llm_only,
        },
        "n_holes_in_topN": len(holes),
        "n_holes_filled": len(hole_labels),
        "n_topN_judged": len(llm_only_labels),
        "audit_rows": audit_rows,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _metrics_table_row(
    config_name: str, channel: str, metrics: dict, hole_rates: dict
) -> dict:
    row = {"config_name": config_name, "channel": channel}
    for k in ("5", "10", "20"):
        row[f"ndcg@{k}"] = metrics.get(f"ndcg@{k}")
        row[f"precision@{k}"] = metrics.get(f"precision@{k}")
        row[f"recall@{k}"] = metrics.get(f"recall@{k}")
        row[f"hole_rate@{k}"] = hole_rates.get(k)
    for k in ("100", "1000"):
        row[f"ndcg@{k}"] = metrics.get(f"ndcg@{k}")
        row[f"precision@{k}"] = metrics.get(f"precision@{k}")
        row[f"recall@{k}"] = metrics.get(f"recall@{k}")
        row[f"hole_rate@{k}"] = hole_rates.get(k)
    row["map"] = metrics.get("map")
    row["bpref"] = metrics.get("bpref")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Track B: re-score retrieval configs")
    parser.add_argument("--config", default=str(EVAL_CONFIG_DIR / "judge.yaml"))
    parser.add_argument(
        "--runs-dir",
        default=str(TRACK_B_RUNS_DIR),
        help="Process every TREC file found here (ignored if --trec-file given)",
    )
    parser.add_argument(
        "--trec-file",
        action="append",
        default=None,
        help="Restrict to specific TREC run file(s); repeatable",
    )
    parser.add_argument("--qrels-path", default=str(QREL_TASK1_2022_ALL))
    parser.add_argument("--topics-path", default=str(TOPICS_JSONL))
    parser.add_argument(
        "--fill-depth",
        type=int,
        default=DEFAULT_FILL_DEPTH,
        help="Shallow-fill boundary",
    )
    parser.add_argument(
        "--hole-rate-cutoffs",
        default=",".join(str(k) for k in DEFAULT_HOLE_RATE_CUTOFFS),
        help="Comma-separated k values for hole rate",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "sync"],
        default=None,
        help="Override execution.mode from config",
    )
    parser.add_argument("--output-dir", default=str(RESULTS_TRACK_B_DIR))
    args = parser.parse_args()

    config = JudgeConfigManager.from_yaml(args.config)
    mode = args.mode or config.execution.mode
    hole_rate_cutoffs = [int(x) for x in args.hole_rate_cutoffs.split(",")]

    # Track B logs into its own W&B group, same project, per the plan --
    # ExperimentLogger reads group/job_type off this WandbConfig instance.
    config.wandb.group = "Track B : Rescoring"
    config.wandb.job_type = "track-b-rescore"

    if args.trec_file:
        trec_files = [Path(p) for p in args.trec_file]
    else:
        trec_files = sorted(p for p in Path(args.runs_dir).iterdir() if p.is_file())
    if not trec_files:
        logger.error(
            f"No TREC files found (runs_dir={args.runs_dir}, trec_file={args.trec_file})"
        )
        return 1
    logger.info(f"Configs to process: {[f.name for f in trec_files]}")

    qrels = _parse_qrels(Path(args.qrels_path))
    topics = _load_topics(Path(args.topics_path))
    client = JudgeClient(config)
    config_sha256 = sha256_file(args.config)

    # One W&B run per config (run name = config name, tables/usage scoped to
    # just that config) -- NOT one combined run per invocation, so a config's
    # run never carries another config's rows, files, or spend.
    results = []
    for trec_file in trec_files:
        usage_before = dict(client.usage)
        result = process_config(
            trec_file=trec_file,
            qrels=qrels,
            client=client,
            topics=topics,
            mode=mode,
            poll_seconds=config.execution.batch_poll_seconds,
            hole_rate_cutoffs=hole_rate_cutoffs,
            fill_depth=args.fill_depth,
        )
        usage_delta = {k: client.usage[k] - usage_before[k] for k in client.usage}
        results.append(result)

        exp_logger = ExperimentLogger(config.wandb)
        manifest = build_manifest(
            config=config,
            execution_mode=mode,
            prompt_sha256=client.prompt_sha256,
            config_sha256=config_sha256,
            usage=usage_delta,
        )
        exp_logger.start_run(
            config=manifest,
            run_name=result["config_name"],
            extra_config={
                "fill_depth": args.fill_depth,
                "hole_rate_cutoffs": hole_rate_cutoffs,
            },
        )
        try:
            metrics_rows = [
                _metrics_table_row(
                    result["config_name"], channel, metrics, result["hole_rate"]
                )
                for channel, metrics in result["channels"].items()
            ]
            exp_logger.log_table("metrics_by_config", metrics_rows)
            exp_logger.log_table("answers_audit", result["audit_rows"])

            metrics_to_log = {}
            for k, v in result["hole_rate"].items():
                metrics_to_log[f"rescoring/hole_rate@{k}"] = v
            for channel, metrics in result["channels"].items():
                for metric_name in ("ndcg@5", "ndcg@10", "ndcg@20"):
                    if metric_name in metrics:
                        metrics_to_log[f"rescoring/{channel}/{metric_name}"] = metrics[
                            metric_name
                        ]
            exp_logger.log_metrics(metrics_to_log)

            exp_logger.save_file(trec_file)
        finally:
            exp_logger.finish()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "rescoring_results.json"
    with open(results_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "audit_rows"} for r in results],
            f,
            indent=2,
        )

    audit_path = output_dir / "answers_audit.jsonl"
    with open(audit_path, "w") as f:
        for r in results:
            for row in r["audit_rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print("Track B re-scoring")
    print(f"{'=' * 70}")
    for r in results:
        print(f"\n[{r['config_name']}]")
        print(f"  hole_rate: {r['hole_rate']}")
        print(f"  holes in top-{args.fill_depth}: {r['n_holes_in_topN']}")
        for channel, metrics in r["channels"].items():
            keys = (
                ["ndcg@5", "ndcg@10", "ndcg@20"]
                if channel != "human_only"
                else [
                    "ndcg@5",
                    "ndcg@10",
                    "ndcg@20",
                    "ndcg@100",
                ]
            )
            shown = {k: round(metrics.get(k, 0.0), 4) for k in keys}
            print(f"  {channel:<15} {shown}")
    print(f"\nWritten to {results_path}")
    print(f"Written to {audit_path}")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
