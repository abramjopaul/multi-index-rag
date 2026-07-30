#!/usr/bin/env python3
"""Track V: the two numbers -- label agreement and system-ranking preservation.

Computed strictly from paired_labels.jsonl (build_paired_labels.py's output),
not a separate path -- every kappa/tau number here is derived from that file.

1. Label agreement vs human qrels: Cohen's kappa + linear-weighted kappa,
   4x4 confusion matrix (graded 0-3), and the same on the H+M/L+N binary
   split. Parse failures are counted separately, never silently treated as
   agreement or disagreement.
2. System-ranking preservation (the load-bearing number): scores each year's
   ARQMath participant runs under human vs. judge qrels via the existing
   pytrec_eval wrapper, then computes Kendall's tau / Spearman's rho between
   the two system orderings (by nDCG'@10, ARQMath's official primary
   metric). Only computed for years where --participant-runs-dir was
   supplied for that year; other years are reported as
   SKIPPED (missing participant runs) rather than fabricated.
3. Segments agreement by topic_type / multi_approach when available (not
   available in this checkout -- see Track V plan's Context section; all
   rows currently fall into a single "unknown" segment, which is reported
   honestly rather than fabricated).

Usage:
    poetry run python experiments/track_v/agreement_analysis.py \\
        --participant-runs-dir 2020=data/runs/arqmath-2020-runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: E402

from multirag.config.judge_config import JudgeConfigManager  # noqa: E402
from multirag.config.path_configs import (
    ARQMATH_2020_RUNS_DIR,  # noqa: E402
    EVAL_CONFIG_DIR,
    QREL_TASK1_2020_ALL,
    QREL_TASK1_2021_ALL,
    QREL_TASK1_2022_ALL,
    RESULTS_TRACK_V_DIR,
)
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402
from multirag.evaluation.metrics import _parse_qrels, evaluate_run  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

_YEAR_TO_HUMAN_QRELS = {
    "2020": QREL_TASK1_2020_ALL,
    "2021": QREL_TASK1_2021_ALL,
    "2022": QREL_TASK1_2022_ALL,
}
_DEFAULT_RUNS_DIRS = {"2020": ARQMATH_2020_RUNS_DIR}


def _load_paired_labels(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def label_agreement(paired: list[dict]) -> dict:
    scored = [p for p in paired if p["judge_label"] is not None]
    n_parse_fail = len(paired) - len(scored)

    human_graded = [p["human_label"] for p in scored]
    judge_graded = [p["judge_label"] for p in scored]
    human_bin = [p["human_bin"] for p in scored]
    judge_bin = [p["judge_bin"] for p in scored]

    graded_labels = [0, 1, 2, 3]
    result = {
        "n_pairs": len(paired),
        "n_scored": len(scored),
        "n_parse_fail": n_parse_fail,
        "graded": {
            "kappa": cohen_kappa_score(human_graded, judge_graded) if scored else None,
            "weighted_kappa": (
                cohen_kappa_score(human_graded, judge_graded, weights="linear")
                if scored
                else None
            ),
            "confusion_matrix": (
                confusion_matrix(
                    human_graded, judge_graded, labels=graded_labels
                ).tolist()
                if scored
                else None
            ),
            "labels": graded_labels,
        },
        "binary": {
            "kappa": cohen_kappa_score(human_bin, judge_bin) if scored else None,
            "confusion_matrix": (
                confusion_matrix(human_bin, judge_bin, labels=[0, 1]).tolist()
                if scored
                else None
            ),
            "labels": [0, 1],
        },
    }

    # Segment by topic_type / multi_approach (honest "unknown" bucket if the
    # per-topic classification data isn't available locally -- see the plan's
    # Context section on this checkout's data gaps).
    segments: dict[str, list[dict]] = defaultdict(list)
    for p in scored:
        key = p.get("topic_type") or "unknown"
        segments[key].append(p)
    result["segments"] = {
        seg: {
            "n": len(rows),
            "graded_agreement": sum(1 for r in rows if r["agree_graded"]) / len(rows),
            "binary_agreement": sum(1 for r in rows if r["agree_binary"]) / len(rows),
        }
        for seg, rows in segments.items()
    }
    multi_approach_rows = [p for p in scored if p.get("multi_approach")]
    result["multi_approach_flagged"] = {
        "n": len(multi_approach_rows),
        "graded_agreement": (
            sum(1 for r in multi_approach_rows if r["agree_graded"])
            / len(multi_approach_rows)
            if multi_approach_rows
            else None
        ),
    }
    return result


def _merge_judge_qrels(
    paired: list[dict], year_prefix_check
) -> dict[str, dict[str, int]]:
    """Build a judge-labelled qrels dict (qid -> {doc_id: label}) restricted
    to pairs matching this year (by topic_id), for pytrec_eval scoring.
    """
    qrels: dict[str, dict[str, int]] = {}
    for p in paired:
        if p["judge_label"] is None:
            continue
        if not year_prefix_check(p["topic_id"]):
            continue
        qrels.setdefault(p["topic_id"], {})[p["answer_id"]] = p["judge_label"]
    return qrels


def system_ranking_preservation(
    paired: list[dict], participant_runs_dirs: dict[str, Path]
) -> dict:
    """For each year with a supplied runs dir: score every participant run
    under human qrels and under judge qrels (both restricted to that year's
    topics), rank systems by ndcg'@10 under each, and compute Kendall's tau /
    Spearman's rho between the two orderings.
    """
    results = {}
    for year in ("2020", "2021", "2022"):
        runs_dir = participant_runs_dirs.get(year)
        if runs_dir is None or not Path(runs_dir).exists():
            results[year] = {"status": "SKIPPED (missing participant runs)"}
            continue

        human_qrels_path = _YEAR_TO_HUMAN_QRELS[year]
        human_qrels = _parse_qrels(human_qrels_path)
        year_topic_ids = set(human_qrels.keys())
        judge_qrels = _merge_judge_qrels(paired, lambda tid: tid in year_topic_ids)

        run_files = sorted(p for p in Path(runs_dir).iterdir() if p.is_file())

        human_scores: dict[str, float] = {}
        judge_scores: dict[str, float] = {}
        with tempfile.TemporaryDirectory() as tmp:
            human_qrels_tmp = Path(tmp) / "human.qrels"
            judge_qrels_tmp = Path(tmp) / "judge.qrels"
            _write_qrels(human_qrels_tmp, human_qrels)
            _write_qrels(judge_qrels_tmp, judge_qrels)

            for run_file in run_files:
                human_metrics = evaluate_run(
                    qrels_path=human_qrels_tmp, run_path=run_file
                )
                judge_metrics = evaluate_run(
                    qrels_path=judge_qrels_tmp, run_path=run_file
                )
                human_scores[run_file.stem] = human_metrics.get("ndcg@10", 0.0)
                judge_scores[run_file.stem] = judge_metrics.get("ndcg@10", 0.0)

        systems = sorted(human_scores.keys())
        if len(systems) < 2:
            results[year] = {
                "status": f"SKIPPED (need >=2 participant runs, found {len(systems)})"
            }
            continue

        human_rank = [human_scores[s] for s in systems]
        judge_rank = [judge_scores[s] for s in systems]
        tau, tau_p = kendalltau(human_rank, judge_rank)
        rho, rho_p = spearmanr(human_rank, judge_rank)

        results[year] = {
            "status": "OK",
            "n_systems": len(systems),
            "kendall_tau": tau,
            "kendall_tau_p": tau_p,
            "spearman_rho": rho,
            "spearman_rho_p": rho_p,
            "human_ndcg10_by_system": human_scores,
            "judge_ndcg10_by_system": judge_scores,
        }
    return results


def _write_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> None:
    with open(path, "w") as f:
        for qid, docs in qrels.items():
            for doc_id, label in docs.items():
                f.write(f"{qid}\t0\t{doc_id}\t{label}\n")


def _parse_runs_dir_args(raw: list[str] | None) -> dict[str, Path]:
    """--participant-runs-dir YEAR=PATH, repeatable. Falls back to the
    checked-in default dirs (currently only 2020) for years not given.
    """
    dirs = dict(_DEFAULT_RUNS_DIRS)
    for item in raw or []:
        year, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--participant-runs-dir must be YEAR=PATH, got {item!r}")
        dirs[year] = Path(path)
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Track V agreement analysis")
    parser.add_argument("--config", default=str(EVAL_CONFIG_DIR / "judge.yaml"))
    parser.add_argument(
        "--paired-labels", default=str(RESULTS_TRACK_V_DIR / "paired_labels.jsonl")
    )
    parser.add_argument(
        "--participant-runs-dir",
        action="append",
        default=None,
        help="YEAR=PATH, repeatable, e.g. 2021=data/runs/arqmath-2021-runs. "
        "2020 defaults to data/runs/arqmath-2020-runs if not overridden.",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_TRACK_V_DIR / "agreement_analysis.json")
    )
    parser.add_argument(
        "--wandb-run-meta",
        default=str(RESULTS_TRACK_V_DIR / "judgements.jsonl"),
        help="File whose _meta row carries the wandb_run_id to resume",
    )
    args = parser.parse_args()

    config = JudgeConfigManager.from_yaml(args.config)
    paired = _load_paired_labels(Path(args.paired_labels))

    agreement = label_agreement(paired)
    runs_dirs = _parse_runs_dir_args(args.participant_runs_dir)
    ranking = system_ranking_preservation(paired, runs_dirs)

    output = {"label_agreement": agreement, "system_ranking_preservation": ranking}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Track V agreement analysis")
    print(f"{'=' * 60}")
    print(
        f"  n pairs: {agreement['n_pairs']} (parse fail: {agreement['n_parse_fail']})"
    )
    print(f"  Graded kappa:          {agreement['graded']['kappa']}")
    print(f"  Graded weighted kappa: {agreement['graded']['weighted_kappa']}")
    print(f"  Binary kappa:          {agreement['binary']['kappa']}")
    for year, r in ranking.items():
        if r["status"] == "OK":
            print(
                f"  {year}: tau={r['kendall_tau']:.4f} rho={r['spearman_rho']:.4f} "
                f"(n_systems={r['n_systems']})"
            )
        else:
            print(f"  {year}: {r['status']}")
    print(f"Written to {output_path}")
    print(f"{'=' * 60}\n")

    # --- W&B: resume the run started by run_judge_over_validation.py ------
    meta_row = {}
    meta_path = Path(args.wandb_run_meta)
    if meta_path.exists():
        with open(meta_path) as f:
            first_line = f.readline()
            if first_line.strip():
                candidate = json.loads(first_line)
                if candidate.get("_meta"):
                    meta_row = candidate

    exp_logger = ExperimentLogger(config.wandb)
    exp_logger.start_run(
        config=meta_row,
        run_name=meta_row.get("run_name", "V-agreement-analysis"),
        run_id=meta_row.get("wandb_run_id"),
    )
    metrics_to_log = {
        "agreement/kappa_graded": agreement["graded"]["kappa"],
        "agreement/kappa_weighted_graded": agreement["graded"]["weighted_kappa"],
        "agreement/kappa_binary": agreement["binary"]["kappa"],
        "agreement/n_parse_fail": agreement["n_parse_fail"],
    }
    for year, r in ranking.items():
        if r["status"] == "OK":
            metrics_to_log[f"ranking/{year}_kendall_tau"] = r["kendall_tau"]
            metrics_to_log[f"ranking/{year}_spearman_rho"] = r["spearman_rho"]
    exp_logger.log_metrics(metrics_to_log)

    if agreement["graded"]["confusion_matrix"] is not None:
        human_graded_full = [
            p["human_label"] for p in paired if p["judge_label"] is not None
        ]
        judge_graded_full = [
            p["judge_label"] for p in paired if p["judge_label"] is not None
        ]
        exp_logger.log_confusion_matrix(
            "confusion_matrix_graded",
            human_graded_full,
            judge_graded_full,
            labels=[0, 1, 2, 3],
        )
        human_bin_full = [p["human_bin"] for p in paired if p["judge_bin"] is not None]
        judge_bin_full = [p["judge_bin"] for p in paired if p["judge_bin"] is not None]
        exp_logger.log_confusion_matrix(
            "confusion_matrix_binary", human_bin_full, judge_bin_full, labels=[0, 1]
        )

    exp_logger.log_table(
        "segment_breakdown",
        [{"segment": seg, **stats} for seg, stats in agreement["segments"].items()],
    )
    # Intentionally no finish() here -- report.py closes the run.

    return 0


if __name__ == "__main__":
    sys.exit(main())
