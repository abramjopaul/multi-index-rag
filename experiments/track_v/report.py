#!/usr/bin/env python3
"""Track V: final verdict report.

Compares this judge's kappa/tau against UMBRELA's published TREC-DL reference
numbers (kappa ~ 0.3-0.5, tau ~ 0.8-0.9) and states that tau -- not kappa --
is decisive: ranking preservation is what licenses using the judge to
compare our own configs, and math relevance judging is expected to be
harder (lower kappa) than the Bing web-relevance task UMBRELA was validated
on.

Emits JUDGE_VALIDATED / JUDGE_MARGINAL / JUDGE_REJECTED, scoped to the years
that have complete tau/rho data (participant runs supplied) -- label
agreement (kappa) is still reported for every year regardless. A year with
SKIPPED ranking data is never silently folded into the verdict.

This is the last script in the V pipeline: it closes the W&B run opened by
run_judge_over_validation.py.

Usage:
    poetry run python experiments/track_v/report.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

# UMBRELA's published TREC-DL reference range (Upadhyay et al. 2024) -- a
# directional reference point, not a pass/fail bar in itself. Math relevance
# judging is expected to be harder (lower kappa) than Bing web relevance;
# tau is what's actually decisive here.
_UMBRELA_KAPPA_RANGE = (0.3, 0.5)
_UMBRELA_TAU_RANGE = (0.8, 0.9)

# Verdict thresholds on mean Kendall's tau across years with complete
# participant-run data. Chosen thresholds, not a received standard: high tau
# preservation licenses judge-based hole-filling for our own config
# comparisons; low tau means the judge would reorder our systems relative to
# ground truth, which is disqualifying regardless of raw label agreement.
_TAU_VALIDATED = 0.7
_TAU_MARGINAL = 0.4


def compute_verdict(ranking: dict) -> tuple[str, dict]:
    complete_years = {year: r for year, r in ranking.items() if r.get("status") == "OK"}
    if not complete_years:
        return "INSUFFICIENT_DATA", {
            "reason": "No year has participant-run data supplied yet -- tau/rho "
            "cannot be computed. Verdict withheld rather than fabricated."
        }

    mean_tau = sum(r["kendall_tau"] for r in complete_years.values()) / len(
        complete_years
    )
    if mean_tau >= _TAU_VALIDATED:
        verdict = "JUDGE_VALIDATED"
    elif mean_tau >= _TAU_MARGINAL:
        verdict = "JUDGE_MARGINAL"
    else:
        verdict = "JUDGE_REJECTED"

    return verdict, {
        "mean_kendall_tau": mean_tau,
        "years_used": sorted(complete_years.keys()),
        "years_skipped": sorted(set(ranking.keys()) - set(complete_years.keys())),
    }


def render_markdown(agreement_data: dict, verdict: str, verdict_detail: dict) -> str:
    la = agreement_data["label_agreement"]
    ranking = agreement_data["system_ranking_preservation"]

    lines = [
        "# Track V: LLM-Judge Validation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        f"- Mean Kendall's tau (years with participant runs): "
        f"{verdict_detail.get('mean_kendall_tau', 'N/A')}",
        f"- Years used: {verdict_detail.get('years_used', [])}",
        f"- Years skipped (no participant runs supplied yet): "
        f"{verdict_detail.get('years_skipped', verdict_detail.get('reason', ''))}",
        "",
        "Ranking preservation (tau), not raw label agreement (kappa), is decisive: "
        "it is what licenses using the judge to compare our own retrieval configs. "
        "Math relevance judging is expected to be harder than the Bing web-relevance "
        "task UMBRELA was validated on, so a lower kappa here is not itself "
        "disqualifying.",
        "",
        "## Label agreement",
        "",
        f"- n pairs: {la['n_pairs']} (parse failures: {la['n_parse_fail']}, "
        f"scored: {la['n_scored']})",
        f"- Graded Cohen's kappa: {la['graded']['kappa']}",
        f"- Graded weighted (linear) kappa: {la['graded']['weighted_kappa']}",
        f"- Binary (H+M vs L+N) kappa: {la['binary']['kappa']}",
        "",
        f"UMBRELA reference (TREC-DL, Bing web relevance): "
        f"kappa ~ {_UMBRELA_KAPPA_RANGE[0]}-{_UMBRELA_KAPPA_RANGE[1]}, "
        f"tau ~ {_UMBRELA_TAU_RANGE[0]}-{_UMBRELA_TAU_RANGE[1]}.",
        "",
        "### Graded confusion matrix (rows=human 0-3, cols=judge 0-3)",
        "",
        "```",
    ]
    cm = la["graded"]["confusion_matrix"]
    if cm is not None:
        for row in cm:
            lines.append(" ".join(f"{v:>6}" for v in row))
    lines.append("```")
    lines.append("")
    lines.append("### Binary confusion matrix (rows=human 0/1, cols=judge 0/1)")
    lines.append("")
    lines.append("```")
    cm_bin = la["binary"]["confusion_matrix"]
    if cm_bin is not None:
        for row in cm_bin:
            lines.append(" ".join(f"{v:>6}" for v in row))
    lines.append("```")
    lines.append("")

    lines.append("### Segment breakdown")
    lines.append("")
    lines.append("| segment | n | graded agreement | binary agreement |")
    lines.append("|---|---|---|---|")
    for seg, stats in la["segments"].items():
        lines.append(
            f"| {seg} | {stats['n']} | {stats['graded_agreement']:.4f} | "
            f"{stats['binary_agreement']:.4f} |"
        )
    lines.append("")
    ma = la["multi_approach_flagged"]
    lines.append(
        f"Multi-approach-flagged topics: n={ma['n']}, graded agreement="
        f"{ma['graded_agreement']}"
    )
    lines.append("")

    lines.append("## System-ranking preservation")
    lines.append("")
    lines.append("| year | status | n systems | Kendall tau | Spearman rho |")
    lines.append("|---|---|---|---|---|")
    for year in sorted(ranking.keys()):
        r = ranking[year]
        if r["status"] == "OK":
            lines.append(
                f"| {year} | OK | {r['n_systems']} | {r['kendall_tau']:.4f} | "
                f"{r['spearman_rho']:.4f} |"
            )
        else:
            lines.append(f"| {year} | {r['status']} | - | - | - |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Track V final verdict report")
    parser.add_argument("--config", default=str(EVAL_CONFIG_DIR / "judge.yaml"))
    parser.add_argument(
        "--agreement-analysis",
        default=str(RESULTS_TRACK_V_DIR / "agreement_analysis.json"),
    )
    parser.add_argument(
        "--wandb-run-meta",
        default=str(RESULTS_TRACK_V_DIR / "judgements.jsonl"),
        help="File whose _meta row carries the wandb_run_id to resume",
    )
    parser.add_argument(
        "--output-jsonl", default=str(RESULTS_TRACK_V_DIR / "agreement.jsonl")
    )
    parser.add_argument("--output-md", default=str(RESULTS_TRACK_V_DIR / "report.md"))
    args = parser.parse_args()

    config = JudgeConfigManager.from_yaml(args.config)

    with open(args.agreement_analysis) as f:
        agreement_data = json.load(f)

    verdict, verdict_detail = compute_verdict(
        agreement_data["system_ranking_preservation"]
    )

    output_record = {
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **agreement_data,
    }
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    md = render_markdown(agreement_data, verdict, verdict_detail)
    output_md = Path(args.output_md)
    with open(output_md, "w") as f:
        f.write(md)

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 60}")
    print(json.dumps(verdict_detail, indent=2, default=str))
    print(f"Written to {output_jsonl}")
    print(f"Written to {output_md}")
    print(f"{'=' * 60}\n")

    # --- W&B: resume + close the run opened by run_judge_over_validation.py --
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
        run_name=meta_row.get("run_name", "V-report"),
        run_id=meta_row.get("wandb_run_id"),
    )
    try:
        la = agreement_data["label_agreement"]
        exp_logger.log_metrics(
            {
                "final/kappa_graded": la["graded"]["kappa"],
                "final/kappa_weighted_graded": la["graded"]["weighted_kappa"],
                "final/kappa_binary": la["binary"]["kappa"],
                "final/parse_fail_count": la["n_parse_fail"],
            }
        )
        exp_logger.log_artifact(output_jsonl, artifact_type="agreement_report")
        exp_logger.log_artifact(output_md, artifact_type="agreement_report")
        exp_logger.set_summary("verdict", verdict)
    finally:
        exp_logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
