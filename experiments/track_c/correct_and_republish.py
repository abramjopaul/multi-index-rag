#!/usr/bin/env python3
"""Post-hoc correction of Track C's refusal-detection + faithfulness/
factual_correctness_precision bugs -- reads already-logged W&B runs, applies
the fix in-memory, republishes a corrected copy to a new W&B group. No
regeneration, no re-judging: zero LLM/API cost.

Bugs fixed (see the investigation this script implements):
1. detect_refusal() (multirag.eval.generate) does a substring-anywhere check
   against the exact canonical refusal sentence, which both undercounts
   paraphrased refusals (5-10x, confirmed) and false-positives on substantive
   answers that merely append the sentence as a trailing caveat. Replaced
   here with a prefix check on the disclaimer opening.
2. faithfulness / factual_correctness_precision are not corrected for
   responses the (fixed) detector now flags as refusals -- ragas often scores
   a refusal-shaped response highly (it's trivially "faithful" if it merely
   paraphrases context instead of answering). Forced to 0.0 for those rows.
   factual_correctness_recall is NEVER forced -- it already behaves
   correctly on refusals (low, not 1.0) and is left as ragas computed it.

Only rows/runs actually affected change; everything else (contexts,
response text, reference construction, claim counts, non-refusal scores) is
carried through unchanged.

Usage:
    poetry run python experiments/track_c/correct_and_republish.py --dry-run
    poetry run python experiments/track_c/correct_and_republish.py
    poetry run python experiments/track_c/correct_and_republish.py \\
        --run-name C-bm25-k5-20260821-094243 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.judge_config import WandbConfig  # noqa: E402
from multirag.eval.generate import GenerationRecord  # noqa: E402
from multirag.eval.references import select_references  # noqa: E402
from multirag.eval.runner import build_report  # noqa: E402
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402
from multirag.generation.sample_builder import build_answer_lookup, load_qrels  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

ENTITY_PROJECT = "abramjopaul-abram/one-last-run"
SOURCE_GROUP = "Track C : Generation"
REFUSAL_PREFIX = "the provided context does not contain enough information to"
METRIC_NAMES = ["faithfulness", "factual_correctness_precision", "factual_correctness_recall"]
FORCED_ON_REFUSAL = {"faithfulness", "factual_correctness_precision"}
STANDARD_FILES = ["generations.jsonl", "scores_per_sample.jsonl", "manifest.json", "report.md", "summary.json"]


def is_refusal_v2(response: str) -> bool:
    return (response or "").strip().lower().startswith(REFUSAL_PREFIX)


def _clean(v):
    """Normalize the pandas-NaN-from-skip artifact (a metric that was never
    computed for this row, e.g. no context / no reference) to None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def _local_dir_has_data(ldir: Path | None) -> bool:
    return (
        ldir is not None
        and (ldir / "generations.jsonl").exists()
        and (ldir / "scores_per_sample.jsonl").exists()
    )


def select_runs(api, group: str, local_root: Path, only_names: list[str] | None) -> list:
    runs = list(api.runs(ENTITY_PROJECT, filters={"group": group}))
    if only_names:
        runs = [r for r in runs if r.name in only_names]
        return runs

    candidates = []
    for r in runs:
        name_lower = r.name.lower()
        if "test" in name_lower or "smoketest" in name_lower:
            logger.info(f"  skip (test/smoketest name): {r.name}")
            continue
        n_topics = r.summary.get("n_topics")
        if r.summary.get("per_topic") is None:
            # The W&B upload can fail after local files were already written
            # (confirmed: fuse_bm25_dense_formula_fused_block k=10 finished
            # with a complete local results/track_c/ dir but zero W&B data).
            # Recoverable via local reconstruction -- only truly skip if
            # there's nothing usable on disk either.
            ldir = local_dir_for(r.name, local_root)
            if not _local_dir_has_data(ldir):
                logger.info(f"  skip (no per_topic table logged, no local fallback): {r.name}")
                continue
            n_topics = sum(1 for _ in open(ldir / "generations.jsonl"))
            logger.info(f"  recovering from local files (no per_topic table in W&B): {r.name}")
        if n_topics is None or n_topics < 10:
            logger.info(f"  skip (n_topics={n_topics} < 10): {r.name}")
            continue
        candidates.append(r)

    # dedupe by (config_name, k), keep latest created_at
    by_ck: dict[tuple, list] = {}
    for r in candidates:
        key = (r.config.get("config_name"), r.config.get("k"))
        by_ck.setdefault(key, []).append(r)
    kept = []
    for key, dupes in by_ck.items():
        dupes.sort(key=lambda r: r.created_at)
        for d in dupes[:-1]:
            logger.info(f"  skip (superseded duplicate of {key}): {d.name}")
        kept.append(dupes[-1])
    return kept


# ---------------------------------------------------------------------------
# Source-file loading (local disk preferred, W&B download fallback)
# ---------------------------------------------------------------------------

import re  # noqa: E402

_RUN_NAME_RE = re.compile(r"^C-(?P<config>.+)-k(?P<k>\d+)-(?P<ts>\d{8}-\d{6})$")


def local_dir_for(run_name: str, local_root: Path) -> Path | None:
    m = _RUN_NAME_RE.match(run_name)
    if not m:
        return None
    candidate = local_root / f"{m['config']}_k{m['k']}_{m['ts']}"
    return candidate if candidate.is_dir() else None


def load_source_files(run, local_root: Path, download_root: Path) -> dict[str, Path]:
    """Returns {filename: local_path} for the 5 standard files + the TREC
    run file (if any) + generation_eval.yaml. Prefers local disk."""
    ldir = local_dir_for(run.name, local_root)
    paths: dict[str, Path] = {}

    if ldir is not None:
        logger.info(f"  using local files: {ldir}")
        for fname in STANDARD_FILES:
            p = ldir / fname
            if p.exists():
                paths[fname] = p
    missing_standard = [f for f in STANDARD_FILES if f not in paths]
    if missing_standard:
        dl_dir = download_root / run.id
        for fname in missing_standard:
            try:
                f = run.file(fname)
                f.download(root=str(dl_dir), replace=True)
                paths[fname] = dl_dir / fname
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  could not obtain {fname} for {run.name}: {e}")

    # TREC run file: prefer manifest's own path if it still exists on disk
    manifest = json.loads(paths["manifest.json"].read_text()) if "manifest.json" in paths else {}
    trec_file = manifest.get("trec_file")
    if trec_file and Path(trec_file).exists():
        paths["_trec_file"] = Path(trec_file)
        paths["_trec_file_name"] = Path(trec_file).name
    elif trec_file:
        dl_dir = download_root / run.id
        try:
            f = run.file(Path(trec_file).name)
            f.download(root=str(dl_dir), replace=True)
            paths["_trec_file"] = dl_dir / Path(trec_file).name
            paths["_trec_file_name"] = Path(trec_file).name
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  could not obtain TREC file for {run.name}: {e}")

    # generation_eval*.yaml: always fetch the exact one attached to this run
    dl_dir = download_root / run.id
    yaml_candidates = [f.name for f in run.files() if f.name.endswith(".yaml")]
    for yname in yaml_candidates:
        try:
            f = run.file(yname)
            f.download(root=str(dl_dir), replace=True)
            paths[f"_yaml:{yname}"] = dl_dir / yname
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  could not obtain {yname} for {run.name}: {e}")

    return paths, manifest


_qrels_cache: dict[str, dict] = {}
_answer_lookup_cache: dict | None = None


def build_per_topic_from_local(gens: list[dict], scores_rows: list[dict], manifest: dict) -> list[dict]:
    """Reconstruct a per_topic table from local generations.jsonl +
    scores_per_sample.jsonl for a run whose W&B upload never happened.
    Everything except the reference-construction fields (n_high_available,
    n_high_used, fallback_used, reference_char_len, reference_token_len) is
    already in those two files; those five are recomputed via the same
    deterministic select_references() the original run used (same qrels,
    same max_reference_answers from the manifest) -- pure local computation,
    no LLM/API call. reference_token_len is left 0 (normally measured
    against the judge's tokenizer via a free but network-dependent
    count_tokens call, skipped here to keep this a zero-network recovery).
    """
    global _answer_lookup_cache
    qrels_path = manifest["qrels_path"]
    if qrels_path not in _qrels_cache:
        _qrels_cache[qrels_path] = load_qrels(Path(qrels_path))
    qrels = _qrels_cache[qrels_path]
    if _answer_lookup_cache is None:
        from multirag.config.path_configs import ANSWERS_JSONL
        _answer_lookup_cache = build_answer_lookup(ANSWERS_JSONL)
    answer_lookup = _answer_lookup_cache
    max_ref = manifest["max_reference_answers"]

    scores_by_topic = {s["topic_id"]: s for s in scores_rows}
    rows = []
    for g in gens:
        s = scores_by_topic.get(g["topic_id"], {})
        ref = select_references(g["topic_id"], qrels, answer_lookup, max_ref, token_counter=None)
        row = {
            "topic_id": g["topic_id"],
            "is_refusal": g["is_refusal"],
            "finish_reason": g["finish_reason"],
            "completion_tokens": g["completion_tokens"],
            "max_new_tokens": g["max_new_tokens"],
            "hit_ceiling": g["hit_ceiling"],
            "response": g["response"],
            "n_contexts": len(g["contexts"]),
            "n_high_available": ref.n_high_available,
            "n_high_used": ref.n_high_used,
            "fallback_used": ref.fallback_used,
            "reference_char_len": ref.reference_char_len,
            "reference_token_len": ref.reference_token_len,
        }
        for k in ("faithfulness", "factual_correctness_precision", "factual_correctness_recall", "n_claims"):
            if k in s:
                row[k] = s[k]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

def correct_per_topic_rows(rows: list[dict], has_faithfulness_col: bool) -> tuple[list[dict], int, int]:
    """Returns (corrected_rows, n_refusal_true_to_false, n_refusal_false_to_true)."""
    corrected = []
    flips_off, flips_on = 0, 0
    for row in rows:
        row = dict(row)
        old_refusal = bool(row.get("is_refusal"))
        new_refusal = is_refusal_v2(row.get("response", ""))
        if old_refusal and not new_refusal:
            flips_off += 1
        elif not old_refusal and new_refusal:
            flips_on += 1
        row["is_refusal"] = new_refusal
        row["is_refusal_changed"] = new_refusal != old_refusal

        if has_faithfulness_col and "faithfulness" in row:
            row["faithfulness"] = 0.0 if new_refusal else _clean(row.get("faithfulness"))
        if "factual_correctness_precision" in row:
            row["factual_correctness_precision"] = (
                0.0 if new_refusal else _clean(row.get("factual_correctness_precision"))
            )
        if "factual_correctness_recall" in row:
            row["factual_correctness_recall"] = _clean(row.get("factual_correctness_recall"))
        corrected.append(row)
    return corrected, flips_off, flips_on


def recompute_summary(generations: list[dict], scores_by_topic: dict[str, dict], original_summary: dict) -> dict:
    """Mirrors build_summary()'s exact arithmetic (src/multirag/eval/runner.py:330-384),
    re-expressed over corrected rows instead of GenerationRecord/ScoreResult objects."""
    n_topics = len(generations)
    n_refusals = sum(1 for g in generations if g["is_refusal"])
    refusal_rate = round(n_refusals / n_topics, 4) if n_topics else 0.0

    summary = dict(original_summary)
    summary["n_refusals"] = n_refusals
    summary["refusal_rate"] = refusal_rate

    for metric_name in METRIC_NAMES:
        if not any(metric_name in scores_by_topic.get(g["topic_id"], {}) for g in generations):
            continue  # this run never had this metric (e.g. faithfulness for no_rag)
        all_vals, answering_vals = [], []
        for g in generations:
            s = scores_by_topic.get(g["topic_id"], {})
            if metric_name not in s:
                continue
            v = s[metric_name]
            if v is None:
                continue
            all_vals.append(v)
            if not g["is_refusal"]:
                answering_vals.append(v)
        summary[f"{metric_name}_mean_all"] = round(sum(all_vals) / len(all_vals), 4) if all_vals else None
        summary[f"{metric_name}_mean_answering"] = (
            round(sum(answering_vals) / len(answering_vals), 4) if answering_vals else None
        )
        summary[f"{metric_name}_n_scored"] = len(all_vals)

    return summary


# ---------------------------------------------------------------------------
# Main per-run processing
# ---------------------------------------------------------------------------

def process_run(run, local_root: Path, download_root: Path, dry_run: bool, new_group: str, corrected_root: Path, api) -> dict:
    logger.info(f"=== {run.name} ===")

    paths, manifest = load_source_files(run, local_root, download_root)

    gens = [json.loads(line) for line in paths["generations.jsonl"].read_text().splitlines() if line.strip()]
    scores_rows = [json.loads(line) for line in paths["scores_per_sample.jsonl"].read_text().splitlines() if line.strip()]
    original_summary = json.loads(paths["summary.json"].read_text())

    # per_topic table: W&B has one for almost every run; reconstruct locally
    # for the rare run whose upload never happened (local files still exist).
    if run.summary.get("per_topic") is not None:
        table_path = run.summary["per_topic"]["path"]
        dl_dir = download_root / run.id
        f = run.file(table_path)
        f.download(root=str(dl_dir), replace=True)
        with open(dl_dir / table_path) as fh:
            table = json.load(fh)
        cols = table["columns"]
        idx = {c: i for i, c in enumerate(cols)}
        per_topic_rows = [{c: row[idx[c]] for c in cols} for row in table["data"]]
        has_faithfulness_col = "faithfulness" in cols
    else:
        logger.info("  no per_topic table in W&B -- reconstructing from local files")
        per_topic_rows = build_per_topic_from_local(gens, scores_rows, manifest)
        has_faithfulness_col = any("faithfulness" in row for row in per_topic_rows)

    # --- correct all three sources of truth consistently ---
    per_topic_corrected, pt_off, pt_on = correct_per_topic_rows(per_topic_rows, has_faithfulness_col)

    for g in gens:
        g["is_refusal"] = is_refusal_v2(g.get("response", ""))

    # scores_per_sample.jsonl doesn't carry response text -- derive refusal
    # status from the matching (already-corrected) generations.jsonl row.
    gens_by_topic = {g["topic_id"]: g for g in gens}
    scores_corrected = []
    for row in scores_rows:
        row = dict(row)
        g = gens_by_topic.get(row["topic_id"])
        new_refusal = g["is_refusal"] if g else bool(row.get("is_refusal"))
        scores_corrected.append(row)
        row["is_refusal"] = new_refusal
        if has_faithfulness_col and "faithfulness" in row:
            row["faithfulness"] = 0.0 if new_refusal else _clean(row.get("faithfulness"))
        if "factual_correctness_precision" in row:
            row["factual_correctness_precision"] = 0.0 if new_refusal else _clean(row.get("factual_correctness_precision"))
        if "factual_correctness_recall" in row:
            row["factual_correctness_recall"] = _clean(row.get("factual_correctness_recall"))

    scores_by_topic = {r["topic_id"]: r for r in scores_corrected}
    new_summary = recompute_summary(gens, scores_by_topic, original_summary)

    gen_records = [GenerationRecord(**{k: v for k, v in g.items() if k in GenerationRecord.__dataclass_fields__}) for g in gens]
    new_report = build_report(manifest, new_summary, gen_records)

    comparison = {
        "run_name": run.name,
        "config_name": run.config.get("config_name"),
        "k": run.config.get("k"),
        "n_topics": len(gens),
        "refusal_rate_old": original_summary.get("refusal_rate"),
        "refusal_rate_new": new_summary.get("refusal_rate"),
        "n_flipped_refusal_to_answering": pt_off,
        "n_flipped_answering_to_refusal": pt_on,
        "faithfulness_mean_all_old": original_summary.get("faithfulness_mean_all"),
        "faithfulness_mean_all_new": new_summary.get("faithfulness_mean_all"),
        "factual_correctness_precision_mean_all_old": original_summary.get("factual_correctness_precision_mean_all"),
        "factual_correctness_precision_mean_all_new": new_summary.get("factual_correctness_precision_mean_all"),
    }

    if dry_run:
        logger.info(f"  [dry-run] {comparison}")
        return comparison

    # --- write corrected local mirror ---
    out_dir = corrected_root / (local_dir_for(run.name, local_root).name if local_dir_for(run.name, local_root) else run.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "generations.jsonl", "w") as fh:
        for g in gens:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(out_dir / "scores_per_sample.jsonl", "w") as fh:
        for r in scores_corrected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(new_summary, fh, indent=2)
    with open(out_dir / "report.md", "w") as fh:
        fh.write(new_report)
    shutil.copy(paths["manifest.json"], out_dir / "manifest.json")
    if "_trec_file" in paths:
        shutil.copy(paths["_trec_file"], out_dir / paths["_trec_file_name"])
    for key, path in paths.items():
        if key.startswith("_yaml:"):
            shutil.copy(path, out_dir / key.split(":", 1)[1])

    # --- publish to the new W&B group ---
    wandb_config = WandbConfig(
        group=new_group,
        job_type="track-c-generation-corrected",
        tags=list(run.tags) + ["corrected"],
    )
    exp_logger = ExperimentLogger(wandb_config)
    new_config = dict(run.config)
    new_config.update({
        "corrected_from_run_id": run.id,
        "corrected_from_run_name": run.name,
        "correction_applied_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    # Resume the existing corrected run if this run_name was already published
    # (idempotent re-run / patching a missed field) -- never create a duplicate.
    existing = [
        r for r in api.runs(ENTITY_PROJECT, filters={"group": new_group})
        if r.name == run.name
    ]
    existing_run_id = existing[0].id if existing else None
    exp_logger.start_run(config=new_config, run_name=run.name, run_id=existing_run_id)
    try:
        exp_logger.log_table("per_topic", per_topic_corrected)
        exp_logger.log_table(
            "metrics_summary",
            [
                {
                    "metric": name,
                    "mean_answering": new_summary.get(f"{name}_mean_answering"),
                    "mean_all": new_summary.get(f"{name}_mean_all"),
                    "n_scored": new_summary.get(f"{name}_n_scored"),
                }
                for name in METRIC_NAMES
            ]
            + [
                # Refusal count/rate surfaced directly in this table (not just
                # buried in per_topic's is_refusal column) -- requested
                # explicitly since it's the single most informative number
                # this correction changes.
                {
                    "metric": "n_refusals",
                    "mean_answering": None,
                    "mean_all": new_summary.get("n_refusals"),
                    "n_scored": new_summary.get("n_topics"),
                },
                {
                    "metric": "refusal_rate",
                    "mean_answering": None,
                    "mean_all": new_summary.get("refusal_rate"),
                    "n_scored": new_summary.get("n_topics"),
                },
            ],
        )
        numeric_summary = {k: v for k, v in new_summary.items() if isinstance(v, (int, float)) and v is not None}
        numeric_summary["usage_cache_hits"] = new_summary.get("usage", {}).get("cache_hits")
        numeric_summary["usage_new_calls"] = new_summary.get("usage", {}).get("new_calls")
        exp_logger.log_metrics(numeric_summary)
        for fname in ["generations.jsonl", "scores_per_sample.jsonl", "summary.json", "report.md"]:
            exp_logger.save_file(out_dir / fname)
        exp_logger.save_file(out_dir / "manifest.json")
        if "_trec_file" in paths:
            exp_logger.save_file(out_dir / paths["_trec_file_name"])
        for key, path in paths.items():
            if key.startswith("_yaml:"):
                exp_logger.save_file(out_dir / key.split(":", 1)[1])
    finally:
        exp_logger.finish()

    logger.info(f"  published -> group={new_group!r} run={run.name!r}")
    return comparison


def main() -> int:
    import wandb

    parser = argparse.ArgumentParser(description="Correct Track C refusal/faithfulness/factual_correctness bugs, republish to a new W&B group")
    parser.add_argument("--group", default=SOURCE_GROUP)
    parser.add_argument("--new-group", default=f"{SOURCE_GROUP} (corrected)")
    parser.add_argument("--run-name", action="append", default=None, help="Restrict to specific original run name(s); repeatable")
    parser.add_argument("--local-root", default="results/track_c")
    parser.add_argument("--corrected-root", default="results/track_c_corrected")
    parser.add_argument("--corrections-report-dir", default="results/track_c_corrections")
    parser.add_argument("--download-root", default="/tmp/track_c_correction_downloads")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    local_root = Path(args.local_root)
    corrected_root = Path(args.corrected_root)
    download_root = Path(args.download_root)

    runs = select_runs(api, args.group, local_root, args.run_name)
    logger.info(f"Processing {len(runs)} runs (dry_run={args.dry_run})")

    comparisons = []
    for run in runs:
        comparisons.append(process_run(run, local_root, download_root, args.dry_run, args.new_group, corrected_root, api))

    report_dir = Path(args.corrections_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "correction_report.json", "w") as fh:
        json.dump(comparisons, fh, indent=2)

    lines = ["# Track C correction report", "", f"{len(comparisons)} runs processed.", "",
             "| run | refusal_rate old->new | faithfulness_mean_all old->new | fc_precision_mean_all old->new |",
             "|---|---|---|---|"]
    for c in comparisons:
        lines.append(
            f"| {c['run_name']} | {c['refusal_rate_old']}->{c['refusal_rate_new']} "
            f"| {c['faithfulness_mean_all_old']}->{c['faithfulness_mean_all_new']} "
            f"| {c['factual_correctness_precision_mean_all_old']}->{c['factual_correctness_precision_mean_all_new']} |"
        )
    (report_dir / "correction_report.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
