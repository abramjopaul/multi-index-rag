#!/usr/bin/env python3
"""Stage-2 formula-aware reranker experiment runner.

Takes a pre-computed stage-1 TREC run file (e.g. from fuse_runs.py) and
rescores the top-N candidates per topic by blending the original text score
with a MaxSim formula-structural relevance score computed from FastText
tuple embeddings. Writes a new TREC run file and evaluates it.

Usage:
    poetry run python experiments/rerank_run.py configs/experiments/rerank_formula_slt.yaml
    poetry run python experiments/rerank_run.py configs/experiments/rerank_formula_slt.yaml --dry-run
    poetry run python experiments/rerank_run.py configs/experiments/rerank_formula_slt.yaml --verbose
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import wandb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logging_config import configure_logging

from multirag.config import RerankerConfig, RerankerConfigManager
from multirag.config.path_configs import (
    ANSWERS_JSONL,
    FORMULA_DIR,
    FORMULA_EMBEDDING_DIR,
    QREL_TASK1_2022_OFFICIAL,
    RUNS_DIR,
    TOPICS_JSONL,
)
from multirag.evaluation.metrics import (
    _is_clearly_trivial,
    _parse_run,
    evaluate_run,
    print_evaluation_report,
)
from multirag.reranking import FormulaMaxSimReranker

configure_logging()
logger = logging.getLogger(__name__)


def _load_topics(topics_path: Path) -> list[dict]:
    topics = []
    with open(topics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                topics.append(json.loads(line))
    return topics


def _load_answer_formulas(
    answers_path: Path,
    candidate_ids: set[str],
) -> dict[str, list[tuple[str, str]]]:
    """Stream answers.jsonl and return non-trivial formulas for candidate posts.

    Returns:
        {answer_id: [(formula_id, latex), ...]} — only non-trivial formulas, only
        for posts that appear in candidate_ids.
    """
    answer_formulas: dict[str, list[tuple[str, str]]] = {}
    remaining = set(candidate_ids)
    with open(answers_path) as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            try:
                answer = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = answer.get("id")
            if aid not in remaining:
                continue
            remaining.discard(aid)
            non_trivial = [
                (str(formula_obj.get("formula_id", "")), formula_obj["latex"])
                for formula_obj in answer.get("formulas", [])
                if not _is_clearly_trivial(formula_obj["latex"])
            ]
            answer_formulas[aid] = non_trivial
    if remaining:
        logger.warning("%d candidate doc_ids not found in answers.jsonl", len(remaining))
    return answer_formulas


def _load_candidate_mathml(
    tsv_base_dir: Path,
    representation: str,
    needed_fids: set[str],
) -> dict[str, str]:
    """Stream collection TSV shards and collect pre-computed MathML for needed formula_ids.

    The TSV files have columns: id, post_id, ..., formula (pre-computed MathML).
    We filter to rows whose `id` (formula_id) is in needed_fids.

    Args:
        tsv_base_dir: data/raw/collection/formula (contains slt_representation_v3/ etc.)
        representation: "slt", "opt", or "slt_type" — selects the TSV subdir.
        needed_fids: formula_ids to collect MathML for.

    Returns:
        {formula_id: mathml_str}
    """
    subdir_map = {
        "slt": "slt_representation_v3",
        "slt_type": "slt_representation_v3",  # SLT-TYPE uses SLT MathML (same PMML)
        "opt": "opt_representation_v3",
    }
    subdir = tsv_base_dir / subdir_map[representation]
    if not subdir.exists():
        logger.warning("TSV dir not found: %s — candidate formulas will use subprocess", subdir)
        return {}

    tsv_files = sorted(subdir.glob("*.tsv"), key=lambda p: int(p.stem))
    formula_mathml: dict[str, str] = {}
    remaining = set(needed_fids)

    csv.field_size_limit(2**31 - 1)

    for tsv_file in tqdm(tsv_files, desc="Loading candidate MathML from TSV shards", unit="shard"):
        if not remaining:
            break
        with open(tsv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if not remaining:
                    break
                fid = row.get("id", "")
                if fid not in remaining:
                    continue
                mathml = row.get("formula", "").strip()
                if mathml:
                    formula_mathml[fid] = mathml
                remaining.discard(fid)

    if remaining:
        logger.warning(
            "%d formula_ids not found in TSV shards — those will fall back to subprocess",
            len(remaining),
        )
    logger.info(
        "Loaded pre-computed MathML for %d/%d candidate formulas",
        len(formula_mathml),
        len(needed_fids),
    )
    return formula_mathml


def _write_run_file(
    reranked: dict[str, list[tuple[str, float]]],
    output_path: Path,
    run_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for topic_id, doc_scores in reranked.items():
            for rank, (doc_id, score) in enumerate(doc_scores, start=1):
                f.write(f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage-2 formula-aware reranker for ARQMath Task 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config_path", type=str, help="Path to reranker YAML config")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--dry-run", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    try:
        logger.info("Loading config from %s", config_path)
        config: RerankerConfig = RerankerConfigManager.from_yaml(config_path)
        run_name = (
            f"rerank_{config.representation}_alpha{config.alpha}_n{config.n_candidates}"
            f"_{datetime.now().strftime('%Y%m%d')}"
        )
        logger.info("Config loaded: %s", run_name)

        if not args.dry_run:
            wandb.init(
                project="multi-index-rag",
                name=run_name,
                group=config.get_experiment_name(),
                tags=["rerank", config.representation, f"alpha={config.alpha}"],
                config=config.model_dump(),
            )
            logger.info("W&B initialized: %s", wandb.run.url)  # type: ignore

        # Load stage-1 run
        stage1_path = Path(config.stage1_run_path)
        logger.info("Parsing stage-1 run from %s", stage1_path)
        raw_run = _parse_run(stage1_path)
        run: dict[str, list[tuple[str, float]]] = {
            qid: sorted(doc_scores.items(), key=lambda x: -x[1])
            for qid, doc_scores in raw_run.items()
        }
        logger.info("Loaded %d topics from stage-1 run", len(run))

        # Load topics
        logger.info("Loading topics from %s", TOPICS_JSONL)
        topics = _load_topics(TOPICS_JSONL)

        # Collect candidate doc_ids
        candidate_ids: set[str] = {
            doc_id for doc_scores in run.values() for doc_id, _ in doc_scores
        }
        logger.info("Collecting formula data for %d unique candidate posts", len(candidate_ids))

        # Stream answers.jsonl → {answer_id: [(formula_id, latex), ...]}
        logger.info("Streaming %s", ANSWERS_JSONL)
        answer_formulas = _load_answer_formulas(ANSWERS_JSONL, candidate_ids)
        logger.info(
            "Loaded formula data for %d/%d candidate posts",
            len(answer_formulas),
            len(candidate_ids),
        )

        # Collect all candidate formula_ids, then load their pre-computed MathML from TSV
        needed_fids: set[str] = {
            fid
            for pairs in answer_formulas.values()
            for fid, _ in pairs
            if fid
        }
        logger.info("Loading pre-computed MathML for %d candidate formula_ids", len(needed_fids))
        tsv_base_dir = Path(config.formula_tsv_base_dir or str(FORMULA_DIR))
        formula_mathml = _load_candidate_mathml(tsv_base_dir, config.representation, needed_fids)

        embedding_dir = Path(config.formula_embedding_dir or str(FORMULA_EMBEDDING_DIR))

        # Instantiate reranker
        reranker = FormulaMaxSimReranker(
            embedding_dir=embedding_dir,
            representation=config.representation,
            alpha=config.alpha,
            aggregation=config.aggregation,
            n_candidates=config.n_candidates,
        )

        # Rerank
        logger.info(
            "Reranking: alpha=%.2f, aggregation=%s, n_candidates=%d, representation=%s",
            config.alpha,
            config.aggregation,
            config.n_candidates,
            config.representation,
        )
        reranked = reranker.rerank(run, topics, answer_formulas, formula_mathml)

        # Write output run file
        out_path = RUNS_DIR / f"{run_name}.tsv"
        logger.info("Writing reranked run to %s", out_path)
        _write_run_file(reranked, out_path, run_name)
        logger.info("Run file written: %s (%d bytes)", out_path, out_path.stat().st_size)

        # Evaluate
        logger.info("Evaluating against qrels...")
        metrics_dict = evaluate_run(
            qrels_path=QREL_TASK1_2022_OFFICIAL,
            run_path=out_path,
        )

        if not args.dry_run:
            metrics_table = wandb.Table(columns=["Metric", "Value"])
            for metric_name, metric_value in sorted(metrics_dict.items()):
                metrics_table.add_data(metric_name, metric_value)
            wandb.log(
                {
                    "metrics_table": metrics_table,
                    "timestamp": datetime.now().isoformat(),
                    "run_file_path": str(out_path),
                    "stage1_run_path": str(stage1_path),
                }
            )
            wandb.save(str(out_path), base_path=RUNS_DIR)
            wandb.save(str(stage1_path), base_path=stage1_path.parent)
            wandb.save(str(config_path), base_path=config_path.parent)

        print_evaluation_report(
            qrels_path=QREL_TASK1_2022_OFFICIAL,
            run_path=out_path,
            run_name=run_name,
        )

        logger.info("=" * 70)
        logger.info("Reranking complete: %s", run_name)
        logger.info("Run file: %s", out_path)
        if not args.dry_run:
            logger.info("W&B URL: %s", wandb.run.url)  # type: ignore
        logger.info("=" * 70)

        if not args.dry_run:
            wandb.finish()

        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        if not args.dry_run:
            try:
                wandb.log({"error": str(e)})
                wandb.finish()
            except Exception:
                pass
        return 1

    except (ValueError, KeyError) as e:
        logger.error("Configuration or data error: %s", e)
        if not args.dry_run:
            try:
                wandb.log({"error": str(e)})
                wandb.finish()
            except Exception:
                pass
        return 1

    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        if not args.dry_run:
            try:
                wandb.log({"error": str(e)})
                wandb.finish()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
