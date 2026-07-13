#!/usr/bin/env python3
"""Unified Task 2 Formula Retrieval experiment runner with W&B logging.

Single-channel:  formula_{v}_{repr}_n{n}_{YYYYMMDD}
Fusion (A3):     a3_{repr_slug}_{method_tag}_{k_tag}_n{n}_{split}_{YYYYMMDD}

Usage:
    poetry run python experiments/run_task2_experiment.py configs/task2/v1_slt.yaml
    poetry run python experiments/run_task2_experiment.py configs/task2/a3_step1_base.yaml
    poetry run python experiments/run_task2_experiment.py configs/task2/v1_slt.yaml --dry-run
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from logging_config import configure_logging
from multirag.config import Task2ConfigManager
from multirag.config.path_configs import (
    COLLECTION_FORMULA_INDEX_DIR,
    MODELS_DIR,
    QREL_TASK2_2021_ALL,
    QREL_TASK2_2022_OFFICIAL,
    RUNS_DIR,
    TOPICS_TASK2_2021_XML,
    TOPICS_TASK2_XML,
)

configure_logging()
logger = logging.getLogger(__name__)

# Minimum ndcg' below which a fusion result is considered broken
_FUSION_REGRESSION_FLOOR = 0.05


def _resolve_topics_qrels(cfg):
    """Return (topics_path, qrels_path) based on year and any explicit overrides."""
    if cfg.topics_path:
        topics_path = Path(cfg.topics_path)
    elif cfg.year == "2021":
        topics_path = TOPICS_TASK2_2021_XML
    else:
        topics_path = TOPICS_TASK2_XML

    if cfg.qrels_path:
        qrels_path = Path(cfg.qrels_path)
    elif cfg.year == "2021":
        qrels_path = QREL_TASK2_2021_ALL
    else:
        qrels_path = QREL_TASK2_2022_OFFICIAL

    return topics_path, qrels_path


def _compute_weights(representations: list[str], channel_weights: dict | None) -> list[float]:
    """Extract ordered weights matching representations; fall back to equal weights."""
    if not channel_weights:
        return [1.0] * len(representations)
    return [channel_weights.get(r, 1.0) for r in representations]


def _make_run_name(cfg) -> str:
    """Build canonical run name: A3 fusion format or single-channel format."""
    reprs = cfg.get_representations()
    date_str = datetime.now().strftime("%Y%m%d")

    if len(reprs) == 1:
        return f"formula_{cfg.model_version}_{reprs[0]}_n{cfg.n}_{date_str}"

    repr_slug = "_".join(reprs)
    k_tag = f"k{cfg.rrf_k}"

    if cfg.fusion_method == "weighted_rrf" and cfg.channel_weights:
        weights = _compute_weights(reprs, cfg.channel_weights)
        w_parts = "x".join(
            str(int(w)) if w == int(w) else f"{w:.1f}" for w in weights
        )
        method_tag = f"wrrf_{w_parts}"
    else:
        method_tag = "rrf"

    return f"a3_{repr_slug}_{method_tag}_{k_tag}_n{cfg.n}_{cfg.year}_{date_str}"


def _build_indices(cfg, embedding_dir: Path, dry_run: bool) -> dict[str, int]:
    """Build collection FAISS index for each representation. Returns {repr: vectors_indexed}."""
    from build_collection_formula_index import build_index, get_tsv_dir

    stats: dict[str, int] = {}
    for representation in cfg.get_representations():
        tsv_dir = get_tsv_dir(representation)
        index_dir = COLLECTION_FORMULA_INDEX_DIR / cfg.model_version / representation

        if dry_run:
            logger.info(f"[dry-run] Would build index: {index_dir}")
            stats[representation] = 0
            continue

        logger.info(f"Building collection index: {representation} @ {index_dir}")
        t0 = time.time()
        build_index(
            representation=representation,
            embedding_dir=embedding_dir,
            tsv_dir=tsv_dir,
            index_dir=index_dir,
            force=cfg.force_rebuild,
            limit=cfg.index_limit,
        )
        elapsed = time.time() - t0
        logger.info(f"Index build complete for {representation} in {elapsed:.1f}s")
        try:
            import faiss as _faiss
            idx = _faiss.read_index(str(index_dir / f"formula_index_sq_{representation}.faiss"))
            stats[representation] = int(idx.ntotal)
            del idx
        except Exception:
            stats[representation] = -1

    return stats


def _run_retrieval(
    cfg,
    embedding_dir: Path,
    output_path: Path,
    run_name: str,
    dry_run: bool,
) -> tuple[dict | None, dict[str, float]]:
    """Run Task 2 retrieval.

    Returns (metrics_dict_or_None, overlap_per_topic).
    overlap_per_topic is empty for single-channel runs.
    """
    from task2_formula_retrieval import (
        build_visual_id_map,
        exclude_and_collapse,
        parse_task2_topics,
        retrieve_instances,
        rrf_fuse_instances,
        run_single_representation,
        score_run,
        write_run,
    )

    topics_path, qrels_path = _resolve_topics_qrels(cfg)

    if dry_run:
        logger.info("[dry-run] Skipping retrieval and scoring")
        return None, {}

    topics = parse_task2_topics(topics_path)
    logger.info(f"Parsed {len(topics)} Task 2 topics from {topics_path.name}")

    representations = cfg.get_representations()
    overlap_per_topic: dict[str, float] = {}

    if len(representations) == 1:
        results = run_single_representation(
            topics=topics,
            representation=representations[0],
            n=cfg.n,
            embedding_dir=embedding_dir,
            model_version=cfg.model_version,
        )
    else:
        # Pre-collapse fusion: retrieve raw instances per channel → fuse → ONE collapse
        channel_results = []
        for repr_ in representations:
            inst = retrieve_instances(
                topics=topics,
                representation=repr_,
                n=cfg.n,
                embedding_dir=embedding_dir,
                model_version=cfg.model_version,
            )
            channel_results.append(inst)

        weights = _compute_weights(representations, cfg.channel_weights)
        logger.info(
            f"Fusing {len(representations)} channels with {cfg.fusion_method}, "
            f"k={cfg.rrf_k}, weights={weights}"
        )
        fused, overlap_per_topic = rrf_fuse_instances(channel_results, cfg.rrf_k, weights)

        # Build fid→vid map from the first representation (IDs are identical across reprs)
        fid_to_vid, fid_to_post = build_visual_id_map(representations[0])

        results = {}
        for topic in topics:
            tid = topic["topic_id"]
            source_fid = topic.get("source_formula_id", "")
            query_vid = fid_to_vid.get(source_fid)
            results[tid] = exclude_and_collapse(
                fused.get(tid, []),
                fid_to_vid=fid_to_vid,
                fid_to_post=fid_to_post,
                source_post_id=None,
                query_visual_id=query_vid,
            )

    write_run(results, output_path, run_name)
    logger.info(f"Run file written: {output_path}")

    if not qrels_path.exists():
        logger.warning(f"Qrels not found: {qrels_path} — skipping scoring")
        return {}, overlap_per_topic

    metrics = score_run(output_path, qrels_path)
    return metrics, overlap_per_topic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Task 2 formula retrieval experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python experiments/run_task2_experiment.py configs/task2/v1_slt.yaml
  poetry run python experiments/run_task2_experiment.py configs/task2/a3_step1_base.yaml
  poetry run python experiments/run_task2_experiment.py configs/task2/v2_slt.yaml --dry-run
        """,
    )
    parser.add_argument("config_path", type=str, help="Path to Task 2 YAML config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip W&B, index build, retrieval, and scoring — validates config only",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    cfg = Task2ConfigManager.from_yaml(config_path)

    representations = cfg.get_representations()
    is_fusion = len(representations) > 1
    run_name = _make_run_name(cfg)

    embedding_dir = Path(cfg.embedding_dir) if cfg.embedding_dir else MODELS_DIR / cfg.model_version
    run_path = RUNS_DIR / f"{run_name}.tsv"

    topics_path, qrels_path = _resolve_topics_qrels(cfg)

    logger.info("=" * 70)
    logger.info("TASK 2 FORMULA RETRIEVAL EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Run name        : {run_name}")
    logger.info(f"Representations : {representations}")
    logger.info(f"Model version   : {cfg.model_version}")
    logger.info(f"Year / split    : {cfg.year}")
    logger.info(f"N               : {cfg.n}")
    logger.info(f"RRF k           : {cfg.rrf_k}")
    if is_fusion:
        logger.info(f"Fusion method   : {cfg.fusion_method}")
        logger.info(f"Channel weights : {cfg.channel_weights}")
    logger.info(f"Topics          : {topics_path}")
    logger.info(f"Qrels           : {qrels_path}")
    logger.info(f"Embedding dir   : {embedding_dir}")
    logger.info(f"Run output      : {run_path}")
    logger.info(f"Dry run         : {args.dry_run}")
    logger.info("=" * 70)

    wandb_tags = ["task2", cfg.model_version, cfg.year]
    if is_fusion:
        wandb_tags.append("a3")

    if not args.dry_run:
        wandb.init(
            project="one-last-run",
            name=run_name,
            group=cfg.get_experiment_name(),
            tags=wandb_tags,
            config=cfg.model_dump(),
        )
        logger.info(f"W&B initialized: {wandb.run.url}")  # type: ignore

    # Step 1: Build collection indices
    logger.info("Building collection FAISS indices...")
    index_stats = _build_indices(cfg, embedding_dir, dry_run=args.dry_run)

    if not args.dry_run:
        wandb.log({"index_stats": index_stats})

    # Step 2: Retrieval + scoring
    logger.info("Running Task 2 retrieval...")
    metrics, overlap_per_topic = _run_retrieval(
        cfg, embedding_dir, run_path, run_name, dry_run=args.dry_run
    )

    if not args.dry_run and metrics is not None:
        mean_overlap = mean(overlap_per_topic.values()) if overlap_per_topic else 0.0

        # Flat metric log
        log_dict: dict = {k: v for k, v in metrics.items()}
        if is_fusion:
            log_dict.update({
                "year": cfg.year,
                "fusion_method": cfg.fusion_method,
                "rrf_k": cfg.rrf_k,
                "channel_weights": str(cfg.channel_weights),
                "slt_type_included": "slt_type" in representations,
                "channel_overlap_mean": mean_overlap,
            })
        wandb.log(log_dict)

        # Per-run summary row table
        primary_ndcg = metrics.get("ndcg'", metrics.get("ndcg'@10", 0.0))
        row = {
            "config": run_name,
            "year": cfg.year,
            "representations": ",".join(representations),
            "n": cfg.n,
            "rrf_k": cfg.rrf_k if is_fusion else None,
            "fusion_method": cfg.fusion_method if is_fusion else "single",
            "weights": str(cfg.channel_weights) if is_fusion else None,
            "slt_type": "slt_type" in representations,
            "ndcg'@10": metrics.get("ndcg'@10"),
            "map'": metrics.get("map'"),
            "P'@10": metrics.get("P'@10"),
            "recall'@1000": metrics.get("recall'@1000"),
            "channel_overlap": mean_overlap if is_fusion else None,
        }
        run_table = wandb.Table(columns=list(row.keys()), data=[list(row.values())])
        wandb.log({
            "run_summary_row": run_table,
            "metrics_table": wandb.Table(
                columns=["Metric", "Value"],
                data=[[k, v] for k, v in sorted(metrics.items())],
            ),
            "run_file_path": str(run_path),
            "config_file_path": str(config_path),
        })

        # Fusion regression guard
        if is_fusion and primary_ndcg is not None and primary_ndcg < _FUSION_REGRESSION_FLOOR:
            logger.warning(
                f"FUSION REGRESSION: ndcg'={primary_ndcg:.4f} is below floor "
                f"{_FUSION_REGRESSION_FLOOR}. Check for encoding mismatch or index bugs."
            )
            wandb.run.tags = list(wandb.run.tags) + ["fusion_regression"]  # type: ignore

        wandb.save(str(run_path), base_path=str(RUNS_DIR))
        wandb.save(str(config_path), base_path=str(config_path.parent))

    # Step 3: Print report
    if metrics:
        print("\n" + "=" * 60)
        print(f"Task 2 Results — {run_name}")
        print("=" * 60)
        for key in sorted(metrics):
            print(f"  {key:<25}: {metrics[key]:.4f}")

    if not args.dry_run:
        wandb.finish()

    logger.info("=" * 70)
    logger.info(f"Experiment complete: {run_name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
