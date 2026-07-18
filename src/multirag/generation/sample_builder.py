"""Shared sample-building for Track C generation + RAGAS evaluation runs.

Loads topics/qrels/answers, resolves ground truth, renders prompts, runs
generation once, and builds RagasSample objects — the common setup phase
shared by every Track C experiment runner (C0.1, C0.3, ...).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from multirag.generation.context_source import build_context_source
from multirag.generation.generator import build_generator
from multirag.generation.prompt_template import PromptTemplate, get_template
from multirag.generation.ragas_eval import RagasSample

logger = logging.getLogger(__name__)


def load_topics(path: Path, n: int | None) -> list[dict]:
    topics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                topics.append(json.loads(line))
    if n is not None:
        topics = topics[:n]
    logger.info(f"Loaded {len(topics)} topics from {path}")
    return topics


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse TREC qrels -> {topic_id: {answer_id: label}}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic_id, _, doc_id, label = parts[0], parts[1], parts[2], parts[3]
            qrels[topic_id][doc_id] = int(label)
    logger.info(f"Loaded qrels for {len(qrels)} topics from {path}")
    return dict(qrels)


def build_answer_lookup(path: Path) -> dict[str, tuple[str, int]]:
    """Stream answers.jsonl -> {answer_id: (body_text, score)}.

    This streams the full 2 GB file once and holds the result in RAM.
    ~30 s on first run; stays in memory for the duration of the run.
    """
    logger.info(f"Building answer lookup from {path} (streaming ~2 GB, ~30 s)...")
    lookup: dict[str, tuple[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answer_id = str(obj.get("id", ""))
            body_text = obj.get("body_text", "")
            score = int(obj.get("score", 0))
            if answer_id:
                lookup[answer_id] = (body_text, score)
    logger.info(f"Answer lookup built: {len(lookup):,} answers")
    return lookup


def select_ground_truth(
    topic_id: str,
    qrels: dict[str, dict[str, int]],
    answer_lookup: dict[str, tuple[str, int]],
    strategy: str = "top_scored",
) -> tuple[str | None, str | None, int | None]:
    """Select a single reference answer for a topic.

    Strategy 'top_scored':
      1. Among judged answer_ids for this topic, take the highest relevance label (3->2->1).
      2. Among those at the highest label, pick the one with the highest SE community score.
      3. Tiebreak: highest numeric answer_id (most recent).

    Returns:
        (body_text, answer_id, se_score) or (None, None, None) if no judged answer found.
    """
    topic_qrels = qrels.get(topic_id, {})
    if not topic_qrels:
        return None, None, None

    # Find highest relevance label with at least one judged answer in the lookup
    for label in (3, 2, 1):
        candidates = [
            aid for aid, lbl in topic_qrels.items()
            if lbl == label and aid in answer_lookup
        ]
        if not candidates:
            continue
        # Pick by highest SE score, then highest answer_id
        best = max(candidates, key=lambda aid: (answer_lookup[aid][1], int(aid)))
        body_text, se_score = answer_lookup[best]
        return body_text, best, se_score

    return None, None, None


def build_generation_samples(
    config, n_topics: int | None, require_ground_truth: bool = False
) -> tuple[list[RagasSample], PromptTemplate, int, int]:
    """Load data, run generation once, and build RagasSamples for a config.

    Generation runs exactly once here regardless of how many times a caller
    later re-evaluates the resulting samples (e.g. C0.3 repeats only the
    judge step against this same fixed sample set).

    Args:
        require_ground_truth: if True, filter topics to ground-truth-having
            ones BEFORE truncating to n_topics, instead of the default
            truncate-then-filter order. Ground-truth availability is NOT
            guaranteed by file order (some topics have zero qrels rows), so a
            naive topics[:n] can silently include GT-less topics — which then
            drop out of every reference-requiring metric, or worse, for
            context_precision specifically, get a real non-NaN
            silently-near-zero score instead of NaN. Off by default so
            existing callers (run_c0_1.py) are unaffected.

    Returns:
        (samples, template, n_topics_loaded, n_topics_with_ground_truth)
    """
    qrels = load_qrels(Path(config.qrels_path))
    answer_lookup = build_answer_lookup(Path(config.answers_path))

    if require_ground_truth:
        all_topics = load_topics(Path(config.topics_path), None)
        gt_available = {
            t["topic_id"]: select_ground_truth(
                t["topic_id"], qrels, answer_lookup, config.ground_truth_strategy
            )
            for t in all_topics
        }
        filtered = [t for t in all_topics if gt_available[t["topic_id"]][0] is not None]
        n_dropped = len(all_topics) - len(filtered)
        if n_dropped:
            logger.info(
                f"require_ground_truth=True: filtered out {n_dropped} topic(s) without "
                f"ground truth before truncation ({len(filtered)}/{len(all_topics)} remain)"
            )
        topics = filtered[:n_topics] if n_topics is not None else filtered
        if n_topics is not None and len(topics) < n_topics:
            logger.warning(
                f"require_ground_truth=True: only {len(topics)} GT-having topics "
                f"available, fewer than requested n_topics={n_topics}"
            )
        gt_map = {t["topic_id"]: gt_available[t["topic_id"]] for t in topics}
    else:
        topics = load_topics(Path(config.topics_path), n_topics)
        gt_map: dict[str, tuple[str | None, str | None, int | None]] = {}
        for topic in topics:
            tid = topic["topic_id"]
            gt_map[tid] = select_ground_truth(tid, qrels, answer_lookup, config.ground_truth_strategy)

    n_with_gt = sum(1 for v in gt_map.values() if v[0] is not None)
    logger.info(f"Ground truth available for {n_with_gt}/{len(topics)} topics")

    template = get_template(config.prompt_template_version)
    context_source = build_context_source(config.context_source, answer_lookup=answer_lookup)

    messages_list: list[list[dict]] = []
    for topic in topics:
        contexts = context_source.get_contexts(topic)
        question = topic["title"] + "\n\n" + topic["question"]
        messages = template.render(question, contexts)
        messages_list.append(messages)

    logger.info("=== GENERATION PHASE ===")
    generator = build_generator(config.generator)
    answers = generator.generate_batch(messages_list)
    generator.unload()

    samples: list[RagasSample] = []
    for topic, answer in zip(topics, answers):
        tid = topic["topic_id"]
        gt_text, gt_aid, gt_score = gt_map.get(tid, (None, None, None))
        question = topic["title"] + "\n\n" + topic["question"]
        contexts = context_source.get_contexts(topic)
        samples.append(RagasSample(
            topic_id=tid,
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=gt_text,
            ground_truth_answer_id=gt_aid,
            ground_truth_score=gt_score,
        ))

    return samples, template, len(topics), n_with_gt
