"""Shared data loaders for Track C generation runs.

Loads topics/qrels/answers — the common data-access layer reused by every
Track C pipeline (generation + reference construction happen downstream, in
src/multirag/eval/generate.py and src/multirag/eval/references.py).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

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

