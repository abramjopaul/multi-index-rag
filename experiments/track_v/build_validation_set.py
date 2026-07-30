#!/usr/bin/env python3
"""Track V: build the human-judged validation set (topic, answer, human label).

Loads human qrels from ARQMath-1 (2020), ARQMath-2 (2021), ARQMath-3 (2022)
Task 1, resolves topic text (title+question) and answer text against the
locally-available topic XML / answers.jsonl, and emits one row per judged
pair: a JudgePair plus its human_label/human_bin gold label.

The --qrel-file/--max-topics/--max-pairs-per-topic/--topic-ids flags compose
so a cheap end-to-end dry run (a few dozen pairs) can prove the whole chain
before the ~100K-pair full run. The full run is the same command with these
flags left at their defaults (all three years, no caps).

Usage:
    # Full run (all three years):
    poetry run python experiments/track_v/build_validation_set.py

    # Cheap dry run (a few cents worth of pairs, for smoke-testing the chain):
    poetry run python experiments/track_v/build_validation_set.py \\
        --qrel-file data/raw/qrels/qrel_task1_2020_all --max-topics 5 --max-pairs-per-topic 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/ (logging_config.py)

import logging  # noqa: E402

from logging_config import configure_logging  # noqa: E402

from multirag.config.path_configs import (
    ANSWERS_JSONL,  # noqa: E402
    QREL_TASK1_2020_ALL,
    QREL_TASK1_2021_ALL,
    QREL_TASK1_2022_ALL,
    RESULTS_TRACK_V_DIR,
    TOPICS_TASK1_2020_XML,
    TOPICS_TASK1_2021_XML,
    TOPICS_TASK1_2022_XML,
)
from multirag.evaluation.metrics import _parse_qrels  # noqa: E402
from multirag.preprocessing.topic_parser import TopicReader  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

DEFAULT_QREL_FILES = [QREL_TASK1_2020_ALL, QREL_TASK1_2021_ALL, QREL_TASK1_2022_ALL]

_QREL_YEAR_TO_TOPICS_XML = {
    "2020": TOPICS_TASK1_2020_XML,
    "2021": TOPICS_TASK1_2021_XML,
    "2022": TOPICS_TASK1_2022_XML,
}


def _infer_topics_xml(qrel_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    for year, xml_path in _QREL_YEAR_TO_TOPICS_XML.items():
        if year in qrel_path.name:
            return xml_path
    raise ValueError(
        f"Cannot infer topics XML for {qrel_path} (no year in filename). "
        "Pass --topics-xml explicitly for a custom qrel file."
    )


def _load_answer_texts(answers_path: Path, needed_ids: set[str]) -> dict[str, str]:
    """Single streaming pass over answers.jsonl, keeping only needed ids
    (the file is ~1.4M lines; loading everything would be wasteful).
    """
    texts: dict[str, str] = {}
    with open(answers_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in needed_ids:
                texts[row["id"]] = row.get("body_text", "")
    return texts


def build_validation_set(
    qrel_files: list[Path],
    answers_path: Path,
    max_topics: int | None,
    max_pairs_per_topic: int | None,
    topic_ids_allowlist: set[str] | None,
    topics_xml_override: Path | None,
) -> tuple[list[dict], dict[str, int]]:
    """Returns (rows, per_year_counts). Each row is a flat dict: JudgePair
    fields + human_label + human_bin + year.
    """
    rows: list[dict] = []
    per_year_counts: dict[str, int] = {}
    unresolved_topics: set[str] = set()
    unresolved_answers: set[str] = set()

    for qrel_path in qrel_files:
        if not qrel_path.exists():
            raise FileNotFoundError(f"Qrel file not found: {qrel_path}")

        qrels = _parse_qrels(qrel_path)
        topics_xml = _infer_topics_xml(qrel_path, topics_xml_override)
        if not topics_xml.exists():
            raise FileNotFoundError(
                f"Topics XML not found: {topics_xml} (needed to resolve {qrel_path})"
            )
        topic_map = TopicReader(topics_xml).map_topics

        topic_ids_in_order = list(qrels.keys())
        if topic_ids_allowlist is not None:
            topic_ids_in_order = [
                t for t in topic_ids_in_order if t in topic_ids_allowlist
            ]
        if max_topics is not None:
            kept = sorted(topic_ids_in_order)[:max_topics]
            logger.info(
                f"{qrel_path.name}: --max-topics {max_topics} -> keeping {kept}"
            )
            topic_ids_in_order = kept

        # Cap per-topic answer counts up front, then do ONE pass over the
        # ~1.4M-line answers.jsonl for this whole qrel file -- resolving
        # per-topic would be O(topics * file size) and is far too slow.
        per_topic_answer_items: dict[str, list[tuple[str, int]]] = {}
        needed_ids: set[str] = set()
        for topic_id in topic_ids_in_order:
            answer_items = list(qrels[topic_id].items())
            if max_pairs_per_topic is not None:
                answer_items = answer_items[:max_pairs_per_topic]
            per_topic_answer_items[topic_id] = answer_items
            needed_ids.update(answer_id for answer_id, _ in answer_items)

        answer_texts = _load_answer_texts(answers_path, needed_ids)

        year_count = 0
        for topic_id in topic_ids_in_order:
            topic = topic_map.get(topic_id)
            if topic is None:
                unresolved_topics.add(topic_id)
                continue
            question = f"{topic.title}\n\n{topic.question}".strip()

            for answer_id, human_label in per_topic_answer_items[topic_id]:
                answer_text = answer_texts.get(answer_id)
                if answer_text is None:
                    unresolved_answers.add(answer_id)
                    continue
                rows.append(
                    {
                        "topic_id": topic_id,
                        "answer_id": answer_id,
                        "question": question,
                        "answer_text": answer_text,
                        "human_label": human_label,
                        "human_bin": 1 if human_label >= 2 else 0,  # relevance_level=2
                        "qrel_source": qrel_path.name,
                    }
                )
                year_count += 1

        per_year_counts[qrel_path.name] = year_count

    if unresolved_topics:
        logger.warning(
            f"{len(unresolved_topics)} topic_id(s) unresolved (no title/question text): "
            f"{sorted(unresolved_topics)[:20]}{'...' if len(unresolved_topics) > 20 else ''}"
        )
    if unresolved_answers:
        logger.warning(
            f"{len(unresolved_answers)} answer_id(s) unresolved (not found in {answers_path.name}): "
            f"{sorted(unresolved_answers)[:20]}{'...' if len(unresolved_answers) > 20 else ''}"
        )

    return rows, per_year_counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Track V judge-validation set"
    )
    parser.add_argument(
        "--qrel-file",
        action="append",
        default=None,
        help="Restrict to specific qrel file(s); repeatable. Default: all three years.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Cap to first N topics per qrel file",
    )
    parser.add_argument(
        "--max-pairs-per-topic",
        type=int,
        default=None,
        help="Cap judged answers per topic",
    )
    parser.add_argument(
        "--topic-ids", default=None, help="Comma-separated explicit topic_id allowlist"
    )
    parser.add_argument(
        "--topics-xml",
        default=None,
        help="Override topics XML path (only valid with a single --qrel-file)",
    )
    parser.add_argument("--answers-path", default=str(ANSWERS_JSONL))
    parser.add_argument(
        "--output", default=str(RESULTS_TRACK_V_DIR / "validation_set.jsonl")
    )
    args = parser.parse_args()

    qrel_files = (
        [Path(p) for p in args.qrel_file]
        if args.qrel_file
        else list(DEFAULT_QREL_FILES)
    )
    if args.topics_xml and len(qrel_files) != 1:
        parser.error("--topics-xml is only valid with exactly one --qrel-file")
    topics_xml_override = Path(args.topics_xml) if args.topics_xml else None

    topic_ids_allowlist = None
    if args.topic_ids:
        topic_ids_allowlist = {
            t.strip() for t in args.topic_ids.split(",") if t.strip()
        }

    rows, per_year_counts = build_validation_set(
        qrel_files=qrel_files,
        answers_path=Path(args.answers_path),
        max_topics=args.max_topics,
        max_pairs_per_topic=args.max_pairs_per_topic,
        topic_ids_allowlist=topic_ids_allowlist,
        topics_xml_override=topics_xml_override,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_topics = len({r["topic_id"] for r in rows})
    n_h = sum(1 for r in rows if r["human_label"] == 3)
    n_m = sum(1 for r in rows if r["human_label"] == 2)
    n_l = sum(1 for r in rows if r["human_label"] == 1)
    n_n = sum(1 for r in rows if r["human_label"] == 0)

    print(f"\n{'=' * 60}")
    print("Track V validation set")
    print(f"{'=' * 60}")
    for name, count in per_year_counts.items():
        print(f"  {name:<35} {count:>8} pairs")
    print(f"  {'TOTAL':<35} {len(rows):>8} pairs across {n_topics} topics")
    print(f"  Graded: H(3)={n_h} M(2)={n_m} L(1)={n_l} N(0)={n_n}")
    print(
        f"  Binary (relevance_level=2): relevant={n_h + n_m} not_relevant={n_l + n_n}"
    )
    print(f"Written to {output_path}")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
