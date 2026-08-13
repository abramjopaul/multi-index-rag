"""Generation phase for Track C: build prompts, run the local generator,
detect refusals/truncation, and write generations.jsonl.

Orchestrates existing, reused-as-is modules (multirag.generation.{context_source,
prompt_template, generator}) plus the topic loader in
multirag.generation.sample_builder -- nothing here re-implements retrieval,
prompting, or generation itself.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from multirag.generation.context_source import ContextSource
from multirag.generation.generator import Generator
from multirag.generation.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class GenerationRecord:
    topic_id: str
    config_name: str
    k: int
    question: str
    contexts: list[str]
    prompt: str
    response: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    max_new_tokens: int
    generator_model: str
    seed: int
    timestamp: str
    is_refusal: bool
    hit_ceiling: bool


def _messages_to_prompt_text(messages: list[dict]) -> str:
    """Flatten rendered chat messages into one string for audit persistence."""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


def detect_refusal(response: str, template: PromptTemplate) -> bool:
    """Primary check: substring match against the template's own canonical
    refusal string (deterministic given a fixed, frozen template -- see
    PromptTemplate.refusal_phrase). Templates with no refusal instruction
    (e.g. v1/no-RAG) never flag a refusal via this path. Documented as a
    heuristic; extend with a broader detector later if needed.
    """
    if template.refusal_phrase and template.refusal_phrase.lower() in response.lower():
        return True
    return False


def run_generation(
    topics: list[dict],
    context_source: ContextSource,
    template: PromptTemplate,
    generator: Generator,
    config_name: str,
    k: int,
    generator_model: str,
    seed: int,
    max_new_tokens: int,
) -> list[GenerationRecord]:
    messages_list: list[list[dict]] = []
    contexts_per_topic: list[list[str]] = []
    for topic in topics:
        contexts = context_source.get_contexts(topic)
        question = topic["title"] + "\n\n" + topic["question"]
        messages = template.render(question, contexts)
        messages_list.append(messages)
        contexts_per_topic.append(contexts)

    logger.info("=== GENERATION PHASE ===")
    metas = generator.generate_batch_with_metadata(messages_list)

    timestamp = datetime.now(timezone.utc).isoformat()
    records: list[GenerationRecord] = []
    for topic, contexts, messages, meta in zip(topics, contexts_per_topic, messages_list, metas):
        question = topic["title"] + "\n\n" + topic["question"]
        hit_ceiling = meta.completion_tokens >= max_new_tokens
        if hit_ceiling and meta.finish_reason != "length":
            logger.warning(
                f"topic {topic['topic_id']}: hit_ceiling=True but "
                f"finish_reason={meta.finish_reason!r} -- disagreement flagged"
            )
        records.append(
            GenerationRecord(
                topic_id=topic["topic_id"],
                config_name=config_name,
                k=k,
                question=question,
                contexts=contexts,
                prompt=_messages_to_prompt_text(messages),
                response=meta.text,
                finish_reason=meta.finish_reason,
                prompt_tokens=meta.prompt_tokens,
                completion_tokens=meta.completion_tokens,
                max_new_tokens=max_new_tokens,
                generator_model=generator_model,
                seed=seed,
                timestamp=timestamp,
                is_refusal=detect_refusal(meta.text, template),
                hit_ceiling=hit_ceiling,
            )
        )
    return records


def write_generations(records: list[GenerationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info(f"Generations written to {path}")


def load_generations(path: Path) -> list[GenerationRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(GenerationRecord(**json.loads(line)))
    return records
