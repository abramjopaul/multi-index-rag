"""Declarative registry of ragas metrics used by Track C generation evaluation.

New metrics are a name-list change (which names appear in the YAML config's
`metrics:` list), not a code change -- add a MetricSpec entry (or uncomment
one of the disabled ones below) and select it by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from multirag.eval._ragas_compat import patch as _patch_ragas

_patch_ragas()

import instructor  # noqa: E402
from google import genai  # noqa: E402
from ragas.llms.base import InstructorLLM, InstructorModelArgs  # noqa: E402
from ragas.metrics.collections import FactualCorrectness, Faithfulness  # noqa: E402

# Declared but disabled for this run -- uncomment the import above and the
# registry entry below to enable one. Each needs its own factory + input
# requirements verified against the installed ragas version before use:
#   from ragas.metrics.collections import (
#       ResponseRelevancy, LLMContextRecall,
#       LLMContextPrecisionWithReference, NoiseSensitivity, AspectCritic,
#   )


def build_judge_llm(model: str, api_key: str, temperature: float = 0.0) -> InstructorLLM:
    """Build an async-capable ragas LLM wrapper around Gemini.

    ragas.llms.llm_factory(provider="google", ...) always builds a SYNC
    instructor client for Google -- its adapter hardcodes
    instructor.from_genai(client) with use_async defaulting to False, so
    calling metric.ascore()/.agenerate() on the result raises TypeError
    ("Cannot use agenerate() with a synchronous client"). Constructing
    InstructorLLM directly with an explicitly async-patched instructor
    client (use_async=True) is the only way to get real async/concurrent
    judge calls out of this ragas version. Confirmed working against a
    real Gemini call.
    """
    genai_client = genai.Client(api_key=api_key)
    async_instructor_client = instructor.from_genai(genai_client, use_async=True)
    return InstructorLLM(
        client=async_instructor_client,
        model=model,
        provider="google",
        model_args=InstructorModelArgs(temperature=temperature),
    )


@dataclass(frozen=True)
class MetricSpec:
    name: str
    factory: Callable[[Any, Any | None], Any]  # (llm, embeddings) -> ragas metric instance
    requires_reference: bool
    requires_embeddings: bool
    requires_context: bool


@dataclass(frozen=True)
class BoundMetric:
    spec: MetricSpec
    instance: Any  # ragas metric object exposing .ascore(...)


_REGISTRY: dict[str, MetricSpec] = {
    "faithfulness": MetricSpec(
        name="faithfulness",
        factory=lambda llm, embeddings: Faithfulness(llm=llm),
        requires_reference=False,
        requires_embeddings=False,
        requires_context=True,
    ),
    "factual_correctness_precision": MetricSpec(
        name="factual_correctness_precision",
        factory=lambda llm, embeddings: FactualCorrectness(llm=llm, mode="precision"),
        requires_reference=True,
        requires_embeddings=False,
        requires_context=False,
    ),
    "factual_correctness_recall": MetricSpec(
        name="factual_correctness_recall",
        factory=lambda llm, embeddings: FactualCorrectness(llm=llm, mode="recall"),
        requires_reference=True,
        requires_embeddings=False,
        requires_context=False,
    ),
    # Disabled -- uncomment the corresponding import above plus an entry here
    # to enable. Left commented rather than deleted so "add a metric" stays a
    # one-line-ish change instead of a from-scratch registration:
    #
    # "response_relevancy": MetricSpec(
    #     name="response_relevancy",
    #     factory=lambda llm, embeddings: ResponseRelevancy(llm=llm, embeddings=embeddings),
    #     requires_reference=False, requires_embeddings=True, requires_context=False,
    # ),
    # "llm_context_recall": MetricSpec(
    #     name="llm_context_recall",
    #     factory=lambda llm, embeddings: LLMContextRecall(llm=llm),
    #     requires_reference=True, requires_embeddings=False, requires_context=True,
    # ),
    # "llm_context_precision_with_reference": MetricSpec(
    #     name="llm_context_precision_with_reference",
    #     factory=lambda llm, embeddings: LLMContextPrecisionWithReference(llm=llm),
    #     requires_reference=True, requires_embeddings=False, requires_context=True,
    # ),
    # "noise_sensitivity": MetricSpec(
    #     name="noise_sensitivity",
    #     factory=lambda llm, embeddings: NoiseSensitivity(llm=llm),
    #     requires_reference=True, requires_embeddings=False, requires_context=True,
    # ),
    # "aspect_critic": MetricSpec(
    #     name="aspect_critic",
    #     factory=lambda llm, embeddings: AspectCritic(llm=llm),
    #     requires_reference=False, requires_embeddings=False, requires_context=False,
    # ),
}


def build_metrics(
    names: list[str], llm: Any, embeddings: Any | None = None
) -> list[BoundMetric]:
    """Instantiate the named metrics, bound to concrete ragas instances.

    Raises a clear ValueError if a selected metric needs embeddings that
    weren't supplied. Per-topic reference availability (a topic with no
    High/Medium qrels answers) is a data concern handled downstream in the
    runner -- reference-requiring metrics are skipped (not scored, not an
    error) for topics whose ReferenceResult.reference_text is None.
    """
    bound: list[BoundMetric] = []
    for name in names:
        spec = _REGISTRY.get(name)
        if spec is None:
            raise ValueError(f"Unknown metric {name!r}. Available: {sorted(_REGISTRY)}")
        if spec.requires_embeddings and embeddings is None:
            raise ValueError(
                f"Metric {name!r} requires an embeddings model, but none was "
                "configured. Set judge.embedding_model in the run config."
            )
        instance = spec.factory(llm, embeddings)
        bound.append(BoundMetric(spec=spec, instance=instance))
    return bound
