"""Track C: generation + RAGAS evaluation layer."""

from multirag.generation.context_source import ContextSource, NoRagSource
from multirag.generation.generator import Generator, HFGenerator, VLLMGenerator
from multirag.generation.prompt_template import PromptTemplate, get_template
from multirag.generation.ragas_eval import RagasEvaluator, RagasSample

__all__ = [
    "ContextSource",
    "NoRagSource",
    "Generator",
    "HFGenerator",
    "VLLMGenerator",
    "PromptTemplate",
    "get_template",
    "RagasEvaluator",
    "RagasSample",
]
