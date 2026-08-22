"""Track C: local generation layer (context assembly, prompting, generator backends).

ragas evaluation lives in src/multirag/eval/ (metrics_registry.py, references.py,
generate.py, runner.py), not here -- this package only covers the generator-side
plumbing that has no ragas dependency.
"""

from multirag.generation.context_source import (
    ContextSource,
    NoRagSource,
    OracleSource,
    RunFileSource,
)
from multirag.generation.generator import Generator, GenerationMeta, HFGenerator, VLLMGenerator
from multirag.generation.prompt_template import PromptTemplate, get_template

__all__ = [
    "ContextSource",
    "NoRagSource",
    "OracleSource",
    "RunFileSource",
    "Generator",
    "GenerationMeta",
    "HFGenerator",
    "VLLMGenerator",
    "PromptTemplate",
    "get_template",
]
