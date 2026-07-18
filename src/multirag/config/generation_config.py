"""Pydantic config models for Track C: generation + RAGAS evaluation runs."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ContextSourceConfig(BaseModel):
    type: str = Field(description='"no_rag" | "run_file" | "oracle"')
    run_path: str | None = None      # for run_file: path to TREC run .tsv
    k: int = 5                       # for run_file / oracle: passages per topic
    relevance_level: int | None = None  # for oracle: minimum qrel label to include


class DecodingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    seed: int = 42


class GeneratorConfig(BaseModel):
    model: str
    revision: str = Field(description="Pinned HuggingFace commit hash for reproducibility")
    dtype: str = "bfloat16"          # "float16" | "bfloat16"
    quantization: str | None = None  # None | "awq" | "gptq" | "int4" | "int8"
    backend: str = "vllm"           # "vllm" | "hf"
    # vLLM only. None = use the model's own config.json max_position_embeddings,
    # which can exceed available KV cache memory for long-context models (e.g.
    # Llama 3.1's 131072 default needs 16GiB of KV cache alone on this task's
    # short no-RAG prompts). Set explicitly to cap it to what actually fits.
    max_model_len: int | None = None
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)


class RagasJudgeConfig(BaseModel):
    backend: str = "gemini"                      # "gemini" | "vllm" | "hf"
    model: str = "gemini-1.5-flash"
    api_key_env: str = "GOOGLE_API_KEY"          # env var name holding the API key
    rpm_limit: int = 14                          # requests/min (free tier: 15 RPM cap)
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    metrics: list[str] = Field(
        default=["answer_relevance", "answer_correctness", "semantic_similarity", "rouge_l"]
    )
    n_repeats: int = 1
    seed: int = 42   # forwarded to ChatGoogleGenerativeAI(seed=...) in ragas_eval.py for
                      # judge-call reproducibility (Gemini docs this as best-effort, not guaranteed)


class GenerationRunConfig(BaseModel):
    run_name: str
    phase: str = "C0.1"               # "C0.1" | "C0.2" | "C1" | "C-Oracle" | ...
    context_source: ContextSourceConfig
    generator: GeneratorConfig
    prompt_template_version: str = "v1"
    topics_path: str
    qrels_path: str
    answers_path: str
    ground_truth_strategy: str = "top_scored"  # "top_scored": highest-label then highest SE score
    ragas: RagasJudgeConfig = Field(default_factory=RagasJudgeConfig)
    n_topics: int | None = None        # None = all; CLI --n-topics overrides this
    # Filter topics to ground-truth-having ones BEFORE truncating to n_topics
    # (see sample_builder.build_generation_samples). Off by default: run_c0_1.py's
    # existing behavior (topics[:n_topics], no GT filtering) is unaffected.
    require_ground_truth: bool = False
    experiment_name: str | None = None


class GenerationRunConfigManager:
    @staticmethod
    def from_yaml(path: str | Path) -> GenerationRunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return GenerationRunConfig(**data)
