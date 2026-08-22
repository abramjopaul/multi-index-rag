"""Pydantic config models for Track C: generation + RAGAS evaluation runs."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from multirag.config.judge_config import WandbConfig


class ContextSourceConfig(BaseModel):
    type: str = Field(description='"no_rag" | "run_file" | "oracle"')
    run_path: str | None = None      # for run_file: path to TREC run .tsv
    k: int = 5                       # for run_file / oracle(disjoint): passages per topic
    relevance_level: int | None = None  # for oracle: minimum qrel label to include
    oracle_mode: str | None = None   # for oracle: "reference" | "disjoint"


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


class TrackCJudgeConfig(BaseModel):
    """ragas judge settings -- see src/multirag/eval/metrics_registry.py for
    how these drive the actual ragas LLM wrapper construction."""

    model: str = "gemini-3.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_concurrency: int = 5   # bounds the asyncio semaphore in eval/runner.py
    cache_dir: str = ".cache/track_c_judge"
    metrics: list[str] = Field(
        default=[
            "faithfulness",
            "factual_correctness_precision",
            "factual_correctness_recall",
        ]
    )


def _track_c_wandb_defaults() -> WandbConfig:
    # Reuses judge_config.WandbConfig (same class ExperimentLogger expects,
    # same one Track B mutates group/job_type on) rather than a parallel
    # duplicate -- only the Track-C-specific defaults differ from V's.
    return WandbConfig(
        group="Track C : Generation",
        job_type="track-c-generation",
        tags=["C", "generation", "arqmath-3"],
    )


class TrackCRunConfig(BaseModel):
    """Stable, per-experiment settings for a Track C generation+eval run.

    Per-invocation knobs (--trec-file, --k, --no-rag, --qrels, --max-topics,
    --topic-ids, --config-name) are CLI arguments in
    experiments/track_c/generate_and_score.py, not here -- this config only
    holds settings that should stay fixed across every config/k in a
    comparison (generator, prompt template, judge, wandb).
    """

    generator: GeneratorConfig
    prompt_template_version: str = "v2"
    judge: TrackCJudgeConfig = Field(default_factory=TrackCJudgeConfig)
    wandb: WandbConfig = Field(default_factory=_track_c_wandb_defaults)


class TrackCConfigManager:
    @staticmethod
    def from_yaml(path: str | Path) -> TrackCRunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return TrackCRunConfig(**data)
