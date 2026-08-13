"""Pydantic config models for the Track V LLM relevance judge."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class JudgeSettings(BaseModel):
    provider: str = "google"  # "google" | "openai_compat"
    model: str = "gemini-3.5-flash-lite"  # exact version string, never a floating alias
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    use_openai_compat_endpoint: bool = False
    openai_compat_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    api_key_env: str = "GEMINI_API_KEY"

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        if v not in {"google", "openai_compat"}:
            raise ValueError(f"provider must be 'google' or 'openai_compat', got '{v}'")
        return v


class ExecutionConfig(BaseModel):
    mode: str = "batch"  # "batch" | "sync"
    batch_poll_seconds: int = 60
    # Empirically determined (live API, 2026-07-30): the TPM quota applies cumulatively
    # to chunks submitted back-to-back in a burst, not just per individual chunk --
    # 2x3,000-pair chunks submitted in immediate succession succeeded (6,000 total), but
    # a 3rd back-to-back submission (pushing the burst to 9,000) failed with 429, even
    # though a single standalone 5,000-pair submission succeeds on its own. Keeping this
    # at 1 means only one chunk is submitted per script invocation, so submissions are
    # naturally spaced out across separate runs (each run polls to completion before
    # exiting) instead of bursting. Only raise after re-confirming headroom empirically.
    batch_max_in_flight: int = 1
    # Empirically determined (live API, 2026-07-30): 5,000-request batch submissions
    # succeed, 8,000/9,781 fail with 429 RESOURCE_EXHAUSTED (TPM quota). Default leaves
    # comfortable headroom below the ~5,000 boundary; raise only after re-confirming
    # against the current account's quota.
    max_batch_chunk_size: int = 3000
    sync_max_concurrency: int = 4
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: int = 120

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in {"batch", "sync"}:
            raise ValueError(f"mode must be 'batch' or 'sync', got '{v}'")
        return v


class CacheConfig(BaseModel):
    dir: str = ".cache/judge"


class PromptConfig(BaseModel):
    template_path: str = "src/multirag/eval/prompts/arqmath_relevance_v1.yaml"


class WandbConfig(BaseModel):
    enabled: bool = True
    entity: str = "abramjopaul-abram"
    project: str = "one-last-run"
    group: str = "V"
    job_type: str = "judge-validation"
    mode: str = "online"  # "online" | "offline" | "disabled"
    tags: list[str] = Field(default_factory=lambda: ["V", "judge", "arqmath-1-2-3"])

    @field_validator("mode")
    @classmethod
    def _validate_wandb_mode(cls, v: str) -> str:
        if v not in {"online", "offline", "disabled"}:
            raise ValueError(f"wandb.mode must be online/offline/disabled, got '{v}'")
        return v


class JudgeConfig(BaseModel):
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class JudgeConfigManager:
    """Manager for loading judge configurations from YAML."""

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> JudgeConfig:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise ValueError(f"Empty config file: {yaml_path}")
        return JudgeConfig(**config_dict)
