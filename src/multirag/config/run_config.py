"""Pydantic models for run configuration with W&B logging."""

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class IndexType(str, Enum):
    """Supported index types for retrieval."""

    SPARSE = "sparse"
    DENSE = "dense"
    FORMULA = "formula"
    # Combinations will be added later
    # SPARSE_DENSE = "sparse_dense"
    # SPARSE_FORMULA = "sparse_formula"
    # DENSE_FORMULA = "dense_formula"
    # ALL = "sparse_dense_formula"


class RunConfig(BaseModel):
    """Configuration for a single retrieval run with W&B logging.

    Attributes:
        run_name: Human-readable name for this run (e.g., "BM25 Baseline").
        index_type: Type(s) of index to use. Can be a single type (sparse, dense, formula)
                   or later a list for fusion systems (["sparse", "dense"]).
        num_hits: Number of hits/results to retrieve from the index per topic.
        index_corpus_limit: Optional limit on number of documents to index (for debugging).
                           None = use all documents in corpus.
        experiment_name: Optional experiment grouping for W&B (defaults to run_name).
    """

    run_name: str = Field(..., description="Human-readable run name")
    index_type: str | list[str] = Field(
        ...,
        description="Single index type (sparse|dense|formula) or list for fusion",
    )
    num_hits: int = Field(
        default=1000,
        ge=1,
        description="Number of hits to retrieve from index per topic",
    )
    index_corpus_limit: int | None = Field(
        default=None, ge=1, description="Optional corpus size limit for indexing"
    )
    force_rebuild: bool = Field(
        default=False,
        description="Force rebuild of index (default: reuse existing index)",
    )
    experiment_name: str | None = Field(
        default=None, description="W&B experiment grouping (defaults to run_name)"
    )

    # Formula indexer fields (only used when index_type == "formula")
    formula_representation: str = Field(
        default="slt",
        description="Formula representation to index: 'slt', 'opt', or 'slt_type'",
    )
    formula_index_path: str | None = Field(
        default=None,
        description="Path to store/load FAISS formula index. Defaults to data/indices/formula/<representation>",
    )
    formula_embedding_dir: str | None = Field(
        default=None,
        description="Directory containing trained FastText models. Defaults to data/formula-indexing",
    )
    formula_tsv_base_dir: str | None = Field(
        default=None,
        description="Base dir containing slt_representation_v3/ and opt_representation_v3/ TSV subdirs. Defaults to data/raw/collection/formula",
    )

    @field_validator("index_type")
    @classmethod
    def validate_index_type(cls, v: str | list[str]) -> str | list[str]:
        """Validate that index_type(s) are supported."""
        valid_types = {t.value for t in IndexType}

        if isinstance(v, str):
            if v not in valid_types:
                raise ValueError(
                    f"Invalid index_type '{v}'. Must be one of: {valid_types}"
                )
        elif isinstance(v, list):
            for idx_type in v:
                if idx_type not in valid_types:
                    raise ValueError(
                        f"Invalid index_type '{idx_type}' in list. Must be one of: {valid_types}"
                    )
        else:
            raise ValueError("index_type must be a string or list of strings")

        return v

    def get_experiment_name(self) -> str:
        """Get the experiment name, with fallback to run_name."""
        return self.experiment_name or self.run_name


class RerankerConfig(BaseModel):
    """Configuration for the stage-2 formula-aware reranker."""

    run_name: str = Field(..., description="Human-readable name for this reranker run")
    stage1_run_path: str = Field(..., description="Path to stage-1 TREC run file (.tsv)")
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Blend weight: 0.0 = pure text, 1.0 = pure formula",
    )
    n_candidates: int = Field(
        default=100,
        ge=1,
        description="Number of stage-1 candidates to rerank per topic",
    )
    aggregation: str = Field(
        default="max",
        description="How to aggregate per-topic-formula MaxSim scores: max | mean | sum",
    )
    representation: str = Field(
        default="slt",
        description="Formula representation for FastText embedding: slt | opt | slt_type",
    )
    formula_embedding_dir: str | None = Field(
        default=None,
        description="Directory containing trained FastText models. Defaults to data/formula-indexing",
    )
    formula_tsv_base_dir: str | None = Field(
        default=None,
        description=(
            "Base dir containing slt_representation_v3/ and opt_representation_v3/ TSV subdirs. "
            "When set, candidate formula embeddings use pre-computed MathML from these files "
            "instead of the subprocess-based LaTeX→MathML conversion. "
            "Defaults to data/raw/collection/formula"
        ),
    )
    experiment_name: str | None = Field(
        default=None,
        description="W&B experiment grouping (defaults to run_name)",
    )

    @field_validator("aggregation")
    @classmethod
    def _validate_aggregation(cls, v: str) -> str:
        if v not in {"max", "mean", "sum"}:
            raise ValueError(f"aggregation must be one of max/mean/sum, got '{v}'")
        return v

    @field_validator("representation")
    @classmethod
    def _validate_representation(cls, v: str) -> str:
        if v not in {"slt", "opt", "slt_type"}:
            raise ValueError(f"representation must be one of slt/opt/slt_type, got '{v}'")
        return v

    def get_experiment_name(self) -> str:
        return self.experiment_name or self.run_name


class RerankerConfigManager:
    """Manager for loading reranker configurations from YAML."""

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> RerankerConfig:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise ValueError(f"Empty config file: {yaml_path}")
        return RerankerConfig(**config_dict)


class RunConfigManager:
    """Manager for loading and saving run configurations."""

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> RunConfig:
        """Load run configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            RunConfig instance.

        Raises:
            FileNotFoundError: If YAML file not found.
            ValueError: If YAML is invalid or doesn't match schema.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Empty config file: {yaml_path}")

        return RunConfig(**config_dict)
