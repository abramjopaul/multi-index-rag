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


class FormulaSearchConfig(BaseModel):
    """Query-time formula selection strategy for Task 1 formula retrieval."""

    strategy: str = Field(
        default="fanout",
        description="fanout (all non-trivial formulas, RRF-fused) | heuristic (single formula)",
    )
    top_n: int = Field(
        default=1000,
        ge=1,
        description="Retrieval depth per formula query (passed as k to batch_search)",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF k constant for fanout fusion",
    )
    trivial_filter: bool = Field(
        default=True,
        description="Drop structurally trivial formulas before querying",
    )
    title_preference: bool = Field(
        default=True,
        description="Heuristic only: prefer title formulas over body formulas",
    )

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if v not in {"fanout", "heuristic"}:
            raise ValueError(f"strategy must be 'fanout' or 'heuristic', got '{v}'")
        return v


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
        description="Path to store/load FAISS formula index. Defaults to data/indices/formula/answer/sq_<representation>",
    )
    formula_embedding_dir: str | None = Field(
        default=None,
        description="Directory containing trained FastText models. Defaults to data/models/v1",
    )
    formula_tsv_base_dir: str | None = Field(
        default=None,
        description="Base dir containing slt_representation_v3/ and opt_representation_v3/ TSV subdirs. Defaults to data/raw/collection/formula",
    )
    formula_search: FormulaSearchConfig = Field(
        default_factory=FormulaSearchConfig,
        description="Query-time formula selection strategy (fanout or heuristic)",
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
        description="Directory containing trained FastText models. Defaults to data/models/v1",
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


class Task2Config(BaseModel):
    """Configuration for a Task 2 formula retrieval experiment with W&B logging."""

    run_name: str = Field(..., description="Human-readable run name")
    representation: str = Field(
        default="opt",
        description="Formula representation(s): slt | opt | slt_type | slt,opt (comma = RRF fusion)",
    )
    model_version: str = Field(
        default="v1",
        description="FastText model version: v1 (n-grams, 300-dim) | v2 (no n-grams, 150-dim)",
    )
    n: int = Field(
        default=2000,
        ge=1,
        description="Retrieval depth per query before visual_id collapse",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF k constant for multi-representation fusion",
    )
    force_rebuild: bool = Field(
        default=False,
        description="Force rebuild of collection formula index",
    )
    index_limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional limit on formulas indexed (for smoke tests)",
    )
    embedding_dir: str | None = Field(
        default=None,
        description="FastText model dir override. Defaults to data/models/{model_version}",
    )
    index_dir: str | None = Field(
        default=None,
        description="Collection FAISS index dir override. Defaults to data/indices/formula/collection/{model_version}",
    )
    topics_path: str | None = Field(
        default=None,
        description="Task 2 topics XML override. Defaults to TOPICS_TASK2_XML",
    )
    qrels_path: str | None = Field(
        default=None,
        description="Qrels TSV override. Defaults to QREL_TASK2_2022_OFFICIAL",
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion method for multi-representation runs: rrf | weighted_rrf",
    )
    channel_weights: dict[str, float] | None = Field(
        default=None,
        description='Per-channel weights for weighted_rrf, e.g. {"slt": 1.0, "opt": 2.0}. '
                    "None = equal weights (standard RRF).",
    )
    year: str = Field(
        default="2022",
        description="Which year's topics/qrels to use: 2021 (tune) | 2022 (final). "
                    "Overridden by explicit topics_path/qrels_path if set.",
    )
    experiment_name: str | None = Field(
        default=None,
        description="W&B experiment group (defaults to run_name)",
    )

    @field_validator("year")
    @classmethod
    def _validate_year(cls, v: str) -> str:
        if v not in {"2021", "2022"}:
            raise ValueError(f"year must be '2021' or '2022', got '{v}'")
        return v

    @field_validator("fusion_method")
    @classmethod
    def _validate_fusion_method(cls, v: str) -> str:
        if v not in {"rrf", "weighted_rrf"}:
            raise ValueError(f"fusion_method must be 'rrf' or 'weighted_rrf', got '{v}'")
        return v

    @field_validator("representation")
    @classmethod
    def _validate_representation(cls, v: str) -> str:
        valid = {"slt", "opt", "slt_type"}
        for token in v.split(","):
            t = token.strip()
            if t not in valid:
                raise ValueError(f"Each representation must be slt|opt|slt_type, got '{t}'")
        return v

    @field_validator("model_version")
    @classmethod
    def _validate_model_version(cls, v: str) -> str:
        if v not in {"v1", "v2"}:
            raise ValueError(f"model_version must be v1 | v2, got '{v}'")
        return v

    def get_experiment_name(self) -> str:
        return self.experiment_name or self.run_name

    def get_representations(self) -> list[str]:
        return [r.strip() for r in self.representation.split(",")]


class Task2ConfigManager:
    """Manager for loading Task 2 experiment configurations from YAML."""

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> Task2Config:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise ValueError(f"Empty config file: {yaml_path}")
        return Task2Config(**config_dict)


class FuseConfig(BaseModel):
    """Configuration for a TREC run fusion experiment."""

    run_name: str = Field(..., description="Human-readable name for this fusion run")
    run_files: list[str] = Field(..., min_length=2, description="Paths to input TREC run files")
    techniques: str = Field(
        default="rrf,combsum,combmnz",
        description="Comma-separated fusion techniques: rrf | combsum | combmnz",
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF k constant")
    top_k: int | None = Field(default=1000, ge=1, description="Results per query to keep")
    output_dir: str = Field(default="data/runs", description="Directory to write fused TREC files")
    experiment_name: str | None = Field(
        default=None, description="W&B experiment group (defaults to run_name)"
    )

    def get_experiment_name(self) -> str:
        return self.experiment_name or self.run_name

    @field_validator("techniques")
    @classmethod
    def _validate_techniques(cls, v: str) -> str:
        valid = {"rrf", "combsum", "combmnz"}
        for t in v.split(","):
            t = t.strip()
            if t not in valid:
                raise ValueError(f"technique must be rrf|combsum|combmnz, got '{t}'")
        return v


class FuseConfigManager:
    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "FuseConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise ValueError(f"Empty config file: {yaml_path}")
        return FuseConfig(**config_dict)


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
