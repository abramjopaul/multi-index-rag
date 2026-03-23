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


class MetricType(str, Enum):
    """Available IR evaluation metrics from ranx.

    Reference: https://amenra.github.io/ranx/metrics/
    """

    PRECISION = "precision"
    RECALL = "recall"
    MAP = "map"
    MEAN_RECIPROCAL_RANK = "mrr"
    NDCG = "ndcg"


class RunConfig(BaseModel):
    """Configuration for a single retrieval run with W&B logging.

    Attributes:
        run_name: Human-readable name for this run (e.g., "BM25 Baseline").
        index_type: Type(s) of index to use. Can be a single type (sparse, dense, formula)
                   or later a list for fusion systems (["sparse", "dense"]).
        num_hits: Number of hits/results to retrieve from the index per topic.
        metrics: List of base metric names (without @k).
                 Examples: ["nDCG", "Recall", "Precision", "MAP"]
        k_values: List of k cutoff values for @k metrics (e.g., [5, 10, 100, 1000]).
                 These apply to all metrics that support @k (nDCG, Precision, Recall, etc).
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
    metrics: list[str] = Field(
        ...,
        description="Base metric names (e.g., ['precsion', 'recall', 'map']). @k variants will be generated automatically.",
    )
    k_values: list[int] = Field(
        default=[5, 10, 100, 1000],
        description="List of k cutoff values for @k metrics",
    )
    index_corpus_limit: int | None = Field(
        default=None, ge=1, description="Optional corpus size limit for indexing"
    )
    force_rebuild: bool = Field(
        default=False, description="Force rebuild of index (default: reuse existing index)"
    )
    experiment_name: str | None = Field(
        default=None, description="W&B experiment grouping (defaults to run_name)"
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

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate that all requested metrics are available in ranx."""
        valid_metrics = {t.value for t in MetricType}

        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid metric '{metric}'. Must be one of: {valid_metrics}"
                )

        return v

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: list[int]) -> list[int]:
        """Validate k_values are positive and sorted."""
        if not v:
            raise ValueError("k_values cannot be empty")

        for k in v:
            if k < 1:
                raise ValueError(f"k_values must be positive, got: {k}")

        return sorted(v)

    def get_experiment_name(self) -> str:
        """Get the experiment name, with fallback to run_name."""
        return self.experiment_name or self.run_name

    def build_eval_metrics(self) -> list[str]:
        """Build full metric names for ranx evaluation.

        Combines base metrics with k_values to create strings like "nDCG@5", "Recall@100".
        MAP, RR, Bpref, and NDCG_burges don't support @k and are added as-is.

        Returns:
            List of metric strings for ranx.evaluate()
        """
        eval_metrics = []

        for metric in self.metrics:
            # if metric in ["map", "rr", "recall", "precision", "mrr"]:  # Metrics that don't use @k
            #     # These metrics don't use @k
            #     eval_metrics.append(metric)
            # else:
            # Add @k variants for metrics that support it
            for k in self.k_values:
                eval_metrics.append(f"{metric}@{k}")

        return eval_metrics


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
