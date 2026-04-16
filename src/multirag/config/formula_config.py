"""Configuration for formula indexing pipeline using TangentCFT."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
import yaml


class DataScopeConfig(BaseModel):
    """Data scope limiting for formula preprocessing.
    
    Attributes:
        max_formulas: Maximum number of formulas to process. None = all.
        sample_rate: Fraction of formulas to use (0.0-1.0).
        skip_first_n: Number of formulas to skip from beginning.
    """
    
    max_formulas: Optional[int] = Field(
        default=None,
        ge=1,
        description="Max formulas to process (None = all)"
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of formulas to sample (0.0-1.0)"
    )
    skip_first_n: int = Field(
        default=0,
        ge=0,
        description="Number of formulas to skip from beginning"
    )


class FastTextConfig(BaseModel):
    """FastText hyperparameters for formula embedding training.
    
    Standard parameters from TangentCFT paper.
    """
    
    vector_size: int = Field(default=300, ge=10, description="Embedding dimension")
    window: int = Field(default=5, ge=1, description="Context window for tuple pairs")
    min_count: int = Field(default=2, ge=1, description="Min tuple occurrences")
    epochs: int = Field(default=30, ge=1, description="Training epochs")
    negative: int = Field(default=20, ge=1, description="Negative samples per example")
    workers: int = Field(default=4, ge=1, description="Parallel thread workers")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class RepresentationConfig(BaseModel):
    """Formula representation configuration.
    
    Attributes:
        type: Representation type ('slt', 'opt', or 'combined').
        tokenization_mode: How to tokenize tuples ('value', 'type', 'both_separated', 'both_combined').
    """
    
    type: Literal['slt', 'opt', 'combined'] = Field(
        default='slt',
        description="Symbol Layout Tree (slt), Operator Tree (opt), or combined"
    )
    tokenization_mode: Literal['value', 'type', 'both_separated', 'both_combined'] = Field(
        default='both_separated',
        description="How to encode tuple elements"
    )


class PathsConfig(BaseModel):
    """Path configuration for formula indexing."""
    
    formula_corpus_tsv: str = Field(
        default="data/raw/collection/formula/latex_representation_v3/",
        description="Input formula TSV files directory"
    )
    output_corpus_csv: str = Field(
        default="data/processed/formula_corpus.csv",
        description="Output prepared formula corpus CSV"
    )
    model_path: str = Field(
        default="data/indices/formula_models/",
        description="FastText model storage directory"
    )
    index_path: str = Field(
        default="data/indices/formula/",
        description="FAISS index storage directory"
    )
    encoder_maps_path: str = Field(
        default="data/indices/formula_models/encoder_maps.pkl",
        description="TupleEncoder node/edge maps pickle file"
    )


class FormulaConfig(BaseModel):
    """Top-level configuration for formula indexing pipeline.
    
    Load from YAML via: FormulaConfig.from_yaml('configs/formula_indexing.yaml')
    """
    
    data_scope: DataScopeConfig = Field(default_factory=DataScopeConfig)
    representation: RepresentationConfig = Field(default_factory=RepresentationConfig)
    fasttext: FastTextConfig = Field(default_factory=FastTextConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "FormulaConfig":
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to formula_indexing.yaml file.
            
        Returns:
            FormulaConfig instance with all settings.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            data_scope=DataScopeConfig(**data.get('data_scope', {})),
            representation=RepresentationConfig(**data.get('representation', {})),
            fasttext=FastTextConfig(**data.get('fasttext', {})),
            paths=PathsConfig(**data.get('paths', {}))
        )
    
    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path where to save the configuration.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'data_scope': self.data_scope.model_dump(),
            'representation': self.representation.model_dump(),
            'fasttext': self.fasttext.model_dump(),
            'paths': self.paths.model_dump()
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def resolve_paths(self, project_root: Optional[Path] = None) -> dict[str, Path]:
        """Resolve relative paths to absolute paths.
        
        Args:
            project_root: Project root directory (auto-detected if None).
            
        Returns:
            Dictionary of resolved absolute paths.
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        
        return {
            'formula_corpus_tsv': project_root / self.paths.formula_corpus_tsv,
            'output_corpus_csv': project_root / self.paths.output_corpus_csv,
            'model_path': project_root / self.paths.model_path,
            'index_path': project_root / self.paths.index_path,
            'encoder_maps_path': project_root / self.paths.encoder_maps_path,
        }
