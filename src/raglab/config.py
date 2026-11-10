"""Configuration management module for RAGLab.

Defines typed configuration schemas using Pydantic Settings and PyYAML.
Supports loading hierarchical configs for reproducible experiments across
ingestion, indexing, retrieval, generation, and evaluation pipelines.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class IngestionConfig(BaseModel):
    """Configuration schema for document ingestion and text splitting.

    Attributes:
        chunker: Chunking strategy to apply ('fixed' or 'recursive').
        chunk_size_tokens: Maximum allowed tokens per chunk.
        chunk_overlap_tokens: Number of overlapping tokens between consecutive chunks.
        tokenizer_encoding: Tiktoken encoding name (e.g. 'cl100k_base').
    """

    chunker: Literal["fixed", "recursive"] = "fixed"
    chunk_size_tokens: int = Field(default=500, gt=0, description="Max tokens per chunk")
    chunk_overlap_tokens: int = Field(
        default=50, ge=0, description="Token overlap between consecutive chunks"
    )
    tokenizer_encoding: str = Field(
        default="cl100k_base", description="Tiktoken encoding name"
    )


class IndexingConfig(BaseModel):
    """Configuration schema for embedding generation and vector/sparse indices."""

    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: Dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    """Configuration schema for dense, sparse, hybrid, and reranking retrieval."""

    strategy: str = "dense"
    top_k: int = 5


class GenerationConfig(BaseModel):
    """Configuration schema for LLM generation parameters and prompt templates."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 1024


class EvaluationConfig(BaseModel):
    """Configuration schema for retrieval and generation evaluation benchmarks."""

    dataset_path: str = "eval/gold_questions.json"
    results_dir: str = "eval/results"


class PipelineConfig(BaseModel):
    """Metadata describing the experiment pipeline."""

    name: str = "baseline_dense"
    description: str = "Standard dense retrieval pipeline"


class AppConfig(BaseSettings):
    """Global application and experiment configuration."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Load and validate an application configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AppConfig: Validated application configuration instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML parsing or validation fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(**data)
