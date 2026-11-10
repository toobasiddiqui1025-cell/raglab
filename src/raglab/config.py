"""Configuration management module for RAGLab.

Defines typed configuration schemas using Pydantic Settings and PyYAML.
Supports loading hierarchical configs for reproducible experiments across
ingestion, indexing, retrieval, generation, and evaluation pipelines.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class IngestionConfig(BaseModel):
    """Configuration schema for document ingestion and text splitting."""


class IndexingConfig(BaseModel):
    """Configuration schema for embedding generation and vector/sparse indices."""


class RetrievalConfig(BaseModel):
    """Configuration schema for dense, sparse, hybrid, and reranking retrieval."""


class GenerationConfig(BaseModel):
    """Configuration schema for LLM generation parameters and prompt templates."""


class EvaluationConfig(BaseModel):
    """Configuration schema for retrieval and generation evaluation benchmarks."""


class AppConfig(BaseSettings):
    """Global application and experiment configuration."""


def load_config(config_path: str) -> AppConfig:
    """Load and validate an application configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AppConfig: Validated application configuration instance.
    """
    raise NotImplementedError
