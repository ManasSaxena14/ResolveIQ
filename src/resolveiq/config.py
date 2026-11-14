"""Configuration loading and validation.

Every tunable setting lives in ``configs/config.yaml``. Secrets such as API keys
are read from environment variables (optionally via a local ``.env`` file) and
are never stored in the YAML file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CONFIG_ENV_VAR = "RESOLVEIQ_CONFIG"

ProviderName = Literal["ollama", "gemini", "groq", "openrouter"]


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys, so typos in the YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_StrictModel):
    name: str = "ResolveIQ"
    company_name: str


class RagConfig(_StrictModel):
    embedding_model: str
    query_prefix: str = ""
    collection: str
    chunk_tokens: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    candidate_k: int = Field(gt=0)
    hybrid: bool
    rerank: bool
    reranker_model: str
    refusal_threshold: float = Field(ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _check_sizes(self) -> RagConfig:
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("rag.chunk_overlap must be smaller than rag.chunk_tokens")
        if self.candidate_k < self.top_k:
            raise ValueError("rag.candidate_k must be greater than or equal to rag.top_k")
        return self


class ProviderConfig(_StrictModel):
    model: str
    base_url: str
    api_key_env: str | None = None


class LLMConfig(_StrictModel):
    provider: ProviderName
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    providers: dict[ProviderName, ProviderConfig]

    @model_validator(mode="after")
    def _check_active_provider(self) -> LLMConfig:
        if self.provider not in self.providers:
            raise ValueError(f"llm.providers has no entry for active provider '{self.provider}'")
        return self

    @property
    def active(self) -> ProviderConfig:
        """Settings of the provider selected by ``llm.provider``."""
        return self.providers[self.provider]


class PathsConfig(_StrictModel):
    kb_dir: Path
    index_dir: Path
    results_dir: Path


class TrackingConfig(_StrictModel):
    enabled: bool = True
    tracking_uri: str


class Settings(_StrictModel):
    project: ProjectConfig
    rag: RagConfig
    llm: LLMConfig
    paths: PathsConfig
    tracking: TrackingConfig

    @staticmethod
    def resolve(path: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> Settings:
    """Load and validate settings.

    The config file is chosen in this order: the ``path`` argument, the
    ``RESOLVEIQ_CONFIG`` environment variable, then ``configs/config.yaml``.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the default settings, loaded once per process."""
    return load_config()
