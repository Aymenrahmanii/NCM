"""Service configuration from environment variables (or a .env file). Nothing hard-coded."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NCM_", env_file=".env", extra="ignore")

    artifacts_dir: str = "artifacts"
    data_dir: str = "data"
    device: str = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    # "real" loads the GPU models at startup; "fake" uses tiny stand-ins (tests, CI, no GPU)
    backend: Literal["real", "fake"] = "real"

    request_timeout_s: float = 60.0
    rag_top_k: int = 3

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
