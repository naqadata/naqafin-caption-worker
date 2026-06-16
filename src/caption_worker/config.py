from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    worker_host: str = "0.0.0.0"
    worker_port: int = 8765
    caption_worker_api_key: str = ""

    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = ""
    model_cache_dir: Path = Path("models")

    job_storage_dir: Path = Path("storage")
    max_concurrent_jobs: int = Field(default=1, ge=1)
    job_retention_seconds: int = Field(default=86400, ge=60)
    max_upload_mb: int = Field(default=2048, ge=1)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.job_storage_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
