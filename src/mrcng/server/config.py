from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRCNG_")

    source_root: Path
    cache_root: Path
    chunk_size: tuple[int, int, int] = (64, 64, 64)
    max_concurrent_reads: int = 32
    fd_cache_size: int = 256
    cors_origins: str = "*"
