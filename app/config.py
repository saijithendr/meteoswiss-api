# app/config.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root (…/app/config.py -> …/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Read from environment variables, optionally overridden by a .env file
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---- App basics
    app_name: str = Field(default="MeteoSwiss API")
    environment: str = Field(default="development")  # e.g. development|staging|production
    debug: bool = Field(default=False)

    # ---- HTTP / server behavior
    request_timeout_seconds: int = Field(default=60, ge=1)
    gzip_minimum_size: int = Field(default=1000, ge=0)

    # ---- CORS (comma-separated in .env, e.g. "http://localhost:3000,https://example.com")
    cors_origins: List[str] = Field(default_factory=list)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    # Cache so the .env parsing and validation happens only once per process
    return Settings()


settings = get_settings()
