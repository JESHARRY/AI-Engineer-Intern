"""Application configuration using Pydantic Settings."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for Autonomous Lead Enrichment Agent."""

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    request_timeout_seconds: int = 15
    max_subpages: int = 5
    max_context_tokens: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def get_settings() -> Settings:
    """Return application settings instance."""
    return settings
