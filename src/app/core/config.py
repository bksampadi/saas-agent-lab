from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables prefixed ``SAL_``."""

    model_config = SettingsConfigDict(
        env_prefix="SAL_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "SaaS Agent Lab"
    database_url: str = "sqlite:///./saas_agent_lab.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
