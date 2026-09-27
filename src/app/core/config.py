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
    # "provider:model", as PydanticAI names models. The provider's own
    # credentials (e.g. ANTHROPIC_API_KEY) are read by its SDK, not here.
    planner_model: str = "anthropic:claude-sonnet-5"
    planner_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
