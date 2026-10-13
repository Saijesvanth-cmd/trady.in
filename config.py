from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = "sqlite:///./trady_dev.db"

    anthropic_api_key: str = ""
    anthropic_model: str = ""

    market_data_provider: str = "demo"  # demo | upstox (upstox arrives in Phase 2)
    upstox_analytics_token: str = ""

    default_paper_balance: float = 100_000
    paper_trading_enabled: bool = True
    live_order_execution_enabled: bool = False

    market_data_cache_seconds: int = 60
    max_upload_mb: int = 5
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _never_live(self):
        if self.live_order_execution_enabled:
            raise ValueError(
                "LIVE_ORDER_EXECUTION_ENABLED must stay false: trady.in never places real orders."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
