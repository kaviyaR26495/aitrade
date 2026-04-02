from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Zerodha
    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    ZERODHA_IP: str = ""  # optional proxy IP

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "password"
    DB_NAME: str = "aitrade"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # App
    SECRET_KEY: str = "change-me"
    DEBUG: bool = False

    # LLM / Chat
    GEMINI_API_KEY: str = ""

    # Trading defaults
    DEFAULT_STOPLOSS_PCT: float = 5.0
    DEFAULT_BUY_LIMIT: float = 10000.0
    MIN_TRADE_CONFIDENCE: float = 0.65
    DEFAULT_SEQ_LEN_DAILY: int = 15
    DEFAULT_SEQ_LEN_WEEKLY: int = 10
    DEFAULT_QUALITY_THRESHOLD: float = 0.8
    DEFAULT_MIN_PROFIT_THRESHOLD: float = 1.2

    # Intervals
    SUPPORTED_INTERVALS: tuple[str, ...] = ("day", "week")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def SYNC_DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def MODEL_DIR(self) -> Path:
        p = Path(__file__).resolve().parent.parent / "model_artifacts"
        p.mkdir(exist_ok=True)
        return p


settings = Settings()
