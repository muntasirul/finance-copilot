from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def default_app_dir() -> Path:
    return Path.home() / ".fincopilot"


class Settings(BaseSettings):
    """Runtime configuration. All values can be overridden with ``FINCOPILOT_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="FINCOPILOT_", env_file=".env", extra="ignore")

    app_dir: Path = default_app_dir()
    db_driver: Literal["sqlite", "sqlcipher"] = "sqlite"
    db_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    session_token: str = secrets.token_urlsafe(32)
    models_dir: Path = Path("models")
    categoriser_confidence_threshold: float = 0.55
    log_level: str = "INFO"

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path or (self.app_dir / "fincopilot.db")


settings = Settings()
