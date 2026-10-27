from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session, sessionmaker

from fincopilot.config import Settings
from fincopilot.ingestion.registry import ParserRegistry
from fincopilot.ml.categoriser.infer import Categoriser


@dataclass
class AppState:
    settings: Settings
    session_factory: sessionmaker[Session]
    registry: ParserRegistry
    categoriser: Categoriser


def get_state(request: Request) -> AppState:
    return request.app.state.ctx  # type: ignore[no-any-return]


def require_token(
    state: AppState = Depends(get_state), authorization: str | None = Header(default=None)
) -> None:
    """Sidecar auth: Tauri passes a per-launch token; reject everything else."""
    expected = f"Bearer {state.settings.session_token}"
    if authorization != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid session token")


def get_db(state: AppState = Depends(get_state)) -> Iterator[Session]:
    db = state.session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
