from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from fincopilot import __version__
from fincopilot.api.deps import AppState
from fincopilot.api.routes import router
from fincopilot.config import Settings, settings
from fincopilot.db.session import init_db, make_engine, make_session_factory
from fincopilot.ingestion.registry import default_registry
from fincopilot.ml.categoriser.infer import load_categoriser


def create_app(cfg: Settings | None = None, db_path: Path | str | None = None) -> FastAPI:
    cfg = cfg or settings
    logging.basicConfig(level=cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(db_path)
        init_db(engine)
        app.state.ctx = AppState(
            settings=cfg,
            session_factory=make_session_factory(engine),
            registry=default_registry(cfg.app_dir / "parsers"),
            categoriser=load_categoriser(cfg.models_dir),
        )
        yield
        engine.dispose()

    app = FastAPI(
        title="Finance Copilot sidecar",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()


def cli() -> None:
    import uvicorn

    print(f"FINCOPILOT_SESSION_TOKEN={settings.session_token}")  # consumed by the Tauri shell
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())
