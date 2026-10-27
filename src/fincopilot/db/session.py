from __future__ import annotations

import base64
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from argon2.low_level import Type, hash_secret_raw
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from fincopilot.config import settings
from fincopilot.db.models import Base

_SALT_FILE = "db.salt"


def derive_key(passphrase: str, salt: bytes) -> str:
    """Argon2id-derive a 32-byte key and return it hex-encoded for SQLCipher's PRAGMA key."""
    raw = hash_secret_raw(
        passphrase.encode(),
        salt,
        time_cost=3,
        memory_cost=64 * 1024,
        parallelism=2,
        hash_len=32,
        type=Type.ID,
    )
    return raw.hex()


def load_or_create_salt(app_dir: Path) -> bytes:
    app_dir.mkdir(parents=True, exist_ok=True)
    p = app_dir / _SALT_FILE
    if p.exists():
        return base64.b64decode(p.read_text())
    salt = secrets.token_bytes(16)
    p.write_text(base64.b64encode(salt).decode())
    return salt


def make_engine(db_path: Path | str | None = None, passphrase: str | None = None) -> Engine:
    """Create an engine. SQLCipher is used only when FINCOPILOT_DB_DRIVER=sqlcipher."""
    path = Path(db_path) if db_path else settings.resolved_db_path
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    if settings.db_driver == "sqlcipher":
        if passphrase is None:
            raise ValueError("passphrase required for sqlcipher driver")
        import sqlcipher3  # type: ignore[import-not-found]

        key = derive_key(passphrase, load_or_create_salt(settings.app_dir))
        engine = create_engine(
            "sqlite+pysqlcipher://", creator=lambda: sqlcipher3.connect(str(path))
        )

        @event.listens_for(engine, "connect")
        def _set_key(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute(f"PRAGMA key = \"x'{key}'\"")
            dbapi_conn.execute("PRAGMA cipher_memory_security = ON")

        return engine

    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        dbapi_conn.execute("PRAGMA journal_mode = WAL")
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
