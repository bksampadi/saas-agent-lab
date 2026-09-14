from collections.abc import Iterator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def create_db_engine(url: str, **kwargs: Any) -> Engine:
    """Create an engine, applying the SQLite-specific settings we rely on."""
    if url.startswith("sqlite"):
        # FastAPI runs sync endpoints in a threadpool, so a pooled connection
        # may be used by a different thread than the one that opened it.
        kwargs.setdefault("connect_args", {"check_same_thread": False})

    engine = create_engine(url, **kwargs)

    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)

    return engine


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    # SQLite ignores foreign keys unless this is set on every new connection.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


engine = create_db_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    with SessionLocal() as session:
        yield session
