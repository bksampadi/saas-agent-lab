from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import create_db_engine, get_session
from app.main import create_app
from app.models import Base


@pytest.fixture
def engine() -> Iterator[Engine]:
    # Each connection to "sqlite://" is a separate empty database. StaticPool
    # hands out one shared connection, so the test and the app (which
    # TestClient runs on another thread) see the same data.
    engine = create_db_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine, autoflush=False, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def client(session: Session) -> Iterator[TestClient]:
    app = create_app()

    def override_get_session() -> Iterator[Session]:
        yield session

    app.dependency_overrides[get_session] = override_get_session
    with TestClient(app) as test_client:
        yield test_client
