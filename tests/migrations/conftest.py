"""Fixtures for the migration tests: Alembic, not the models, builds the
schema here."""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.pool import StaticPool

from app.core.database import create_db_engine


@pytest.fixture
def engine() -> Iterator[Engine]:
    """An empty in-memory database, with no tables until a migration runs."""
    engine = create_db_engine("sqlite://", poolclass=StaticPool)
    yield engine
    engine.dispose()
