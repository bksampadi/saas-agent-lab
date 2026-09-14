"""Migrations must produce exactly the schema the models describe.

These run against in-memory SQLite only.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection, Engine, inspect
from sqlalchemy.pool import StaticPool

from app.core.database import create_db_engine
from app.models import Base

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


@pytest.fixture
def empty_engine() -> Iterator[Engine]:
    engine = create_db_engine("sqlite://", poolclass=StaticPool)
    yield engine
    engine.dispose()


def alembic_config(connection: Connection) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.attributes["connection"] = connection
    return config


def schema_names(engine: Engine) -> dict[str, dict[str, list[Any]]]:
    """Names of every constraint and index, per table."""
    inspector = inspect(engine)
    return {
        table: {
            "check": sorted(c["name"] for c in inspector.get_check_constraints(table)),
            "unique": sorted(
                c["name"] for c in inspector.get_unique_constraints(table)
            ),
            "foreign_key": sorted(c["name"] for c in inspector.get_foreign_keys(table)),
            "index": sorted(c["name"] for c in inspector.get_indexes(table)),
        }
        for table in inspector.get_table_names()
        if table != "alembic_version"
    }


def test_upgrade_head_matches_models(empty_engine: Engine) -> None:
    with empty_engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    assert diff == []


def test_upgrade_head_matches_model_constraint_names(empty_engine: Engine) -> None:
    # compare_metadata does not look at CHECK constraints or constraint names,
    # so compare the reflected names against a create_all() schema directly.
    with empty_engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")

    reference = create_db_engine("sqlite://", poolclass=StaticPool)
    try:
        Base.metadata.create_all(reference)
        assert schema_names(empty_engine) == schema_names(reference)
    finally:
        reference.dispose()


def test_downgrade_base_removes_all_tables(empty_engine: Engine) -> None:
    with empty_engine.begin() as connection:
        config = alembic_config(connection)
        command.upgrade(config, "head")
        command.downgrade(config, "base")

    assert inspect(empty_engine).get_table_names() == ["alembic_version"]
