"""Migrations must produce exactly the schema the models describe.

These run against in-memory SQLite only.
"""

from typing import Any

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect
from sqlalchemy.pool import StaticPool

from app.core.database import create_db_engine
from app.models import Base
from support import alembic_config


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


def test_upgrade_head_matches_models(engine: Engine) -> None:
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    assert diff == []


def test_upgrade_head_matches_model_constraint_names(engine: Engine) -> None:
    # compare_metadata does not look at CHECK constraints or constraint names,
    # so compare the reflected names against a create_all() schema directly.
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")

    reference = create_db_engine("sqlite://", poolclass=StaticPool)
    try:
        Base.metadata.create_all(reference)
        assert schema_names(engine) == schema_names(reference)
    finally:
        reference.dispose()


def test_downgrade_base_removes_all_tables(engine: Engine) -> None:
    with engine.begin() as connection:
        config = alembic_config(connection)
        command.upgrade(config, "head")
        command.downgrade(config, "base")

    assert inspect(engine).get_table_names() == ["alembic_version"]


def check_constraint_sql(engine: Engine) -> dict[str, dict[str | None, str]]:
    inspector = inspect(engine)
    return {
        table: {c["name"]: c["sqltext"] for c in inspector.get_check_constraints(table)}
        for table in inspector.get_table_names()
        if table != "alembic_version"
    }


def test_upgrade_head_matches_model_check_constraint_sql(engine: Engine) -> None:
    # Names alone would not catch a migration whose CHECK text drifted from
    # the model's.
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")

    reference = create_db_engine("sqlite://", poolclass=StaticPool)
    try:
        Base.metadata.create_all(reference)
        assert check_constraint_sql(engine) == check_constraint_sql(reference)
    finally:
        reference.dispose()


def test_downgrade_0002_leaves_the_initial_schema_and_upgrades_again(
    engine: Engine,
) -> None:
    with engine.begin() as connection:
        config = alembic_config(connection)
        command.upgrade(config, "head")
        command.downgrade(config, "0001")

    assert sorted(inspect(engine).get_table_names()) == [
        "alembic_version",
        "assignments",
        "audit_events",
        "licences",
        "users",
    ]

    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    assert diff == []
