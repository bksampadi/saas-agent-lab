"""Migration 0006 on a database that already holds licences, and assignments
and agent runs that reference them.

The column is added in place, not by rebuilding the table, so nothing that
references a licence has to move. In memory only.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from app.core.database import create_db_engine

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


@pytest.fixture
def engine() -> Iterator[Engine]:
    engine = create_db_engine("sqlite://", poolclass=StaticPool)
    yield engine
    engine.dispose()


def migrate(engine: Engine, revision: str, *, down: bool = False) -> None:
    with engine.begin() as connection:
        config = alembic_config(connection)
        if down:
            command.downgrade(config, revision)
        else:
            command.upgrade(config, revision)


def alembic_config(connection: Connection) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.attributes["connection"] = connection
    return config


def seed_0005(engine: Engine) -> None:
    """Two licences, one of them assigned and the target of a completed run
    with a tool call, as revision 0005 stores them."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, email, name, status, created_at) "
                "VALUES (1, 'ada@example.com', 'Ada', 'active', '2026-09-28')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO licences (id, product, seats_total) "
                "VALUES (1, 'Figma', 5), (2, 'Slack', 0)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO assignments (id, user_id, licence_id, assigned_at) "
                "VALUES (1, 1, 1, '2026-09-28')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO agent_runs (id, instruction, requesting_actor, status, "
                "goal_type, desired_state, extracted_user_email, extracted_product, "
                "resolved_user_id, resolved_licence_id, last_sequence_no, "
                "outcome_reason, outcome_detail, created_at, updated_at, "
                "completed_at) VALUES (7, 'Give ada@example.com Figma.', "
                "'admin@example.com', 'completed', 'ensure_assignment', 'assigned', "
                "'ada@example.com', 'Figma', 1, 1, 1, 'goal_satisfied', '{}', "
                "'2026-09-28', '2026-09-28', '2026-09-28')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO tool_calls (agent_run_id, sequence_no, tool_name, "
                "arguments, status, result, created_at, completed_at) "
                "VALUES (7, 1, 'assign_licence', '{}', 'succeeded', '{}', "
                "'2026-09-28', '2026-09-28')"
            )
        )


def migrated(engine: Engine) -> None:
    migrate(engine, "0005")
    seed_0005(engine)
    migrate(engine, "0006")


def test_upgrade_gives_every_existing_licence_the_allow_policy(
    engine: Engine,
) -> None:
    migrated(engine)

    with engine.connect() as connection:
        licences = connection.execute(
            text(
                "SELECT id, product, seats_total, agent_policy "
                "FROM licences ORDER BY id"
            )
        ).all()
        assignments = connection.execute(
            text("SELECT id, user_id, licence_id FROM assignments")
        ).all()
        runs = connection.execute(
            text("SELECT id, status, resolved_licence_id FROM agent_runs")
        ).all()
        tool_calls = connection.execute(
            text("SELECT agent_run_id, sequence_no FROM tool_calls")
        ).all()
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
        foreign_keys_on = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert [tuple(licence) for licence in licences] == [
        (1, "Figma", 5, "allow"),
        (2, "Slack", 0, "allow"),
    ]
    # Everything that references a licence is untouched.
    assert [tuple(row) for row in assignments] == [(1, 1, 1)]
    assert [tuple(row) for row in runs] == [(7, "completed", 1)]
    assert [tuple(row) for row in tool_calls] == [(7, 1)]
    assert violations == []
    assert foreign_keys_on == 1


@pytest.mark.parametrize("policy", ["allow", "require_approval", "deny"])
def test_the_upgraded_schema_accepts_every_policy(engine: Engine, policy: str) -> None:
    migrated(engine)

    with engine.begin() as connection:
        connection.execute(
            text("UPDATE licences SET agent_policy = :policy WHERE id = 1"),
            {"policy": policy},
        )


@pytest.mark.parametrize("policy", ["sometimes", "ALLOW", None])
def test_the_upgraded_schema_rejects_an_unknown_or_missing_policy(
    engine: Engine, policy: str | None
) -> None:
    migrated(engine)

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text("UPDATE licences SET agent_policy = :policy WHERE id = 1"),
            {"policy": policy},
        )


def test_downgrade_drops_the_policy_and_keeps_every_licence(engine: Engine) -> None:
    migrated(engine)

    migrate(engine, "0005", down=True)

    with engine.connect() as connection:
        licences = connection.execute(
            text("SELECT id, product, seats_total FROM licences ORDER BY id")
        ).all()
        columns = [
            row[1] for row in connection.execute(text("PRAGMA table_info(licences)"))
        ]
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
    assert [tuple(licence) for licence in licences] == [
        (1, "Figma", 5),
        (2, "Slack", 0),
    ]
    assert columns == ["id", "product", "seats_total"]
    assert violations == []


@pytest.mark.parametrize("policy", ["require_approval", "deny"])
def test_downgrade_refuses_to_drop_a_policy_other_than_allow(
    engine: Engine, policy: str
) -> None:
    migrated(engine)
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE licences SET agent_policy = :policy WHERE id = 2"),
            {"policy": policy},
        )

    with pytest.raises(RuntimeError, match="revision 0005 cannot store it"):
        migrate(engine, "0005", down=True)

    # Nothing was changed: the policy is still there, at 0006.
    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        stored = connection.execute(
            text("SELECT agent_policy FROM licences WHERE id = 2")
        ).scalar_one()
    assert version == "0006"
    assert stored == policy
