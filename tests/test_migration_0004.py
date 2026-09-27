"""Migration 0004 on a database that already holds agent runs, model calls
and tool calls. SQLite rebuilds tool_calls here. In memory only."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, text
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


def seed_0003(engine: Engine) -> None:
    """One resolved run with a model call and two finished tool calls in one
    trace, as revision 0003 stores it."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, email, name, status, created_at) "
                "VALUES (1, 'ada@example.com', 'Ada', 'active', '2026-09-27')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO licences (id, product, seats_total) VALUES (1, 'Figma', 5)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO agent_runs (id, instruction, requesting_actor, status, "
                "goal_type, desired_state, extracted_user_email, extracted_product, "
                "resolved_user_id, resolved_licence_id, last_sequence_no, "
                "created_at, updated_at) VALUES (7, 'Give ada@example.com Figma.', "
                "'admin@example.com', 'executing', 'ensure_assignment', 'assigned', "
                "'ada@example.com', 'Figma', 1, 1, 3, '2026-09-27', '2026-09-27')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO model_calls (agent_run_id, sequence_no, stage, "
                "model_name, status, input_tokens, output_tokens, latency_ms, "
                "output, created_at) VALUES (7, 1, 'extraction', "
                "'claude-sonnet-5', 'succeeded', 120, 15, 250, "
                "'{\"kind\": \"ensure_assignment\"}', '2026-09-27')"
            )
        )
        for sequence_no in (2, 3):
            connection.execute(
                text(
                    "INSERT INTO tool_calls (agent_run_id, sequence_no, tool_name, "
                    "arguments, status, result, created_at, completed_at) "
                    "VALUES (7, :n, 'get_licence', '{\"licence_id\": 1}', "
                    "'succeeded', '{}', '2026-09-27', '2026-09-27')"
                ),
                {"n": sequence_no},
            )


def test_upgrade_keeps_every_tool_call_without_an_observation(engine: Engine) -> None:
    migrate(engine, "0003")
    seed_0003(engine)

    migrate(engine, "0004")

    with engine.connect() as connection:
        calls = connection.execute(
            text(
                "SELECT agent_run_id, sequence_no, status, observation "
                "FROM tool_calls ORDER BY sequence_no"
            )
        ).all()
        model_calls = connection.execute(
            text("SELECT COUNT(*) FROM model_calls")
        ).scalar_one()
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
        foreign_keys_on = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert [tuple(call) for call in calls] == [
        (7, 2, "succeeded", None),
        (7, 3, "succeeded", None),
    ]
    assert model_calls == 1
    assert violations == []
    assert foreign_keys_on == 1


def test_downgrade_keeps_tool_calls_without_an_observation(engine: Engine) -> None:
    migrate(engine, "0003")
    seed_0003(engine)
    migrate(engine, "0004")

    migrate(engine, "0003", down=True)

    with engine.connect() as connection:
        calls = connection.execute(text("SELECT COUNT(*) FROM tool_calls")).scalar()
        columns = [
            row[1] for row in connection.execute(text("PRAGMA table_info(tool_calls)"))
        ]
    assert calls == 2
    assert "observation" not in columns


def test_downgrade_refuses_to_drop_an_observation(engine: Engine) -> None:
    migrate(engine, "0003")
    seed_0003(engine)
    migrate(engine, "0004")
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE tool_calls SET observation = '{}' WHERE sequence_no = 2")
        )

    with pytest.raises(RuntimeError, match="revision 0003 cannot store them"):
        migrate(engine, "0003", down=True)

    with engine.connect() as connection:
        kept = connection.execute(
            text("SELECT COUNT(*) FROM tool_calls WHERE observation IS NOT NULL")
        ).scalar_one()
    assert kept == 1
