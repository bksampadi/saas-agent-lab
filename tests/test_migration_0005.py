"""Migration 0005 on databases that already hold agent runs, model calls and
tool calls, including model-visible observations.

SQLite rebuilds agent_runs in this migration, which it refuses while rows
reference the table and foreign keys are enforced (as create_db_engine
enforces them). In memory only.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, text
from sqlalchemy.pool import StaticPool

from app.core.database import create_db_engine

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"
OBSERVATION = '{"seats_active":0,"seats_available":5,"seats_total":5}'


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


def seed_0004(engine: Engine) -> None:
    """One extracted, resolved run with a model call and two tool calls in one
    trace, one of them with an observation, and one run failed at
    extraction, as revision 0004 stores them."""
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
                "outcome_reason, outcome_detail, created_at, updated_at, "
                "completed_at) VALUES (7, 'Give ada@example.com Figma.', "
                "'admin@example.com', 'completed', 'ensure_assignment', 'assigned', "
                "'ada@example.com', 'Figma', 1, 1, 3, 'goal_satisfied', '{}', "
                "'2026-09-27', '2026-09-27', '2026-09-27')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO agent_runs (id, instruction, requesting_actor, status, "
                "last_sequence_no, outcome_reason, created_at, updated_at, "
                "completed_at) VALUES (8, 'Hm.', 'admin@example.com', 'failed', 1, "
                "'planner_error', '2026-09-27', '2026-09-27', '2026-09-27')"
            )
        )
        for run_id in (7, 8):
            connection.execute(
                text(
                    "INSERT INTO model_calls (agent_run_id, sequence_no, stage, "
                    "model_name, status, input_tokens, output_tokens, latency_ms, "
                    "output, created_at) VALUES (:run, 1, 'extraction', "
                    "'claude-sonnet-5', 'succeeded', 120, 15, 250, "
                    "'{\"kind\": \"unsupported\"}', '2026-09-27')"
                ),
                {"run": run_id},
            )
        for sequence_no, tool_name, observation in (
            (2, "get_licence", OBSERVATION),
            (3, "assign_licence", None),
        ):
            connection.execute(
                text(
                    "INSERT INTO tool_calls (agent_run_id, sequence_no, tool_name, "
                    "arguments, status, result, observation, created_at, "
                    "completed_at) VALUES (7, :n, :tool, '{\"licence_id\": 1}', "
                    "'succeeded', '{}', :observation, '2026-09-27', '2026-09-27')"
                ),
                {"n": sequence_no, "tool": tool_name, "observation": observation},
            )


def migrated(engine: Engine) -> None:
    migrate(engine, "0004")
    seed_0004(engine)
    migrate(engine, "0005")


def test_upgrade_keeps_every_run_and_trace_row(engine: Engine) -> None:
    migrated(engine)

    with engine.connect() as connection:
        runs = connection.execute(
            text(
                "SELECT id, status, outcome_reason, last_sequence_no, "
                "decision_context, decision_proposal, decision_reason_code "
                "FROM agent_runs ORDER BY id"
            )
        ).all()
        model_calls = connection.execute(
            text("SELECT agent_run_id, sequence_no, stage FROM model_calls ORDER BY id")
        ).all()
        tool_calls = connection.execute(
            text(
                "SELECT agent_run_id, sequence_no, tool_name, observation "
                "FROM tool_calls ORDER BY sequence_no"
            )
        ).all()
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
        foreign_keys_on = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert [tuple(run) for run in runs] == [
        (7, "completed", "goal_satisfied", 3, None, None, None),
        (8, "failed", "planner_error", 1, None, None, None),
    ]
    assert [tuple(call) for call in model_calls] == [
        (7, 1, "extraction"),
        (8, 1, "extraction"),
    ]
    # The observation is kept byte for byte.
    assert [tuple(call) for call in tool_calls] == [
        (7, 2, "get_licence", OBSERVATION),
        (7, 3, "assign_licence", None),
    ]
    assert violations == []
    assert foreign_keys_on == 1


def test_the_upgraded_schema_accepts_decision_stage_rows(engine: Engine) -> None:
    migrated(engine)

    with engine.begin() as connection:
        connection.execute(
            text(
                'UPDATE agent_runs SET decision_context = \'{"prompt": "p"}\', '
                "decision_proposal = 'cannot_proceed', "
                "decision_reason_code = 'no_seats_available' WHERE id = 7"
            )
        )


def test_downgrade_keeps_runs_and_observations_that_0004_can_store(
    engine: Engine,
) -> None:
    migrated(engine)

    migrate(engine, "0004", down=True)

    with engine.connect() as connection:
        runs = connection.execute(text("SELECT id FROM agent_runs ORDER BY id")).all()
        model_calls = connection.execute(
            text("SELECT COUNT(*) FROM model_calls")
        ).scalar()
        observations = connection.execute(
            text("SELECT observation FROM tool_calls ORDER BY sequence_no")
        ).all()
        columns = [
            row[1] for row in connection.execute(text("PRAGMA table_info(agent_runs)"))
        ]
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
    assert [tuple(run) for run in runs] == [(7,), (8,)]
    assert model_calls == 2
    assert [tuple(row) for row in observations] == [(OBSERVATION,), (None,)]
    assert "decision_context" not in columns
    assert violations == []


@pytest.mark.parametrize(
    "decision_data",
    [
        "UPDATE agent_runs SET status = 'executing', outcome_reason = NULL, "
        "outcome_detail = NULL, completed_at = NULL, "
        'decision_context = \'{"prompt": "p"}\' WHERE id = 7',
        "UPDATE agent_runs SET status = 'failed', outcome_reason = 'step_limit' "
        "WHERE id = 7",
    ],
    ids=["decision-context", "step-limit"],
)
def test_downgrade_refuses_to_drop_decision_stage_data(
    engine: Engine, decision_data: str
) -> None:
    migrated(engine)
    with engine.begin() as connection:
        connection.execute(text(decision_data))

    with pytest.raises(RuntimeError, match="revision 0004 cannot store them"):
        migrate(engine, "0004", down=True)

    # Nothing was changed: the data is still there, at 0005.
    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
    assert version == "0005"
