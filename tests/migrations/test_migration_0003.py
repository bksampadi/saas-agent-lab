"""Migration 0003 on databases that already hold agent runs and tool calls.

SQLite rebuilds agent_runs in this migration, which it refuses while rows
reference the table and foreign keys are enforced (as create_db_engine
enforces them). In memory only.
"""

import pytest
from sqlalchemy import Engine, text

from support import migrate


def seed_0002(engine: Engine) -> None:
    """One resolved run with two finished tool calls, as revision 0002 stores it."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, email, name, status, created_at) "
                "VALUES (1, 'ada@example.com', 'Ada', 'active', '2026-09-26')"
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
                "resolved_user_id, resolved_licence_id, created_at, updated_at) "
                "VALUES (7, 'Give Ada Figma.', 'admin@example.com', 'executing', "
                "'ensure_assignment', 'assigned', 'ada@example.com', 'Figma', "
                "1, 1, '2026-09-26', '2026-09-26')"
            )
        )
        for sequence_no in (1, 2):
            connection.execute(
                text(
                    "INSERT INTO tool_calls (agent_run_id, sequence_no, tool_name, "
                    "arguments, status, result, created_at, completed_at) "
                    "VALUES (7, :n, 'get_user', '{\"user_id\": 1}', 'succeeded', "
                    "'{}', '2026-09-26', '2026-09-26')"
                ),
                {"n": sequence_no},
            )


def test_upgrade_keeps_runs_and_tool_calls_and_continues_their_sequence(
    engine: Engine,
) -> None:
    migrate(engine, "0002")
    seed_0002(engine)

    migrate(engine, "head")

    with engine.connect() as connection:
        run = connection.execute(
            text(
                "SELECT status, goal_type, extracted_user_email, last_sequence_no "
                "FROM agent_runs WHERE id = 7"
            )
        ).one()
        calls = connection.execute(
            text(
                "SELECT agent_run_id, sequence_no, status FROM tool_calls "
                "ORDER BY sequence_no"
            )
        ).all()
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
        foreign_keys_on = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert tuple(run) == ("executing", "ensure_assignment", "ada@example.com", 2)
    assert [tuple(call) for call in calls] == [(7, 1, "succeeded"), (7, 2, "succeeded")]
    assert violations == []
    assert foreign_keys_on == 1


def test_downgrade_keeps_runs_that_revision_0002_can_store(engine: Engine) -> None:
    migrate(engine, "0002")
    seed_0002(engine)
    migrate(engine, "head")

    migrate(engine, "0002", down=True)

    with engine.connect() as connection:
        runs = connection.execute(text("SELECT id FROM agent_runs")).all()
        calls = connection.execute(text("SELECT COUNT(*) FROM tool_calls")).scalar()
    assert [tuple(run) for run in runs] == [(7,)]
    assert calls == 2


def test_downgrade_refuses_runs_without_an_extracted_goal(engine: Engine) -> None:
    migrate(engine, "head")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO agent_runs (instruction, requesting_actor, status, "
                "created_at, updated_at) VALUES ('Give Alice GitHub.', "
                "'admin@example.com', 'received', '2026-09-27', '2026-09-27')"
            )
        )

    with pytest.raises(RuntimeError, match="revision 0002 cannot store them"):
        migrate(engine, "0002", down=True)
