"""Migration 0007 on a database that already holds agent runs, model calls
and tool calls.

SQLite rebuilds agent_runs and tool_calls in this migration, with the trace
rows set aside meanwhile (as in 0005). In memory only.
"""

import pytest
from sqlalchemy import Connection, Engine, text

from support import migrate

OBSERVATION = '{"seats_active":0,"seats_available":5,"seats_total":5}'
ASSIGN_ARGUMENTS = '{"user_id": 1, "licence_id": 1}'


def seed_0006(engine: Engine) -> None:
    """A completed model-directed run with a model call and two tool calls,
    and a second resolved run still executing, as revision 0006 stores them."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, email, name, status, created_at) "
                "VALUES (1, 'ada@example.com', 'Ada', 'active', '2026-09-29')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO licences (id, product, seats_total, agent_policy) "
                "VALUES (1, 'Figma', 5, 'allow')"
            )
        )
        for run_id, status, reason, completed_at in (
            (7, "completed", "goal_satisfied", "2026-09-29"),
            (8, "executing", None, None),
        ):
            connection.execute(
                text(
                    "INSERT INTO agent_runs (id, instruction, requesting_actor, "
                    "status, goal_type, desired_state, extracted_user_email, "
                    "extracted_product, resolved_user_id, resolved_licence_id, "
                    "last_sequence_no, outcome_reason, decision_context, "
                    "created_at, updated_at, completed_at) VALUES (:id, "
                    "'Give ada@example.com Figma.', 'admin@example.com', :status, "
                    "'ensure_assignment', 'assigned', 'ada@example.com', 'Figma', "
                    "1, 1, 3, :reason, '{\"prompt\": \"p\"}', '2026-09-29', "
                    "'2026-09-29', :completed_at)"
                ),
                {
                    "id": run_id,
                    "status": status,
                    "reason": reason,
                    "completed_at": completed_at,
                },
            )
        connection.execute(
            text(
                "INSERT INTO model_calls (agent_run_id, sequence_no, stage, "
                "model_name, status, input_tokens, output_tokens, latency_ms, "
                "output, created_at) VALUES (7, 1, 'decision', 'claude-sonnet-5', "
                "'succeeded', 120, 15, 250, '{\"kind\": \"tool_calls\"}', "
                "'2026-09-29')"
            )
        )
        for sequence_no, tool_name, observation in (
            (2, "get_licence", OBSERVATION),
            (3, "assign_licence", '{"outcome":"assigned","reason_code":null}'),
        ):
            connection.execute(
                text(
                    "INSERT INTO tool_calls (agent_run_id, sequence_no, tool_name, "
                    "arguments, status, result, observation, created_at, "
                    "completed_at) VALUES (7, :n, :tool, :arguments, 'succeeded', "
                    "'{}', :observation, '2026-09-29', '2026-09-29')"
                ),
                {
                    "n": sequence_no,
                    "tool": tool_name,
                    "arguments": ASSIGN_ARGUMENTS,
                    "observation": observation,
                },
            )


def migrated(engine: Engine) -> None:
    migrate(engine, "0006")
    seed_0006(engine)
    migrate(engine, "0007")


def column_types(connection: Connection, table: str) -> dict[str, str]:
    return {
        row[1]: row[2]
        for row in connection.execute(text(f"PRAGMA table_info({table})"))
    }


def test_upgrade_keeps_every_run_and_trace_row_with_no_policy_decision(
    engine: Engine,
) -> None:
    migrated(engine)

    with engine.connect() as connection:
        runs = connection.execute(
            text("SELECT id, status, outcome_reason FROM agent_runs ORDER BY id")
        ).all()
        model_calls = connection.execute(
            text("SELECT agent_run_id, sequence_no FROM model_calls")
        ).all()
        tool_calls = connection.execute(
            text(
                "SELECT sequence_no, tool_name, status, policy_decision, observation "
                "FROM tool_calls ORDER BY sequence_no"
            )
        ).all()
        types = column_types(connection, "tool_calls")
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
        foreign_keys_on = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert [tuple(run) for run in runs] == [
        (7, "completed", "goal_satisfied"),
        (8, "executing", None),
    ]
    assert [tuple(call) for call in model_calls] == [(7, 1)]
    # No policy was evaluated for these calls, so none is recorded.
    assert [tuple(call) for call in tool_calls] == [
        (2, "get_licence", "succeeded", None, OBSERVATION),
        (
            3,
            "assign_licence",
            "succeeded",
            None,
            '{"outcome":"assigned","reason_code":null}',
        ),
    ]
    assert types["status"] == "VARCHAR(32)"
    assert types["policy_decision"] == "VARCHAR(32)"
    assert violations == []
    assert foreign_keys_on == 1


def test_downgrade_keeps_what_0006_can_store(engine: Engine) -> None:
    migrated(engine)

    migrate(engine, "0006", down=True)

    with engine.connect() as connection:
        runs = connection.execute(text("SELECT id FROM agent_runs ORDER BY id")).all()
        tool_calls = connection.execute(
            text("SELECT sequence_no, observation FROM tool_calls ORDER BY sequence_no")
        ).all()
        types = column_types(connection, "tool_calls")
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
    assert [tuple(run) for run in runs] == [(7,), (8,)]
    assert [tuple(call) for call in tool_calls] == [
        (2, OBSERVATION),
        (3, '{"outcome":"assigned","reason_code":null}'),
    ]
    assert "policy_decision" not in types
    assert types["status"] == "VARCHAR(16)"
    assert violations == []


@pytest.mark.parametrize(
    "policy_data",
    [
        "UPDATE agent_runs SET status = 'awaiting_approval' WHERE id = 8",
        "UPDATE agent_runs SET status = 'blocked', outcome_reason = 'policy_denied', "
        "completed_at = '2026-09-29' WHERE id = 8",
        "UPDATE tool_calls SET policy_decision = 'allow' WHERE sequence_no = 3",
    ],
    ids=["paused-run", "policy-denied-run", "recorded-decision"],
)
def test_downgrade_refuses_to_drop_policy_data(
    engine: Engine, policy_data: str
) -> None:
    migrated(engine)
    with engine.begin() as connection:
        connection.execute(text(policy_data))

    with pytest.raises(RuntimeError, match="revision 0006 cannot store them"):
        migrate(engine, "0006", down=True)

    # Nothing was changed: the data is still there, at 0007.
    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
    assert version == "0007"
