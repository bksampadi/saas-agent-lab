"""policy checks

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-29 12:30:00.000000

"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import sqlalchemy as sa
from alembic import op
from alembic.operations import BatchOperations

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The columns as revision 0006 has them: rows set aside are put back into
# them, and a new column starts NULL.
TOOL_CALL_COLUMNS = (
    "id, agent_run_id, sequence_no, tool_name, arguments, status, result, error, "
    "observation, created_at, completed_at"
)
MODEL_CALL_COLUMNS = (
    "id, agent_run_id, sequence_no, stage, model_name, status, input_tokens, "
    "output_tokens, latency_ms, output, error, created_at"
)
TRACE_TABLES = (("tool_calls", TOOL_CALL_COLUMNS), ("model_calls", MODEL_CALL_COLUMNS))


@contextmanager
def trace_rows_set_aside() -> Iterator[None]:
    """Hold tool_calls and model_calls rows in temporary tables while SQLite
    rebuilds agent_runs (and tool_calls, then empty).

    As in 0005: batch mode on SQLite rebuilds a table (copy, drop, rename),
    and SQLite will not drop agent_runs while rows reference it with foreign
    keys enforced. Putting the rows back afterwards checks every reference,
    and every new CHECK, again. Other databases alter the tables in place and
    need none of this.
    """
    if op.get_bind().dialect.name != "sqlite":
        yield
        return
    for table, columns in TRACE_TABLES:
        op.execute(
            f"CREATE TEMPORARY TABLE {table}_0007 AS SELECT {columns} FROM {table}"
        )
        op.execute(f"DELETE FROM {table}")
    yield
    for table, columns in TRACE_TABLES:
        op.execute(
            f"INSERT INTO {table} ({columns}) SELECT {columns} FROM {table}_0007"
        )
        op.execute(f"DROP TABLE {table}_0007")


# Each CHECK is written as the models write it, so the schema matches
# create_all() exactly: {name: (revision 0006, revision 0007)}.
AGENT_RUN_CHECKS = {
    "ck_agent_runs_agent_run_status": (
        "status IN ('received', 'resolved', 'executing', 'verifying', "
        "'completed', 'needs_clarification', 'blocked', 'failed')",
        "status IN ('received', 'resolved', 'executing', 'awaiting_approval', "
        "'verifying', 'completed', 'needs_clarification', 'blocked', 'failed')",
    ),
    "ck_agent_runs_agent_outcome_reason": (
        "outcome_reason IN ('goal_satisfied', 'already_satisfied', "
        "'unsupported_request', 'instruction_unclear', 'invalid_input', "
        "'user_not_found', 'licence_not_found', 'licence_ambiguous', "
        "'no_seats_available', 'user_inactive', 'planner_error', "
        "'goal_scope_violation', 'tool_failed', 'verification_failed', "
        "'step_limit', 'unexpected_error')",
        "outcome_reason IN ('goal_satisfied', 'already_satisfied', "
        "'unsupported_request', 'instruction_unclear', 'invalid_input', "
        "'user_not_found', 'licence_not_found', 'licence_ambiguous', "
        "'no_seats_available', 'user_inactive', 'policy_denied', "
        "'planner_error', 'goal_scope_violation', 'tool_failed', "
        "'verification_failed', 'step_limit', 'unexpected_error')",
    ),
    "ck_agent_runs_resolved_ids_match_status": (
        "(status IN ('resolved', 'executing', 'verifying', 'completed', "
        "'blocked') AND resolved_user_id IS NOT NULL) "
        "OR (status IN ('received', 'needs_clarification') "
        "AND resolved_user_id IS NULL) "
        "OR status = 'failed'",
        "(status IN ('resolved', 'executing', 'awaiting_approval', "
        "'verifying', 'completed', 'blocked') AND resolved_user_id IS NOT NULL) "
        "OR (status IN ('received', 'needs_clarification') "
        "AND resolved_user_id IS NULL) "
        "OR status = 'failed'",
    ),
    "ck_agent_runs_completed_at_iff_terminal": (
        "(completed_at IS NULL) = "
        "(status IN ('received', 'resolved', 'executing', 'verifying'))",
        "(completed_at IS NULL) = (status IN ('received', 'resolved', "
        "'executing', 'awaiting_approval', 'verifying'))",
    ),
    "ck_agent_runs_outcome_matches_status": (
        "(status IN ('received', 'resolved', 'executing', 'verifying') "
        "AND outcome_reason IS NULL) "
        "OR (outcome_reason IS NOT NULL AND ("
        "(status = 'completed' "
        "AND outcome_reason IN ('goal_satisfied', 'already_satisfied')) "
        "OR (status = 'needs_clarification' AND outcome_reason IN "
        "('unsupported_request', 'instruction_unclear', "
        "'invalid_input', 'user_not_found', 'licence_not_found', "
        "'licence_ambiguous')) "
        "OR (status = 'blocked' "
        "AND outcome_reason IN ('no_seats_available', 'user_inactive')) "
        "OR (status = 'failed' AND outcome_reason IN ('planner_error', "
        "'goal_scope_violation', 'tool_failed', 'verification_failed', "
        "'step_limit', 'unexpected_error'))))",
        "(status IN ('received', 'resolved', 'executing', 'awaiting_approval', "
        "'verifying') AND outcome_reason IS NULL) "
        "OR (outcome_reason IS NOT NULL AND ("
        "(status = 'completed' "
        "AND outcome_reason IN ('goal_satisfied', 'already_satisfied')) "
        "OR (status = 'needs_clarification' AND outcome_reason IN "
        "('unsupported_request', 'instruction_unclear', "
        "'invalid_input', 'user_not_found', 'licence_not_found', "
        "'licence_ambiguous')) "
        "OR (status = 'blocked' AND outcome_reason IN "
        "('no_seats_available', 'user_inactive', 'policy_denied')) "
        "OR (status = 'failed' AND outcome_reason IN ('planner_error', "
        "'goal_scope_violation', 'tool_failed', 'verification_failed', "
        "'step_limit', 'unexpected_error'))))",
    ),
}
TOOL_CALL_CHECKS = {
    "ck_tool_calls_tool_call_status": (
        "status IN ('started', 'succeeded', 'failed')",
        "status IN ('awaiting_approval', 'started', 'succeeded', 'failed')",
    ),
    "ck_tool_calls_outcome_matches_status": (
        "(status = 'started' AND completed_at IS NULL "
        "AND result IS NULL AND error IS NULL) "
        "OR (status = 'succeeded' AND completed_at IS NOT NULL "
        "AND result IS NOT NULL AND error IS NULL) "
        "OR (status = 'failed' AND completed_at IS NOT NULL "
        "AND result IS NULL AND error IS NOT NULL)",
        "(status IN ('awaiting_approval', 'started') AND completed_at IS NULL "
        "AND result IS NULL AND error IS NULL) "
        "OR (status = 'succeeded' AND completed_at IS NOT NULL "
        "AND result IS NOT NULL AND error IS NULL) "
        "OR (status = 'failed' AND completed_at IS NOT NULL "
        "AND result IS NULL AND error IS NOT NULL)",
    ),
    "ck_tool_calls_observation_needs_outcome": (
        "observation IS NULL OR status != 'started'",
        "observation IS NULL OR status NOT IN ('awaiting_approval', 'started')",
    ),
}
NEW_TOOL_CALL_CHECKS = {
    "ck_tool_calls_policy_decision": (
        "policy_decision IN ('allow', 'require_approval', 'deny')"
    ),
    # NULL-safe: a NULL decision on an awaiting call must fail, not pass.
    "ck_tool_calls_awaiting_approval_was_required": (
        "status != 'awaiting_approval' OR (policy_decision IS NOT NULL "
        "AND policy_decision = 'require_approval')"
    ),
    "ck_tool_calls_denied_call_failed": (
        "policy_decision IS NULL OR policy_decision != 'deny' OR status = 'failed'"
    ),
}


def replace_checks(
    batch_op: BatchOperations, checks: dict[str, tuple[str, str]], *, upgrade: bool
) -> None:
    for name, (old, new) in checks.items():
        batch_op.drop_constraint(op.f(name), type_="check")
        batch_op.create_check_constraint(op.f(name), new if upgrade else old)


def upgrade() -> None:
    with trace_rows_set_aside():
        with op.batch_alter_table("tool_calls", schema=None) as batch_op:
            # "awaiting_approval" does not fit the old 16 characters.
            batch_op.alter_column(
                "status",
                existing_type=sa.VARCHAR(length=16),
                type_=sa.String(length=32),
                existing_nullable=False,
            )
            batch_op.add_column(
                sa.Column("policy_decision", sa.String(length=32), nullable=True)
            )
            replace_checks(batch_op, TOOL_CALL_CHECKS, upgrade=True)
            for name, condition in NEW_TOOL_CALL_CHECKS.items():
                batch_op.create_check_constraint(op.f(name), condition)
        with op.batch_alter_table("agent_runs", schema=None) as batch_op:
            replace_checks(batch_op, AGENT_RUN_CHECKS, upgrade=True)


def downgrade() -> None:
    # Revision 0006 cannot store a paused run, a held call, a policy denial
    # or any recorded policy decision. Refuse rather than delete them.
    remaining = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT (SELECT COUNT(*) FROM agent_runs "
                "WHERE status = 'awaiting_approval' "
                "OR outcome_reason = 'policy_denied') + "
                "(SELECT COUNT(*) FROM tool_calls "
                "WHERE status = 'awaiting_approval' OR policy_decision IS NOT NULL)"
            )
        )
        .scalar_one()
    )
    if remaining:
        raise RuntimeError(
            f"{remaining} agent run(s) or tool call(s) hold policy data; "
            "revision 0006 cannot store them. Remove them first."
        )

    with trace_rows_set_aside():
        with op.batch_alter_table("tool_calls", schema=None) as batch_op:
            for name in NEW_TOOL_CALL_CHECKS:
                batch_op.drop_constraint(op.f(name), type_="check")
            replace_checks(batch_op, TOOL_CALL_CHECKS, upgrade=False)
            batch_op.drop_column("policy_decision")
            batch_op.alter_column(
                "status",
                existing_type=sa.String(length=32),
                type_=sa.VARCHAR(length=16),
                existing_nullable=False,
            )
        with op.batch_alter_table("agent_runs", schema=None) as batch_op:
            replace_checks(batch_op, AGENT_RUN_CHECKS, upgrade=False)
