"""decision stage

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-27 23:10:00.000000

"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

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
    rebuilds agent_runs.

    As in 0003: batch mode on SQLite rebuilds a table (copy, drop, rename),
    and SQLite will not drop agent_runs while rows reference it with foreign
    keys enforced. Both trace tables reference it now. Putting the rows back
    afterwards checks every reference again. Other databases alter the table
    in place and need none of this.
    """
    if op.get_bind().dialect.name != "sqlite":
        yield
        return
    for table, columns in TRACE_TABLES:
        op.execute(
            f"CREATE TEMPORARY TABLE {table}_0005 AS SELECT {columns} FROM {table}"
        )
        op.execute(f"DELETE FROM {table}")
    yield
    for table, columns in TRACE_TABLES:
        op.execute(
            f"INSERT INTO {table} ({columns}) SELECT {columns} FROM {table}_0005"
        )
        op.execute(f"DROP TABLE {table}_0005")


OUTCOME_REASON_0004 = (
    "outcome_reason IN ('goal_satisfied', 'already_satisfied', "
    "'unsupported_request', 'instruction_unclear', 'invalid_input', "
    "'user_not_found', 'licence_not_found', 'licence_ambiguous', "
    "'no_seats_available', 'user_inactive', 'planner_error', "
    "'goal_scope_violation', 'tool_failed', 'verification_failed', "
    "'unexpected_error')"
)
OUTCOME_REASON_0005 = (
    "outcome_reason IN ('goal_satisfied', 'already_satisfied', "
    "'unsupported_request', 'instruction_unclear', 'invalid_input', "
    "'user_not_found', 'licence_not_found', 'licence_ambiguous', "
    "'no_seats_available', 'user_inactive', 'planner_error', "
    "'goal_scope_violation', 'tool_failed', 'verification_failed', "
    "'step_limit', 'unexpected_error')"
)

OUTCOME_MATCHES_STATUS_0004 = (
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
    "'unexpected_error'))))"
)
OUTCOME_MATCHES_STATUS_0005 = (
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
    "'step_limit', 'unexpected_error'))))"
)

# The new columns' own CHECKs are written as SQLAlchemy writes an Enum's, so
# the schema matches create_all() exactly.
NEW_AGENT_RUN_CHECKS = {
    "ck_agent_runs_agent_decision_proposal": (
        "decision_proposal IN ('goal_reached', 'no_action_needed', 'cannot_proceed')"
    ),
    "ck_agent_runs_agent_cannot_proceed_reason": (
        "decision_reason_code IN ('no_seats_available', 'user_inactive')"
    ),
    "ck_agent_runs_decision_context_needs_resolution": (
        "decision_context IS NULL OR resolved_user_id IS NOT NULL"
    ),
    "ck_agent_runs_decision_proposal_needs_context": (
        "decision_proposal IS NULL OR decision_context IS NOT NULL"
    ),
    "ck_agent_runs_decision_reason_matches_proposal": (
        "CASE WHEN decision_proposal = 'cannot_proceed' "
        "THEN decision_reason_code IS NOT NULL "
        "ELSE decision_reason_code IS NULL END"
    ),
}


def upgrade() -> None:
    with (
        trace_rows_set_aside(),
        op.batch_alter_table("agent_runs", schema=None) as batch_op,
    ):
        batch_op.add_column(
            sa.Column("decision_context", sa.JSON(none_as_null=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("decision_proposal", sa.String(length=32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("decision_reason_code", sa.String(length=32), nullable=True)
        )
        # step_limit: the enum's CHECK and the status pairing.
        batch_op.drop_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), OUTCOME_REASON_0005
        )
        batch_op.drop_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), OUTCOME_MATCHES_STATUS_0005
        )
        for name, condition in NEW_AGENT_RUN_CHECKS.items():
            batch_op.create_check_constraint(op.f(name), condition)


def downgrade() -> None:
    # Revision 0004 cannot store a decision stage (its context and proposal)
    # or a step_limit outcome. Refuse rather than delete them.
    remaining = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM agent_runs WHERE decision_context IS NOT NULL "
                "OR decision_proposal IS NOT NULL OR outcome_reason = 'step_limit'"
            )
        )
        .scalar_one()
    )
    if remaining:
        raise RuntimeError(
            f"{remaining} agent run(s) hold decision-stage data; revision 0004 "
            "cannot store them. Remove them first."
        )

    with (
        trace_rows_set_aside(),
        op.batch_alter_table("agent_runs", schema=None) as batch_op,
    ):
        for name in NEW_AGENT_RUN_CHECKS:
            batch_op.drop_constraint(op.f(name), type_="check")
        batch_op.drop_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), OUTCOME_MATCHES_STATUS_0004
        )
        batch_op.drop_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), OUTCOME_REASON_0004
        )
        batch_op.drop_column("decision_reason_code")
        batch_op.drop_column("decision_proposal")
        batch_op.drop_column("decision_context")
