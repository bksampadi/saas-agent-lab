"""model calls and extraction

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-27 20:47:49.338173

"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TOOL_CALL_COLUMNS = (
    "id, agent_run_id, sequence_no, tool_name, arguments, status, result, error, "
    "created_at, completed_at"
)


@contextmanager
def tool_calls_set_aside() -> Iterator[None]:
    """Hold tool_calls rows in a temporary table while SQLite rebuilds
    agent_runs.

    Batch mode on SQLite rebuilds the table: copy, drop, rename. SQLite will
    not drop a table that rows still reference while foreign keys are
    enforced, and PRAGMA foreign_keys cannot change inside a transaction.
    Putting the rows back afterwards checks every reference again. Other
    databases alter the table in place and need none of this.
    """
    if op.get_bind().dialect.name != "sqlite":
        yield
        return
    op.execute(
        f"CREATE TEMPORARY TABLE tool_calls_0003 AS "
        f"SELECT {TOOL_CALL_COLUMNS} FROM tool_calls"
    )
    op.execute("DELETE FROM tool_calls")
    yield
    op.execute(
        f"INSERT INTO tool_calls ({TOOL_CALL_COLUMNS}) "
        f"SELECT {TOOL_CALL_COLUMNS} FROM tool_calls_0003"
    )
    op.execute("DROP TABLE tool_calls_0003")


OUTCOME_REASON_0002 = (
    "outcome_reason IN ('goal_satisfied', 'already_satisfied', 'invalid_input', "
    "'user_not_found', 'licence_not_found', 'licence_ambiguous', "
    "'no_seats_available', 'user_inactive', 'goal_scope_violation', "
    "'tool_failed', 'verification_failed', 'unexpected_error')"
)
OUTCOME_REASON_0003 = (
    "outcome_reason IN ('goal_satisfied', 'already_satisfied', "
    "'unsupported_request', 'instruction_unclear', 'invalid_input', "
    "'user_not_found', 'licence_not_found', 'licence_ambiguous', "
    "'no_seats_available', 'user_inactive', 'planner_error', "
    "'goal_scope_violation', 'tool_failed', 'verification_failed', "
    "'unexpected_error')"
)

OUTCOME_MATCHES_STATUS_0002 = (
    "(status IN ('received', 'resolved', 'executing', 'verifying') "
    "AND outcome_reason IS NULL) "
    "OR (outcome_reason IS NOT NULL AND ("
    "(status = 'completed' "
    "AND outcome_reason IN ('goal_satisfied', 'already_satisfied')) "
    "OR (status = 'needs_clarification' AND outcome_reason IN "
    "('invalid_input', 'user_not_found', 'licence_not_found', "
    "'licence_ambiguous')) "
    "OR (status = 'blocked' "
    "AND outcome_reason IN ('no_seats_available', 'user_inactive')) "
    "OR (status = 'failed' AND outcome_reason IN ('goal_scope_violation', "
    "'tool_failed', 'verification_failed', 'unexpected_error'))))"
)
OUTCOME_MATCHES_STATUS_0003 = (
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

NEW_AGENT_RUN_CHECKS = {
    "ck_agent_runs_goal_columns_together": (
        "(goal_type IS NULL) = (desired_state IS NULL) "
        "AND (goal_type IS NULL) = (extracted_user_email IS NULL) "
        "AND (goal_type IS NULL) = (extracted_product IS NULL)"
    ),
    "ck_agent_runs_resolved_ids_need_goal": (
        "resolved_user_id IS NULL OR goal_type IS NOT NULL"
    ),
    "ck_agent_runs_extraction_outcome_has_no_goal": (
        "outcome_reason NOT IN ('unsupported_request', 'instruction_unclear') "
        "OR goal_type IS NULL"
    ),
    "ck_agent_runs_last_sequence_no_non_negative": "last_sequence_no >= 0",
}


def upgrade() -> None:
    op.create_table(
        "model_calls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_run_id", sa.Integer(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column(
            "stage",
            sa.Enum(
                "extraction",
                "decision",
                name="model_call_stage",
                native_enum=False,
                create_constraint=True,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "succeeded",
                "failed",
                name="model_call_status",
                native_enum=False,
                create_constraint=True,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("output", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("error", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(status = 'succeeded' AND output IS NOT NULL AND error IS NULL "
            "AND input_tokens IS NOT NULL) "
            "OR (status = 'failed' AND output IS NULL AND error IS NOT NULL)",
            name=op.f("ck_model_calls_outcome_matches_status"),
        ),
        sa.CheckConstraint(
            "(input_tokens IS NULL) = (output_tokens IS NULL)",
            name=op.f("ck_model_calls_token_counts_together"),
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR (input_tokens >= 0 AND output_tokens >= 0)",
            name=op.f("ck_model_calls_token_counts_non_negative"),
        ),
        sa.CheckConstraint(
            "latency_ms >= 0", name=op.f("ck_model_calls_latency_ms_non_negative")
        ),
        sa.CheckConstraint(
            "sequence_no >= 1", name=op.f("ck_model_calls_sequence_no_positive")
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            name=op.f("fk_model_calls_agent_run_id_agent_runs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_calls")),
        sa.UniqueConstraint(
            "agent_run_id",
            "sequence_no",
            name="uq_model_calls_agent_run_id_sequence_no",
        ),
    )

    with (
        tool_calls_set_aside(),
        op.batch_alter_table("agent_runs", schema=None) as batch_op,
    ):
        batch_op.add_column(
            sa.Column(
                "last_sequence_no", sa.Integer(), server_default="0", nullable=False
            )
        )
        # NULL until a natural-language run's intent is extracted.
        batch_op.alter_column(
            "goal_type", existing_type=sa.VARCHAR(length=32), nullable=True
        )
        batch_op.alter_column(
            "desired_state", existing_type=sa.VARCHAR(length=32), nullable=True
        )
        batch_op.alter_column(
            "extracted_user_email", existing_type=sa.TEXT(), nullable=True
        )
        batch_op.alter_column(
            "extracted_product", existing_type=sa.TEXT(), nullable=True
        )
        # New outcome reasons: the enum's CHECK and the status pairing.
        batch_op.drop_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), OUTCOME_REASON_0003
        )
        batch_op.drop_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), OUTCOME_MATCHES_STATUS_0003
        )
        for name, condition in NEW_AGENT_RUN_CHECKS.items():
            batch_op.create_check_constraint(op.f(name), condition)

    # Existing runs have only tool calls, numbered 1..n: continue from n.
    op.execute(
        "UPDATE agent_runs SET last_sequence_no = COALESCE("
        "(SELECT MAX(tool_calls.sequence_no) FROM tool_calls "
        "WHERE tool_calls.agent_run_id = agent_runs.id), 0)"
    )


def downgrade() -> None:
    # Revision 0002 cannot represent a run without a goal or with an
    # extraction outcome. Refuse rather than delete them.
    remaining = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM agent_runs WHERE goal_type IS NULL "
                "OR outcome_reason IN "
                "('unsupported_request', 'instruction_unclear', 'planner_error')"
            )
        )
        .scalar_one()
    )
    if remaining:
        raise RuntimeError(
            f"{remaining} agent run(s) have no extracted goal or an extraction "
            "outcome; revision 0002 cannot store them. Remove them first."
        )

    op.drop_table("model_calls")

    with (
        tool_calls_set_aside(),
        op.batch_alter_table("agent_runs", schema=None) as batch_op,
    ):
        for name in NEW_AGENT_RUN_CHECKS:
            batch_op.drop_constraint(op.f(name), type_="check")
        batch_op.drop_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_outcome_matches_status"), OUTCOME_MATCHES_STATUS_0002
        )
        batch_op.drop_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_agent_runs_agent_outcome_reason"), OUTCOME_REASON_0002
        )
        batch_op.alter_column(
            "extracted_product", existing_type=sa.TEXT(), nullable=False
        )
        batch_op.alter_column(
            "extracted_user_email", existing_type=sa.TEXT(), nullable=False
        )
        batch_op.alter_column(
            "desired_state", existing_type=sa.VARCHAR(length=32), nullable=False
        )
        batch_op.alter_column(
            "goal_type", existing_type=sa.VARCHAR(length=32), nullable=False
        )
        batch_op.drop_column("last_sequence_no")
