from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UTCDateTime, enum_values, utcnow


class AgentRunStatus(StrEnum):
    RECEIVED = "received"
    RESOLVED = "resolved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    BLOCKED = "blocked"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset(
    {
        AgentRunStatus.COMPLETED,
        AgentRunStatus.NEEDS_CLARIFICATION,
        AgentRunStatus.BLOCKED,
        AgentRunStatus.FAILED,
    }
)


class GoalType(StrEnum):
    ENSURE_ASSIGNMENT = "ensure_assignment"


class DesiredState(StrEnum):
    """The state the verifier checks for. Redundant with the goal type today,
    but stored explicitly so the persisted goal says what "done" means."""

    ASSIGNED = "assigned"


class OutcomeReason(StrEnum):
    # COMPLETED
    GOAL_SATISFIED = "goal_satisfied"  # a mutating tool call succeeded
    ALREADY_SATISFIED = "already_satisfied"  # no mutating tool call succeeded
    # NEEDS_CLARIFICATION
    INVALID_INPUT = "invalid_input"
    USER_NOT_FOUND = "user_not_found"
    LICENCE_NOT_FOUND = "licence_not_found"
    LICENCE_AMBIGUOUS = "licence_ambiguous"
    # BLOCKED
    NO_SEATS_AVAILABLE = "no_seats_available"
    USER_INACTIVE = "user_inactive"
    # FAILED
    GOAL_SCOPE_VIOLATION = "goal_scope_violation"
    TOOL_FAILED = "tool_failed"
    VERIFICATION_FAILED = "verification_failed"
    UNEXPECTED_ERROR = "unexpected_error"


class AgentRun(Base):
    """One attempt to reach a goal on a person's behalf.

    Written by the agent executor in its own short transactions, never in a
    business transaction, so the run's history survives a rolled-back change.
    The CHECK constraints keep each row internally consistent; which status
    may follow which is enforced in code (``app.agent.status``).
    """

    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "(resolved_user_id IS NULL) = (resolved_licence_id IS NULL)",
            name="resolved_ids_together",
        ),
        # A failed run may have failed before or after resolution.
        CheckConstraint(
            "(status IN ('resolved', 'executing', 'verifying', 'completed', "
            "'blocked') AND resolved_user_id IS NOT NULL) "
            "OR (status IN ('received', 'needs_clarification') "
            "AND resolved_user_id IS NULL) "
            "OR status = 'failed'",
            name="resolved_ids_match_status",
        ),
        CheckConstraint(
            "(completed_at IS NULL) = "
            "(status IN ('received', 'resolved', 'executing', 'verifying'))",
            name="completed_at_iff_terminal",
        ),
        # "IS NOT NULL" is needed: NULL IN (...) is NULL, and a CHECK that
        # evaluates to NULL passes, so a terminal run with no reason would.
        CheckConstraint(
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
            "'tool_failed', 'verification_failed', 'unexpected_error'))))",
            name="outcome_matches_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Stored exactly as given. Text columns: extracted text is untrusted and
    # may not fit the domain's column sizes (the resolver rejects that).
    instruction: Mapped[str] = mapped_column(Text)
    requesting_actor: Mapped[str] = mapped_column(String(320))
    status: Mapped[AgentRunStatus] = mapped_column(
        Enum(
            AgentRunStatus,
            name="agent_run_status",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    goal_type: Mapped[GoalType] = mapped_column(
        Enum(
            GoalType,
            name="agent_goal_type",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    desired_state: Mapped[DesiredState] = mapped_column(
        Enum(
            DesiredState,
            name="agent_desired_state",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    extracted_user_email: Mapped[str] = mapped_column(Text)
    extracted_product: Mapped[str] = mapped_column(Text)
    resolved_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    resolved_licence_id: Mapped[int | None] = mapped_column(ForeignKey("licences.id"))
    outcome_reason: Mapped[OutcomeReason | None] = mapped_column(
        Enum(
            OutcomeReason,
            name="agent_outcome_reason",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    # none_as_null: None is stored as SQL NULL, not the JSON text 'null'.
    outcome_detail: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True)
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )
    # Set when the run reaches any terminal status, not only COMPLETED.
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
