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
    # Paused before a mutation that policy says needs a person's approval.
    AWAITING_APPROVAL = "awaiting_approval"
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
    # NEEDS_CLARIFICATION, at extraction: no goal was extracted
    UNSUPPORTED_REQUEST = "unsupported_request"
    INSTRUCTION_UNCLEAR = "instruction_unclear"
    # NEEDS_CLARIFICATION, at resolution
    INVALID_INPUT = "invalid_input"
    USER_NOT_FOUND = "user_not_found"
    LICENCE_NOT_FOUND = "licence_not_found"
    LICENCE_AMBIGUOUS = "licence_ambiguous"
    # BLOCKED
    NO_SEATS_AVAILABLE = "no_seats_available"
    USER_INACTIVE = "user_inactive"
    POLICY_DENIED = "policy_denied"  # policy denied the run's mutation
    # FAILED
    PLANNER_ERROR = "planner_error"  # at any model stage; the detail names it
    # No longer produced: tool calls take their ids from the persisted goal.
    # Kept for runs recorded before that, which the CHECK constraint allows.
    GOAL_SCOPE_VIOLATION = "goal_scope_violation"
    TOOL_FAILED = "tool_failed"
    VERIFICATION_FAILED = "verification_failed"
    STEP_LIMIT = "step_limit"  # a decision-stage limit was reached
    UNEXPECTED_ERROR = "unexpected_error"


class DecisionProposalKind(StrEnum):
    """What a decision model concluded. Its opinion, never the run's outcome."""

    GOAL_REACHED = "goal_reached"
    NO_ACTION_NEEDED = "no_action_needed"
    CANNOT_PROCEED = "cannot_proceed"


class CannotProceedReason(StrEnum):
    """The blocking conditions a decision model may claim. Each one is checked
    against current application state before it can block a run."""

    NO_SEATS_AVAILABLE = "no_seats_available"
    USER_INACTIVE = "user_inactive"


class AgentRun(Base):
    """One attempt to reach a goal on a person's behalf.

    Written by the agent executor in its own short transactions, never in a
    business transaction, so the run's history survives a rolled-back change.
    The CHECK constraints keep each row internally consistent; which status
    may follow which is enforced in code (``app.agent.status``).

    The goal columns (goal type, desired state, extracted text) are NULL
    while a natural-language instruction waits for extraction, and stay NULL
    if extraction ends the run. A RECEIVED run with them set is ready for
    resolution.
    """

    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "(resolved_user_id IS NULL) = (resolved_licence_id IS NULL)",
            name="resolved_ids_together",
        ),
        # A goal is extracted whole or not at all.
        CheckConstraint(
            "(goal_type IS NULL) = (desired_state IS NULL) "
            "AND (goal_type IS NULL) = (extracted_user_email IS NULL) "
            "AND (goal_type IS NULL) = (extracted_product IS NULL)",
            name="goal_columns_together",
        ),
        CheckConstraint(
            "resolved_user_id IS NULL OR goal_type IS NOT NULL",
            name="resolved_ids_need_goal",
        ),
        # These two are decided only by extraction, before any goal exists.
        # (planner_error is not among them: a later model stage can fail too.)
        # NULL NOT IN (...) is NULL and passes, which is right: a run without
        # an outcome yet may or may not have a goal.
        CheckConstraint(
            "outcome_reason NOT IN ('unsupported_request', 'instruction_unclear') "
            "OR goal_type IS NULL",
            name="extraction_outcome_has_no_goal",
        ),
        CheckConstraint("last_sequence_no >= 0", name="last_sequence_no_non_negative"),
        # A failed run may have failed before or after resolution.
        CheckConstraint(
            "(status IN ('resolved', 'executing', 'awaiting_approval', "
            "'verifying', 'completed', 'blocked') AND resolved_user_id IS NOT NULL) "
            "OR (status IN ('received', 'needs_clarification') "
            "AND resolved_user_id IS NULL) "
            "OR status = 'failed'",
            name="resolved_ids_match_status",
        ),
        CheckConstraint(
            "(completed_at IS NULL) = (status IN ('received', 'resolved', "
            "'executing', 'awaiting_approval', 'verifying'))",
            name="completed_at_iff_terminal",
        ),
        # "IS NOT NULL" is needed: NULL IN (...) is NULL, and a CHECK that
        # evaluates to NULL passes, so a terminal run with no reason would.
        CheckConstraint(
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
            name="outcome_matches_status",
        ),
        # The decision stage starts only on a resolved goal, and a proposal
        # is recorded only after its context is.
        CheckConstraint(
            "decision_context IS NULL OR resolved_user_id IS NOT NULL",
            name="decision_context_needs_resolution",
        ),
        CheckConstraint(
            "decision_proposal IS NULL OR decision_context IS NOT NULL",
            name="decision_proposal_needs_context",
        ),
        # A reason exactly when the proposal is cannot_proceed. A CASE,
        # because a comparison with NULL is NULL and NULL passes a CHECK; a
        # NULL proposal takes the ELSE branch.
        CheckConstraint(
            "CASE WHEN decision_proposal = 'cannot_proceed' "
            "THEN decision_reason_code IS NOT NULL "
            "ELSE decision_reason_code IS NULL END",
            name="decision_reason_matches_proposal",
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
    goal_type: Mapped[GoalType | None] = mapped_column(
        Enum(
            GoalType,
            name="agent_goal_type",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    desired_state: Mapped[DesiredState | None] = mapped_column(
        Enum(
            DesiredState,
            name="agent_desired_state",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    extracted_user_email: Mapped[str | None] = mapped_column(Text)
    extracted_product: Mapped[str | None] = mapped_column(Text)
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
    # The highest trace position handed out so far (0 before the first step).
    # Model calls and tool calls both take their sequence_no from this one
    # counter, so the run's whole trace has a single order; see
    # AgentRunRepository.next_sequence_no. It orders the trace and nothing
    # else: it is not a budget or limit on model requests or tool calls.
    last_sequence_no: Mapped[int] = mapped_column(default=0, server_default="0")
    # {"instructions": ..., "prompt": ...}: exactly what the decision model is
    # sent before its first request, so what it was told is known later
    # without re-rendering anything. Never contains a row id.
    decision_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True)
    )
    decision_proposal: Mapped[DecisionProposalKind | None] = mapped_column(
        Enum(
            DecisionProposalKind,
            name="agent_decision_proposal",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    decision_reason_code: Mapped[CannotProceedReason | None] = mapped_column(
        Enum(
            CannotProceedReason,
            name="agent_cannot_proceed_reason",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )
    # Set when the run reaches any terminal status, not only COMPLETED.
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
