from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Enum,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UTCDateTime, enum_values, utcnow
from app.models.licence import PolicyDecision


class ToolCallStatus(StrEnum):
    # Held before it runs: policy requires a person's approval. No business
    # transaction has opened for it.
    AWAITING_APPROVAL = "awaiting_approval"
    STARTED = "started"  # its business transaction may have opened
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ToolCall(Base):
    """One tool invocation within an agent run, in the order it was made.

    ``sequence_no`` (1, 2, 3, ... per run) is the order of the trace; it never
    depends on timestamps. Runs execute their tool calls one at a time, so the
    executor assigns the next number; the unique constraint turns any
    accidental concurrent allocation into an error instead of a tie.

    ``result`` and ``error`` are internal and may hold row ids.
    ``observation`` is set only on a call a decision model made: the exact
    text the model was given as the call's result, id-free by construction.
    ``policy_decision`` is what policy decided when the call was admitted
    (app.agent.policy): set for a mutating call that passed the goal-scope
    and limit checks, NULL for a read and for a call refused before policy.
    """

    __tablename__ = "tool_calls"
    __table_args__ = (
        # Named explicitly: the "uq" naming convention would only include the
        # first column. Its index also serves lookups by agent_run_id.
        UniqueConstraint(
            "agent_run_id", "sequence_no", name="uq_tool_calls_agent_run_id_sequence_no"
        ),
        CheckConstraint("sequence_no >= 1", name="sequence_no_positive"),
        CheckConstraint(
            "(status IN ('awaiting_approval', 'started') AND completed_at IS NULL "
            "AND result IS NULL AND error IS NULL) "
            "OR (status = 'succeeded' AND completed_at IS NOT NULL "
            "AND result IS NOT NULL AND error IS NULL) "
            "OR (status = 'failed' AND completed_at IS NOT NULL "
            "AND result IS NULL AND error IS NOT NULL)",
            name="outcome_matches_status",
        ),
        # A model sees a call's result only once the call has one.
        CheckConstraint(
            "observation IS NULL OR status NOT IN ('awaiting_approval', 'started')",
            name="observation_needs_outcome",
        ),
        # A call waits for approval only when policy required it. "IS NOT
        # NULL" is needed: NULL = 'require_approval' is NULL, and a CHECK
        # that evaluates to NULL passes, so a NULL decision would.
        CheckConstraint(
            "status != 'awaiting_approval' OR (policy_decision IS NOT NULL "
            "AND policy_decision = 'require_approval')",
            name="awaiting_approval_was_required",
        ),
        # A denied call never runs. A NULL decision (a read, or a call refused
        # before policy) takes the first branch explicitly.
        CheckConstraint(
            "policy_decision IS NULL OR policy_decision != 'deny' OR status = 'failed'",
            name="denied_call_failed",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # No ON DELETE: deleting a run that has tool calls fails, so history is
    # never removed silently.
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"))
    sequence_no: Mapped[int]
    tool_name: Mapped[str] = mapped_column(String(64))
    # none_as_null: None is stored as SQL NULL, not the JSON text 'null', so
    # the CHECK constraint above can test it.
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON(none_as_null=True))
    status: Mapped[ToolCallStatus] = mapped_column(
        Enum(
            ToolCallStatus,
            name="tool_call_status",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    policy_decision: Mapped[PolicyDecision | None] = mapped_column(
        Enum(
            PolicyDecision,
            name="policy_decision",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        )
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    # {"code", "message", "error_type"}: never a traceback or raw exception text.
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    # Text, not JSON: stored exactly as sent, so it can be compared byte for
    # byte with what crossed the model boundary.
    observation: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
