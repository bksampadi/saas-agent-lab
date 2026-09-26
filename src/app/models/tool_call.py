from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Enum, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UTCDateTime, enum_values, utcnow


class ToolCallStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ToolCall(Base):
    """One tool invocation within an agent run, in the order it was made.

    ``sequence_no`` (1, 2, 3, ... per run) is the order of the trace; it never
    depends on timestamps. Runs execute their tool calls one at a time, so the
    executor assigns the next number; the unique constraint turns any
    accidental concurrent allocation into an error instead of a tie.
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
            "(status = 'started' AND completed_at IS NULL "
            "AND result IS NULL AND error IS NULL) "
            "OR (status = 'succeeded' AND completed_at IS NOT NULL "
            "AND result IS NOT NULL AND error IS NULL) "
            "OR (status = 'failed' AND completed_at IS NOT NULL "
            "AND result IS NULL AND error IS NOT NULL)",
            name="outcome_matches_status",
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
            length=16,
        )
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    # {"code", "message", "error_type"}: never a traceback or raw exception text.
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
