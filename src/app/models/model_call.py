from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Enum, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UTCDateTime, enum_values, utcnow


class ModelCallStage(StrEnum):
    EXTRACTION = "extraction"  # instruction -> intent
    DECISION = "decision"  # choosing tool calls after resolution


class ModelCallStatus(StrEnum):
    SUCCEEDED = "succeeded"  # a response the application accepted
    FAILED = "failed"  # no response, or one the application rejected


class ModelCall(Base):
    """One request to a language model within an agent run.

    One row per request, retries included, written as soon as the request
    has an outcome. ``sequence_no`` shares one counter with the run's tool
    calls (AgentRun.last_sequence_no), so model calls and tool calls
    together form one ordered trace that never depends on timestamps.

    Nothing here is authoritative: ``output`` records what the model
    proposed and the application accepted as well-formed, not a fact about
    application state.
    """

    __tablename__ = "model_calls"
    __table_args__ = (
        UniqueConstraint(
            "agent_run_id",
            "sequence_no",
            name="uq_model_calls_agent_run_id_sequence_no",
        ),
        CheckConstraint("sequence_no >= 1", name="sequence_no_positive"),
        CheckConstraint("latency_ms >= 0", name="latency_ms_non_negative"),
        CheckConstraint(
            "(input_tokens IS NULL) = (output_tokens IS NULL)",
            name="token_counts_together",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR (input_tokens >= 0 AND output_tokens >= 0)",
            name="token_counts_non_negative",
        ),
        # A failed call may have no token counts: the request itself failed.
        CheckConstraint(
            "(status = 'succeeded' AND output IS NOT NULL AND error IS NULL "
            "AND input_tokens IS NOT NULL) "
            "OR (status = 'failed' AND output IS NULL AND error IS NOT NULL)",
            name="outcome_matches_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # No ON DELETE, as for tool calls: history is never removed silently.
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"))
    sequence_no: Mapped[int]
    stage: Mapped[ModelCallStage] = mapped_column(
        Enum(
            ModelCallStage,
            name="model_call_stage",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=16,
        )
    )
    model_name: Mapped[str] = mapped_column(String(200))
    status: Mapped[ModelCallStatus] = mapped_column(
        Enum(
            ModelCallStatus,
            name="model_call_status",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=16,
        )
    )
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    latency_ms: Mapped[int]
    # none_as_null, as for tool calls, so the CHECK constraint can test it.
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    # {"code", "message", "error_type"}: never a traceback or raw exception text.
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
