from collections.abc import Collection

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ToolCall, ToolCallStatus


class ToolCallRepository:
    """Persistence for tool calls. Never commits or rolls back."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, call: ToolCall) -> ToolCall:
        self._session.add(call)
        self._session.flush()  # sends the INSERT so call.id is assigned
        return call

    def get(self, call_id: int) -> ToolCall | None:
        return self._session.get(ToolCall, call_id)

    def next_sequence_no(self, agent_run_id: int) -> int:
        """1 for a run's first call, then one more than the highest so far.

        Correct only while a run's calls are made one at a time; two
        concurrent callers would both get the same number and the second
        INSERT would fail on the (agent_run_id, sequence_no) unique constraint.
        """
        statement = select(func.max(ToolCall.sequence_no)).where(
            ToolCall.agent_run_id == agent_run_id
        )
        return (self._session.scalar(statement) or 0) + 1

    def list_for_run(self, agent_run_id: int) -> list[ToolCall]:
        statement = (
            select(ToolCall)
            .where(ToolCall.agent_run_id == agent_run_id)
            .order_by(ToolCall.sequence_no)
        )
        return list(self._session.scalars(statement))

    def any_started(self, agent_run_id: int) -> bool:
        """True if a call of this run has no recorded outcome yet."""
        statement = select(ToolCall.id).where(
            ToolCall.agent_run_id == agent_run_id,
            ToolCall.status == ToolCallStatus.STARTED,
        )
        return self._session.scalars(statement).first() is not None

    def any_succeeded(self, agent_run_id: int, tool_names: Collection[str]) -> bool:
        statement = select(ToolCall.id).where(
            ToolCall.agent_run_id == agent_run_id,
            ToolCall.tool_name.in_(tool_names),
            ToolCall.status == ToolCallStatus.SUCCEEDED,
        )
        return self._session.scalars(statement).first() is not None
