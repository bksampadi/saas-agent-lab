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

    def count_for_tools(self, agent_run_id: int, tool_names: Collection[str]) -> int:
        """How many calls of these tools the run has recorded, whatever their
        outcome: each one was an attempt."""
        statement = select(func.count(ToolCall.id)).where(
            ToolCall.agent_run_id == agent_run_id, ToolCall.tool_name.in_(tool_names)
        )
        return self._session.execute(statement).scalar_one()

    def latest_for_tools(
        self, agent_run_id: int, tool_names: Collection[str]
    ) -> ToolCall | None:
        """The run's most recent call of any of these tools, in trace order."""
        statement = (
            select(ToolCall)
            .where(
                ToolCall.agent_run_id == agent_run_id,
                ToolCall.tool_name.in_(tool_names),
            )
            .order_by(ToolCall.sequence_no.desc())
        )
        return self._session.scalars(statement).first()
