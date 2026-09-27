from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import AgentRun, ModelCall, ToolCall


class AgentRunRepository:
    """Persistence for agent runs. Never commits or rolls back."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, run: AgentRun) -> AgentRun:
        self._session.add(run)
        self._session.flush()  # sends the INSERT so run.id is assigned
        return run

    def get(self, run_id: int) -> AgentRun | None:
        return self._session.get(AgentRun, run_id)

    def next_sequence_no(self, run_id: int) -> int:
        """Hand out the run's next trace position: 1, 2, 3, ...

        One counter per run, shared by model calls and tool calls. The
        increment happens in the database, in a single UPDATE, so two
        concurrent callers get different numbers: the second waits for the
        first's row lock (Postgres) or write lock (SQLite). A number is used
        by at most one row: each table's (agent_run_id, sequence_no) unique
        constraint catches a repeat within it, and the counter prevents one
        across the two tables.
        """
        statement = (
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(last_sequence_no=AgentRun.last_sequence_no + 1)
            .returning(AgentRun.last_sequence_no)
        )
        return self._session.execute(statement).scalar_one()

    def list_trace(self, run_id: int) -> list[ModelCall | ToolCall]:
        """The run's model calls and tool calls, in sequence_no order."""
        entries: list[ModelCall | ToolCall] = [
            *self._session.scalars(
                select(ModelCall).where(ModelCall.agent_run_id == run_id)
            ),
            *self._session.scalars(
                select(ToolCall).where(ToolCall.agent_run_id == run_id)
            ),
        ]
        return sorted(entries, key=lambda entry: entry.sequence_no)
