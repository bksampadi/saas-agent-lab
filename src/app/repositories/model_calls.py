from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ModelCall
from app.repositories.agent_runs import AgentRunRepository


class ModelCallRepository:
    """Persistence for model calls. Never commits or rolls back."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, call: ModelCall) -> ModelCall:
        self._session.add(call)
        self._session.flush()  # sends the INSERT so call.id is assigned
        return call

    def next_sequence_no(self, agent_run_id: int) -> int:
        """The run's next trace position, shared with its tool calls."""
        return AgentRunRepository(self._session).next_sequence_no(agent_run_id)

    def list_for_run(self, agent_run_id: int) -> list[ModelCall]:
        statement = (
            select(ModelCall)
            .where(ModelCall.agent_run_id == agent_run_id)
            .order_by(ModelCall.sequence_no)
        )
        return list(self._session.scalars(statement))
