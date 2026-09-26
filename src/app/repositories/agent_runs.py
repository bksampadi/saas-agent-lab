from sqlalchemy.orm import Session

from app.models import AgentRun


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
