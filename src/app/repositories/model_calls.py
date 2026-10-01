from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ModelCall, ModelCallStage


class ModelCallRepository:
    """Persistence for model calls. Never commits or rolls back."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, call: ModelCall) -> ModelCall:
        self._session.add(call)
        self._session.flush()  # sends the INSERT so call.id is assigned
        return call

    def count_for_stage(self, agent_run_id: int, stage: ModelCallStage) -> int:
        """How many requests the run's ``stage`` has made, failed ones included."""
        statement = select(func.count(ModelCall.id)).where(
            ModelCall.agent_run_id == agent_run_id, ModelCall.stage == stage
        )
        return self._session.execute(statement).scalar_one()
