"""Natural-language agent runs, as callers outside the agent layer use them:
start one and read one back.

Nothing here decides anything. ``start`` is AgentExecutor.run, so the
executor opens and closes its own short transactions, and no session is
open while a model runs. That is why this takes a session factory rather
than a session: a caller's transaction must never wrap a run. ``get`` reads
what the executor persisted, never current application state.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from app.agent.executor import AgentExecutor, AgentRunNotFound
from app.agent.planner import DecisionPlanner, IntentPlanner
from app.models import AgentRun, ModelCall, ToolCall
from app.repositories.agent_runs import AgentRunRepository


@dataclass(frozen=True)
class AgentRunRecord:
    """A run and its trace, as persisted, read in one session. Internal: the
    rows carry resolved ids and internal results."""

    run: AgentRun
    trace: list[ModelCall | ToolCall]  # in sequence_no order


class AgentRuns:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def start(
        self,
        *,
        instruction: str,
        requesting_actor: str,
        intent_planner: IntentPlanner,
        decision_planner: DecisionPlanner,
    ) -> AgentRunRecord:
        """Drive one natural-language run to a terminal status, or to an
        approval pause (AWAITING_APPROVAL), and return it.

        Raises InvalidInput, recording nothing, for a blank or over-long
        instruction or an invalid requesting actor. Once the run exists, an
        expected failure (unclear instruction, unknown user, planner error,
        limit, failed verification...) is its persisted outcome, not an
        exception.
        """
        run_id = AgentExecutor(self._sessions).run(
            instruction=instruction,
            requesting_actor=requesting_actor,
            intent_planner=intent_planner,
            decision_planner=decision_planner,
        )
        return self.get(run_id)

    def get(self, run_id: int) -> AgentRunRecord:
        """The run and its trace. Raises AgentRunNotFound."""
        with self._sessions() as read:
            runs = AgentRunRepository(read)
            run = runs.get(run_id)
            if run is None:
                raise AgentRunNotFound(run_id)
            return AgentRunRecord(run=run, trace=runs.list_trace(run_id))
