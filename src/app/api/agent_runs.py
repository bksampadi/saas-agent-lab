"""Agent runs over HTTP: transport only.

POST runs the whole natural-language flow (app.agent.runs) and returns the
run's summary. GET returns what was persisted, with the trace as the models
saw it (app.schemas.agent_runs).

Both endpoints are plain ``def``: FastAPI runs them in a worker thread, so a
run's synchronous model requests never block the event loop. Neither takes
the request transaction: an agent run opens its own short transactions, and
none may be open while a model runs.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.agent.executor import AgentRunNotFound
from app.agent.planner import DecisionPlanner, IntentPlanner
from app.agent.runs import AgentRuns
from app.api.deps import (
    get_actor,
    get_agent_runs,
    get_decision_planner,
    get_intent_planner,
)
from app.schemas.agent_runs import AgentRunCreate, AgentRunDetail, AgentRunRead
from app.schemas.assignment import ID_MAX
from app.services.errors import InvalidInput

router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])

AgentRunsDep = Annotated[AgentRuns, Depends(get_agent_runs)]
IntentPlannerDep = Annotated[IntentPlanner, Depends(get_intent_planner)]
DecisionPlannerDep = Annotated[DecisionPlanner, Depends(get_decision_planner)]
ActorDep = Annotated[str, Depends(get_actor)]


@router.post("", response_model=AgentRunRead, status_code=status.HTTP_201_CREATED)
def create_agent_run(
    body: AgentRunCreate,
    runs: AgentRunsDep,
    intent_planner: IntentPlannerDep,
    decision_planner: DecisionPlannerDep,
    actor: ActorDep,
) -> AgentRunRead:
    """Run the instruction to a terminal status, or to an approval pause,
    synchronously.

    201 whenever a run was created: how it ended (completed, blocked, needs
    clarification, failed), or that it awaits approval, is in the body, not
    the status code.
    """
    try:
        record = runs.start(
            instruction=body.instruction,
            requesting_actor=actor,
            intent_planner=intent_planner,
            decision_planner=decision_planner,
        )
    except InvalidInput as error:
        # Reachable for a blank instruction or a whitespace-only actor
        # header; no run was created.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return AgentRunRead.of(record.run)


@router.get("/{run_id}", response_model=AgentRunDetail)
def get_agent_run(
    run_id: Annotated[int, Path(ge=1, le=ID_MAX)], runs: AgentRunsDep
) -> AgentRunDetail:
    try:
        record = runs.get(run_id)
    except AgentRunNotFound as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return AgentRunDetail.of(record.run, record.trace)
