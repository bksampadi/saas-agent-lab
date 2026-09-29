"""Runs of one ensure-assignment goal, from start to a terminal status, or
to an approval pause if policy holds the run's mutation (AWAITING_APPROVAL).

``run_ensure_assignment`` and ``run_instruction`` are deterministic: test
scaffolding and the reference path, not a planner. Every step goes through
the executor, so the same checks apply as apply to a model: the goal is
resolved and persisted first, tool calls are scoped to it, and only the
verifier can complete the run. ``run_instruction`` starts from natural
language: a planner extracts the intent, and nothing after that step
involves a model.

``run_directed_instruction`` also starts from natural language, and after
resolution a decision planner chooses the tool calls (AgentExecutor.decide).
"""

from app.agent.executor import AgentExecutor
from app.agent.planner import DecisionPlanner, IntentPlanner
from app.models import AgentRunStatus
from app.schemas.agent import (
    AssignLicenceInput,
    ExtractedAssignmentIntent,
    GetLicenceInput,
    ListUserAssignmentsInput,
    UserAssignmentsSnapshot,
)


def run_ensure_assignment(
    executor: AgentExecutor,
    *,
    instruction: str,
    requesting_actor: str,
    intent: ExtractedAssignmentIntent,
) -> int:
    """Drive one run to a terminal status or an approval pause, and return
    its id."""
    run_id = executor.create_run(
        instruction=instruction, requesting_actor=requesting_actor, intent=intent
    )
    _resolve_and_execute(executor, run_id)
    return run_id


def run_instruction(
    executor: AgentExecutor,
    planner: IntentPlanner,
    *,
    instruction: str,
    requesting_actor: str,
) -> int:
    """Drive one natural-language run to a terminal status or an approval
    pause, and return its id.

    Only an extracted assignment intent goes on to resolution; any other
    extraction outcome has already ended the run, with nothing attempted.
    """
    run_id = executor.receive_run(
        instruction=instruction, requesting_actor=requesting_actor
    )
    if executor.extract_intent(run_id, planner) is AgentRunStatus.RECEIVED:
        _resolve_and_execute(executor, run_id)
    return run_id


def run_directed_instruction(
    executor: AgentExecutor,
    intent_planner: IntentPlanner,
    decision_planner: DecisionPlanner,
    *,
    instruction: str,
    requesting_actor: str,
) -> int:
    """Drive one natural-language run to a terminal status or an approval
    pause, letting ``decision_planner`` choose its tool calls, and return
    its id.

    Extraction and resolution are exactly as in run_instruction; only a
    RESOLVED run reaches the decision stage.
    """
    run_id = executor.receive_run(
        instruction=instruction, requesting_actor=requesting_actor
    )
    if executor.extract_intent(run_id, intent_planner) is not AgentRunStatus.RECEIVED:
        return run_id
    if executor.resolve_run(run_id) is AgentRunStatus.RESOLVED:
        executor.decide(run_id, decision_planner)
    return run_id


def _resolve_and_execute(executor: AgentExecutor, run_id: int) -> None:
    if executor.resolve_run(run_id) is not AgentRunStatus.RESOLVED:
        return
    goal = executor.get_goal(run_id)

    listing = executor.call_tool(run_id, ListUserAssignmentsInput(user_id=goal.user_id))
    if listing.run_status is not AgentRunStatus.EXECUTING:
        return
    # This only decides whether to attempt the change. It proves nothing:
    # the verifier reads current state again before the run can complete.
    already_assigned = isinstance(listing.output, UserAssignmentsSnapshot) and any(
        assignment.active and assignment.licence_id == goal.licence_id
        for assignment in listing.output.assignments
    )

    if not already_assigned:
        # Recorded, never acted on: a free seat now is not a reservation, and
        # assign_licence checks capacity again when it inserts.
        capacity = executor.call_tool(
            run_id, GetLicenceInput(licence_id=goal.licence_id)
        )
        if capacity.run_status is not AgentRunStatus.EXECUTING:
            return
        assignment = executor.call_tool(
            run_id,
            AssignLicenceInput(user_id=goal.user_id, licence_id=goal.licence_id),
        )
        if assignment.run_status is not AgentRunStatus.EXECUTING:
            return

    executor.verify_and_finish(run_id)
