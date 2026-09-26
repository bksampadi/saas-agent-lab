"""A deterministic, model-free run of one ensure-assignment goal.

Test scaffolding and the reference path a planner will follow; not a planner.
Every step goes through the executor, so the same checks apply as will apply
to a model: the goal is resolved and persisted first, tool calls are scoped
to it, and only the verifier can complete the run.
"""

from app.agent.executor import AgentExecutor
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
    """Drive one run to a terminal status and return its id."""
    run_id = executor.create_run(
        instruction=instruction, requesting_actor=requesting_actor, intent=intent
    )
    if executor.resolve_run(run_id) is not AgentRunStatus.RESOLVED:
        return run_id
    goal = executor.get_goal(run_id)

    listing = executor.call_tool(run_id, ListUserAssignmentsInput(user_id=goal.user_id))
    if listing.run_status is not AgentRunStatus.EXECUTING:
        return run_id
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
            return run_id
        assignment = executor.call_tool(
            run_id,
            AssignLicenceInput(user_id=goal.user_id, licence_id=goal.licence_id),
        )
        if assignment.run_status is not AgentRunStatus.EXECUTING:
            return run_id

    executor.verify_and_finish(run_id)
    return run_id
