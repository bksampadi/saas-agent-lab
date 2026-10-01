"""The four tools a decision model can call, and what it is shown of each.

No tool takes an argument. The model chooses which tool; every call acts on
the run's resolved goal, its user and its licence, so nothing a model says
can name a row. A call's result exists in three forms, kept apart:

    run()        the internal result: a snapshot, or a ToolError, carrying
                 row ids; for application code only
    observe()    the model-visible observation: a closed, id-free DTO
    serialize()  its exact text, persisted with the call
                 (ToolCall.observation) and given to the model as it is

A rejection is shown as a closed code chosen from the error's code, never
from an exception's message, which may name row ids. Business rules
(capacity, duplicates, user status) stay in the services; nothing here
re-checks them.
"""

import json
from typing import assert_never

from app.schemas.agent import (
    AssignmentAttemptObservation,
    AssignmentRejectionCode,
    AssignmentSnapshot,
    LicenceCapacityObservation,
    LicenceSnapshot,
    ModelObservation,
    ResolvedAssignmentGoal,
    TargetAssignmentsObservation,
    TargetToolName,
    TargetUserObservation,
    ToolError,
    ToolOutput,
    UserAssignmentsSnapshot,
    UserSnapshot,
)
from app.services.assignments import AssignmentService
from app.services.users import UserService

# Each tool's calls are recorded under the name of the application operation
# it runs (ToolCall.tool_name). The model, and the public trace, use the
# tool's own name.
RECORDED_NAMES: dict[TargetToolName, str] = {
    "get_target_user": "get_user",
    "get_target_licence_capacity": "get_licence",
    "list_target_user_assignments": "list_user_assignments",
    "assign_target_licence": "assign_licence",
}

# The one tool that changes anything. Policy governs it, and it counts
# against the run's mutation limit; the others count as reads.
MUTATION: TargetToolName = "assign_target_licence"
RECORDED_MUTATIONS = frozenset({RECORDED_NAMES[MUTATION]})
RECORDED_READS = frozenset(RECORDED_NAMES.values()) - RECORDED_MUTATIONS

# The assignment rejections a model is shown, by ToolError code: the domain
# rules it may react to. Any other failure is shown nothing; it ends the run.
REJECTION_CODES: dict[str, AssignmentRejectionCode] = {
    "no_seats_available": "no_seats_available",
    "user_inactive": "user_inactive",
    "assignment_already_exists": "already_assigned",
}


def arguments(tool: TargetToolName, goal: ResolvedAssignmentGoal) -> dict[str, int]:
    """The ids a call of ``tool`` acts on, as recorded with the call."""
    match tool:
        case "get_target_user" | "list_target_user_assignments":
            return {"user_id": goal.user_id}
        case "get_target_licence_capacity":
            return {"licence_id": goal.licence_id}
        case "assign_target_licence":
            return {"user_id": goal.user_id, "licence_id": goal.licence_id}
        case _:
            assert_never(tool)


def run(
    tool: TargetToolName,
    goal: ResolvedAssignmentGoal,
    *,
    users: UserService,
    assignments: AssignmentService,
    actor: str,
) -> ToolOutput:
    """Run ``tool`` on the goal's user and licence, through services bound
    to the caller's business transaction. ``actor`` ("agent:run-<id>") is
    the audit actor of a change."""
    match tool:
        case "get_target_user":
            return UserSnapshot.of(users.get_user(goal.user_id))
        case "get_target_licence_capacity":
            return LicenceSnapshot.of(assignments.get_seat_usage(goal.licence_id))
        case "list_target_user_assignments":
            rows = assignments.list_assignments_for_user(goal.user_id)
            return UserAssignmentsSnapshot(
                user_id=goal.user_id,
                assignments=[AssignmentSnapshot.of(row) for row in rows],
            )
        case "assign_target_licence":
            # The existing service call, audit event included, unchanged.
            assignment = assignments.assign_licence(
                user_id=goal.user_id, licence_id=goal.licence_id, actor=actor
            )
            return AssignmentSnapshot.of(assignment)
        case _:
            assert_never(tool)


def observe(
    tool: TargetToolName,
    goal: ResolvedAssignmentGoal,
    outcome: ToolOutput | ToolError,
) -> ModelObservation | None:
    """What the model is shown of a call's ``outcome``, or None for a
    failure it is not shown (the executor then ends the run).

    ``goal`` is used only to pick out the target licence among the user's
    assignments; its ids never reach the observation.
    """
    if isinstance(outcome, ToolError):
        code = REJECTION_CODES.get(outcome.code)
        if tool == MUTATION and code is not None:
            return AssignmentAttemptObservation(outcome="rejected", reason_code=code)
        return None
    match outcome:
        case UserSnapshot():
            return TargetUserObservation(status=outcome.status)
        case LicenceSnapshot():
            return LicenceCapacityObservation(
                seats_total=outcome.seats_total,
                seats_active=outcome.seats_active,
                seats_available=outcome.seats_available,
            )
        case UserAssignmentsSnapshot():
            return TargetAssignmentsObservation(
                holds_active_seat=any(
                    assignment.active and assignment.licence_id == goal.licence_id
                    for assignment in outcome.assignments
                )
            )
        case AssignmentSnapshot():
            return AssignmentAttemptObservation(outcome="assigned", reason_code=None)
        case _:
            assert_never(outcome)


def serialize(observation: ModelObservation) -> str:
    """The observation as text: JSON with sorted keys, no spaces, ASCII only.
    The same observation always gives the same string."""
    return json.dumps(
        observation.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
