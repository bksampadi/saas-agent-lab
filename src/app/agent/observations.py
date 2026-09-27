"""The model boundary for tool results: internal result in, observation out.

A tool call's internal result (a snapshot, or a ToolError) is for
application code and may carry row ids. A decision model is shown something
else: a closed, id-free observation built here, serialized once, persisted
with the call, and handed to the model as that same text.

    internal result or ToolError
            |  observe()
    observation DTO (no id fields, extra="forbid")
            |  serialize()
    text  ->  ToolCall.observation  ->  the model's tool result

A rejection is described by a closed code chosen from the error's code,
never from an exception's message, which may name row ids.
"""

import json
from typing import assert_never

from app.schemas.agent import (
    AssignLicenceInput,
    AssignmentAttemptObservation,
    AssignmentRejectionCode,
    AssignmentSnapshot,
    LicenceCapacityObservation,
    LicenceSnapshot,
    ModelObservation,
    ResolvedAssignmentGoal,
    TargetAssignmentsObservation,
    TargetUserObservation,
    ToolError,
    ToolInput,
    ToolOutput,
    UserAssignmentsSnapshot,
    UserSnapshot,
)

# The assignment rejections a model is shown, by ToolError code: the domain
# rules it may react to. Any other failure is shown nothing; it ends the run.
REJECTION_CODES: dict[str, AssignmentRejectionCode] = {
    "no_seats_available": "no_seats_available",
    "user_inactive": "user_inactive",
    "assignment_already_exists": "already_assigned",
}


def observe(
    goal: ResolvedAssignmentGoal, args: ToolInput, outcome: ToolOutput | ToolError
) -> ModelObservation | None:
    """What the model is shown for a call's ``outcome``, or None if the call
    failed in a way the model is not shown (the executor then ends the run).

    ``goal`` is used only to pick out the target licence among the user's
    assignments; its ids never reach the observation.
    """
    if isinstance(outcome, ToolError):
        code = REJECTION_CODES.get(outcome.code)
        if isinstance(args, AssignLicenceInput) and code is not None:
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
