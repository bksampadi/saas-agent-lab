"""The four tools a decision model can call, and what it is shown of each.

No tool takes an argument. The model chooses which tool; every call acts on
the run's resolved goal, its user and its licence, so nothing a model says
can name a row. A call's result exists in three forms, kept apart:

    run()        the internal result: a snapshot, or the ToolError
                 tool_error() makes of what it raised, carrying row ids;
                 for application code only
    observe()    the model-visible observation: a closed, id-free DTO
    serialize()  its exact text, persisted with the call
                 (ToolCall.observation) and given to the model as it is

What a domain rule's rejection means to a run, from the code recorded for
it to how the run ends, is one table: DOMAIN_RULES. A rejection is shown as
a closed code chosen there, never from an exception's message, which may
name row ids. Business rules (capacity, duplicates, user status) stay in the
services; nothing here re-checks them.
"""

import json
from dataclasses import dataclass
from typing import assert_never

from app.models import CannotProceedReason, OutcomeReason
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
from app.services.errors import (
    AssignmentAlreadyExists,
    DomainError,
    InvalidInput,
    LicenceNotFound,
    NoSeatsAvailable,
    UserInactive,
    UserNotFound,
)
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

UNEXPECTED_ERROR = "unexpected_error"


@dataclass(frozen=True)
class DomainRule:
    """What one domain rule's rejection of a tool call means to a run.

    ``code``: the ToolCall's recorded error code, also shown in the public
    trace. ``shown_as``: the code the model is shown when the rule rejects
    the assignment; None, and the model is shown nothing: the run ends
    FAILED (tool_failed). ``blocks``: the run ends BLOCKED with this reason
    if the rule rejected its latest assignment attempt and the goal does
    not hold. ``claim``: the cannot_proceed reason a model gives for the
    rule; confirmed against current state, it blocks the run the same way.
    """

    code: str
    shown_as: AssignmentRejectionCode | None = None
    blocks: OutcomeReason | None = None
    claim: CannotProceedReason | None = None


# The domain errors a tool can raise. Any other domain error is recorded as
# "domain_error", and any other exception as "unexpected_error"; neither is
# shown to the model.
DOMAIN_RULES: dict[type[DomainError], DomainRule] = {
    InvalidInput: DomainRule("invalid_input"),
    UserNotFound: DomainRule("user_not_found"),
    LicenceNotFound: DomainRule("licence_not_found"),
    AssignmentAlreadyExists: DomainRule(
        "assignment_already_exists", shown_as="already_assigned"
    ),
    NoSeatsAvailable: DomainRule(
        "no_seats_available",
        shown_as="no_seats_available",
        blocks=OutcomeReason.NO_SEATS_AVAILABLE,
        claim=CannotProceedReason.NO_SEATS_AVAILABLE,
    ),
    UserInactive: DomainRule(
        "user_inactive",
        shown_as="user_inactive",
        blocks=OutcomeReason.USER_INACTIVE,
        claim=CannotProceedReason.USER_INACTIVE,
    ),
}

# The same rules, by recorded error code and by a model's claim.
_RULES_BY_CODE = {rule.code: rule for rule in DOMAIN_RULES.values()}
BLOCKS_BY_CLAIM: dict[CannotProceedReason, OutcomeReason] = {
    rule.claim: rule.blocks
    for rule in DOMAIN_RULES.values()
    if rule.claim is not None and rule.blocks is not None
}


def tool_error(error: Exception, tool_name: str) -> ToolError:
    """Normalize an exception raised while running a tool."""
    if isinstance(error, DomainError):
        rule = DOMAIN_RULES.get(type(error))
        # Domain messages are written for callers and safe to keep.
        return ToolError(
            code="domain_error" if rule is None else rule.code,
            message=str(error),
            error_type=type(error).__name__,
        )
    # Never str(error): SQLAlchemy errors include SQL and parameter values,
    # and arbitrary exceptions may include anything.
    return ToolError(
        code=UNEXPECTED_ERROR,
        message=f"Unexpected error while running {tool_name}.",
        error_type=type(error).__name__,
    )


def blocking_reason(error_code: str) -> OutcomeReason | None:
    """The BLOCKED reason for a rejection recorded as ``error_code``, or None
    if its rule does not block the goal."""
    rule = _RULES_BY_CODE.get(error_code)
    return None if rule is None else rule.blocks


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
            # The service writes the audit event with the change.
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
        rule = _RULES_BY_CODE.get(outcome.code)
        if tool == MUTATION and rule is not None and rule.shown_as is not None:
            return AssignmentAttemptObservation(
                outcome="rejected", reason_code=rule.shown_as
            )
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
