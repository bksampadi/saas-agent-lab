"""The tools an agent run can call.

Each tool calls an existing service and returns structured facts. Business
rules (capacity, duplicates, user status) stay in the services; nothing here
re-checks them. Tools never see a Session: the executor opens the business
transaction and hands them services bound to it, as routers receive them.
"""

from dataclasses import dataclass
from typing import assert_never

from app.schemas.agent import (
    AssignLicenceInput,
    AssignmentSnapshot,
    GetLicenceInput,
    GetUserInput,
    LicenceSnapshot,
    ListUserAssignmentsInput,
    ToolInput,
    ToolOutput,
    UserAssignmentsSnapshot,
    UserSnapshot,
)
from app.services.assignments import AssignmentService
from app.services.licences import LicenceService
from app.services.users import UserService

TOOL_INPUTS = (
    GetUserInput,
    GetLicenceInput,
    ListUserAssignmentsInput,
    AssignLicenceInput,
)
MUTATING_TOOL_NAMES = frozenset(
    tool_input.tool_name for tool_input in TOOL_INPUTS if tool_input.mutating
)
READ_TOOL_NAMES = frozenset(
    tool_input.tool_name for tool_input in TOOL_INPUTS if not tool_input.mutating
)


@dataclass(frozen=True)
class ToolContext:
    users: UserService
    licences: LicenceService
    assignments: AssignmentService
    actor: str  # the audit actor for changes: "agent:run-<id>"


def get_user(context: ToolContext, args: GetUserInput) -> UserSnapshot:
    return UserSnapshot.of(context.users.get_user(args.user_id))


def get_licence(context: ToolContext, args: GetLicenceInput) -> LicenceSnapshot:
    return LicenceSnapshot.of(context.assignments.get_seat_usage(args.licence_id))


def list_user_assignments(
    context: ToolContext, args: ListUserAssignmentsInput
) -> UserAssignmentsSnapshot:
    assignments = context.assignments.list_assignments_for_user(args.user_id)
    return UserAssignmentsSnapshot(
        user_id=args.user_id,
        assignments=[AssignmentSnapshot.of(assignment) for assignment in assignments],
    )


def assign_licence(
    context: ToolContext, args: AssignLicenceInput
) -> AssignmentSnapshot:
    """The existing service call, audit event included, unchanged."""
    assignment = context.assignments.assign_licence(
        user_id=args.user_id, licence_id=args.licence_id, actor=context.actor
    )
    return AssignmentSnapshot.of(assignment)


def run_tool(context: ToolContext, args: ToolInput) -> ToolOutput:
    # Tools are looked up by name when called, so a test can replace one.
    match args:
        case GetUserInput():
            return get_user(context, args)
        case GetLicenceInput():
            return get_licence(context, args)
        case ListUserAssignmentsInput():
            return list_user_assignments(context, args)
        case AssignLicenceInput():
            return assign_licence(context, args)
        case _:
            assert_never(args)


def target_ids(args: ToolInput) -> tuple[int | None, int | None]:
    """The (user_id, licence_id) a call names; None where it names neither."""
    match args:
        case GetUserInput() | ListUserAssignmentsInput():
            return args.user_id, None
        case GetLicenceInput():
            return None, args.licence_id
        case AssignLicenceInput():
            return args.user_id, args.licence_id
        case _:
            assert_never(args)
