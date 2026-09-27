"""Goal verification: the only source of a run's success.

Takes the persisted goal and reads current application state. It never sees
tool results or planner output, so neither can make a run succeed.

check_block does the same for a blocking condition a decision model claims:
it reads current state, never what the model saw earlier, so a claim can
block a run only when the application finds the condition itself.
"""

from typing import assert_never

from app.models import CannotProceedReason, DesiredState, UserStatus
from app.schemas.agent import (
    AssignmentSnapshot,
    BlockCheck,
    LicenceSnapshot,
    ResolvedAssignmentGoal,
    UserSnapshot,
    VerificationEvidence,
    VerificationResult,
)
from app.services.assignments import AssignmentService
from app.services.users import UserService


def verify(
    assignments: AssignmentService, goal: ResolvedAssignmentGoal
) -> VerificationResult:
    if goal.desired_state is DesiredState.ASSIGNED:
        # Only an active assignment of exactly this licence to exactly this
        # user counts; revoked rows are ignored by the query itself.
        active = assignments.get_active_assignment(
            user_id=goal.user_id, licence_id=goal.licence_id
        )
        return VerificationResult(
            satisfied=active is not None,
            evidence=VerificationEvidence(
                goal_type=goal.goal_type,
                desired_state=goal.desired_state,
                user_id=goal.user_id,
                licence_id=goal.licence_id,
                active_assignment=(
                    None if active is None else AssignmentSnapshot.of(active)
                ),
            ),
        )
    assert_never(goal.desired_state)


def check_block(
    users: UserService,
    assignments: AssignmentService,
    goal: ResolvedAssignmentGoal,
    reason: CannotProceedReason,
) -> BlockCheck:
    """Whether ``reason`` holds for the goal's user and licence now, by the
    same rules assign_licence applies."""
    match reason:
        case CannotProceedReason.NO_SEATS_AVAILABLE:
            usage = assignments.get_seat_usage(goal.licence_id)
            return BlockCheck(
                reason=reason,
                confirmed=usage.seats_available == 0,
                user=None,
                licence=LicenceSnapshot.of(usage),
            )
        case CannotProceedReason.USER_INACTIVE:
            user = users.get_user(goal.user_id)
            return BlockCheck(
                reason=reason,
                confirmed=user.status is not UserStatus.ACTIVE,
                user=UserSnapshot.of(user),
                licence=None,
            )
        case _:
            assert_never(reason)
