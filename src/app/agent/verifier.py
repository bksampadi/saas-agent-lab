"""Goal verification: the only source of a run's success.

Takes the persisted goal and reads current application state. It never sees
tool results or planner output, so neither can make a run succeed.
"""

from typing import assert_never

from app.models import DesiredState
from app.schemas.agent import (
    AssignmentSnapshot,
    ResolvedAssignmentGoal,
    VerificationEvidence,
    VerificationResult,
)
from app.services.assignments import AssignmentService


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
