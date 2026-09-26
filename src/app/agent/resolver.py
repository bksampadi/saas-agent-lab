"""Deterministic identity resolution: extracted text in, row ids out.

Users match on the exact normalized email; licences on the exact product
name ignoring case. No fuzzy matching, and never a choice between candidates:
zero matches is "not found", more than one is "ambiguous". Reads only.
"""

from app.models import DesiredState, GoalType, OutcomeReason
from app.schemas.agent import (
    ExtractedAssignmentIntent,
    ResolutionFailure,
    ResolvedAssignmentGoal,
)
from app.services.errors import InvalidInput
from app.services.licences import LicenceService
from app.services.users import UserService


def resolve_assignment_goal(
    users: UserService, licences: LicenceService, intent: ExtractedAssignmentIntent
) -> ResolvedAssignmentGoal | ResolutionFailure:
    try:
        # Both inputs are validated before either "not found" is reported.
        user = users.find_user_by_email(intent.user_email)
        matches = licences.find_licences_by_product(intent.product)
    except InvalidInput as error:
        return ResolutionFailure(
            code=OutcomeReason.INVALID_INPUT, detail={"message": str(error)}
        )

    if user is None:
        return ResolutionFailure(
            code=OutcomeReason.USER_NOT_FOUND,
            detail={"user_email": intent.user_email},
        )
    if not matches:
        return ResolutionFailure(
            code=OutcomeReason.LICENCE_NOT_FOUND, detail={"product": intent.product}
        )
    if len(matches) > 1:
        # Names only: enough to ask which one was meant, without handing out
        # row ids.
        return ResolutionFailure(
            code=OutcomeReason.LICENCE_AMBIGUOUS,
            detail={
                "product": intent.product,
                "candidates": [licence.product for licence in matches],
            },
        )

    (licence,) = matches
    return ResolvedAssignmentGoal(
        goal_type=GoalType.ENSURE_ASSIGNMENT,
        desired_state=DesiredState.ASSIGNED,
        user_id=user.id,
        licence_id=licence.id,
        extracted_user_email=intent.user_email,
        extracted_product=intent.product,
    )
