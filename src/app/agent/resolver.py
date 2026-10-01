"""Deterministic identity resolution: extracted text in, row ids out.

Users match on the exact normalized email; licences on the exact product
name ignoring case. No fuzzy matching, and never a choice between candidates:
zero matches is "not found", more than one is "ambiguous". Reads only.
"""

from app.models import OutcomeReason
from app.schemas.agent import ResolutionFailure, ResolvedAssignmentGoal
from app.services.errors import InvalidInput
from app.services.licences import LicenceService
from app.services.users import UserService


def resolve_assignment_goal(
    users: UserService, licences: LicenceService, user_email: str, product: str
) -> ResolvedAssignmentGoal | ResolutionFailure:
    """The goal that ``user_email`` holds a seat of ``product``, as
    extracted from the instruction, with the rows it names."""
    try:
        # Both inputs are validated before either "not found" is reported.
        user = users.find_user_by_email(user_email)
        matches = licences.find_licences_by_product(product)
    except InvalidInput as error:
        return ResolutionFailure(
            code=OutcomeReason.INVALID_INPUT, detail={"message": str(error)}
        )

    if user is None:
        return ResolutionFailure(
            code=OutcomeReason.USER_NOT_FOUND,
            detail={"user_email": user_email},
        )
    if not matches:
        return ResolutionFailure(
            code=OutcomeReason.LICENCE_NOT_FOUND, detail={"product": product}
        )
    if len(matches) > 1:
        # Names only: enough to ask which one was meant, without handing out
        # row ids.
        return ResolutionFailure(
            code=OutcomeReason.LICENCE_AMBIGUOUS,
            detail={
                "product": product,
                "candidates": [licence.product for licence in matches],
            },
        )

    (licence,) = matches
    return ResolvedAssignmentGoal(
        user_id=user.id,
        licence_id=licence.id,
        extracted_user_email=user_email,
        extracted_product=product,
    )
