"""Contracts of the agent layer: the text a planner may hand in, the goal the
resolver produces, tool arguments and results, and the verifier's result.

Tool results are structured facts read from the application at one moment
(snapshots), never prose summaries. A later read may disagree with them.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.models import (
    Assignment,
    DesiredState,
    GoalType,
    OutcomeReason,
)

# A storage bound only; the resolver applies the domain's own limits.
EXTRACTED_TEXT_MAX_LENGTH = 1000

ExtractedText = Annotated[str, StringConstraints(max_length=EXTRACTED_TEXT_MAX_LENGTH)]


# --- resolution ---------------------------------------------------------------


class ExtractedAssignmentIntent(BaseModel):
    """Text extracted from an instruction, before any identity is known.

    Text only: there are deliberately no id fields, and extra="forbid" rejects
    any that are supplied, so which rows a run acts on is decided by the
    resolver alone. Content is checked by the resolver (invalid_input).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_email: ExtractedText
    product: ExtractedText


class ResolvedAssignmentGoal(BaseModel):
    """The contract a run executes and is verified against, as persisted on
    the AgentRun. Built only by the resolver or loaded from the run."""

    model_config = ConfigDict(frozen=True)

    goal_type: GoalType
    desired_state: DesiredState
    user_id: int
    licence_id: int
    extracted_user_email: str
    extracted_product: str


ResolutionFailureCode = Literal[
    OutcomeReason.INVALID_INPUT,
    OutcomeReason.USER_NOT_FOUND,
    OutcomeReason.LICENCE_NOT_FOUND,
    OutcomeReason.LICENCE_AMBIGUOUS,
]


class ResolutionFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ResolutionFailureCode
    detail: dict[str, Any]  # JSON-safe values only


# --- tool results -------------------------------------------------------------


class AssignmentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    assignment_id: int
    user_id: int
    licence_id: int
    active: bool
    assigned_at: datetime
    revoked_at: datetime | None

    @classmethod
    def of(cls, assignment: Assignment) -> Self:
        return cls(
            assignment_id=assignment.id,
            user_id=assignment.user_id,
            licence_id=assignment.licence_id,
            active=assignment.revoked_at is None,
            assigned_at=assignment.assigned_at,
            revoked_at=assignment.revoked_at,
        )


# --- verification -------------------------------------------------------------


class VerificationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    goal_type: GoalType
    desired_state: DesiredState
    user_id: int
    licence_id: int
    active_assignment: AssignmentSnapshot | None


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    satisfied: bool
    evidence: VerificationEvidence
