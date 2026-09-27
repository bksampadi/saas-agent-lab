"""Contracts of the agent layer: what a planner may extract from an
instruction, the goal the resolver produces, tool arguments and results,
model-call records, and the verifier's result.

Tool results are structured facts read from the application at one moment
(snapshots), never prose summaries. A later read may disagree with them.
They are internal results for application code and carry row ids; they are
never shown to a model as they are. Anything a model sees is a separate,
id-free observation.
"""

from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.models import (
    Assignment,
    DesiredState,
    GoalType,
    OutcomeReason,
    User,
    UserStatus,
)
from app.schemas.assignment import EntityId
from app.services.assignments import SeatUsage

# A storage bound only; the resolver applies the domain's own limits.
EXTRACTED_TEXT_MAX_LENGTH = 1000

ExtractedText = Annotated[str, StringConstraints(max_length=EXTRACTED_TEXT_MAX_LENGTH)]


# --- extraction ---------------------------------------------------------------
#
# What a planner may say an instruction asks for: exactly one of three closed
# shapes. Text and reason codes only. There are no id fields and no free-form
# fields, and extra="forbid" rejects any that are supplied, so a planner can
# neither name rows nor carry anything the application did not ask for.


class EnsureAssignmentIntent(BaseModel):
    """The instruction asks for exactly one thing: that one user, identified
    by an email address written in the instruction, holds a seat of one
    product. Both values are copied from the instruction, never inferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ensure_assignment"] = "ensure_assignment"
    user_email: ExtractedText
    product: ExtractedText


NeedsClarificationCode = Literal[
    "missing_user_email",  # a user is named or described, but no email given
    "missing_product",
    "multiple_users",
    "multiple_products",
    "conflicting_request",  # the instruction contradicts itself
]


class NeedsClarification(BaseModel):
    """The instruction asks for a licence assignment but cannot be carried
    out as written."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["needs_clarification"] = "needs_clarification"
    reason_code: NeedsClarificationCode


UnsupportedCode = Literal[
    "unsupported_action",  # asks for something other than a licence assignment
    "additional_request",  # an assignment plus anything else
    "not_a_request",
]


class Unsupported(BaseModel):
    """The instruction asks for something this agent does not do. The whole
    instruction is refused; no part of it is carried out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["unsupported"] = "unsupported"
    reason_code: UnsupportedCode


ExtractedIntent = Annotated[
    EnsureAssignmentIntent | NeedsClarification | Unsupported,
    Field(discriminator="kind"),
]


# --- model calls --------------------------------------------------------------


class ModelCallError(BaseModel):
    """A failed or rejected model request, normalized like ToolError: a code,
    a message written by the application, and the exception class name."""

    model_config = ConfigDict(frozen=True)

    code: str  # invalid_output, timeout, provider_error or unexpected_error
    message: str
    error_type: str | None


class ModelCallRecord(BaseModel):
    """One model request's outcome, as a planner reports it for persistence."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    input_tokens: int | None  # None if the request produced no response
    output_tokens: int | None
    latency_ms: int
    output: dict[str, Any] | None  # the accepted structured output, as JSON
    error: ModelCallError | None  # set exactly when output is None


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


# --- tool arguments -----------------------------------------------------------


class ToolInputBase(BaseModel):
    """Arguments of one tool. Every id must equal the run's resolved goal;
    the executor checks that before anything runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: ClassVar[str]
    mutating: ClassVar[bool] = False


class GetUserInput(ToolInputBase):
    tool_name: ClassVar[str] = "get_user"

    user_id: EntityId


class GetLicenceInput(ToolInputBase):
    tool_name: ClassVar[str] = "get_licence"

    licence_id: EntityId


class ListUserAssignmentsInput(ToolInputBase):
    tool_name: ClassVar[str] = "list_user_assignments"

    user_id: EntityId


class AssignLicenceInput(ToolInputBase):
    tool_name: ClassVar[str] = "assign_licence"
    mutating: ClassVar[bool] = True

    user_id: EntityId
    licence_id: EntityId


ToolInput = (
    GetUserInput | GetLicenceInput | ListUserAssignmentsInput | AssignLicenceInput
)


# --- tool results -------------------------------------------------------------


class UserSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int
    email: str
    name: str
    status: UserStatus

    @classmethod
    def of(cls, user: User) -> Self:
        return cls(
            user_id=user.id, email=user.email, name=user.name, status=user.status
        )


class LicenceSnapshot(BaseModel):
    """Seats as counted when read: an observation, never a reservation."""

    model_config = ConfigDict(frozen=True)

    licence_id: int
    product: str
    seats_total: int
    seats_active: int
    seats_available: int

    @classmethod
    def of(cls, usage: SeatUsage) -> Self:
        return cls(
            licence_id=usage.licence.id,
            product=usage.licence.product,
            seats_total=usage.licence.seats_total,
            seats_active=usage.seats_active,
            seats_available=usage.seats_available,
        )


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


class UserAssignmentsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int
    assignments: list[AssignmentSnapshot]  # active and revoked, ordered by id


ToolOutput = (
    UserSnapshot | LicenceSnapshot | UserAssignmentsSnapshot | AssignmentSnapshot
)


class ToolError(BaseModel):
    """A failed tool call, normalized. Never a traceback or an exception
    object; for unexpected errors not even the exception's own message."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    error_type: str | None  # the exception class name, if an exception was raised


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
