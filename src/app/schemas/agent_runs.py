"""The agent-run HTTP contract: what a caller sends to start a run, and what
it is shown of one.

A run is shown by what it means, never by its internal rows: no resolved
user or licence id, no tool-call arguments or internal results, and no
exception message. Every field is read from what was persisted when it
happened; nothing is re-derived from current application state.

The trace shows the run as its models knew it: each model request with its
accepted output, and each tool call under the model-facing name that made it,
with the exact observation the model was given, and, for a mutation, what
policy decided when the call was admitted.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.agent.decision_tools import MODEL_TOOL_NAMES
from app.agent.executor import INSTRUCTION_MAX_LENGTH
from app.models import (
    AgentRun,
    AgentRunStatus,
    CannotProceedReason,
    DecisionProposalKind,
    GoalType,
    ModelCall,
    ModelCallStage,
    ModelCallStatus,
    OutcomeReason,
    PolicyDecision,
    ToolCall,
    ToolCallStatus,
)
from app.schemas.agent import (
    CannotProceed,
    DecisionContext,
    DecisionProposal,
    DecisionToolRequest,
    EnsureAssignmentIntent,
    GoalReached,
    NeedsClarification,
    NoActionNeeded,
    TargetToolName,
    Unsupported,
)

# --- request ------------------------------------------------------------------


class AgentRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Not stripped: a run stores its instruction exactly as given. A blank
    # one passes here and is refused by the executor (422), as a blank actor
    # header is by the services.
    instruction: Annotated[str, Field(min_length=1, max_length=INSTRUCTION_MAX_LENGTH)]


# --- run ----------------------------------------------------------------------


class GoalRead(BaseModel):
    """The goal as extracted from the instruction. Resolved ids stay internal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: GoalType
    user_email: str
    product: str


BlockingReason = Literal["no_seats_available", "user_inactive"]


class BlockCheckRead(BaseModel):
    """The model's cannot_proceed claim, as the application checked it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: CannotProceedReason
    confirmed: bool


class VerificationRead(BaseModel):
    """What the application found when the run ended, as recorded then.

    ``satisfied``: the goal held in committed state. ``blocking_rejection``:
    a blocking domain rule rejected the run's latest assignment attempt.
    ``block_check``: the model claimed a block, and whether current state
    confirmed it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    satisfied: bool
    blocking_rejection: BlockingReason | None
    block_check: BlockCheckRead | None


class _AgentRunSummary(BaseModel):
    """A run's summary: what was asked, by whom, and how it ended.

    ``outcome_code`` refines ``outcome_reason`` with the application-chosen
    code the run recorded, where it has one: the extraction's reason code
    (instruction_unclear, unsupported_request), the planner error's code
    (planner_error) or the limit reached (step_limit).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    instruction: str
    requesting_actor: str
    status: AgentRunStatus
    outcome_reason: OutcomeReason | None
    outcome_code: str | None
    goal: GoalRead | None
    decision_proposal: DecisionProposal | None
    created_at: datetime
    completed_at: datetime | None


class AgentRunRead(_AgentRunSummary):
    """A run's summary, as POST /agent-runs returns it."""

    @classmethod
    def of(cls, run: AgentRun) -> Self:
        return cls(**_summary(run))


# --- trace --------------------------------------------------------------------

# What a model request's accepted output may be: an extraction result, a
# request for tools, or a decision proposal. Closed, and id-free by
# construction; each member rejects undeclared fields.
ModelCallOutput = Annotated[
    EnsureAssignmentIntent
    | NeedsClarification
    | Unsupported
    | DecisionToolRequest
    | GoalReached
    | NoActionNeeded
    | CannotProceed,
    Field(discriminator="kind"),
]
_MODEL_CALL_OUTPUT: TypeAdapter[ModelCallOutput] = TypeAdapter(ModelCallOutput)


class ModelCallTraceEntry(BaseModel):
    """One model request. A failed or rejected one shows only its error code:
    error messages and exception types stay internal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["model_call"] = "model_call"
    sequence_no: int
    stage: ModelCallStage
    model: str
    status: ModelCallStatus
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    output: ModelCallOutput | None
    error_code: str | None

    @classmethod
    def of(cls, call: ModelCall) -> Self:
        return cls(
            sequence_no=call.sequence_no,
            stage=call.stage,
            model=call.model_name,
            status=call.status,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            latency_ms=call.latency_ms,
            output=(
                None
                if call.output is None
                else _MODEL_CALL_OUTPUT.validate_python(call.output)
            ),
            error_code=None if call.error is None else call.error["code"],
        )


class ToolCallTraceEntry(BaseModel):
    """One tool call, by the model-facing tool that made it. ``observation``
    is the exact text the model was given, or None if it was given nothing
    (a refused or denied call, a call held for approval, or a failure that
    ended the run). ``policy`` is what policy decided when the call was
    admitted: None for a read, and for a call refused before policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_call"] = "tool_call"
    sequence_no: int
    tool: TargetToolName
    status: ToolCallStatus
    policy: PolicyDecision | None
    error_code: str | None
    observation: str | None

    @classmethod
    def of(cls, call: ToolCall) -> Self:
        return cls(
            sequence_no=call.sequence_no,
            tool=MODEL_TOOL_NAMES[call.tool_name],
            status=call.status,
            policy=call.policy_decision,
            error_code=None if call.error is None else call.error["code"],
            observation=call.observation,
        )


TraceEntry = Annotated[
    ModelCallTraceEntry | ToolCallTraceEntry, Field(discriminator="kind")
]


class AgentRunDetail(_AgentRunSummary):
    """A run's summary, what its decision model was told first, what the
    application verified at the end, and the whole trace in order.
    ``decision_context`` is None if the run never reached its decision
    stage."""

    decision_context: DecisionContext | None
    verification: VerificationRead | None
    trace: list[TraceEntry]

    @classmethod
    def of(cls, run: AgentRun, trace: list[ModelCall | ToolCall]) -> Self:
        return cls(
            **_summary(run),
            decision_context=(
                None
                if run.decision_context is None
                else DecisionContext.model_validate(run.decision_context)
            ),
            verification=_verification(run.outcome_detail),
            trace=[
                ModelCallTraceEntry.of(entry)
                if isinstance(entry, ModelCall)
                else ToolCallTraceEntry.of(entry)
                for entry in sorted(trace, key=lambda entry: entry.sequence_no)
            ],
        )


# --- projections from the persisted run ----------------------------------------


def _summary(run: AgentRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "instruction": run.instruction,
        "requesting_actor": run.requesting_actor,
        "status": run.status,
        "outcome_reason": run.outcome_reason,
        "outcome_code": _outcome_code(run.outcome_reason, run.outcome_detail),
        "goal": _goal(run),
        "decision_proposal": _proposal(run),
        "created_at": run.created_at,
        "completed_at": run.completed_at,
    }


def _goal(run: AgentRun) -> GoalRead | None:
    if (
        run.goal_type is None
        or run.extracted_user_email is None
        or run.extracted_product is None
    ):
        return None  # extraction ended the run, or has not happened yet
    return GoalRead(
        kind=run.goal_type,
        user_email=run.extracted_user_email,
        product=run.extracted_product,
    )


def _proposal(run: AgentRun) -> DecisionProposal | None:
    match run.decision_proposal:
        case None:
            return None
        case DecisionProposalKind.GOAL_REACHED:
            return GoalReached()
        case DecisionProposalKind.NO_ACTION_NEEDED:
            return NoActionNeeded()
        case DecisionProposalKind.CANNOT_PROCEED:
            if run.decision_reason_code is None:
                raise ValueError(f"Agent run {run.id}: cannot_proceed has no reason.")
            return CannotProceed(reason_code=run.decision_reason_code)
        case _:
            assert_never(run.decision_proposal)


# The outcome_detail key holding each reason's refining code. Only these keys
# are read: the rest of the detail is internal and may name rows.
_OUTCOME_CODE_KEYS: dict[OutcomeReason, str] = {
    OutcomeReason.INSTRUCTION_UNCLEAR: "reason_code",
    OutcomeReason.UNSUPPORTED_REQUEST: "reason_code",
    OutcomeReason.PLANNER_ERROR: "code",
    OutcomeReason.STEP_LIMIT: "limit",
}


def _outcome_code(
    reason: OutcomeReason | None, detail: dict[str, Any] | None
) -> str | None:
    key = None if reason is None else _OUTCOME_CODE_KEYS.get(reason)
    if key is None or detail is None:
        return None
    return detail.get(key)


def _verification(detail: dict[str, Any] | None) -> VerificationRead | None:
    """The id-free part of the verification the executor recorded in the
    run's terminal transaction, or None if the run ended before its goal
    was verified."""
    if detail is None or not isinstance(detail.get("satisfied"), bool):
        return None
    rejection = detail.get("rejection")  # set by a decision run only
    block_check = detail.get("block_check")  # likewise
    return VerificationRead(
        satisfied=detail["satisfied"],
        blocking_rejection=None if rejection is None else rejection["error"]["code"],
        block_check=(
            None
            if block_check is None
            else BlockCheckRead(
                reason=block_check["reason"], confirmed=block_check["confirmed"]
            )
        ),
    )
