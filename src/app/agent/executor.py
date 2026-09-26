"""Runs an agent's steps with explicit transaction boundaries.

There are two kinds of transaction, and they are never open at the same time:

- log transactions write AgentRun and ToolCall rows;
- a business transaction wraps one tool's service call and, for a change,
  its AuditEvent. Services and repositories still never commit.

Every step opens a new session and closes it before the next one opens:

    LOG       ToolCall STARTED (+ run -> EXECUTING)          COMMIT, close
    BUSINESS  service call (+ AuditEvent)                    COMMIT or ROLLBACK, close
    LOG       ToolCall SUCCEEDED/FAILED (+ run outcome)      COMMIT, close

So a rolled-back change leaves the run's history intact, and on SQLite no
log write holds a lock while a business write waits for it. No ORM object
crosses from one session to the next; only ids and plain values do.

Known crash windows (no recovery yet): a crash after the business commit
but before the ToolCall's outcome is written leaves the call STARTED; the
change is still traceable to the run through its audit actor
"agent:run-<id>". A run with a STARTED call refuses further tool calls and
verification (UnfinishedToolCall), so it is never retried blindly or
labelled without knowing whether it made a change.

Tool calls within a run are serial. Nothing here coordinates concurrent
callers on the same run: the status checks are read-then-write, not
compare-and-set.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.agent import tools
from app.agent.resolver import resolve_assignment_goal
from app.agent.status import transition
from app.agent.verifier import verify
from app.models import (
    AgentRun,
    AgentRunStatus,
    DesiredState,
    GoalType,
    OutcomeReason,
    ToolCall,
    ToolCallStatus,
)
from app.models.base import utcnow
from app.repositories.agent_runs import AgentRunRepository
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import (
    ExtractedAssignmentIntent,
    ResolutionFailure,
    ResolvedAssignmentGoal,
    ToolError,
    ToolInput,
    ToolOutput,
    VerificationResult,
)
from app.services.assignments import AssignmentService
from app.services.errors import (
    AssignmentAlreadyExists,
    AssignmentAlreadyRevoked,
    AssignmentNotFound,
    DomainError,
    EmailAlreadyExists,
    InvalidInput,
    LicenceNotFound,
    NoSeatsAvailable,
    ProductAlreadyExists,
    UserInactive,
    UserNotFound,
)
from app.services.licences import LicenceService
from app.services.users import UserService
from app.services.validation import agent_run_actor, validated_external_actor

INSTRUCTION_MAX_LENGTH = 2000

GOAL_SCOPE_VIOLATION = "goal_scope_violation"
UNEXPECTED_ERROR = "unexpected_error"

_DOMAIN_ERROR_CODES: dict[type[DomainError], str] = {
    InvalidInput: "invalid_input",
    EmailAlreadyExists: "email_already_exists",
    UserNotFound: "user_not_found",
    ProductAlreadyExists: "product_already_exists",
    LicenceNotFound: "licence_not_found",
    UserInactive: "user_inactive",
    AssignmentAlreadyExists: "assignment_already_exists",
    NoSeatsAvailable: "no_seats_available",
    AssignmentNotFound: "assignment_not_found",
    AssignmentAlreadyRevoked: "assignment_already_revoked",
}

# Expected domain rules that stop the goal: the run ends BLOCKED.
_BLOCKING_ERRORS: dict[str, OutcomeReason] = {
    "no_seats_available": OutcomeReason.NO_SEATS_AVAILABLE,
    "user_inactive": OutcomeReason.USER_INACTIVE,
}

# After this failure the goal may hold anyway: another transaction made the
# same assignment first. The call is still recorded as FAILED, because it
# did fail, but the run stays EXECUTING so the verifier can decide.
_VERIFY_AFTER_ERRORS = frozenset({"assignment_already_exists"})


class AgentRunNotFound(Exception):
    def __init__(self, run_id: int) -> None:
        super().__init__(f"Agent run {run_id} not found.")
        self.run_id = run_id


class RunNotExecutable(Exception):
    """The run's status does not allow the step. Nothing was recorded."""

    def __init__(self, run_id: int, status: AgentRunStatus) -> None:
        super().__init__(f"Agent run {run_id} is {status}; it cannot do this now.")
        self.run_id = run_id
        self.status = status


class UnfinishedToolCall(RunNotExecutable):
    """The run has a tool call with no recorded outcome (a crash window).

    Its change may or may not have committed, so the run must not carry on
    (or be verified and labelled) until that is reconciled. Recovery is not
    implemented; this only refuses to continue.
    """

    def __init__(self, run_id: int, status: AgentRunStatus) -> None:
        Exception.__init__(
            self,
            f"Agent run {run_id} has a tool call with no recorded outcome; "
            "it must be reconciled before the run can continue.",
        )
        self.run_id = run_id
        self.status = status


@dataclass(frozen=True)
class ToolCallOutcome:
    tool_call_id: int
    sequence_no: int
    output: ToolOutput | None  # set if the call succeeded
    error: ToolError | None  # set if it failed
    run_status: AgentRunStatus  # the run's status after the call


def tool_error(error: Exception, tool_name: str) -> ToolError:
    """Normalize an exception raised while running a tool."""
    if isinstance(error, DomainError):
        # Domain messages are written for callers and safe to keep.
        return ToolError(
            code=_DOMAIN_ERROR_CODES.get(type(error), "domain_error"),
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


def goal_scope_error(goal: ResolvedAssignmentGoal, args: ToolInput) -> ToolError | None:
    """A ToolError if ``args`` names any user or licence other than the goal's,
    or names none at all (such a call could not be checked, so it is refused)."""
    user_id, licence_id = tools.target_ids(args)
    if user_id is None and licence_id is None:
        return ToolError(
            code=GOAL_SCOPE_VIOLATION,
            message=f"{args.tool_name} names no target to check against the goal.",
            error_type=None,
        )
    outside = []
    if user_id is not None and user_id != goal.user_id:
        outside.append("user_id")
    if licence_id is not None and licence_id != goal.licence_id:
        outside.append("licence_id")
    if not outside:
        return None
    return ToolError(
        code=GOAL_SCOPE_VIOLATION,
        message=(
            f"{args.tool_name} named {' and '.join(outside)} outside the run's "
            "resolved goal."
        ),
        error_type=None,
    )


def _unexpected_detail(stage: str, error: Exception) -> dict[str, Any]:
    return {"stage": stage, "error_type": type(error).__name__}


def _persisted_goal(run: AgentRun) -> ResolvedAssignmentGoal:
    if run.resolved_user_id is None or run.resolved_licence_id is None:
        raise RunNotExecutable(run.id, run.status)
    return ResolvedAssignmentGoal(
        goal_type=run.goal_type,
        desired_state=run.desired_state,
        user_id=run.resolved_user_id,
        licence_id=run.resolved_licence_id,
        extracted_user_email=run.extracted_user_email,
        extracted_product=run.extracted_product,
    )


def _tool_context(session: Session, actor: str) -> tools.ToolContext:
    return tools.ToolContext(
        users=UserService(session),
        licences=LicenceService(session),
        assignments=AssignmentService(session),
        actor=actor,
    )


class AgentExecutor:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    # --- run lifecycle ----------------------------------------------------------

    def create_run(
        self,
        *,
        instruction: str,
        requesting_actor: str,
        intent: ExtractedAssignmentIntent,
    ) -> int:
        """Persist a RECEIVED ensure-assignment run and return its id.

        Raises InvalidInput, and records nothing, for a blank or over-long
        instruction or an invalid requesting actor ("agent:" is reserved).
        The instruction and extracted text are stored exactly as given.
        """
        requesting_actor = validated_external_actor(requesting_actor)
        if not instruction.strip():
            raise InvalidInput("Instruction must not be empty.")
        if len(instruction) > INSTRUCTION_MAX_LENGTH:
            raise InvalidInput(
                f"Instruction must be at most {INSTRUCTION_MAX_LENGTH} characters."
            )

        with self._sessions.begin() as log:
            run = AgentRunRepository(log).add(
                AgentRun(
                    instruction=instruction,
                    requesting_actor=requesting_actor,
                    status=AgentRunStatus.RECEIVED,
                    goal_type=GoalType.ENSURE_ASSIGNMENT,
                    desired_state=DesiredState.ASSIGNED,
                    extracted_user_email=intent.user_email,
                    extracted_product=intent.product,
                )
            )
            return run.id

    def resolve_run(self, run_id: int) -> AgentRunStatus:
        """Resolve the run's persisted extracted text to row ids.

        RESOLVED stores the goal on the run; an unknown, ambiguous or invalid
        entity ends the run NEEDS_CLARIFICATION with nothing attempted.
        """
        with self._sessions() as log:
            run = self._get_run(log, run_id)
            if run.status is not AgentRunStatus.RECEIVED:
                raise RunNotExecutable(run_id, run.status)
            intent = ExtractedAssignmentIntent(
                user_email=run.extracted_user_email, product=run.extracted_product
            )

        resolution: ResolvedAssignmentGoal | ResolutionFailure
        try:
            with self._sessions() as read:
                resolution = resolve_assignment_goal(
                    UserService(read), LicenceService(read), intent
                )
        except Exception as error:
            return self._fail(run_id, _unexpected_detail("resolution", error))

        with self._sessions.begin() as log:
            run = self._get_run(log, run_id)
            if isinstance(resolution, ResolutionFailure):
                transition(
                    run,
                    AgentRunStatus.NEEDS_CLARIFICATION,
                    reason=resolution.code,
                    detail=resolution.detail,
                )
            else:
                run.resolved_user_id = resolution.user_id
                run.resolved_licence_id = resolution.licence_id
                transition(run, AgentRunStatus.RESOLVED)
            return run.status

    def get_goal(self, run_id: int) -> ResolvedAssignmentGoal:
        """The run's persisted resolved goal."""
        with self._sessions() as log:
            return _persisted_goal(self._get_run(log, run_id))

    # --- tool calls -------------------------------------------------------------

    def call_tool(self, run_id: int, args: ToolInput) -> ToolCallOutcome:
        """Record and run one tool call; see the module docstring for the
        three transactions.

        A call naming anything outside the run's persisted goal is recorded
        as FAILED (goal_scope_violation) and fails the run; no business
        transaction is opened. Raises RunNotExecutable, recording nothing,
        unless the run is RESOLVED or EXECUTING with no unfinished call.
        """
        # LOG: check, then record the call as STARTED.
        with self._sessions.begin() as log:
            run = self._get_run(log, run_id)
            if run.status not in (AgentRunStatus.RESOLVED, AgentRunStatus.EXECUTING):
                raise RunNotExecutable(run_id, run.status)
            calls = ToolCallRepository(log)
            if calls.any_started(run.id):
                raise UnfinishedToolCall(run_id, run.status)
            goal = _persisted_goal(run)
            call = ToolCall(
                agent_run_id=run.id,
                sequence_no=calls.next_sequence_no(run.id),
                tool_name=args.tool_name,
                arguments=args.model_dump(mode="json"),
            )

            scope_error = goal_scope_error(goal, args)
            if scope_error is not None:
                call.status = ToolCallStatus.FAILED
                call.error = scope_error.model_dump(mode="json")
                call.completed_at = utcnow()
                calls.add(call)
                transition(
                    run,
                    AgentRunStatus.FAILED,
                    reason=OutcomeReason.GOAL_SCOPE_VIOLATION,
                    detail=_call_detail(call, scope_error),
                )
                return ToolCallOutcome(
                    tool_call_id=call.id,
                    sequence_no=call.sequence_no,
                    output=None,
                    error=scope_error,
                    run_status=run.status,
                )

            call.status = ToolCallStatus.STARTED
            calls.add(call)
            if run.status is AgentRunStatus.RESOLVED:
                transition(run, AgentRunStatus.EXECUTING)
            call_id = call.id
            actor = agent_run_actor(run.id)

        # BUSINESS: the service call and its audit event commit or roll back
        # together. The log session above is already committed and closed.
        outcome: ToolOutput | ToolError
        try:
            with self._sessions() as session, session.begin():
                outcome = tools.run_tool(_tool_context(session, actor), args)
        except Exception as error:
            # Also reached if the commit fails after the tool returned.
            outcome = tool_error(error, args.tool_name)

        # LOG: record what happened, and end the run if the failure calls for it.
        with self._sessions.begin() as log:
            run = self._get_run(log, run_id)
            call = ToolCallRepository(log).get(call_id)
            if call is None:
                raise RuntimeError(f"Tool call {call_id} disappeared.")
            call.completed_at = utcnow()
            if isinstance(outcome, ToolError):
                call.status = ToolCallStatus.FAILED
                call.error = outcome.model_dump(mode="json")
                # Run serially, the run is still EXECUTING here. If an
                # uncoordinated concurrent caller ended it meanwhile, the
                # call's outcome is still recorded; the run's is left alone.
                if run.status is AgentRunStatus.EXECUTING:
                    _end_run_after_failure(run, call, outcome)
            else:
                call.status = ToolCallStatus.SUCCEEDED
                call.result = outcome.model_dump(mode="json")
            return ToolCallOutcome(
                tool_call_id=call.id,
                sequence_no=call.sequence_no,
                output=None if isinstance(outcome, ToolError) else outcome,
                error=outcome if isinstance(outcome, ToolError) else None,
                run_status=run.status,
            )

    # --- verification -----------------------------------------------------------

    def verify_and_finish(self, run_id: int) -> AgentRunStatus:
        """Verify the persisted goal against current state and end the run.

        The only way a run becomes COMPLETED. The verifier reads through a
        new session, so it sees committed state, never an earlier session's
        objects or any tool's result. Refused while a tool call has no
        recorded outcome: whether this run made a change would be unknown.
        """
        with self._sessions.begin() as log:
            run = self._get_run(log, run_id)
            if ToolCallRepository(log).any_started(run.id):
                raise UnfinishedToolCall(run_id, run.status)
            transition(run, AgentRunStatus.VERIFYING)
            goal = _persisted_goal(run)

        result: VerificationResult
        try:
            with self._sessions() as read:
                result = verify(AssignmentService(read), goal)
        except Exception as error:
            return self._fail(run_id, _unexpected_detail("verification", error))

        with self._sessions.begin() as log:
            run = self._get_run(log, run_id)
            detail = result.model_dump(mode="json")
            if not result.satisfied:
                transition(
                    run,
                    AgentRunStatus.FAILED,
                    reason=OutcomeReason.VERIFICATION_FAILED,
                    detail=detail,
                )
            else:
                changed = ToolCallRepository(log).any_succeeded(
                    run.id, tools.MUTATING_TOOL_NAMES
                )
                transition(
                    run,
                    AgentRunStatus.COMPLETED,
                    reason=(
                        OutcomeReason.GOAL_SATISFIED
                        if changed
                        else OutcomeReason.ALREADY_SATISFIED
                    ),
                    detail=detail,
                )
            return run.status

    # --- helpers ----------------------------------------------------------------

    def _get_run(self, session: Session, run_id: int) -> AgentRun:
        run = AgentRunRepository(session).get(run_id)
        if run is None:
            raise AgentRunNotFound(run_id)
        return run

    def _fail(self, run_id: int, detail: dict[str, Any]) -> AgentRunStatus:
        with self._sessions.begin() as log:
            run = self._get_run(log, run_id)
            transition(
                run,
                AgentRunStatus.FAILED,
                reason=OutcomeReason.UNEXPECTED_ERROR,
                detail=detail,
            )
            return run.status


def _call_detail(call: ToolCall, error: ToolError) -> dict[str, Any]:
    return {
        "tool_call_id": call.id,
        "sequence_no": call.sequence_no,
        "tool_name": call.tool_name,
        "error": error.model_dump(mode="json"),
    }


def _end_run_after_failure(run: AgentRun, call: ToolCall, error: ToolError) -> None:
    blocked_reason = _BLOCKING_ERRORS.get(error.code)
    if blocked_reason is not None:
        transition(
            run,
            AgentRunStatus.BLOCKED,
            reason=blocked_reason,
            detail=_call_detail(call, error),
        )
    elif error.code not in _VERIFY_AFTER_ERRORS:
        transition(
            run,
            AgentRunStatus.FAILED,
            reason=OutcomeReason.TOOL_FAILED,
            detail=_call_detail(call, error),
        )
