"""Runs an agent's steps with explicit transaction boundaries.

A run's whole life is AgentExecutor.run: receive, extract, resolve, decide.
Each step ends the run itself when it cannot go on.

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

The first LOG transaction admits the call. Its arguments are the run's
persisted goal: the model names only the tool. The run's status and limits
are checked, and then, for the mutation, policy (app.agent.policy), which
reads the target licence in that same transaction; the decision is recorded
with the call. A call over a limit is recorded FAILED (step_limit) and ends
the run FAILED; one policy denies is recorded FAILED (policy_denied) and
ends the run BLOCKED; one policy holds for approval is recorded
AWAITING_APPROVAL and pauses the run. None of them opens a business
transaction. The policy read and the recorded decision are atomic together,
not with the business transaction that runs an admitted call afterwards:
policy is decided at admission.

Model calls and tool calls take their sequence_no from one per-run counter,
so a run's trace has a single order that never depends on timestamps.

Extraction follows the same rule. The run is persisted RECEIVED before any
model is called. The planner then runs with no session open, and each model
request it makes is recorded in its own LOG transaction as soon as the
request has an outcome:

    LOG       ModelCall (per request, retries included)      COMMIT, close
    LOG       run's goal, or its outcome                     COMMIT, close

A crash during a model request leaves no row for it (the request changed no
application state); the run stays RECEIVED without a goal.

Decision keeps every rule above. What the model will be told is persisted,
and the run starts EXECUTING, before any model request. The planner then
runs with no session open. Each of its model requests is checked against
the request limit and recorded in its own LOG transaction; each tool call
it makes goes through call_tool (the three transactions above), which
persists the exact observation the model will be given in the last:

    LOG       decision context (+ run -> EXECUTING)          COMMIT, close
              ... model requests and tool calls, as above ...
    LOG       the model's proposal (+ run -> VERIFYING)      COMMIT, close
    READ      verifier, and any claimed block's check        close
    LOG       run outcome                                    COMMIT, close

Known crash windows (no recovery yet): a crash after the business commit
but before the ToolCall's outcome is written leaves the call STARTED; the
change is still traceable to the run through its audit actor
"agent:run-<id>". The crash ends the decision loop, leaving the run
EXECUTING, and nothing decides it again. Should a planner carry on anyway,
a run with a STARTED call refuses further tool calls and verification
(RunNotExecutable), so it is never labelled without knowing whether it made
a change. A paused run's held call stays AWAITING_APPROVAL; nothing
resumes it yet.

Tool calls within a run are serial. Nothing here coordinates concurrent
callers on the same run: the status checks are read-then-write, not
compare-and-set.
"""

from typing import Any, assert_never

from sqlalchemy.orm import Session, sessionmaker

from app.agent import policy, tools
from app.agent.decision import (
    MAX_DECISION_MODEL_REQUESTS,
    MAX_DECISION_MUTATION_CALLS,
    MAX_DECISION_READ_CALLS,
    DecisionLimit,
    DecisionStopped,
    decision_context,
    decision_outcome,
)
from app.agent.planner import (
    DecisionPlanner,
    IntentPlanner,
    PlannerError,
    ungrounded_fields,
)
from app.agent.resolver import resolve_assignment_goal
from app.agent.verifier import check_block, verify
from app.models import (
    AgentRun,
    AgentRunStatus,
    DecisionProposalKind,
    DesiredState,
    GoalType,
    ModelCall,
    ModelCallStage,
    ModelCallStatus,
    OutcomeReason,
    PolicyDecision,
    ToolCall,
    ToolCallStatus,
)
from app.models.base import utcnow
from app.repositories.agent_runs import AgentRunRepository
from app.repositories.model_calls import ModelCallRepository
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import (
    BlockCheck,
    CannotProceed,
    DecisionProposal,
    EnsureAssignmentIntent,
    ExtractedIntent,
    ModelCallRecord,
    NeedsClarification,
    ResolutionFailure,
    ResolvedAssignmentGoal,
    TargetToolName,
    ToolError,
    ToolOutput,
    Unsupported,
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
from app.services.licences import LicenceService
from app.services.users import UserService
from app.services.validation import agent_run_actor, validated_external_actor

INSTRUCTION_MAX_LENGTH = 2000

STEP_LIMIT = "step_limit"
POLICY_DENIED = "policy_denied"
UNEXPECTED_ERROR = "unexpected_error"

# The domain errors a tool can raise, by the code recorded for each. Any
# other domain error would be recorded as "domain_error".
_DOMAIN_ERROR_CODES: dict[type[DomainError], str] = {
    InvalidInput: "invalid_input",
    UserNotFound: "user_not_found",
    LicenceNotFound: "licence_not_found",
    UserInactive: "user_inactive",
    AssignmentAlreadyExists: "assignment_already_exists",
    NoSeatsAvailable: "no_seats_available",
}

# Domain rules that block the goal: a run whose latest assignment attempt
# one of them rejected, and whose goal does not hold, ends BLOCKED.
_BLOCKING_ERRORS: dict[str, OutcomeReason] = {
    "no_seats_available": OutcomeReason.NO_SEATS_AVAILABLE,
    "user_inactive": OutcomeReason.USER_INACTIVE,
}


class AgentRunNotFound(Exception):
    def __init__(self, run_id: int) -> None:
        super().__init__(f"Agent run {run_id} not found.")
        self.run_id = run_id


class RunNotExecutable(Exception):
    """The run's state does not allow the step. Nothing was recorded."""

    def __init__(
        self, run_id: int, status: AgentRunStatus, why: str = "it cannot do this now"
    ) -> None:
        super().__init__(f"Agent run {run_id} is {status}; {why}.")
        self.run_id = run_id
        self.status = status


# A tool call with no recorded outcome (a crash window): its change may or
# may not have committed, so the run must not carry on, or be verified and
# labelled, until that is reconciled. Recovery is not implemented.
UNFINISHED_CALL = "a tool call has no recorded outcome"


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


def _unexpected_detail(stage: str, error: Exception) -> dict[str, Any]:
    return {"stage": stage, "error_type": type(error).__name__}


def _persisted_goal(run: AgentRun) -> ResolvedAssignmentGoal:
    if (
        run.resolved_user_id is None
        or run.resolved_licence_id is None
        or run.extracted_user_email is None
        or run.extracted_product is None
    ):
        raise RunNotExecutable(run.id, run.status)
    return ResolvedAssignmentGoal(
        user_id=run.resolved_user_id,
        licence_id=run.resolved_licence_id,
        extracted_user_email=run.extracted_user_email,
        extracted_product=run.extracted_product,
    )


def _validated_request(instruction: str, requesting_actor: str) -> str:
    """The normalized requesting actor, or raise InvalidInput."""
    requesting_actor = validated_external_actor(requesting_actor)
    if not instruction.strip():
        raise InvalidInput("Instruction must not be empty.")
    if len(instruction) > INSTRUCTION_MAX_LENGTH:
        raise InvalidInput(
            f"Instruction must be at most {INSTRUCTION_MAX_LENGTH} characters."
        )
    return requesting_actor


def _require_awaiting_extraction(run: AgentRun) -> None:
    if run.status is not AgentRunStatus.RECEIVED or run.goal_type is not None:
        raise RunNotExecutable(run.id, run.status)


class _ModelCallLog:
    """The ModelCallRecorder handed to a planner: each record() is one short
    log transaction, and the call takes the run's next sequence_no.

    With ``max_requests``, before_request() enforces the stage's request
    limit, counted from the requests already recorded for the run and stage
    (retries and failed requests included). At the limit it ends the run
    FAILED (step_limit) and raises DecisionStopped: the request is never
    made.
    """

    def __init__(
        self,
        sessions: sessionmaker[Session],
        run_id: int,
        stage: ModelCallStage,
        *,
        max_requests: int | None = None,
    ) -> None:
        self._sessions = sessions
        self._run_id = run_id
        self._stage = stage
        self._max_requests = max_requests

    def before_request(self) -> None:
        if self._max_requests is None:
            return
        with self._sessions.begin() as log:
            made = ModelCallRepository(log).count_for_stage(self._run_id, self._stage)
            if made < self._max_requests:
                return
            run = _get_run(log, self._run_id)
            _end_at_limit(run, DecisionLimit.MODEL_REQUESTS, self._max_requests)
        # Raised after the commit, so the run's outcome stays recorded.
        raise DecisionStopped(self._run_id)

    def record(self, call: ModelCallRecord) -> None:
        with self._sessions.begin() as log:
            ModelCallRepository(log).add(
                ModelCall(
                    agent_run_id=self._run_id,
                    sequence_no=AgentRunRepository(log).next_sequence_no(self._run_id),
                    stage=self._stage,
                    model_name=call.model_name,
                    status=(
                        ModelCallStatus.SUCCEEDED
                        if call.error is None
                        else ModelCallStatus.FAILED
                    ),
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    latency_ms=call.latency_ms,
                    output=call.output,
                    error=(
                        None
                        if call.error is None
                        else call.error.model_dump(mode="json")
                    ),
                )
            )


def _record_extraction(
    run: AgentRun, instruction: str, outcome: ExtractedIntent | PlannerError
) -> None:
    """Store an assignment intent as the run's goal, or end the run."""
    match outcome:
        case EnsureAssignmentIntent():
            # The planner is asked to copy text, but whatever it is, text that
            # is not in the instruction was invented and is never resolved.
            missing = ungrounded_fields(outcome, instruction)
            if missing:
                _end(
                    run,
                    AgentRunStatus.FAILED,
                    reason=OutcomeReason.PLANNER_ERROR,
                    detail={
                        "stage": ModelCallStage.EXTRACTION.value,
                        "code": "ungrounded_output",
                        "error_type": None,
                        "fields": missing,
                    },
                )
                return
            run.goal_type = GoalType.ENSURE_ASSIGNMENT
            run.desired_state = DesiredState.ASSIGNED
            run.extracted_user_email = outcome.user_email
            run.extracted_product = outcome.product
        case NeedsClarification():
            _end(
                run,
                AgentRunStatus.NEEDS_CLARIFICATION,
                reason=OutcomeReason.INSTRUCTION_UNCLEAR,
                detail={"reason_code": outcome.reason_code},
            )
        case Unsupported():
            _end(
                run,
                AgentRunStatus.NEEDS_CLARIFICATION,
                reason=OutcomeReason.UNSUPPORTED_REQUEST,
                detail={"reason_code": outcome.reason_code},
            )
        case PlannerError():
            _end(
                run,
                AgentRunStatus.FAILED,
                reason=OutcomeReason.PLANNER_ERROR,
                detail={
                    "stage": ModelCallStage.EXTRACTION.value,
                    "code": outcome.code,
                    "error_type": outcome.error_type,
                },
            )
        case _:
            assert_never(outcome)


class AgentExecutor:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def run(
        self,
        *,
        instruction: str,
        requesting_actor: str,
        intent_planner: IntentPlanner,
        decision_planner: DecisionPlanner,
    ) -> int:
        """Drive one natural-language run to a terminal status, or to an
        approval pause (AWAITING_APPROVAL), and return its id.

        Raises InvalidInput, recording nothing, for a blank or over-long
        instruction or an invalid requesting actor. After that, a step that
        cannot go on ends the run itself, and the next step is not taken.
        """
        run_id = self.receive_run(
            instruction=instruction, requesting_actor=requesting_actor
        )
        if self.extract_intent(run_id, intent_planner) is not AgentRunStatus.RECEIVED:
            return run_id
        if self.resolve_run(run_id) is not AgentRunStatus.RESOLVED:
            return run_id
        self.decide(run_id, decision_planner)
        return run_id

    # --- run lifecycle ----------------------------------------------------------

    def receive_run(self, *, instruction: str, requesting_actor: str) -> int:
        """Persist a RECEIVED run for a natural-language instruction and return
        its id, before any model is called.

        Nothing is extracted yet: the goal columns stay NULL until the
        instruction's intent is extracted. Raises InvalidInput, and records
        nothing, for a blank or over-long instruction or an invalid
        requesting actor ("agent:" is reserved). The instruction is stored
        exactly as given.
        """
        requesting_actor = _validated_request(instruction, requesting_actor)
        with self._sessions.begin() as log:
            run = AgentRunRepository(log).add(
                AgentRun(
                    instruction=instruction,
                    requesting_actor=requesting_actor,
                    status=AgentRunStatus.RECEIVED,
                )
            )
            return run.id

    def extract_intent(self, run_id: int, planner: IntentPlanner) -> AgentRunStatus:
        """Ask ``planner`` what the run's instruction asks for, and record it.

        An assignment intent becomes the run's goal, with its text stored
        exactly as extracted; the run stays RECEIVED, ready for resolve_run.
        NeedsClarification and Unsupported end the run NEEDS_CLARIFICATION
        (instruction_unclear, unsupported_request). A PlannerError, or
        extracted text that is not in the instruction, ends it FAILED
        (planner_error). Nothing is resolved or attempted in any case.

        Raises RunNotExecutable, calling nothing, unless the run is RECEIVED
        and not yet extracted.
        """
        with self._sessions() as log:
            run = _get_run(log, run_id)
            _require_awaiting_extraction(run)
            instruction = run.instruction

        calls = _ModelCallLog(self._sessions, run_id, ModelCallStage.EXTRACTION)
        outcome: ExtractedIntent | PlannerError
        try:
            outcome = planner.plan(instruction, calls)
        except PlannerError as error:
            outcome = error
        except Exception as error:
            return self._fail(run_id, _unexpected_detail("extraction", error))

        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            _require_awaiting_extraction(run)
            _record_extraction(run, instruction, outcome)
            return run.status

    def resolve_run(self, run_id: int) -> AgentRunStatus:
        """Resolve the run's persisted extracted text to row ids.

        RESOLVED stores the goal on the run; an unknown, ambiguous or invalid
        entity ends the run NEEDS_CLARIFICATION with nothing attempted.
        """
        with self._sessions() as log:
            run = _get_run(log, run_id)
            if (
                run.status is not AgentRunStatus.RECEIVED
                or run.extracted_user_email is None
                or run.extracted_product is None
            ):
                # Not RECEIVED, or not extracted yet: there is nothing to resolve.
                raise RunNotExecutable(run_id, run.status)
            user_email, product = run.extracted_user_email, run.extracted_product

        resolution: ResolvedAssignmentGoal | ResolutionFailure
        try:
            with self._sessions() as read:
                resolution = resolve_assignment_goal(
                    UserService(read), LicenceService(read), user_email, product
                )
        except Exception as error:
            return self._fail(run_id, _unexpected_detail("resolution", error))

        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            if isinstance(resolution, ResolutionFailure):
                _end(
                    run,
                    AgentRunStatus.NEEDS_CLARIFICATION,
                    reason=resolution.code,
                    detail=resolution.detail,
                )
            else:
                run.resolved_user_id = resolution.user_id
                run.resolved_licence_id = resolution.licence_id
                run.status = AgentRunStatus.RESOLVED
            return run.status

    # --- decision ---------------------------------------------------------------

    def decide(self, run_id: int, planner: DecisionPlanner) -> AgentRunStatus:
        """Let ``planner`` choose the run's tool calls, then end or pause the
        run.

        What the model will be told is built from the persisted goal's
        semantic values and persisted, and the run moves to EXECUTING, before
        any model request. The planner then runs with no session open. Each
        of its requests is recorded as a decision ModelCall and refused past
        MAX_DECISION_MODEL_REQUESTS; each tool it calls goes through
        call_tool. Nothing is read or attempted unless the model asks for it.

        When the model concludes, its proposal is recorded and the run ends
        as decision_outcome says: the verifier and the application's own
        checks decide, never the proposal. A PlannerError ends the run FAILED
        (planner_error), and any other planner exception FAILED
        (unexpected_error). A limit, policy, or a tool failure the model is
        not shown has already ended or paused the run when DecisionStopped
        reaches here: the loop stops, no proposal is recorded, and the model
        is asked nothing more.

        Raises RunNotExecutable, calling nothing, unless the run is RESOLVED.
        """
        # LOG: persist what the model will be told, and start executing.
        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            if run.status is not AgentRunStatus.RESOLVED:
                raise RunNotExecutable(run_id, run.status)
            goal = _persisted_goal(run)
            context = decision_context(
                goal.extracted_user_email, goal.extracted_product
            )
            run.decision_context = context.model_dump(mode="json")
            run.status = AgentRunStatus.EXECUTING

        # No session is open while the planner runs; each request and each
        # tool call opens and closes its own.
        calls = _ModelCallLog(
            self._sessions,
            run_id,
            ModelCallStage.DECISION,
            max_requests=MAX_DECISION_MODEL_REQUESTS,
        )
        try:
            proposal = planner.decide(
                context, lambda tool: self.call_tool(run_id, tool), calls
            )
        except DecisionStopped:
            # The application ended or paused the run; that status stands.
            with self._sessions() as log:
                return _get_run(log, run_id).status
        except PlannerError as error:
            return self._end_decision(
                run_id,
                OutcomeReason.PLANNER_ERROR,
                {
                    "stage": ModelCallStage.DECISION.value,
                    "code": error.code,
                    "error_type": error.error_type,
                },
            )
        except Exception as error:
            return self._end_decision(
                run_id,
                OutcomeReason.UNEXPECTED_ERROR,
                _unexpected_detail("decision", error),
            )
        return self._finish_decision(run_id, proposal)

    def _finish_decision(
        self, run_id: int, proposal: DecisionProposal
    ) -> AgentRunStatus:
        """Record the model's proposal, verify, and end the run."""
        # LOG: the proposal is recorded as what it is, the model's opinion.
        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            if run.status is not AgentRunStatus.EXECUTING:
                raise RunNotExecutable(run_id, run.status)
            calls = ToolCallRepository(log)
            if calls.any_started(run.id):
                raise RunNotExecutable(run_id, run.status, UNFINISHED_CALL)
            run.decision_proposal = DecisionProposalKind(proposal.kind)
            if isinstance(proposal, CannotProceed):
                run.decision_reason_code = proposal.reason_code
            run.status = AgentRunStatus.VERIFYING
            goal = _persisted_goal(run)
            changed = calls.any_succeeded(run.id, tools.RECORDED_MUTATIONS)
            attempt = calls.latest_for_tools(run.id, tools.RECORDED_MUTATIONS)
            rejection = _blocking_rejection(attempt)
            rejection_detail = (
                None if rejection is None or attempt is None else _call_detail(attempt)
            )

        # READ: current state only. A claimed block is checked afresh, never
        # taken from what the model observed.
        block_check: BlockCheck | None = None
        try:
            with self._sessions() as read:
                result = verify(AssignmentService(read), goal)
                if not result.satisfied and isinstance(proposal, CannotProceed):
                    block_check = check_block(
                        UserService(read),
                        AssignmentService(read),
                        goal,
                        proposal.reason_code,
                    )
        except Exception as error:
            return self._fail(run_id, _unexpected_detail("verification", error))

        status, reason = decision_outcome(
            proposal=proposal,
            satisfied=result.satisfied,
            changed=changed,
            rejection=rejection,
            claim_confirmed=block_check is not None and block_check.confirmed,
        )
        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            _end(
                run,
                status,
                reason=reason,
                detail={
                    **result.model_dump(mode="json"),
                    "rejection": rejection_detail,
                    "block_check": (
                        None
                        if block_check is None
                        else block_check.model_dump(mode="json")
                    ),
                },
            )
            return run.status

    def _end_decision(
        self, run_id: int, reason: OutcomeReason, detail: dict[str, Any]
    ) -> AgentRunStatus:
        """End a run whose decision loop stopped without a proposal."""
        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            # A tool call may have ended the run already; that outcome stands.
            if run.status is AgentRunStatus.EXECUTING:
                _end(run, AgentRunStatus.FAILED, reason=reason, detail=detail)
            return run.status

    # --- tool calls -------------------------------------------------------------

    def call_tool(self, run_id: int, tool: TargetToolName) -> str:
        """Record and run one call of ``tool`` the decision model chose, and
        return the observation the model is given; see the module docstring
        for its three transactions.

        The call acts on the run's persisted goal. Admission (_start_call)
        may refuse it over a limit, or policy may deny it or hold it for
        approval: then no business transaction opens, the run has ended or
        paused, and DecisionStopped is raised. Otherwise the call's outcome
        and its observation (tools.observe) are persisted together. A
        rejection the model is shown leaves the run EXECUTING, so the model
        may react; a blocking one counts when the run finishes. A failure it
        is not shown ends the run FAILED (tool_failed), and DecisionStopped
        is raised.

        Raises RunNotExecutable, recording nothing, unless the run is in its
        decision stage (EXECUTING) with no unfinished call.
        """
        call_id, goal = self._start_call(run_id, tool)

        # BUSINESS: the service call and its audit event commit or roll back
        # together. The log session above is already committed and closed.
        outcome: ToolOutput | ToolError
        try:
            with self._sessions() as session, session.begin():
                outcome = tools.run(
                    tool,
                    goal,
                    users=UserService(session),
                    assignments=AssignmentService(session),
                    actor=agent_run_actor(run_id),
                )
        except Exception as error:
            # Also reached if the commit fails after the tool returned.
            outcome = tool_error(error, tools.RECORDED_NAMES[tool])

        # LOG: record what happened, and end the run if the model cannot be
        # shown it.
        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            call = ToolCallRepository(log).get(call_id)
            if call is None:
                raise RuntimeError(f"Tool call {call_id} disappeared.")
            call.completed_at = utcnow()
            observation = tools.observe(tool, goal, outcome)
            if isinstance(outcome, ToolError):
                call.status = ToolCallStatus.FAILED
                call.error = outcome.model_dump(mode="json")
                if observation is None:
                    _end(
                        run,
                        AgentRunStatus.FAILED,
                        reason=OutcomeReason.TOOL_FAILED,
                        detail=_call_detail(call),
                    )
            else:
                call.status = ToolCallStatus.SUCCEEDED
                call.result = outcome.model_dump(mode="json")
            if observation is not None:
                # Serialized once, here: this exact text is what the model
                # is given.
                call.observation = tools.serialize(observation)
            shown = call.observation
        if shown is None:
            # Raised after the commit, so the call and the run's outcome
            # stay recorded.
            raise DecisionStopped(run_id)
        return shown

    def _start_call(
        self, run_id: int, tool: TargetToolName
    ) -> tuple[int, ResolvedAssignmentGoal]:
        """LOG: admit the call, record it STARTED, and return its id and the
        goal it acts on. Or record why it was refused, denied or held, end or
        pause the run, and raise DecisionStopped.

        Checked in this order, in this one transaction: the run's status and
        unfinished calls, the run's limits, and last, for the mutation,
        policy. So a call refused for a limit is never evaluated, and its
        policy_decision stays NULL.
        """
        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            if run.status is not AgentRunStatus.EXECUTING:
                raise RunNotExecutable(run_id, run.status)
            calls = ToolCallRepository(log)
            if calls.any_started(run.id):
                raise RunNotExecutable(run_id, run.status, UNFINISHED_CALL)
            goal = _persisted_goal(run)
            call = ToolCall(
                agent_run_id=run.id,
                sequence_no=AgentRunRepository(log).next_sequence_no(run.id),
                tool_name=tools.RECORDED_NAMES[tool],
                arguments=tools.arguments(tool, goal),
            )
            exceeded = _exceeded_limit(calls, run.id, tool)
            if exceeded is not None:
                _refuse_over_limit(run, calls, call, *exceeded)
            else:
                # Read from committed state, here: atomic with the recorded
                # decision, not with the business transaction that follows.
                policy_decision = policy.evaluate(tool, goal, LicenceService(log))
                call.policy_decision = policy_decision
                match policy_decision:
                    case None | PolicyDecision.ALLOW:
                        call.status = ToolCallStatus.STARTED
                        calls.add(call)
                        return call.id, goal
                    case PolicyDecision.DENY:
                        _deny(run, calls, call)
                    case PolicyDecision.REQUIRE_APPROVAL:
                        _hold_for_approval(run, calls, call)
                    case _:
                        assert_never(policy_decision)
        # Raised after the commit, so the call and the run's outcome stay
        # recorded.
        raise DecisionStopped(run_id)

    # --- helpers ----------------------------------------------------------------

    def _fail(self, run_id: int, detail: dict[str, Any]) -> AgentRunStatus:
        with self._sessions.begin() as log:
            run = _get_run(log, run_id)
            _end(
                run,
                AgentRunStatus.FAILED,
                reason=OutcomeReason.UNEXPECTED_ERROR,
                detail=detail,
            )
            return run.status


def _end(
    run: AgentRun,
    status: AgentRunStatus,
    *,
    reason: OutcomeReason,
    detail: dict[str, Any] | None,
) -> None:
    """End the run: its final status, why, and when. A run that has ended
    stays ended. Which reason may go with which status is the agent_runs
    outcome_matches_status CHECK constraint."""
    if run.completed_at is not None:
        raise RunNotExecutable(run.id, run.status, "it has already ended")
    run.status = status
    run.outcome_reason = reason
    run.outcome_detail = detail
    run.completed_at = utcnow()


def _get_run(session: Session, run_id: int) -> AgentRun:
    run = AgentRunRepository(session).get(run_id)
    if run is None:
        raise AgentRunNotFound(run_id)
    return run


def _call_detail(call: ToolCall) -> dict[str, Any]:
    """The run's outcome detail for a call that ended or blocked it: which
    call, and the error recorded with it."""
    return {
        "tool_call_id": call.id,
        "sequence_no": call.sequence_no,
        "tool_name": call.tool_name,
        "error": call.error,
    }


def _exceeded_limit(
    calls: ToolCallRepository, run_id: int, tool: TargetToolName
) -> tuple[DecisionLimit, int] | None:
    """The limit a call of ``tool`` would exceed, and its maximum, counting
    the run's recorded calls of its kind (reads, or the mutation), whatever
    their outcome."""
    if tool == tools.MUTATION:
        limit, maximum = DecisionLimit.MUTATION_CALLS, MAX_DECISION_MUTATION_CALLS
        kind = tools.RECORDED_MUTATIONS
    else:
        limit, maximum = DecisionLimit.READ_CALLS, MAX_DECISION_READ_CALLS
        kind = tools.RECORDED_READS
    if calls.count_for_tools(run_id, kind) >= maximum:
        return limit, maximum
    return None


def _end_at_limit(run: AgentRun, limit: DecisionLimit, maximum: int) -> None:
    _end(
        run,
        AgentRunStatus.FAILED,
        reason=OutcomeReason.STEP_LIMIT,
        detail={"limit": limit.value, "maximum": maximum},
    )


def _refuse_over_limit(
    run: AgentRun,
    calls: ToolCallRepository,
    call: ToolCall,
    limit: DecisionLimit,
    maximum: int,
) -> None:
    """Record a call over a limit as FAILED, never run, so the trace shows
    the attempt, and end the run FAILED (step_limit)."""
    call.status = ToolCallStatus.FAILED
    call.error = ToolError(
        code=STEP_LIMIT,
        message=f"Decision limit reached: at most {maximum} {limit.value}.",
        error_type=None,
    ).model_dump(mode="json")
    call.completed_at = utcnow()
    calls.add(call)
    _end_at_limit(run, limit, maximum)


def _deny(run: AgentRun, calls: ToolCallRepository, call: ToolCall) -> None:
    """Record a call policy denied as FAILED, never run, and end the run
    BLOCKED (policy_denied)."""
    error = ToolError(
        code=POLICY_DENIED,
        message=f"Policy denies {call.tool_name} for this run's target.",
        error_type=None,
    )
    call.status = ToolCallStatus.FAILED
    call.error = error.model_dump(mode="json")
    call.completed_at = utcnow()
    calls.add(call)
    _end(
        run,
        AgentRunStatus.BLOCKED,
        reason=OutcomeReason.POLICY_DENIED,
        detail=_call_detail(call),
    )


def _hold_for_approval(
    run: AgentRun, calls: ToolCallRepository, call: ToolCall
) -> None:
    """Record a call policy holds for approval as AWAITING_APPROVAL, unrun,
    with its arguments, and pause the run."""
    call.status = ToolCallStatus.AWAITING_APPROVAL
    calls.add(call)
    run.status = AgentRunStatus.AWAITING_APPROVAL


def _blocking_rejection(attempt: ToolCall | None) -> OutcomeReason | None:
    """The BLOCKED reason if a run's latest mutation attempt was rejected by
    a blocking domain rule. Read from the error code the executor recorded
    from the typed domain error, never from a message."""
    if attempt is None or attempt.status is not ToolCallStatus.FAILED:
        return None
    if attempt.error is None:
        return None
    return _BLOCKING_ERRORS.get(attempt.error["code"])
