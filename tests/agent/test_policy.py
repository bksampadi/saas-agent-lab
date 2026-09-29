"""Policy at call admission: allow, deny and require_approval, on the Day 1
(direct) path and the model-directed path.

A mutation is checked against policy when AgentExecutor._start_call admits
it, after its goal scope and the run's limits. Denied, it never runs and the
run ends BLOCKED; held for approval, it never runs and the run pauses. In
both cases no business transaction opens, and a model is asked nothing more.
The model-directed tests drive the real PydanticAI planner with a scripted
model (FunctionModel), so the stop is checked through the actual tool
scheduling and exception propagation.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

import app.agent.executor as executor_module
from app.agent import policy, tools
from app.agent.decision_tools import GoalBoundTools
from app.agent.executor import AgentExecutor, RunNotExecutable, ToolCallOutcome
from app.agent.harness import run_ensure_assignment
from app.agent.pydantic_ai_decision import PydanticAIDecisionPlanner
from app.agent.status import IllegalTransition
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    AuditEvent,
    Licence,
    OutcomeReason,
    PolicyDecision,
    ToolCall,
    ToolCallStatus,
    User,
)
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import (
    AssignLicenceInput,
    ExtractedAssignmentIntent,
    GetLicenceInput,
    GetUserInput,
    ListUserAssignmentsInput,
    ToolError,
    ToolInput,
)
from app.services.licences import LicenceService

HUMAN = "admin@example.com"
Sessions = sessionmaker[Session]
S = AgentRunStatus
R = OutcomeReason
ALLOW = PolicyDecision.ALLOW
DENY = PolicyDecision.DENY
REQUIRE = PolicyDecision.REQUIRE_APPROVAL
AWAITING = ToolCallStatus.AWAITING_APPROVAL

ADA = 48213
FIGMA = 97531
SLACK = 97532

Step = ModelResponse | Callable[[], ModelResponse]


# --- scripted model -----------------------------------------------------------


class Script:
    """A FunctionModel that answers with ``steps`` in order, and keeps every
    request it was sent. A callable step runs when the request arrives (to
    change state between requests). Steps left over were never requested."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[list[ModelMessage]] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(list(messages))
        step = self.steps.pop(0)
        return step if isinstance(step, ModelResponse) else step()

    def planner(self) -> PydanticAIDecisionPlanner:
        return PydanticAIDecisionPlanner(
            FunctionModel(self.respond, model_name="scripted-model"), timeout_seconds=5
        )


def call(*tool_names: str) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(name, {}) for name in tool_names],
        usage=RequestUsage(input_tokens=100, output_tokens=10),
    )


GOAL_REACHED = ModelResponse(
    parts=[ToolCallPart("goal_reached", {})],
    usage=RequestUsage(input_tokens=100, output_tokens=10),
)
USER = "get_target_user"
CAPACITY = "get_target_licence_capacity"
ASSIGNMENTS = "list_target_user_assignments"
ASSIGN = "assign_target_licence"


# --- database -----------------------------------------------------------------


def seed(sessions: Sessions, agent_policy: PolicyDecision, *, seats: int = 5) -> None:
    with sessions.begin() as session:
        session.add(User(id=ADA, email="ada@example.com", name="Ada"))
        session.add(
            Licence(
                id=FIGMA, product="Figma", seats_total=seats, agent_policy=agent_policy
            )
        )


def set_policy(sessions: Sessions, agent_policy: PolicyDecision) -> None:
    with sessions.begin() as session:
        session.execute(
            update(Licence).where(Licence.id == FIGMA).values(agent_policy=agent_policy)
        )


def resolved_run(executor: AgentExecutor) -> int:
    run_id = executor.create_run(
        instruction="Give ada@example.com a Figma seat.",
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product="Figma"),
    )
    assert executor.resolve_run(run_id) is S.RESOLVED
    return run_id


def harness_run(executor: AgentExecutor) -> int:
    return run_ensure_assignment(
        executor,
        instruction="Give ada@example.com a Figma seat.",
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product="Figma"),
    )


def get_run(sessions: Sessions, run_id: int) -> AgentRun:
    with sessions() as session:
        run = session.get(AgentRun, run_id)
        assert run is not None
        return run


def tool_calls(sessions: Sessions, run_id: int) -> list[ToolCall]:
    with sessions() as session:
        return ToolCallRepository(session).list_for_run(run_id)


def mutations(sessions: Sessions) -> tuple[int, int]:
    """(assignment rows, audit rows): both zero means nothing was changed."""
    with sessions() as session:
        return (
            len(session.scalars(select(Assignment)).all()),
            len(session.scalars(select(AuditEvent)).all()),
        )


@pytest.fixture
def business(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The tool of every call that entered a business transaction, in order.
    tools.run_tool is called only inside that transaction."""
    entered: list[str] = []
    real = tools.run_tool

    def spy(context: tools.ToolContext, args: ToolInput) -> Any:
        entered.append(args.tool_name)
        return real(context, args)

    monkeypatch.setattr(tools, "run_tool", spy)
    return entered


@pytest.fixture
def evaluations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The tool of every call policy was evaluated for, in order."""
    evaluated: list[str] = []
    real = policy.evaluate

    def spy(args: ToolInput, licences: LicenceService) -> PolicyDecision | None:
        evaluated.append(args.tool_name)
        return real(args, licences)

    monkeypatch.setattr(policy, "evaluate", spy)
    return evaluated


# --- the policy itself --------------------------------------------------------


@pytest.mark.parametrize("agent_policy", list(PolicyDecision))
def test_an_assignment_is_decided_by_its_licence_policy(
    session: Session, agent_policy: PolicyDecision
) -> None:
    licence = Licence(product="Figma", seats_total=5, agent_policy=agent_policy)
    session.add(licence)
    session.flush()

    decision = policy.evaluate(
        AssignLicenceInput(user_id=1, licence_id=licence.id), LicenceService(session)
    )

    assert decision is agent_policy


@pytest.mark.parametrize(
    "args",
    [
        GetUserInput(user_id=1),
        GetLicenceInput(licence_id=1),
        ListUserAssignmentsInput(user_id=1),
    ],
    ids=lambda args: args.tool_name,
)
def test_reads_are_not_governed_by_policy(session: Session, args: ToolInput) -> None:
    # Nothing is read for them either: licence 1 does not exist.
    assert policy.evaluate(args, LicenceService(session)) is None


def test_assign_licence_is_the_only_mutation_policy_governs_today() -> None:
    # A tripwire: a new mutating tool needs its own policy case and tests.
    assert tools.MUTATING_TOOL_NAMES == {AssignLicenceInput.tool_name}


def test_an_empty_outcome_can_only_mean_a_call_held_for_approval() -> None:
    held = ToolCallOutcome(
        tool_call_id=1,
        sequence_no=1,
        output=None,
        error=None,
        run_status=S.AWAITING_APPROVAL,
    )
    assert held.run_status is S.AWAITING_APPROVAL

    error = ToolError(code="policy_denied", message="m", error_type=None)
    for output_error, run_status in [
        ((None, None), S.EXECUTING),  # not an empty success
        ((None, error), S.AWAITING_APPROVAL),  # held calls carry no error
    ]:
        with pytest.raises(ValueError, match="held for approval"):
            ToolCallOutcome(
                tool_call_id=1,
                sequence_no=1,
                output=output_error[0],
                error=output_error[1],
                run_status=run_status,
            )


# --- the Day 1 (direct) path -----------------------------------------------------


def test_an_allowed_assignment_runs_as_before_and_records_the_decision(
    executor: AgentExecutor, session_factory: Sessions, business: list[str]
) -> None:
    seed(session_factory, ALLOW)

    run_id = harness_run(executor)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.COMPLETED, R.GOAL_SATISFIED)
    assert [
        (c.tool_name, c.policy_decision) for c in tool_calls(session_factory, run_id)
    ] == [
        ("list_user_assignments", None),
        ("get_licence", None),
        ("assign_licence", ALLOW),
    ]
    assert business == ["list_user_assignments", "get_licence", "assign_licence"]
    assert mutations(session_factory) == (1, 1)
    with session_factory() as session:
        event = session.scalars(select(AuditEvent)).one()
    assert event.actor == f"agent:run-{run_id}"


def test_a_denied_assignment_never_runs_and_blocks_the_run(
    executor: AgentExecutor, session_factory: Sessions, business: list[str]
) -> None:
    seed(session_factory, DENY)

    run_id = harness_run(executor)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.BLOCKED, R.POLICY_DENIED)
    assert run.completed_at is not None
    *reads, denied = tool_calls(session_factory, run_id)
    assert [c.policy_decision for c in reads] == [None, None]
    assert (denied.tool_name, denied.status, denied.policy_decision) == (
        "assign_licence",
        ToolCallStatus.FAILED,
        DENY,
    )
    assert denied.error is not None and denied.error["code"] == "policy_denied"
    assert (denied.result, denied.observation) == (None, None)
    assert denied.completed_at is not None
    # Zero mutation: the call never entered a business transaction.
    assert "assign_licence" not in business
    assert mutations(session_factory) == (0, 0)
    # Ended at admission, not verified.
    assert run.outcome_detail is not None and "satisfied" not in run.outcome_detail


def test_a_held_assignment_never_runs_and_pauses_the_run(
    executor: AgentExecutor, session_factory: Sessions, business: list[str]
) -> None:
    seed(session_factory, REQUIRE)

    run_id = harness_run(executor)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason, run.outcome_detail, run.completed_at) == (
        S.AWAITING_APPROVAL,
        None,
        None,
        None,
    )
    held = tool_calls(session_factory, run_id)[-1]
    assert (held.tool_name, held.status, held.policy_decision) == (
        "assign_licence",
        AWAITING,
        REQUIRE,
    )
    # The operation is persisted as it was requested, and nothing happened yet.
    assert held.arguments == {"user_id": ADA, "licence_id": FIGMA}
    assert (held.result, held.error, held.observation, held.completed_at) == (
        None,
        None,
        None,
        None,
    )
    assert "assign_licence" not in business
    assert mutations(session_factory) == (0, 0)


@pytest.mark.parametrize("agent_policy", [DENY, REQUIRE])
def test_a_denied_or_held_mutation_opens_no_business_transaction(
    executor: AgentExecutor,
    session_factory: Sessions,
    statements: list[str],
    monkeypatch: pytest.MonkeyPatch,
    agent_policy: PolicyDecision,
) -> None:
    seed(session_factory, agent_policy)
    run_id = resolved_run(executor)
    contexts: list[str] = []

    def no_business(session: Session, actor: str) -> Any:
        contexts.append(actor)  # the first thing a business transaction does
        raise AssertionError("a business transaction was opened")

    monkeypatch.setattr(executor_module, "_tool_context", no_business)
    first = len(statements)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=ADA, licence_id=FIGMA)
    )

    assert contexts == []
    sent = " ".join(statements[first:])
    # Only the log tables and the licence policy were touched.
    assert "assignments" not in sent and "audit_events" not in sent
    assert "licences" in sent
    assert mutations(session_factory) == (0, 0)
    if agent_policy is DENY:
        assert outcome.run_status is S.BLOCKED
        assert outcome.error is not None and outcome.error.code == "policy_denied"
        assert outcome.output is None
    else:
        # Distinguished by run_status: held, not an empty success.
        assert outcome.run_status is S.AWAITING_APPROVAL
        assert (outcome.output, outcome.error, outcome.observation) == (
            None,
            None,
            None,
        )


def paused_by_harness(executor: AgentExecutor, sessions: Sessions) -> int:
    return harness_run(executor)


def paused_by_model(executor: AgentExecutor, sessions: Sessions) -> int:
    run_id = resolved_run(executor)
    executor.decide(run_id, Script(call(ASSIGN)).planner())
    return run_id


@pytest.mark.parametrize(
    "pause", [paused_by_harness, paused_by_model], ids=["direct", "model-directed"]
)
def test_a_paused_run_refuses_every_further_step(
    executor: AgentExecutor,
    session_factory: Sessions,
    pause: Callable[[AgentExecutor, Sessions], int],
) -> None:
    seed(session_factory, REQUIRE)
    run_id = pause(executor, session_factory)
    before = get_run(session_factory, run_id)
    assert before.status is S.AWAITING_APPROVAL
    calls_before = len(tool_calls(session_factory, run_id))

    assign = AssignLicenceInput(user_id=ADA, licence_id=FIGMA)
    steps: list[Callable[[], object]] = [
        lambda: executor.call_tool(run_id, GetUserInput(user_id=ADA)),
        lambda: executor.call_tool(run_id, assign),
        lambda: executor.call_decision_tool(run_id, assign),
        lambda: executor.decide(run_id, Script(GOAL_REACHED).planner()),
        lambda: executor.resolve_run(run_id),
    ]
    for step in steps:
        with pytest.raises(RunNotExecutable):
            step()
    # A model-directed run is refused before anything else (it ends through
    # decide); a direct one at the transition, as for any status that
    # verification cannot follow.
    with pytest.raises((RunNotExecutable, IllegalTransition)):
        executor.verify_and_finish(run_id)

    after = get_run(session_factory, run_id)
    assert (after.status, after.last_sequence_no) == (
        S.AWAITING_APPROVAL,
        before.last_sequence_no,
    )
    assert len(tool_calls(session_factory, run_id)) == calls_before
    assert mutations(session_factory) == (0, 0)


# --- the order of admission checks -------------------------------------------------


def test_a_call_outside_the_goal_is_refused_before_policy_is_evaluated(
    executor: AgentExecutor, session_factory: Sessions, evaluations: list[str]
) -> None:
    seed(session_factory, DENY)
    with session_factory.begin() as session:
        session.add(Licence(id=SLACK, product="Slack", seats_total=5))
    run_id = resolved_run(executor)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=ADA, licence_id=SLACK)
    )

    assert outcome.run_status is S.FAILED
    assert get_run(session_factory, run_id).outcome_reason is R.GOAL_SCOPE_VIOLATION
    (refused,) = tool_calls(session_factory, run_id)
    assert refused.error is not None
    assert (refused.error["code"], refused.policy_decision) == (
        "goal_scope_violation",
        None,
    )
    assert evaluations == []


def test_a_mutation_over_the_limit_is_refused_before_policy_is_evaluated(
    executor: AgentExecutor, session_factory: Sessions, evaluations: list[str]
) -> None:
    # No seats: the first attempt is admitted, runs and is rejected, which
    # the model is shown. Then policy turns to deny, and the model tries again.
    seed(session_factory, ALLOW, seats=0)

    def deny_then_try_again() -> ModelResponse:
        set_policy(session_factory, DENY)
        return call(ASSIGN)

    run_id = resolved_run(executor)
    status = executor.decide(
        run_id, Script(call(ASSIGN), deny_then_try_again).planner()
    )

    assert (status, get_run(session_factory, run_id).outcome_reason) == (
        S.FAILED,
        R.STEP_LIMIT,
    )
    first, second = tool_calls(session_factory, run_id)
    assert first.error is not None and second.error is not None
    assert (first.error["code"], first.policy_decision) == ("no_seats_available", ALLOW)
    assert (second.error["code"], second.policy_decision) == ("step_limit", None)
    assert evaluations == ["assign_licence"]  # the refused call never was


def test_policy_is_read_when_the_call_is_admitted_not_at_resolution(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, ALLOW)

    def hold_then_assign() -> ModelResponse:
        set_policy(session_factory, REQUIRE)  # committed after resolution
        return call(ASSIGN)

    run_id = resolved_run(executor)
    status = executor.decide(
        run_id, Script(call(ASSIGNMENTS), hold_then_assign).planner()
    )

    assert status is S.AWAITING_APPROVAL
    assert tool_calls(session_factory, run_id)[-1].policy_decision is REQUIRE
    assert mutations(session_factory) == (0, 0)


# --- the model-directed path: the model gets no second chance ------------------


@pytest.mark.parametrize(
    ("agent_policy", "status", "reason"),
    [(DENY, S.BLOCKED, R.POLICY_DENIED), (REQUIRE, S.AWAITING_APPROVAL, None)],
    ids=["deny", "require-approval"],
)
def test_the_model_is_asked_nothing_more_after_a_denial_or_hold(
    executor: AgentExecutor,
    session_factory: Sessions,
    business: list[str],
    agent_policy: PolicyDecision,
    status: AgentRunStatus,
    reason: OutcomeReason | None,
) -> None:
    seed(session_factory, agent_policy)
    # What it would do next, given the chance: look again, then claim success.
    look_again = call(ASSIGNMENTS)
    script = Script(call(ASSIGNMENTS), call(ASSIGN), look_again, GOAL_REACHED)
    run_id = resolved_run(executor)

    returned = executor.decide(run_id, script.planner())

    run = get_run(session_factory, run_id)
    assert (returned, run.status, run.outcome_reason) == (status, status, reason)
    assert len(script.requests) == 2  # the last chose the mutation
    assert script.steps == [look_again, GOAL_REACHED]  # never requested
    assert run.decision_proposal is None
    last = tool_calls(session_factory, run_id)[-1]
    assert (last.tool_name, last.policy_decision, last.observation) == (
        "assign_licence",
        agent_policy,
        None,
    )
    assert business == ["list_user_assignments"]
    assert mutations(session_factory) == (0, 0)


@pytest.mark.parametrize("agent_policy", [DENY, REQUIRE])
def test_a_later_tool_in_the_same_response_never_runs(
    executor: AgentExecutor,
    session_factory: Sessions,
    business: list[str],
    monkeypatch: pytest.MonkeyPatch,
    agent_policy: PolicyDecision,
) -> None:
    seed(session_factory, agent_policy)
    entered: list[str] = []
    real_target_user = GoalBoundTools.get_target_user
    real_get_user = tools.get_user

    def target_user_spy(self: GoalBoundTools) -> str:
        entered.append("get_target_user")  # the model-facing tool's body
        return real_target_user(self)

    def get_user_spy(context: tools.ToolContext, args: GetUserInput) -> Any:
        entered.append("get_user")  # the application tool's body
        return real_get_user(context, args)

    monkeypatch.setattr(GoalBoundTools, "get_target_user", target_user_spy)
    monkeypatch.setattr(tools, "get_user", get_user_spy)
    # One response asks for the mutation, then a read.
    script = Script(call(ASSIGN, USER), GOAL_REACHED)
    run_id = resolved_run(executor)

    executor.decide(run_id, script.planner())

    assert entered == []
    assert business == []
    assert [c.tool_name for c in tool_calls(session_factory, run_id)] == [
        "assign_licence"
    ]
    assert len(script.requests) == 1
    assert script.steps == [GOAL_REACHED]
    assert mutations(session_factory) == (0, 0)


def test_the_same_response_still_runs_both_tools_when_policy_allows(
    executor: AgentExecutor,
    session_factory: Sessions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The control for the test above: the spies do see a later tool when
    # nothing stops the loop.
    seed(session_factory, ALLOW)
    entered: list[str] = []
    real_target_user = GoalBoundTools.get_target_user

    def target_user_spy(self: GoalBoundTools) -> str:
        entered.append("get_target_user")
        return real_target_user(self)

    monkeypatch.setattr(GoalBoundTools, "get_target_user", target_user_spy)
    script = Script(call(ASSIGN, USER), GOAL_REACHED)
    run_id = resolved_run(executor)

    assert executor.decide(run_id, script.planner()) is S.COMPLETED
    assert entered == ["get_target_user"]
    assert [c.tool_name for c in tool_calls(session_factory, run_id)] == [
        "assign_licence",
        "get_user",
    ]
