"""Policy at call admission: allow, deny and require_approval.

A mutation is checked against policy when AgentExecutor._start_call admits
it, after its goal scope and the run's limits. Denied, it never runs and the
run ends BLOCKED; held for approval, it never runs and the run pauses. In
both cases no business transaction opens, and a model is asked nothing more.
The tests drive the real PydanticAI planner with a scripted model
(FunctionModel), so the stop is checked through the actual tool scheduling
and exception propagation.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.agent import policy, tools
from app.agent.executor import AgentExecutor, RunNotExecutable
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
    EnsureAssignmentIntent,
    ResolvedAssignmentGoal,
    TargetToolName,
)
from app.services.licences import LicenceService
from support import (
    ASSIGN,
    ASSIGNMENTS,
    CAPACITY,
    GOAL_REACHED,
    USER,
    FixedIntent,
    Script,
    call,
    directed_run,
    resolved_run,
)

Sessions = sessionmaker[Session]
S = AgentRunStatus
R = OutcomeReason
ALLOW = PolicyDecision.ALLOW
DENY = PolicyDecision.DENY
REQUIRE = PolicyDecision.REQUIRE_APPROVAL
AWAITING = ToolCallStatus.AWAITING_APPROVAL

ADA = 48213
FIGMA = 97531


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


def full_run(executor: AgentExecutor) -> int:
    """A whole run whose model reads, then tries to assign, then claims
    success."""
    return directed_run(
        executor, Script(call(ASSIGNMENTS), call(CAPACITY), call(ASSIGN), GOAL_REACHED)
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
    tools.run is called only inside that transaction."""
    entered: list[str] = []
    real = tools.run

    def spy(tool: TargetToolName, goal: ResolvedAssignmentGoal, **services: Any) -> Any:
        entered.append(tool)
        return real(tool, goal, **services)

    monkeypatch.setattr(tools, "run", spy)
    return entered


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The tool of every call the planner asked the application to run."""
    asked: list[str] = []
    real = AgentExecutor.call_tool

    def spy(self: AgentExecutor, run_id: int, tool: TargetToolName) -> str:
        asked.append(tool)
        return real(self, run_id, tool)

    monkeypatch.setattr(AgentExecutor, "call_tool", spy)
    return asked


@pytest.fixture
def evaluations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The tool of every call policy was evaluated for, in order."""
    evaluated: list[str] = []
    real = policy.evaluate

    def spy(
        tool: TargetToolName, goal: ResolvedAssignmentGoal, licences: LicenceService
    ) -> PolicyDecision | None:
        evaluated.append(tool)
        return real(tool, goal, licences)

    monkeypatch.setattr(policy, "evaluate", spy)
    return evaluated


# --- the policy itself --------------------------------------------------------


def goal_for(licence_id: int) -> ResolvedAssignmentGoal:
    return ResolvedAssignmentGoal(
        user_id=1,
        licence_id=licence_id,
        extracted_user_email="ada@example.com",
        extracted_product="Figma",
    )


@pytest.mark.parametrize("agent_policy", list(PolicyDecision))
def test_an_assignment_is_decided_by_its_licence_policy(
    session: Session, agent_policy: PolicyDecision
) -> None:
    licence = Licence(product="Figma", seats_total=5, agent_policy=agent_policy)
    session.add(licence)
    session.flush()

    decision = policy.evaluate(ASSIGN, goal_for(licence.id), LicenceService(session))

    assert decision is agent_policy


@pytest.mark.parametrize("tool", [USER, CAPACITY, ASSIGNMENTS])
def test_reads_are_not_governed_by_policy(
    session: Session, tool: TargetToolName
) -> None:
    # Nothing is read for them either: licence 1 does not exist.
    assert policy.evaluate(tool, goal_for(1), LicenceService(session)) is None


# --- each decision, recorded with its call ---------------------------------------


def test_an_allowed_assignment_runs_as_before_and_records_the_decision(
    executor: AgentExecutor, session_factory: Sessions, business: list[str]
) -> None:
    seed(session_factory, ALLOW)

    run_id = full_run(executor)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.COMPLETED, R.GOAL_SATISFIED)
    assert [
        (c.tool_name, c.policy_decision) for c in tool_calls(session_factory, run_id)
    ] == [
        ("list_user_assignments", None),
        ("get_licence", None),
        ("assign_licence", ALLOW),
    ]
    assert business == [ASSIGNMENTS, CAPACITY, ASSIGN]
    assert mutations(session_factory) == (1, 1)
    with session_factory() as session:
        event = session.scalars(select(AuditEvent)).one()
    assert event.actor == f"agent:run-{run_id}"


def test_a_denied_assignment_never_runs_and_blocks_the_run(
    executor: AgentExecutor, session_factory: Sessions, business: list[str]
) -> None:
    seed(session_factory, DENY)

    run_id = full_run(executor)

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
    assert ASSIGN not in business
    assert mutations(session_factory) == (0, 0)
    # Ended at admission, not verified.
    assert run.outcome_detail is not None and "satisfied" not in run.outcome_detail


def test_a_held_assignment_never_runs_and_pauses_the_run(
    executor: AgentExecutor, session_factory: Sessions, business: list[str]
) -> None:
    seed(session_factory, REQUIRE)

    run_id = full_run(executor)

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
    assert ASSIGN not in business
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
    entered: list[str] = []

    def no_business(tool: TargetToolName, *_: Any, **__: Any) -> Any:
        entered.append(tool)  # called only inside a business transaction
        raise AssertionError("a business transaction was opened")

    monkeypatch.setattr(tools, "run", no_business)
    first = len(statements)

    status = executor.decide(run_id, Script(call(ASSIGN), GOAL_REACHED).planner())

    assert entered == []
    sent = " ".join(statements[first:])
    # Only the log tables and the licence policy were touched.
    assert "assignments" not in sent and "audit_events" not in sent
    assert "licences" in sent
    assert mutations(session_factory) == (0, 0)
    (stopped,) = tool_calls(session_factory, run_id)
    assert (stopped.result, stopped.observation) == (None, None)
    if agent_policy is DENY:
        assert status is S.BLOCKED
        assert stopped.error is not None and stopped.error["code"] == "policy_denied"
    else:
        assert status is S.AWAITING_APPROVAL
        assert stopped.error is None


def test_a_paused_run_refuses_every_further_step(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, REQUIRE)
    run_id = resolved_run(executor)
    executor.decide(run_id, Script(call(ASSIGN)).planner())
    before = get_run(session_factory, run_id)
    assert before.status is S.AWAITING_APPROVAL
    calls_before = len(tool_calls(session_factory, run_id))

    intent = EnsureAssignmentIntent(user_email="ada@example.com", product="Figma")
    steps: list[Callable[[], object]] = [
        lambda: executor.extract_intent(run_id, FixedIntent(intent)),
        lambda: executor.resolve_run(run_id),
        lambda: executor.decide(run_id, Script(GOAL_REACHED).planner()),
        lambda: executor.call_tool(run_id, USER),
        lambda: executor.call_tool(run_id, ASSIGN),
    ]
    for step in steps:
        with pytest.raises(RunNotExecutable):
            step()

    after = get_run(session_factory, run_id)
    assert (after.status, after.last_sequence_no) == (
        S.AWAITING_APPROVAL,
        before.last_sequence_no,
    )
    assert len(tool_calls(session_factory, run_id)) == calls_before
    assert mutations(session_factory) == (0, 0)


# --- the order of admission checks -------------------------------------------------


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
    assert evaluations == [ASSIGN]  # the refused call never was


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
    assert business == [ASSIGNMENTS]
    assert mutations(session_factory) == (0, 0)


@pytest.mark.parametrize("agent_policy", [DENY, REQUIRE])
def test_a_later_tool_in_the_same_response_never_runs(
    executor: AgentExecutor,
    session_factory: Sessions,
    business: list[str],
    requested: list[str],
    agent_policy: PolicyDecision,
) -> None:
    seed(session_factory, agent_policy)
    # One response asks for the mutation, then a read.
    script = Script(call(ASSIGN, USER), GOAL_REACHED)
    run_id = resolved_run(executor)

    executor.decide(run_id, script.planner())

    # The read never reached the application, let alone a business transaction.
    assert requested == [ASSIGN]
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
    business: list[str],
    requested: list[str],
) -> None:
    # The control for the test above: the spies do see a later tool when
    # nothing stops the loop.
    seed(session_factory, ALLOW)
    script = Script(call(ASSIGN, USER), GOAL_REACHED)
    run_id = resolved_run(executor)

    assert executor.decide(run_id, script.planner()) is S.COMPLETED
    assert requested == business == [ASSIGN, USER]
    assert [c.tool_name for c in tool_calls(session_factory, run_id)] == [
        "assign_licence",
        "get_user",
    ]
