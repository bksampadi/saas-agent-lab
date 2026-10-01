"""Transaction isolation against real SQLite locking (``locking_engine``):
the executor never has two transactions open at once, on any path."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse
from sqlalchemy import Engine, event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

import app.agent.executor as executor_module
from app.agent import tools
from app.agent.executor import AgentExecutor
from app.agent.planner import CallTool, DecisionModelCallRecorder
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    DesiredState,
    GoalType,
    Licence,
    PolicyDecision,
    User,
)
from app.repositories.assignments import AssignmentRepository
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import (
    AssignmentSnapshot,
    DecisionContext,
    DecisionProposal,
    GoalReached,
    ResolvedAssignmentGoal,
    TargetToolName,
    ToolOutput,
)
from app.services.assignments import AssignmentService
from app.services.users import UserService
from support import (
    ASSIGN,
    ASSIGNMENTS,
    CAPACITY,
    GOAL_REACHED,
    HUMAN,
    Crash,
    FakeDecisionPlanner,
    Script,
    TransactionCounter,
    add,
    call,
    directed_run,
    resolved_run,
)

Sessions = sessionmaker[Session]


@pytest.fixture
def sessions(locking_engine: Engine) -> Sessions:
    return sessionmaker(bind=locking_engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def executor(sessions: Sessions) -> AgentExecutor:
    return AgentExecutor(sessions)


@pytest.fixture
def ada(sessions: Sessions) -> int:
    return add(sessions, User(email="ada@example.com", name="Ada"))


@pytest.fixture
def transactions(locking_engine: Engine) -> Iterator[TransactionCounter]:
    """Counts open transactions, reading or writing, across all sessions."""
    counter = TransactionCounter()
    began, ended = counter.began, counter.ended
    event.listen(locking_engine, "begin", began)
    event.listen(locking_engine, "commit", ended)
    event.listen(locking_engine, "rollback", ended)
    yield counter
    event.remove(locking_engine, "begin", began)
    event.remove(locking_engine, "commit", ended)
    event.remove(locking_engine, "rollback", ended)


def run(executor: AgentExecutor, product: str = "Figma") -> int:
    """A whole run whose model reads, then tries to assign, then claims
    success."""
    return directed_run(
        executor,
        Script(call(ASSIGNMENTS), call(CAPACITY), call(ASSIGN), GOAL_REACHED),
        product=product,
    )


def test_this_database_detects_a_log_write_left_open_during_a_business_write(
    sessions: Sessions, ada: int
) -> None:
    # The mistake the executor must never make, shown to be detectable here.
    figma = add(sessions, Licence(product="Figma", seats_total=5))
    with sessions() as log, sessions() as business:
        log.add(
            AgentRun(
                instruction="i",
                requesting_actor=HUMAN,
                status=AgentRunStatus.RECEIVED,
                goal_type=GoalType.ENSURE_ASSIGNMENT,
                desired_state=DesiredState.ASSIGNED,
                extracted_user_email="ada@example.com",
                extracted_product="Figma",
            )
        )
        log.flush()  # a log write transaction is now open

        with pytest.raises(OperationalError, match="locked"):
            AssignmentService(business).assign_licence(
                user_id=ada, licence_id=figma, actor=HUMAN
            )


# --- every path ---------------------------------------------------------------


def completes(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5))
    run(e)


def already_satisfied(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5))
    run(e)
    run(e)


def blocked(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Zoom", seats_total=0))
    run(e, product="Zoom")


def policy_denies(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5, agent_policy=PolicyDecision.DENY))
    run(e)


def policy_holds(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(
        s,
        Licence(
            product="Figma",
            seats_total=5,
            agent_policy=PolicyDecision.REQUIRE_APPROVAL,
        ),
    )
    run(e)


def needs_clarification(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    run(e, product="Sketch")


def resolver_crashes(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5))

    def fail(users: UserService, licences: Any, user_email: str, product: str) -> None:
        users.find_user_by_email(user_email)  # fails after reading
        raise RuntimeError("resolver failure")

    m.setattr(executor_module, "resolve_assignment_goal", fail)
    run(e)


def verifier_crashes(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5))

    def fail(assignments: AssignmentService, goal: Any) -> None:
        assignments.get_active_assignment(  # fails after reading
            user_id=goal.user_id, licence_id=goal.licence_id
        )
        raise RuntimeError("verifier failure")

    m.setattr(executor_module, "verify", fail)
    run(e)


def verification_fails(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5))

    real_run = tools.run

    def lying_run(
        tool: TargetToolName, goal: ResolvedAssignmentGoal, **services: Any
    ) -> ToolOutput:
        if tool != ASSIGN:
            return real_run(tool, goal, **services)
        return AssignmentSnapshot(
            assignment_id=999,
            user_id=goal.user_id,
            licence_id=goal.licence_id,
            active=True,
            assigned_at=datetime.now(UTC),
            revoked_at=None,
        )

    m.setattr(tools, "run", lying_run)
    run(e)


def assignment_rolls_back(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5))
    real_add = AssignmentRepository.add

    def add_then_fail(self: AssignmentRepository, assignment: Assignment) -> Assignment:
        real_add(self, assignment)
        raise RuntimeError("after the flush")

    m.setattr(AssignmentRepository, "add", add_then_fail)
    run(e)


def assigned_elsewhere(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    figma = add(s, Licence(product="Figma", seats_total=5))

    def assigned_by_a_person_then_assign() -> ModelResponse:
        # Between the model's requests: no transaction of the run is open.
        with s.begin() as session:
            ada = session.scalars(
                select(User.id).where(User.email == "ada@example.com")
            ).one()
            AssignmentService(session).assign_licence(
                user_id=ada, licence_id=figma, actor=HUMAN
            )
        return call(ASSIGN)

    directed_run(
        e, Script(call(ASSIGNMENTS), assigned_by_a_person_then_assign, GOAL_REACHED)
    )


def crash_after_business_commit(
    e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch
) -> None:
    add(s, Licence(product="Figma", seats_total=5))
    run_id = resolved_run(e)

    def crash(*_: Any) -> None:
        raise Crash

    def assign(
        context: DecisionContext, call_tool: CallTool, calls: DecisionModelCallRecorder
    ) -> DecisionProposal:
        call_tool(ASSIGN)
        return GoalReached()

    m.setattr(ToolCallRepository, "get", crash)
    with pytest.raises(Crash):
        e.decide(run_id, FakeDecisionPlanner(assign))


SCENARIOS = {
    "completes": completes,
    "already-satisfied": already_satisfied,
    "blocked": blocked,
    "policy-denies": policy_denies,
    "policy-holds": policy_holds,
    "needs-clarification": needs_clarification,
    "resolver-crashes": resolver_crashes,
    "verifier-crashes": verifier_crashes,
    "verification-fails": verification_fails,
    "assignment-rolls-back": assignment_rolls_back,
    "assigned-elsewhere": assigned_elsewhere,
    "crash-after-business-commit": crash_after_business_commit,
}


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS)
def test_executor_never_has_two_transactions_open_at_once(
    executor: AgentExecutor,
    sessions: Sessions,
    transactions: TransactionCounter,
    ada: int,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Callable[[AgentExecutor, Sessions, pytest.MonkeyPatch], None],
) -> None:
    scenario(executor, sessions, monkeypatch)

    # Every session, reading or writing, was closed before the next opened,
    # on every path, including failures and a crash. A lock left open would
    # have made a later write fail here, on this engine.
    assert (transactions.peak, transactions.open_now) == (1, 0)
