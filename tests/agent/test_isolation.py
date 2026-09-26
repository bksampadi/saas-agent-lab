"""Transaction isolation against real SQLite locking.

The default test engine (StaticPool) gives every session the same connection,
so overlapping transactions there cannot lock; they silently share one
transaction instead. These tests use a shared-cache in-memory database with
a normal pool, where every session has its own connection and a second
writer fails at once with "database table is locked". Still in memory.
"""

import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

import app.agent.executor as executor_module
from app.agent import tools
from app.agent.executor import AgentExecutor
from app.agent.harness import run_ensure_assignment
from app.core.database import create_db_engine
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    AuditEvent,
    Base,
    DesiredState,
    GoalType,
    Licence,
    OutcomeReason,
    User,
)
from app.repositories.assignments import AssignmentRepository
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import (
    AssignLicenceInput,
    AssignmentSnapshot,
    ExtractedAssignmentIntent,
    GetLicenceInput,
    ListUserAssignmentsInput,
)
from app.services.assignments import AssignmentService
from app.services.users import UserService

HUMAN = "admin@example.com"
Sessions = sessionmaker[Session]


@pytest.fixture
def locking_engine() -> Iterator[Engine]:
    name = f"agentlab-{uuid4().hex}"
    # The database lives while at least one connection to it is open.
    keeper = sqlite3.connect(f"file:{name}?mode=memory&cache=shared", uri=True)
    engine = create_db_engine(
        f"sqlite:///file:{name}?mode=memory&cache=shared&uri=true",
        poolclass=QueuePool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    keeper.close()


@pytest.fixture
def sessions(locking_engine: Engine) -> Sessions:
    return sessionmaker(bind=locking_engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def executor(sessions: Sessions) -> AgentExecutor:
    return AgentExecutor(sessions)


def add(sessions: Sessions, row: User | Licence | Assignment) -> int:
    with sessions.begin() as session:
        session.add(row)
        session.flush()
        return row.id


def run(executor: AgentExecutor, product: str = "Figma") -> int:
    return run_ensure_assignment(
        executor,
        instruction="Give Ada a seat.",
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product=product),
    )


def get_run(sessions: Sessions, run_id: int) -> AgentRun:
    with sessions() as session:
        agent_run = session.get(AgentRun, run_id)
        assert agent_run is not None
        return agent_run


def rows(sessions: Sessions, model: type[Assignment] | type[AuditEvent]) -> int:
    with sessions() as session:
        return len(session.scalars(select(model)).all())


@pytest.fixture
def ada(sessions: Sessions) -> int:
    return add(sessions, User(email="ada@example.com", name="Ada"))


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


def test_successful_run_takes_no_conflicting_locks(
    executor: AgentExecutor, sessions: Sessions, ada: int
) -> None:
    add(sessions, Licence(product="Figma", seats_total=5))

    run_id = run(executor)

    agent_run = get_run(sessions, run_id)
    assert agent_run.outcome_reason is OutcomeReason.GOAL_SATISFIED
    assert (rows(sessions, Assignment), rows(sessions, AuditEvent)) == (1, 1)


def test_blocked_run_takes_no_conflicting_locks(
    executor: AgentExecutor, sessions: Sessions, ada: int
) -> None:
    add(sessions, Licence(product="Zoom", seats_total=0))

    run_id = run(executor, product="Zoom")

    assert get_run(sessions, run_id).status is AgentRunStatus.BLOCKED
    assert rows(sessions, Assignment) == 0


def test_rolled_back_run_takes_no_conflicting_locks_and_the_next_run_works(
    executor: AgentExecutor,
    sessions: Sessions,
    ada: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add(sessions, Licence(product="Figma", seats_total=5))
    real_add = AssignmentRepository.add

    def add_then_fail(self: AssignmentRepository, assignment: Assignment) -> Assignment:
        real_add(self, assignment)
        raise RuntimeError("after the flush")

    with monkeypatch.context() as patch:
        patch.setattr(AssignmentRepository, "add", add_then_fail)
        failed = run(executor)

    assert get_run(sessions, failed).status is AgentRunStatus.FAILED
    assert (rows(sessions, Assignment), rows(sessions, AuditEvent)) == (0, 0)

    succeeded = run(executor)

    assert get_run(sessions, succeeded).status is AgentRunStatus.COMPLETED
    assert (rows(sessions, Assignment), rows(sessions, AuditEvent)) == (1, 1)


def test_goal_scope_violation_takes_no_conflicting_locks(
    executor: AgentExecutor, sessions: Sessions, ada: int
) -> None:
    figma = add(sessions, Licence(product="Figma", seats_total=5))
    bob = add(sessions, User(email="bob@example.com", name="Bob"))
    run_id = executor.create_run(
        instruction="Give Ada a seat.",
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product="Figma"),
    )
    executor.resolve_run(run_id)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=bob, licence_id=figma)
    )

    assert outcome.run_status is AgentRunStatus.FAILED
    assert rows(sessions, Assignment) == 0


def test_stale_capacity_run_takes_no_conflicting_locks(
    executor: AgentExecutor, sessions: Sessions, ada: int
) -> None:
    zoom = add(sessions, Licence(product="Zoom", seats_total=1))
    bob = add(sessions, User(email="bob@example.com", name="Bob"))
    run_id = executor.create_run(
        instruction="Give Ada a seat.",
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product="Zoom"),
    )
    executor.resolve_run(run_id)
    executor.call_tool(run_id, GetLicenceInput(licence_id=zoom))
    with sessions.begin() as session:
        AssignmentService(session).assign_licence(
            user_id=bob, licence_id=zoom, actor=HUMAN
        )

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=ada, licence_id=zoom)
    )

    assert outcome.run_status is AgentRunStatus.BLOCKED


# --- never two transactions at once, on every path ----------------------------


class Crash(BaseException):
    """Stands in for the process dying: not an Exception, so nothing catches it."""


@dataclass
class TransactionCounter:
    open_now: int = 0
    peak: int = 0

    def began(self, *_: Any) -> None:
        self.open_now += 1
        self.peak = max(self.peak, self.open_now)

    def ended(self, *_: Any) -> None:
        self.open_now -= 1


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


def resolved(executor: AgentExecutor, product: str = "Figma") -> int:
    run_id = executor.create_run(
        instruction="Give Ada a seat.",
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product=product),
    )
    executor.resolve_run(run_id)
    return run_id


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


def needs_clarification(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    run(e, product="Sketch")


def resolver_crashes(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    add(s, Licence(product="Figma", seats_total=5))

    def fail(users: UserService, licences: Any, intent: Any) -> None:
        users.find_user_by_email(intent.user_email)  # fails after reading
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

    def lying_assign(context: Any, args: AssignLicenceInput) -> AssignmentSnapshot:
        return AssignmentSnapshot(
            assignment_id=999,
            user_id=args.user_id,
            licence_id=args.licence_id,
            active=True,
            assigned_at=datetime.now(UTC),
            revoked_at=None,
        )

    m.setattr(tools, "assign_licence", lying_assign)
    run(e)


def scope_violation(e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch) -> None:
    figma = add(s, Licence(product="Figma", seats_total=5))
    bob = add(s, User(email="bob@example.com", name="Bob"))
    e.call_tool(resolved(e), AssignLicenceInput(user_id=bob, licence_id=figma))


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
    run_id = resolved(e)
    goal = e.get_goal(run_id)
    e.call_tool(run_id, ListUserAssignmentsInput(user_id=goal.user_id))
    with s.begin() as session:
        AssignmentService(session).assign_licence(
            user_id=goal.user_id, licence_id=figma, actor=HUMAN
        )
    e.call_tool(run_id, AssignLicenceInput(user_id=goal.user_id, licence_id=figma))
    e.verify_and_finish(run_id)


def crash_after_business_commit(
    e: AgentExecutor, s: Sessions, m: pytest.MonkeyPatch
) -> None:
    figma = add(s, Licence(product="Figma", seats_total=5))
    run_id = resolved(e)
    goal = e.get_goal(run_id)

    def crash(*_: Any) -> None:
        raise Crash

    m.setattr(ToolCallRepository, "get", crash)
    with pytest.raises(Crash):
        e.call_tool(run_id, AssignLicenceInput(user_id=goal.user_id, licence_id=figma))


SCENARIOS = {
    "completes": completes,
    "already-satisfied": already_satisfied,
    "blocked": blocked,
    "needs-clarification": needs_clarification,
    "resolver-crashes": resolver_crashes,
    "verifier-crashes": verifier_crashes,
    "verification-fails": verification_fails,
    "scope-violation": scope_violation,
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
    # on every path, including failures and a crash.
    assert (transactions.peak, transactions.open_now) == (1, 0)
