"""The executor's steps on the one real path (receive, extract, resolve,
decide): resolution, the business transaction around each tool call, what
verification rests on, and the state each crash window leaves.

Decisions are made by a scripted model behind the production decision
planner, or by a fake planner where only the executor's side matters.
"""

import contextlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse
from sqlalchemy import event, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import app.agent.executor as executor_module
from app.agent import tools
from app.agent.executor import AgentExecutor, RunNotExecutable, UnfinishedToolCall
from app.agent.planner import DecisionModelCallRecorder, TargetTools
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    AuditEvent,
    Licence,
    OutcomeReason,
    ToolCall,
    ToolCallStatus,
    User,
    UserStatus,
)
from app.repositories.assignments import AssignmentRepository
from app.repositories.audit_events import AuditEventRepository
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import (
    AssignLicenceInput,
    AssignmentSnapshot,
    DecisionContext,
    DecisionProposal,
    GoalReached,
    LicenceSnapshot,
    ToolError,
    UserAssignmentsSnapshot,
    UserSnapshot,
)
from app.services.assignments import AssignmentService
from support import (
    ASSIGN,
    ASSIGNMENTS,
    CAPACITY,
    GOAL_REACHED,
    HUMAN,
    NO_ACTION_NEEDED,
    USER,
    FakeDecisionPlanner,
    Script,
    Step,
    call,
    cannot_proceed,
    extracted_run,
    resolved_run,
)

S = AgentRunStatus
R = OutcomeReason
Sessions = sessionmaker[Session]


# --- helpers ------------------------------------------------------------------


def add(sessions: Sessions, row: User | Licence | Assignment) -> int:
    with sessions.begin() as session:
        session.add(row)
        session.flush()
        return row.id


def make_user(
    sessions: Sessions, email: str, status: UserStatus = UserStatus.ACTIVE
) -> int:
    return add(sessions, User(email=email, name="Someone", status=status))


def make_licence(sessions: Sessions, product: str, seats: int = 5) -> int:
    return add(sessions, Licence(product=product, seats_total=seats))


def assign_as_human(sessions: Sessions, user_id: int, licence_id: int) -> int:
    """Another, separately committed business transaction."""
    with sessions.begin() as session:
        return (
            AssignmentService(session)
            .assign_licence(user_id=user_id, licence_id=licence_id, actor=HUMAN)
            .id
        )


def get_run(sessions: Sessions, run_id: int) -> AgentRun:
    with sessions() as session:
        run = session.get(AgentRun, run_id)
        assert run is not None
        return run


def outcome(sessions: Sessions, run_id: int) -> tuple[AgentRunStatus, R | None]:
    run = get_run(sessions, run_id)
    return run.status, run.outcome_reason


def tool_calls(sessions: Sessions, run_id: int) -> list[ToolCall]:
    with sessions() as session:
        return list(
            session.scalars(
                select(ToolCall)
                .where(ToolCall.agent_run_id == run_id)
                .order_by(ToolCall.sequence_no)
            )
        )


def assignments(sessions: Sessions) -> list[Assignment]:
    with sessions() as session:
        return list(session.scalars(select(Assignment).order_by(Assignment.id)))


def audit_events(sessions: Sessions) -> list[AuditEvent]:
    with sessions() as session:
        return list(session.scalars(select(AuditEvent).order_by(AuditEvent.id)))


def decided(
    executor: AgentExecutor,
    *steps: Step,
    user_email: str = "ada@example.com",
    product: str = "Figma",
) -> int:
    """A resolved run, decided by a model that answers with ``steps``."""
    run_id = resolved_run(executor, user_email, product)
    executor.decide(run_id, Script(*steps).planner())
    return run_id


def assign_then_conclude(
    context: DecisionContext, target: TargetTools, calls: DecisionModelCallRecorder
) -> DecisionProposal:
    target.assign_target_licence()
    return GoalReached()


class Crash(BaseException):
    """Stands in for the process dying: not an Exception, so nothing catches it."""


def crash(*_: Any) -> None:
    raise Crash


@dataclass(frozen=True)
class Seed:
    user_id: int  # ada@example.com
    licence_id: int  # Figma, 5 seats
    other_user_id: int  # bob@example.com
    other_licence_id: int  # Slack, 5 seats


@pytest.fixture
def seed(session_factory: Sessions) -> Seed:
    return Seed(
        user_id=make_user(session_factory, "ada@example.com"),
        licence_id=make_licence(session_factory, "Figma"),
        other_user_id=make_user(session_factory, "bob@example.com"),
        other_licence_id=make_licence(session_factory, "Slack"),
    )


# --- resolution ---------------------------------------------------------------


def test_resolve_run_persists_the_resolved_goal(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = extracted_run(executor, "ADA@example.com", "figma")

    assert executor.resolve_run(run_id) is S.RESOLVED

    run = get_run(session_factory, run_id)
    assert (run.resolved_user_id, run.resolved_licence_id) == (
        seed.user_id,
        seed.licence_id,
    )
    # The text stays as extracted; only the ids are added.
    assert (run.extracted_user_email, run.extracted_product) == (
        "ADA@example.com",
        "figma",
    )
    assert run.completed_at is None
    assert tool_calls(session_factory, run_id) == []


@pytest.mark.parametrize(
    ("user_email", "product", "reason"),
    [
        ("nobody@example.com", "Figma", R.USER_NOT_FOUND),
        ("ada@example.com", "Unknown", R.LICENCE_NOT_FOUND),
        ("ada@example.com", "figma", R.LICENCE_AMBIGUOUS),
        ("not-an-email", "Figma", R.INVALID_INPUT),
    ],
)
def test_an_unresolved_run_needs_clarification_and_is_never_decided(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    user_email: str,
    product: str,
    reason: OutcomeReason,
) -> None:
    make_licence(session_factory, "FIGMA")  # makes "figma" ambiguous
    run_id = extracted_run(executor, user_email, product)

    assert executor.resolve_run(run_id) is S.NEEDS_CLARIFICATION

    run = get_run(session_factory, run_id)
    assert run.outcome_reason is reason
    assert run.outcome_detail
    assert run.completed_at is not None
    assert (run.resolved_user_id, run.resolved_licence_id) == (None, None)
    planner = FakeDecisionPlanner(assign_then_conclude)
    with pytest.raises(RunNotExecutable):
        executor.decide(run_id, planner)
    assert planner.contexts == []
    assert tool_calls(session_factory, run_id) == []


def test_resolution_reads_the_persisted_extracted_text(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = extracted_run(executor, product="Figma")
    with session_factory.begin() as session:
        session.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(extracted_product="Slack")
        )

    executor.resolve_run(run_id)

    assert get_run(session_factory, run_id).resolved_licence_id == (
        seed.other_licence_id
    )


def test_a_run_is_resolved_only_once(executor: AgentExecutor, seed: Seed) -> None:
    run_id = resolved_run(executor)

    with pytest.raises(RunNotExecutable):
        executor.resolve_run(run_id)


def test_a_resolver_crash_fails_the_run_without_keeping_its_message(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any) -> None:
        raise RuntimeError("secret-token in the message")

    monkeypatch.setattr(executor_module, "resolve_assignment_goal", fail)
    run_id = extracted_run(executor)

    assert executor.resolve_run(run_id) is S.FAILED

    run = get_run(session_factory, run_id)
    assert run.outcome_reason is R.UNEXPECTED_ERROR
    assert run.outcome_detail == {"stage": "resolution", "error_type": "RuntimeError"}


# --- the business transaction -------------------------------------------------


def test_an_inactive_users_rejected_assignment_blocks_the_run(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    make_user(session_factory, "carol@example.com", UserStatus.INACTIVE)

    run_id = decided(
        executor, call(ASSIGN), GOAL_REACHED, user_email="carol@example.com"
    )

    assert outcome(session_factory, run_id) == (S.BLOCKED, R.USER_INACTIVE)
    (attempt,) = tool_calls(session_factory, run_id)
    assert attempt.error is not None and attempt.error["code"] == "user_inactive"
    assert attempt.observation == '{"outcome":"rejected","reason_code":"user_inactive"}'
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])


def test_a_duplicate_insert_race_rolls_back_and_is_left_to_the_verifier(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assign_as_human(session_factory, seed.user_id, seed.licence_id)
    real_get_active = AssignmentRepository.get_active
    missed: list[int] = []

    def miss_once(
        self: AssignmentRepository, user_id: int, licence_id: int
    ) -> Assignment | None:
        # The service's pre-check misses the committed row, as in a race, so
        # the INSERT reaches the real partial unique index. The verifier,
        # later, reads normally.
        if not missed:
            missed.append(1)
            return None
        return real_get_active(self, user_id, licence_id)

    monkeypatch.setattr(AssignmentRepository, "get_active", miss_once)

    run_id = decided(executor, call(ASSIGN), GOAL_REACHED)

    assert missed == [1]
    (attempt,) = tool_calls(session_factory, run_id)
    assert attempt.error is not None
    assert attempt.error["code"] == "assignment_already_exists"
    assert attempt.observation == (
        '{"outcome":"rejected","reason_code":"already_assigned"}'
    )
    # The goal holds, so the run is complete; this run changed nothing.
    assert outcome(session_factory, run_id) == (S.COMPLETED, R.ALREADY_SATISFIED)
    assert len(assignments(session_factory)) == 1
    assert [e.actor for e in audit_events(session_factory)] == [HUMAN]


def test_an_error_after_the_assignment_flush_leaves_nothing_partial(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_add = AssignmentRepository.add
    flushed: list[int] = []

    def add_then_fail(self: AssignmentRepository, assignment: Assignment) -> Assignment:
        real_add(self, assignment)  # the INSERT is sent, not committed
        flushed.append(len(self._session.scalars(select(Assignment)).all()))
        raise RuntimeError("boom: secret-token")

    with monkeypatch.context() as patch:
        patch.setattr(AssignmentRepository, "add", add_then_fail)
        run_id = decided(executor, call(ASSIGN), GOAL_REACHED)

    assert flushed == [1]
    (failed,) = tool_calls(session_factory, run_id)
    assert (failed.status, failed.error, failed.observation) == (
        ToolCallStatus.FAILED,
        {
            "code": "unexpected_error",
            "message": "Unexpected error while running assign_licence.",
            "error_type": "RuntimeError",
        },
        None,
    )
    assert outcome(session_factory, run_id) == (S.FAILED, R.TOOL_FAILED)
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])

    # The next run works normally, and the failed one keeps its history.
    retry = decided(executor, call(ASSIGN), GOAL_REACHED)
    assert outcome(session_factory, retry) == (S.COMPLETED, R.GOAL_SATISFIED)
    assert [e.actor for e in audit_events(session_factory)] == [f"agent:run-{retry}"]
    assert len(tool_calls(session_factory, run_id)) == 1


def test_an_audit_failure_rolls_back_the_assignment_as_well(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_add = AuditEventRepository.add
    flushed: list[tuple[int, int]] = []

    def add_then_fail(self: AuditEventRepository, event_: AuditEvent) -> AuditEvent:
        real_add(self, event_)
        flushed.append(
            (
                len(self._session.scalars(select(Assignment)).all()),
                len(self._session.scalars(select(AuditEvent)).all()),
            )
        )
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(AuditEventRepository, "add", add_then_fail)

    run_id = decided(executor, call(ASSIGN), GOAL_REACHED)

    assert flushed == [(1, 1)]
    (failed,) = tool_calls(session_factory, run_id)
    assert failed.error is not None and failed.error["code"] == "unexpected_error"
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])
    assert outcome(session_factory, run_id) == (S.FAILED, R.TOOL_FAILED)


def test_a_failed_commit_after_the_tool_returned_is_a_failed_call(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)

    def fail_business_commit(session: Session) -> None:
        # Only the business session can see an (uncommitted) assignment here.
        if session.scalar(select(func.count()).select_from(Assignment)):
            raise RuntimeError("commit failed")

    event.listen(Session, "before_commit", fail_business_commit)
    try:
        executor.decide(run_id, Script(call(ASSIGN), GOAL_REACHED).planner())
    finally:
        event.remove(Session, "before_commit", fail_business_commit)

    # What the tool returned was never committed, so it is not the result.
    (failed,) = tool_calls(session_factory, run_id)
    assert (failed.status, failed.result) == (ToolCallStatus.FAILED, None)
    assert failed.error is not None and failed.error["code"] == "unexpected_error"
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])
    assert outcome(session_factory, run_id) == (S.FAILED, R.TOOL_FAILED)


def test_database_error_text_is_never_persisted(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(self: AuditEventRepository, event_: AuditEvent) -> AuditEvent:
        raise IntegrityError(
            "INSERT INTO audit_events (actor) VALUES ('secret-sql')",
            {"actor": "secret-param"},
            Exception("secret-orig"),
        )

    monkeypatch.setattr(AuditEventRepository, "add", fail)

    run_id = decided(executor, call(ASSIGN), GOAL_REACHED)

    (failed,) = tool_calls(session_factory, run_id)
    assert ToolError.model_validate(failed.error) == ToolError(
        code="unexpected_error",
        message="Unexpected error while running assign_licence.",
        error_type="IntegrityError",
    )
    with session_factory() as session:
        persisted = session.execute(
            text(
                "SELECT arguments, result, error, observation FROM tool_calls "
                "UNION ALL SELECT outcome_detail, decision_context, NULL, NULL "
                "FROM agent_runs UNION ALL SELECT output, error, NULL, NULL "
                "FROM model_calls"
            )
        ).all()
    assert "secret" not in repr(persisted)


def test_arguments_and_results_are_stored_as_json(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = decided(
        executor, call(USER, CAPACITY, ASSIGNMENTS), call(ASSIGN), GOAL_REACHED
    )

    with session_factory() as session:
        raw = session.execute(
            text(
                "SELECT arguments, result IS NULL, error IS NULL "
                "FROM tool_calls WHERE agent_run_id = :run ORDER BY sequence_no"
            ),
            {"run": run_id},
        ).all()
    # Stored as JSON text; an absent error is SQL NULL, not 'null'.
    assert [json.loads(row[0]) for row in raw] == [
        {"user_id": seed.user_id},
        {"licence_id": seed.licence_id},
        {"user_id": seed.user_id},
        {"user_id": seed.user_id, "licence_id": seed.licence_id},
    ]
    assert [(row[1], row[2]) for row in raw] == [(0, 1)] * 4

    calls = tool_calls(session_factory, run_id)
    UserSnapshot.model_validate(calls[0].result)
    LicenceSnapshot.model_validate(calls[1].result)
    UserAssignmentsSnapshot.model_validate(calls[2].result)
    assigned = AssignmentSnapshot.model_validate(calls[3].result)
    assert assigned.assigned_at.tzinfo is not None


# --- verification decides -----------------------------------------------------


def test_a_tool_claiming_success_without_writing_cannot_complete_the_run(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lying_assign(
        context: tools.ToolContext, args: AssignLicenceInput
    ) -> AssignmentSnapshot:
        return AssignmentSnapshot(
            assignment_id=999,
            user_id=args.user_id,
            licence_id=args.licence_id,
            active=True,
            assigned_at=datetime.now(UTC),
            revoked_at=None,
        )

    monkeypatch.setattr(tools, "assign_licence", lying_assign)

    run_id = decided(executor, call(ASSIGN), GOAL_REACHED)

    assert outcome(session_factory, run_id) == (S.FAILED, R.VERIFICATION_FAILED)
    # The trace records what the tool claimed; the verifier found otherwise.
    (claimed,) = tool_calls(session_factory, run_id)
    assert claimed.status is ToolCallStatus.SUCCEEDED
    assert claimed.observation == '{"outcome":"assigned","reason_code":null}'
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])


def test_a_stale_favourable_observation_does_not_complete_the_run(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    assignment_id = assign_as_human(session_factory, seed.user_id, seed.licence_id)

    def revoked_then_concluded() -> ModelResponse:
        # After the model saw the seat held, before it concludes.
        with session_factory.begin() as session:
            AssignmentService(session).revoke_assignment(
                assignment_id=assignment_id, actor=HUMAN
            )
        return NO_ACTION_NEEDED

    run_id = decided(executor, call(ASSIGNMENTS), revoked_then_concluded)

    assert outcome(session_factory, run_id) == (S.FAILED, R.VERIFICATION_FAILED)
    run = get_run(session_factory, run_id)
    assert run.outcome_detail is not None
    assert run.outcome_detail["satisfied"] is False
    assert run.outcome_detail["evidence"]["active_assignment"] is None
    # The earlier observation stays in the trace as it was.
    (listing,) = tool_calls(session_factory, run_id)
    assert listing.observation == '{"holds_active_seat":true}'
    assert listing.result is not None
    assert listing.result["assignments"][0]["active"] is True


def test_a_verifier_crash_fails_the_run_without_keeping_its_message(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any) -> None:
        raise RuntimeError("secret-token")

    monkeypatch.setattr(executor_module, "verify", fail)

    run_id = decided(executor, call(ASSIGN), GOAL_REACHED)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.FAILED, R.UNEXPECTED_ERROR)
    assert run.outcome_detail == {"stage": "verification", "error_type": "RuntimeError"}


def test_a_favourable_capacity_observation_is_not_a_reservation(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    last_seat = make_licence(session_factory, "Zoom", seats=1)
    taken: list[int] = []

    def seat_taken_then_assign() -> ModelResponse:
        # After the model saw one free seat: another valid transaction takes it.
        taken.append(assign_as_human(session_factory, seed.other_user_id, last_seat))
        return call(ASSIGN)

    run_id = decided(
        executor,
        call(CAPACITY),
        seat_taken_then_assign,
        cannot_proceed("no_seats_available"),
        product="Zoom",
    )

    # The trace shows the favourable observation, then the rejection after it.
    observed, attempt = tool_calls(session_factory, run_id)
    assert observed.observation == (
        '{"seats_active":0,"seats_available":1,"seats_total":1}'
    )
    assert attempt.error is not None
    assert attempt.error["code"] == "no_seats_available"
    # Blocked by the service's check at mutation time; nothing invalid exists.
    assert outcome(session_factory, run_id) == (S.BLOCKED, R.NO_SEATS_AVAILABLE)
    run = get_run(session_factory, run_id)
    assert run.outcome_detail is not None
    assert run.outcome_detail["rejection"]["tool_call_id"] == attempt.id
    assert [(a.id, a.user_id) for a in assignments(session_factory)] == [
        (taken[0], seed.other_user_id)
    ]
    assert [(e.actor, e.entity_id) for e in audit_events(session_factory)] == [
        (HUMAN, taken[0])
    ]


@pytest.mark.parametrize("change", ["text-edited", "text-now-ambiguous"])
def test_execution_and_verification_use_the_persisted_ids_not_the_text(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed, change: str
) -> None:
    run_id = resolved_run(executor)  # "Figma" -> seed.licence_id
    if change == "text-edited":
        with session_factory.begin() as session:
            session.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(extracted_product="Slack")
            )
    else:
        make_licence(session_factory, "FIGMA")  # re-resolving would be ambiguous

    executor.decide(
        run_id, Script(call(ASSIGNMENTS), call(ASSIGN), GOAL_REACHED).planner()
    )

    assert outcome(session_factory, run_id) == (S.COMPLETED, R.GOAL_SATISFIED)
    run = get_run(session_factory, run_id)
    assert run.outcome_detail is not None
    assert run.outcome_detail["evidence"]["licence_id"] == seed.licence_id
    assert [(a.user_id, a.licence_id) for a in assignments(session_factory)] == [
        (seed.user_id, seed.licence_id)
    ]


# --- crash windows: the state each leaves (documented, not recovered) ---------


def test_a_crash_after_a_call_starts_leaves_it_started_and_changes_nothing(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    with monkeypatch.context() as patch:
        patch.setattr(tools, "run_tool", crash)
        with pytest.raises(Crash):
            executor.decide(run_id, FakeDecisionPlanner(assign_then_conclude))

    assert get_run(session_factory, run_id).status is S.EXECUTING
    (started,) = tool_calls(session_factory, run_id)
    assert (started.status, started.completed_at, started.result, started.error) == (
        ToolCallStatus.STARTED,
        None,
        None,
        None,
    )
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])
    # Nothing decides the run again.
    with pytest.raises(RunNotExecutable):
        executor.decide(run_id, FakeDecisionPlanner(assign_then_conclude))
    assert len(tool_calls(session_factory, run_id)) == 1


def test_a_planner_that_carries_on_after_a_crashed_call_cannot_call_or_conclude(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    refused: list[str] = []

    def carry_on(
        context: DecisionContext, target: TargetTools, calls: DecisionModelCallRecorder
    ) -> DecisionProposal:
        with monkeypatch.context() as patch:
            patch.setattr(tools, "run_tool", crash)
            with contextlib.suppress(Crash):  # the planner swallows the crash
                target.assign_target_licence()
        with pytest.raises(UnfinishedToolCall):
            target.get_target_user()
        refused.append("tool call")
        return GoalReached()

    # Whether the crashed call made its change is unknown, so the run is
    # neither carried on nor verified and labelled.
    with pytest.raises(UnfinishedToolCall):
        executor.decide(run_id, FakeDecisionPlanner(carry_on))

    assert refused == ["tool call"]
    run = get_run(session_factory, run_id)
    assert (run.status, run.decision_proposal, run.outcome_reason) == (
        S.EXECUTING,
        None,
        None,
    )
    (started,) = tool_calls(session_factory, run_id)
    assert started.status is ToolCallStatus.STARTED


def test_a_crash_after_the_business_commit_is_attributable_to_the_run(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    with monkeypatch.context() as patch:
        # The last LOG transaction is the first to look a tool call up by id.
        patch.setattr(ToolCallRepository, "get", crash)
        with pytest.raises(Crash):
            executor.decide(run_id, FakeDecisionPlanner(assign_then_conclude))

    assert get_run(session_factory, run_id).status is S.EXECUTING
    (started,) = tool_calls(session_factory, run_id)
    assert started.status is ToolCallStatus.STARTED
    # The change committed, and its audit event names the run, so the stale
    # STARTED call can be reconciled rather than guessed at.
    (assignment,) = assignments(session_factory)
    (audit,) = audit_events(session_factory)
    assert audit.actor == f"agent:run-{run_id}"
    assert (audit.action, audit.entity_id) == ("assignment.create", assignment.id)
    assert audit.after == started.arguments
    # No blind retry.
    with pytest.raises(RunNotExecutable):
        executor.decide(run_id, FakeDecisionPlanner(assign_then_conclude))
    assert len(tool_calls(session_factory, run_id)) == 1
    assert len(assignments(session_factory)) == 1
