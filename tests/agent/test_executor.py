"""The executor, driven step by step: run lifecycle, goal-scoped tool calls,
transaction boundaries, the ordered ToolCall trace and run outcomes."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import event, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import app.agent.executor as executor_module
from app.agent import tools
from app.agent.executor import AgentExecutor, RunNotExecutable, UnfinishedToolCall
from app.agent.status import IllegalTransition, transition
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    AuditEvent,
    DesiredState,
    GoalType,
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
    ExtractedAssignmentIntent,
    GetLicenceInput,
    GetUserInput,
    LicenceSnapshot,
    ListUserAssignmentsInput,
    ResolvedAssignmentGoal,
    ToolError,
    ToolInput,
    UserAssignmentsSnapshot,
    UserSnapshot,
    VerificationEvidence,
    VerificationResult,
)
from app.services.assignments import AssignmentService
from app.services.errors import InvalidInput

HUMAN = "admin@example.com"
INSTRUCTION = "Give Ada a Figma seat."
BUSINESS_TABLES = ("users", "licences", "assignments", "audit_events")

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


def revoke_as_human(sessions: Sessions, assignment_id: int) -> None:
    with sessions.begin() as session:
        AssignmentService(session).revoke_assignment(
            assignment_id=assignment_id, actor=HUMAN
        )


def get_run(sessions: Sessions, run_id: int) -> AgentRun:
    with sessions() as session:
        run = session.get(AgentRun, run_id)
        assert run is not None
        return run


def all_runs(sessions: Sessions) -> list[AgentRun]:
    with sessions() as session:
        return list(session.scalars(select(AgentRun)))


def tool_calls(sessions: Sessions, run_id: int) -> list[ToolCall]:
    with sessions() as session:
        return ToolCallRepository(session).list_for_run(run_id)


def all_tool_calls(sessions: Sessions) -> list[ToolCall]:
    with sessions() as session:
        return list(session.scalars(select(ToolCall)))


def assignments(sessions: Sessions) -> list[Assignment]:
    with sessions() as session:
        return list(session.scalars(select(Assignment).order_by(Assignment.id)))


def audit_events(sessions: Sessions) -> list[AuditEvent]:
    with sessions() as session:
        return list(session.scalars(select(AuditEvent).order_by(AuditEvent.id)))


def intent(
    user_email: str = "ada@example.com", product: str = "Figma"
) -> ExtractedAssignmentIntent:
    return ExtractedAssignmentIntent(user_email=user_email, product=product)


def new_run(
    executor: AgentExecutor, user_email: str = "ada@example.com", product: str = "Figma"
) -> int:
    return executor.create_run(
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
        intent=intent(user_email, product),
    )


def resolved_run(
    executor: AgentExecutor, user_email: str = "ada@example.com", product: str = "Figma"
) -> int:
    run_id = new_run(executor, user_email, product)
    assert executor.resolve_run(run_id) is AgentRunStatus.RESOLVED
    return run_id


def touching_business_tables(sent: list[str]) -> list[str]:
    return [s for s in sent if any(f" {table}" in s for table in BUSINESS_TABLES)]


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


# --- create_run ---------------------------------------------------------------


def test_create_run_persists_a_received_run_exactly_as_given(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = executor.create_run(
        instruction="  give ADA a figma seat ",
        requesting_actor="  admin@example.com ",
        intent=intent("  ADA@Example.com ", "figma "),
    )

    run = get_run(session_factory, run_id)
    assert run.status is AgentRunStatus.RECEIVED
    assert run.instruction == "  give ADA a figma seat "
    assert run.requesting_actor == HUMAN
    assert run.goal_type is GoalType.ENSURE_ASSIGNMENT
    assert run.desired_state is DesiredState.ASSIGNED
    assert (run.extracted_user_email, run.extracted_product) == (
        "  ADA@Example.com ",
        "figma ",
    )
    assert (run.resolved_user_id, run.resolved_licence_id) == (None, None)
    assert (run.outcome_reason, run.outcome_detail, run.completed_at) == (
        None,
        None,
        None,
    )
    assert run.created_at.tzinfo is UTC
    assert run.updated_at.tzinfo is UTC


@pytest.mark.parametrize(
    "actor", ["", "   ", "a" * 321, "agent:run-1", " Agent:run-1", "AGENT:x"]
)
def test_create_run_rejects_invalid_or_reserved_requesting_actor(
    executor: AgentExecutor, session_factory: Sessions, actor: str
) -> None:
    with pytest.raises(InvalidInput):
        executor.create_run(
            instruction=INSTRUCTION, requesting_actor=actor, intent=intent()
        )

    assert all_runs(session_factory) == []


@pytest.mark.parametrize("instruction", ["", "   ", "x" * 2001])
def test_create_run_rejects_blank_or_over_long_instruction(
    executor: AgentExecutor, session_factory: Sessions, instruction: str
) -> None:
    with pytest.raises(InvalidInput):
        executor.create_run(
            instruction=instruction, requesting_actor=HUMAN, intent=intent()
        )

    assert all_runs(session_factory) == []


# --- resolve_run --------------------------------------------------------------


def test_resolve_run_persists_the_resolved_goal(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = new_run(executor, "ADA@example.com", "figma")

    assert executor.resolve_run(run_id) is AgentRunStatus.RESOLVED

    run = get_run(session_factory, run_id)
    assert (run.resolved_user_id, run.resolved_licence_id) == (
        seed.user_id,
        seed.licence_id,
    )
    assert run.completed_at is None
    assert tool_calls(session_factory, run_id) == []
    assert executor.get_goal(run_id) == ResolvedAssignmentGoal(
        goal_type=GoalType.ENSURE_ASSIGNMENT,
        desired_state=DesiredState.ASSIGNED,
        user_id=seed.user_id,
        licence_id=seed.licence_id,
        extracted_user_email="ADA@example.com",
        extracted_product="figma",
    )


@pytest.mark.parametrize(
    ("user_email", "product", "reason"),
    [
        ("nobody@example.com", "Figma", OutcomeReason.USER_NOT_FOUND),
        ("ada@example.com", "Unknown", OutcomeReason.LICENCE_NOT_FOUND),
        ("ada@example.com", "figma", OutcomeReason.LICENCE_AMBIGUOUS),
        ("not-an-email", "Figma", OutcomeReason.INVALID_INPUT),
    ],
)
def test_unresolved_run_needs_clarification_and_can_call_no_tools(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    user_email: str,
    product: str,
    reason: OutcomeReason,
) -> None:
    make_licence(session_factory, "FIGMA")  # makes "figma" ambiguous
    run_id = new_run(executor, user_email, product)

    assert executor.resolve_run(run_id) is AgentRunStatus.NEEDS_CLARIFICATION

    run = get_run(session_factory, run_id)
    assert run.outcome_reason is reason
    assert run.outcome_detail
    assert run.completed_at is not None
    assert (run.resolved_user_id, run.resolved_licence_id) == (None, None)
    with pytest.raises(RunNotExecutable):
        executor.call_tool(run_id, GetUserInput(user_id=seed.user_id))
    assert tool_calls(session_factory, run_id) == []


def test_resolution_reads_the_persisted_extracted_text(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = new_run(executor, product="Figma")
    with session_factory.begin() as session:
        session.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(extracted_product="Slack")
        )

    executor.resolve_run(run_id)

    assert executor.get_goal(run_id).licence_id == seed.other_licence_id


def test_a_run_is_resolved_only_once(executor: AgentExecutor, seed: Seed) -> None:
    run_id = resolved_run(executor)

    with pytest.raises(RunNotExecutable):
        executor.resolve_run(run_id)


def test_resolver_crash_fails_the_run_without_keeping_its_message(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crash(*args: Any) -> None:
        raise RuntimeError("secret-token in the message")

    monkeypatch.setattr(executor_module, "resolve_assignment_goal", crash)
    run_id = new_run(executor)

    assert executor.resolve_run(run_id) is AgentRunStatus.FAILED

    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.UNEXPECTED_ERROR
    assert run.outcome_detail == {"stage": "resolution", "error_type": "RuntimeError"}


# --- when tool calls are allowed ----------------------------------------------


def test_tool_call_on_an_unresolved_run_is_rejected_and_not_recorded(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = new_run(executor)

    with pytest.raises(RunNotExecutable):
        executor.call_tool(run_id, GetUserInput(user_id=seed.user_id))

    assert tool_calls(session_factory, run_id) == []
    assert get_run(session_factory, run_id).status is AgentRunStatus.RECEIVED


def test_tool_call_on_a_finished_run_is_rejected_and_not_recorded(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)
    executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )
    assert executor.verify_and_finish(run_id) is AgentRunStatus.COMPLETED

    with pytest.raises(RunNotExecutable):
        executor.call_tool(run_id, GetUserInput(user_id=seed.user_id))

    assert len(tool_calls(session_factory, run_id)) == 1


# --- goal scope ---------------------------------------------------------------

OUT_OF_SCOPE_CALLS = {
    "assign-other-user": lambda s: AssignLicenceInput(
        user_id=s.other_user_id, licence_id=s.licence_id
    ),
    "assign-other-licence": lambda s: AssignLicenceInput(
        user_id=s.user_id, licence_id=s.other_licence_id
    ),
    "assign-other-both": lambda s: AssignLicenceInput(
        user_id=s.other_user_id, licence_id=s.other_licence_id
    ),
    "get-other-user": lambda s: GetUserInput(user_id=s.other_user_id),
    "get-other-licence": lambda s: GetLicenceInput(licence_id=s.other_licence_id),
    "list-other-user": lambda s: ListUserAssignmentsInput(user_id=s.other_user_id),
}


@pytest.mark.parametrize(
    "make_call", OUT_OF_SCOPE_CALLS.values(), ids=OUT_OF_SCOPE_CALLS
)
def test_call_outside_the_resolved_goal_fails_before_any_business_transaction(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    statements: list[str],
    monkeypatch: pytest.MonkeyPatch,
    make_call: Any,
) -> None:
    run_id = resolved_run(executor)
    args: ToolInput = make_call(seed)  # names rows that exist, just not the goal's

    def forbidden(*_: Any) -> None:
        raise AssertionError("no tool may run for an out-of-scope call")

    monkeypatch.setattr(tools, "run_tool", forbidden)
    statements.clear()

    outcome = executor.call_tool(run_id, args)

    assert touching_business_tables(statements) == []
    assert outcome.error is not None
    assert outcome.error.code == "goal_scope_violation"
    assert outcome.run_status is AgentRunStatus.FAILED
    (call,) = tool_calls(session_factory, run_id)
    assert call.status is ToolCallStatus.FAILED
    assert call.sequence_no == 1
    assert call.tool_name == args.tool_name
    assert call.arguments == args.model_dump(mode="json")
    assert call.result is None
    assert call.error == {
        "code": "goal_scope_violation",
        "message": outcome.error.message,
        "error_type": None,
    }
    run = get_run(session_factory, run_id)
    assert run.status is AgentRunStatus.FAILED
    assert run.outcome_reason is OutcomeReason.GOAL_SCOPE_VIOLATION
    assert run.outcome_detail is not None
    assert run.outcome_detail["tool_call_id"] == call.id
    assert assignments(session_factory) == []
    assert audit_events(session_factory) == []


def test_scope_violation_after_earlier_calls_keeps_the_trace_and_ends_the_run(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)
    executor.call_tool(run_id, ListUserAssignmentsInput(user_id=seed.user_id))

    executor.call_tool(
        run_id,
        AssignLicenceInput(user_id=seed.other_user_id, licence_id=seed.licence_id),
    )

    calls = tool_calls(session_factory, run_id)
    assert [(c.sequence_no, c.tool_name, c.status) for c in calls] == [
        (1, "list_user_assignments", ToolCallStatus.SUCCEEDED),
        (2, "assign_licence", ToolCallStatus.FAILED),
    ]
    with pytest.raises(RunNotExecutable):
        executor.call_tool(
            run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
        )
    assert assignments(session_factory) == []


# --- sequence_no --------------------------------------------------------------


def test_sequence_numbers_follow_call_order_not_timestamps(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)
    goal_calls: list[ToolInput] = [
        GetUserInput(user_id=seed.user_id),
        GetLicenceInput(licence_id=seed.licence_id),
        ListUserAssignmentsInput(user_id=seed.user_id),
        AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id),
    ]
    outcomes = [executor.call_tool(run_id, args) for args in goal_calls]
    # Scramble the timestamps: the order must not depend on them.
    with session_factory.begin() as session:
        session.execute(
            update(ToolCall)
            .where(ToolCall.agent_run_id == run_id, ToolCall.sequence_no == 1)
            .values(created_at=datetime(2100, 1, 1, tzinfo=UTC))
        )

    calls = tool_calls(session_factory, run_id)

    assert [o.sequence_no for o in outcomes] == [1, 2, 3, 4]
    assert [(c.sequence_no, c.tool_name) for c in calls] == [
        (1, "get_user"),
        (2, "get_licence"),
        (3, "list_user_assignments"),
        (4, "assign_licence"),
    ]


def test_sequence_numbers_are_counted_per_run(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    first = resolved_run(executor)
    second = resolved_run(executor)

    executor.call_tool(first, GetUserInput(user_id=seed.user_id))
    executor.call_tool(second, GetUserInput(user_id=seed.user_id))
    executor.call_tool(first, GetLicenceInput(licence_id=seed.licence_id))

    assert [c.sequence_no for c in tool_calls(session_factory, first)] == [1, 2]
    assert [c.sequence_no for c in tool_calls(session_factory, second)] == [1]


def test_a_repeated_sequence_number_is_refused_by_the_database(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What two concurrent callers on one run would do: allocate the same number.
    run_id = resolved_run(executor)
    executor.call_tool(run_id, GetUserInput(user_id=seed.user_id))
    monkeypatch.setattr(ToolCallRepository, "next_sequence_no", lambda self, run_id: 1)

    with pytest.raises(IntegrityError, match="UNIQUE constraint failed"):
        executor.call_tool(
            run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
        )

    assert [c.sequence_no for c in tool_calls(session_factory, run_id)] == [1]
    assert assignments(session_factory) == []


# --- a successful mutation ----------------------------------------------------


def test_successful_assignment_commits_change_audit_event_and_trace(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )

    assert isinstance(outcome.output, AssignmentSnapshot)
    assert outcome.error is None
    # A successful call does not finish the run; only the verifier can.
    assert outcome.run_status is AgentRunStatus.EXECUTING
    (call,) = tool_calls(session_factory, run_id)
    assert call.status is ToolCallStatus.SUCCEEDED
    assert call.completed_at is not None
    assert call.result == outcome.output.model_dump(mode="json")
    assert call.error is None
    (assignment,) = assignments(session_factory)
    assert (assignment.id, assignment.user_id, assignment.licence_id) == (
        outcome.output.assignment_id,
        seed.user_id,
        seed.licence_id,
    )
    (audit,) = audit_events(session_factory)
    assert audit.actor == f"agent:run-{run_id}"
    assert (audit.action, audit.entity_id) == ("assignment.create", assignment.id)
    assert get_run(session_factory, run_id).requesting_actor == HUMAN


# --- expected domain failures -------------------------------------------------


def test_no_free_seat_blocks_the_run_and_writes_nothing(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    full = make_licence(session_factory, "Zoom", seats=0)
    run_id = resolved_run(executor, product="Zoom")

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=full)
    )

    expected_error = {
        "code": "no_seats_available",
        "message": f"Licence {full} has no seats available.",
        "error_type": "NoSeatsAvailable",
    }
    assert outcome.error == ToolError.model_validate(expected_error)
    assert outcome.run_status is AgentRunStatus.BLOCKED
    (call,) = tool_calls(session_factory, run_id)
    assert (call.status, call.error, call.result) == (
        ToolCallStatus.FAILED,
        expected_error,
        None,
    )
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.NO_SEATS_AVAILABLE
    assert run.outcome_detail == {
        "tool_call_id": call.id,
        "sequence_no": 1,
        "tool_name": "assign_licence",
        "error": expected_error,
    }
    assert run.completed_at is not None
    assert assignments(session_factory) == []
    assert audit_events(session_factory) == []


def test_inactive_user_blocks_the_run(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    carol = make_user(session_factory, "carol@example.com", UserStatus.INACTIVE)
    run_id = resolved_run(executor, user_email="carol@example.com")

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=carol, licence_id=seed.licence_id)
    )

    assert outcome.error is not None and outcome.error.code == "user_inactive"
    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (
        AgentRunStatus.BLOCKED,
        OutcomeReason.USER_INACTIVE,
    )
    assert assignments(session_factory) == []
    assert audit_events(session_factory) == []


def test_assignment_made_elsewhere_is_recorded_truthfully_and_left_to_the_verifier(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)
    listing = executor.call_tool(run_id, ListUserAssignmentsInput(user_id=seed.user_id))
    assert isinstance(listing.output, UserAssignmentsSnapshot)
    assert listing.output.assignments == []
    assign_as_human(session_factory, seed.user_id, seed.licence_id)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )

    # The call failed, and says so; the run does not claim failure yet.
    assert outcome.error is not None
    assert outcome.error.code == "assignment_already_exists"
    assert outcome.run_status is AgentRunStatus.EXECUTING
    assert executor.verify_and_finish(run_id) is AgentRunStatus.COMPLETED
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.ALREADY_SATISFIED
    assert len(assignments(session_factory)) == 1
    assert [e.actor for e in audit_events(session_factory)] == [HUMAN]


def test_duplicate_insert_race_rolls_back_and_is_left_to_the_verifier(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    assign_as_human(session_factory, seed.user_id, seed.licence_id)

    with monkeypatch.context() as patch:
        # The service's pre-check misses the committed row, as in a race, so
        # the INSERT reaches the real partial unique index.
        patch.setattr(AssignmentRepository, "get_active", lambda self, u, lic: None)
        outcome = executor.call_tool(
            run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
        )

    assert outcome.error is not None
    assert outcome.error.code == "assignment_already_exists"
    assert outcome.run_status is AgentRunStatus.EXECUTING
    assert len(assignments(session_factory)) == 1
    assert [e.actor for e in audit_events(session_factory)] == [HUMAN]
    assert executor.verify_and_finish(run_id) is AgentRunStatus.COMPLETED


# --- unexpected failures: nothing partial, history survives -------------------


def test_unexpected_error_after_the_assignment_flush_leaves_nothing_partial(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    real_add = AssignmentRepository.add
    flushed: list[int] = []

    def add_then_fail(self: AssignmentRepository, assignment: Assignment) -> Assignment:
        real_add(self, assignment)  # the INSERT is sent, not committed
        flushed.append(len(self._session.scalars(select(Assignment)).all()))
        raise RuntimeError("boom: secret-token")

    with monkeypatch.context() as patch:
        patch.setattr(AssignmentRepository, "add", add_then_fail)
        outcome = executor.call_tool(
            run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
        )

    assert flushed == [1]
    assert outcome.output is None
    expected_error = {
        "code": "unexpected_error",
        "message": "Unexpected error while running assign_licence.",
        "error_type": "RuntimeError",
    }
    (call,) = tool_calls(session_factory, run_id)
    assert (call.status, call.error) == (ToolCallStatus.FAILED, expected_error)
    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (
        AgentRunStatus.FAILED,
        OutcomeReason.TOOL_FAILED,
    )
    assert assignments(session_factory) == []
    assert audit_events(session_factory) == []

    # The next operation, on a fresh run, works normally.
    retry = resolved_run(executor)
    executor.call_tool(
        retry, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )
    assert executor.verify_and_finish(retry) is AgentRunStatus.COMPLETED
    assert [e.actor for e in audit_events(session_factory)] == [f"agent:run-{retry}"]
    assert len(tool_calls(session_factory, run_id)) == 1  # history kept


def test_audit_failure_rolls_back_the_assignment_as_well(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
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

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )

    assert flushed == [(1, 1)]
    assert outcome.error is not None and outcome.error.code == "unexpected_error"
    assert assignments(session_factory) == []
    assert audit_events(session_factory) == []
    assert get_run(session_factory, run_id).status is AgentRunStatus.FAILED


def test_failed_commit_after_the_tool_returned_is_a_failed_call(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)

    def fail_business_commit(session: Session) -> None:
        # Only the business session can see an (uncommitted) assignment here.
        if session.scalar(select(func.count()).select_from(Assignment)):
            raise RuntimeError("commit failed")

    event.listen(Session, "before_commit", fail_business_commit)
    try:
        outcome = executor.call_tool(
            run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
        )
    finally:
        event.remove(Session, "before_commit", fail_business_commit)

    assert outcome.output is None  # what the tool returned was never committed
    assert outcome.error is not None and outcome.error.code == "unexpected_error"
    (call,) = tool_calls(session_factory, run_id)
    assert (call.status, call.result) == (ToolCallStatus.FAILED, None)
    assert assignments(session_factory) == []
    assert audit_events(session_factory) == []


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
    run_id = resolved_run(executor)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )

    assert outcome.error == ToolError(
        code="unexpected_error",
        message="Unexpected error while running assign_licence.",
        error_type="IntegrityError",
    )
    with session_factory() as session:
        persisted = session.execute(
            text(
                "SELECT arguments, result, error FROM tool_calls UNION ALL "
                "SELECT outcome_detail, NULL, NULL FROM agent_runs"
            )
        ).all()
    assert "secret" not in repr(persisted)


# --- verification decides -----------------------------------------------------


def test_verification_is_refused_before_any_tool_call(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)

    with pytest.raises(IllegalTransition):
        executor.verify_and_finish(run_id)

    assert get_run(session_factory, run_id).status is AgentRunStatus.RESOLVED


def test_successful_tool_calls_cannot_complete_a_run_the_verifier_rejects(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )
    seen: list[ResolvedAssignmentGoal] = []

    def unsatisfied(
        assignments: AssignmentService, goal: ResolvedAssignmentGoal
    ) -> VerificationResult:
        seen.append(goal)
        return VerificationResult(
            satisfied=False,
            evidence=VerificationEvidence(
                goal_type=goal.goal_type,
                desired_state=goal.desired_state,
                user_id=goal.user_id,
                licence_id=goal.licence_id,
                active_assignment=None,
            ),
        )

    monkeypatch.setattr(executor_module, "verify", unsatisfied)

    assert executor.verify_and_finish(run_id) is AgentRunStatus.FAILED

    assert seen == [executor.get_goal(run_id)]  # the persisted goal, once
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.VERIFICATION_FAILED


def test_stale_favourable_observation_does_not_complete_the_run(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    assignment_id = assign_as_human(session_factory, seed.user_id, seed.licence_id)
    run_id = resolved_run(executor)
    listing = executor.call_tool(run_id, ListUserAssignmentsInput(user_id=seed.user_id))
    assert isinstance(listing.output, UserAssignmentsSnapshot)
    assert [a.active for a in listing.output.assignments] == [True]

    revoke_as_human(session_factory, assignment_id)  # the observation goes stale

    assert executor.verify_and_finish(run_id) is AgentRunStatus.FAILED
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.VERIFICATION_FAILED
    assert run.outcome_detail is not None
    assert run.outcome_detail["satisfied"] is False
    assert run.outcome_detail["evidence"]["active_assignment"] is None
    # The earlier observation stays in the trace as it was.
    (call,) = tool_calls(session_factory, run_id)
    assert call.result is not None
    assert call.result["assignments"][0]["active"] is True


def test_verifier_crash_fails_the_run_without_keeping_its_message(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )

    def crash(*args: Any) -> None:
        raise RuntimeError("secret-token")

    monkeypatch.setattr(executor_module, "verify", crash)

    assert executor.verify_and_finish(run_id) is AgentRunStatus.FAILED
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.UNEXPECTED_ERROR
    assert run.outcome_detail == {"stage": "verification", "error_type": "RuntimeError"}


# --- I12: a favourable observation is not a guarantee -------------------------


def test_favourable_capacity_observation_is_not_a_reservation(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    last_seat = make_licence(session_factory, "Zoom", seats=1)
    run_id = resolved_run(executor, product="Zoom")

    # 1. The run observes one free seat.
    observed = executor.call_tool(run_id, GetLicenceInput(licence_id=last_seat))
    assert isinstance(observed.output, LicenceSnapshot)
    assert observed.output.seats_available == 1

    # 2. State changes elsewhere: another valid transaction takes that seat.
    bobs = assign_as_human(session_factory, seed.other_user_id, last_seat)

    # 3. The run's assignment is rejected by the service at mutation time.
    attempt = executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=last_seat)
    )
    assert attempt.error is not None and attempt.error.code == "no_seats_available"

    # 4. The trace shows the favourable observation, then the rejection after it.
    calls = tool_calls(session_factory, run_id)
    assert [(c.sequence_no, c.tool_name, c.status) for c in calls] == [
        (1, "get_licence", ToolCallStatus.SUCCEEDED),
        (2, "assign_licence", ToolCallStatus.FAILED),
    ]
    assert calls[0].result is not None
    assert calls[0].result["seats_available"] == 1
    assert calls[1].error is not None
    assert calls[1].error["code"] == "no_seats_available"

    # 5. The run is BLOCKED; no invalid assignment or partial audit row exists.
    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (
        AgentRunStatus.BLOCKED,
        OutcomeReason.NO_SEATS_AVAILABLE,
    )
    assert run.outcome_detail is not None
    assert run.outcome_detail["sequence_no"] == 2
    assert [(a.id, a.user_id) for a in assignments(session_factory)] == [
        (bobs, seed.other_user_id)
    ]
    assert [(e.actor, e.entity_id) for e in audit_events(session_factory)] == [
        (HUMAN, bobs)
    ]


# --- serialization ------------------------------------------------------------


def test_arguments_results_and_errors_round_trip_as_json(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = resolved_run(executor)
    outcomes = [
        executor.call_tool(run_id, GetUserInput(user_id=seed.user_id)),
        executor.call_tool(run_id, GetLicenceInput(licence_id=seed.licence_id)),
        executor.call_tool(run_id, ListUserAssignmentsInput(user_id=seed.user_id)),
        executor.call_tool(
            run_id,
            AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id),
        ),
        # Fails: Ada now holds the licence.
        executor.call_tool(
            run_id,
            AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id),
        ),
    ]

    with session_factory() as session:
        raw = session.execute(
            text(
                "SELECT arguments, result, error, result IS NULL, error IS NULL "
                "FROM tool_calls WHERE agent_run_id = :run ORDER BY sequence_no"
            ),
            {"run": run_id},
        ).all()
    # Stored as JSON text; an absent result or error is SQL NULL, not 'null'.
    assert [json.loads(row[0]) for row in raw] == [
        {"user_id": seed.user_id},
        {"licence_id": seed.licence_id},
        {"user_id": seed.user_id},
        {"user_id": seed.user_id, "licence_id": seed.licence_id},
        {"user_id": seed.user_id, "licence_id": seed.licence_id},
    ]
    assert [(row[3], row[4]) for row in raw] == [(0, 1)] * 4 + [(1, 0)]

    calls = tool_calls(session_factory, run_id)
    parsed = [
        UserSnapshot.model_validate(calls[0].result),
        LicenceSnapshot.model_validate(calls[1].result),
        UserAssignmentsSnapshot.model_validate(calls[2].result),
        AssignmentSnapshot.model_validate(calls[3].result),
    ]
    assert parsed == [o.output for o in outcomes[:4]]
    assert isinstance(parsed[3], AssignmentSnapshot)
    assert parsed[3].assigned_at.tzinfo is not None
    assert ToolError.model_validate(calls[4].error) == outcomes[4].error


# --- crash windows: the state each leaves (documented, not recovered) ---------


class Crash(BaseException):
    """Stands in for the process dying: not an Exception, so nothing catches it."""


def crash(*_: Any) -> None:
    raise Crash


def test_crash_window_a_after_the_run_commit_leaves_only_a_received_run(
    executor: AgentExecutor, session_factory: Sessions, seed: Seed
) -> None:
    run_id = new_run(executor)  # a crash right after this commit leaves...

    run = get_run(session_factory, run_id)
    assert run.status is AgentRunStatus.RECEIVED
    assert (run.resolved_user_id, run.completed_at) == (None, None)
    assert tool_calls(session_factory, run_id) == []
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])


def test_crash_window_b_after_the_started_commit_leaves_no_change_and_stops(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    with monkeypatch.context() as patch:
        patch.setattr(tools, "run_tool", crash)
        with pytest.raises(Crash):
            executor.call_tool(
                run_id,
                AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id),
            )

    assert get_run(session_factory, run_id).status is AgentRunStatus.EXECUTING
    (call,) = tool_calls(session_factory, run_id)
    assert (call.status, call.completed_at, call.result, call.error) == (
        ToolCallStatus.STARTED,
        None,
        None,
        None,
    )
    assert (assignments(session_factory), audit_events(session_factory)) == ([], [])
    # Nothing carries on past an unfinished call.
    with pytest.raises(UnfinishedToolCall):
        executor.call_tool(run_id, GetUserInput(user_id=seed.user_id))
    with pytest.raises(UnfinishedToolCall):
        executor.verify_and_finish(run_id)
    assert len(tool_calls(session_factory, run_id)) == 1


def test_crash_window_c_after_the_business_commit_is_attributable_to_the_run(
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
            executor.call_tool(
                run_id,
                AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id),
            )

    assert get_run(session_factory, run_id).status is AgentRunStatus.EXECUTING
    (call,) = tool_calls(session_factory, run_id)
    assert call.status is ToolCallStatus.STARTED
    # The change committed, and its audit event names the run, so the stale
    # STARTED call can be reconciled rather than guessed at.
    (assignment,) = assignments(session_factory)
    (audit,) = audit_events(session_factory)
    assert audit.actor == f"agent:run-{run_id}"
    assert (audit.action, audit.entity_id) == ("assignment.create", assignment.id)
    assert audit.after == call.arguments
    # No blind retry, and no completion labelled "already satisfied".
    with pytest.raises(UnfinishedToolCall):
        executor.call_tool(
            run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
        )
    with pytest.raises(UnfinishedToolCall):
        executor.verify_and_finish(run_id)
    assert len(tool_calls(session_factory, run_id)) == 1
    assert len(assignments(session_factory)) == 1


def test_crash_window_d_after_verification_leaves_the_run_verifying(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = resolved_run(executor)
    executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )
    with monkeypatch.context() as patch:
        # Only the terminal LOG transaction asks this.
        patch.setattr(ToolCallRepository, "any_succeeded", crash)
        with pytest.raises(Crash):
            executor.verify_and_finish(run_id)

    run = get_run(session_factory, run_id)
    assert run.status is AgentRunStatus.VERIFYING
    assert (run.outcome_reason, run.outcome_detail, run.completed_at) == (
        None,
        None,
        None,
    )
    with pytest.raises(IllegalTransition):
        executor.verify_and_finish(run_id)
    with pytest.raises(RunNotExecutable):
        executor.call_tool(run_id, GetUserInput(user_id=seed.user_id))
    assert len(assignments(session_factory)) == 1


# --- review regressions -------------------------------------------------------


@pytest.mark.parametrize("seats", [5, 0], ids=["succeeds", "no-seats"])
def test_call_outcome_is_recorded_even_if_the_run_was_ended_meanwhile(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
    seats: int,
) -> None:
    # Concurrent callers are not coordinated (a documented limitation), but
    # a call's outcome must still be recorded, never left STARTED.
    licence = make_licence(session_factory, "Zoom", seats=seats)
    run_id = resolved_run(executor, product="Zoom")
    real_context = executor_module._tool_context

    def end_run_first(session: Session, actor: str) -> tools.ToolContext:
        # Runs after the STARTED commit and before the business transaction
        # has touched the database: another caller ends the run here.
        with session_factory.begin() as other:
            run = other.get(AgentRun, run_id)
            assert run is not None
            transition(
                run,
                AgentRunStatus.FAILED,
                reason=OutcomeReason.VERIFICATION_FAILED,
                detail={},
            )
        return real_context(session, actor)

    monkeypatch.setattr(executor_module, "_tool_context", end_run_first)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=licence)
    )

    (call,) = tool_calls(session_factory, run_id)
    expected = ToolCallStatus.SUCCEEDED if seats else ToolCallStatus.FAILED
    assert call.status is expected
    assert call.completed_at is not None
    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (
        AgentRunStatus.FAILED,
        OutcomeReason.VERIFICATION_FAILED,
    )
    assert outcome.run_status is AgentRunStatus.FAILED


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

    goal = executor.get_goal(run_id)
    assert (goal.user_id, goal.licence_id) == (seed.user_id, seed.licence_id)
    executor.call_tool(run_id, ListUserAssignmentsInput(user_id=seed.user_id))
    executor.call_tool(
        run_id, AssignLicenceInput(user_id=seed.user_id, licence_id=seed.licence_id)
    )

    assert executor.verify_and_finish(run_id) is AgentRunStatus.COMPLETED
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.GOAL_SATISFIED
    assert run.outcome_detail is not None
    assert run.outcome_detail["evidence"]["licence_id"] == seed.licence_id
    assert [(a.user_id, a.licence_id) for a in assignments(session_factory)] == [
        (seed.user_id, seed.licence_id)
    ]


def test_a_call_that_names_no_target_is_refused(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tool that names nothing cannot be checked against the goal, so the
    # check fails closed rather than open.
    run_id = resolved_run(executor)
    monkeypatch.setattr(tools, "target_ids", lambda args: (None, None))

    outcome = executor.call_tool(run_id, GetUserInput(user_id=seed.user_id))

    assert outcome.error is not None
    assert outcome.error.code == "goal_scope_violation"
    assert outcome.run_status is AgentRunStatus.FAILED
