"""The deterministic harness end to end: extracted intent in, terminal run out."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agent import tools
from app.agent.executor import AgentExecutor
from app.agent.harness import run_ensure_assignment
from app.agent.tools import ToolContext
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
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import (
    AssignLicenceInput,
    AssignmentSnapshot,
    ExtractedAssignmentIntent,
    ListUserAssignmentsInput,
    UserAssignmentsSnapshot,
)

HUMAN = "admin@example.com"
Sessions = sessionmaker[Session]


def add(sessions: Sessions, row: User | Licence | Assignment) -> int:
    with sessions.begin() as session:
        session.add(row)
        session.flush()
        return row.id


def run(
    executor: AgentExecutor, user_email: str = "ada@example.com", product: str = "Figma"
) -> int:
    return run_ensure_assignment(
        executor,
        instruction=f"Give {user_email} a {product} seat.",
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email=user_email, product=product),
    )


def get_run(sessions: Sessions, run_id: int) -> AgentRun:
    with sessions() as session:
        agent_run = session.get(AgentRun, run_id)
        assert agent_run is not None
        return agent_run


def trace(sessions: Sessions, run_id: int) -> list[tuple[int, str, ToolCallStatus]]:
    with sessions() as session:
        return [
            (call.sequence_no, call.tool_name, call.status)
            for call in ToolCallRepository(session).list_for_run(run_id)
        ]


def tool_calls(sessions: Sessions, run_id: int) -> list[ToolCall]:
    with sessions() as session:
        return ToolCallRepository(session).list_for_run(run_id)


def assignments(sessions: Sessions) -> list[Assignment]:
    with sessions() as session:
        return list(session.scalars(select(Assignment).order_by(Assignment.id)))


def audit_actors(sessions: Sessions) -> list[str]:
    with sessions() as session:
        return list(session.scalars(select(AuditEvent.actor).order_by(AuditEvent.id)))


@pytest.fixture
def ada(session_factory: Sessions) -> int:
    return add(session_factory, User(email="ada@example.com", name="Ada"))


@pytest.fixture
def figma(session_factory: Sessions) -> int:
    return add(session_factory, Licence(product="Figma", seats_total=5))


FULL_TRACE = [
    (1, "list_user_assignments", ToolCallStatus.SUCCEEDED),
    (2, "get_licence", ToolCallStatus.SUCCEEDED),
    (3, "assign_licence", ToolCallStatus.SUCCEEDED),
]


# --- completed ----------------------------------------------------------------


def test_run_assigns_the_licence_and_completes_through_the_verifier(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    run_id = run(executor)

    agent_run = get_run(session_factory, run_id)
    assert agent_run.status is AgentRunStatus.COMPLETED
    assert agent_run.outcome_reason is OutcomeReason.GOAL_SATISFIED
    assert agent_run.completed_at is not None
    assert agent_run.requesting_actor == HUMAN
    assert trace(session_factory, run_id) == FULL_TRACE
    (assignment,) = assignments(session_factory)
    assert (assignment.user_id, assignment.licence_id) == (ada, figma)
    assert audit_actors(session_factory) == [f"agent:run-{run_id}"]
    assert agent_run.outcome_detail is not None
    assert agent_run.outcome_detail["satisfied"] is True
    evidence = agent_run.outcome_detail["evidence"]
    assert evidence["active_assignment"]["assignment_id"] == assignment.id


def test_second_run_for_the_same_goal_is_already_satisfied_without_mutating(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    first = run(executor)
    assert get_run(session_factory, first).status is AgentRunStatus.COMPLETED

    second = run(executor)

    agent_run = get_run(session_factory, second)
    assert agent_run.status is AgentRunStatus.COMPLETED
    assert agent_run.outcome_reason is OutcomeReason.ALREADY_SATISFIED
    assert trace(session_factory, second) == [
        (1, "list_user_assignments", ToolCallStatus.SUCCEEDED)
    ]
    assert len(assignments(session_factory)) == 1
    assert audit_actors(session_factory) == [f"agent:run-{first}"]


def test_assignment_made_by_a_person_is_already_satisfied(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    add(session_factory, Assignment(user_id=ada, licence_id=figma))

    run_id = run(executor)

    agent_run = get_run(session_factory, run_id)
    assert agent_run.outcome_reason is OutcomeReason.ALREADY_SATISFIED
    assert [name for _, name, _ in trace(session_factory, run_id)] == [
        "list_user_assignments"
    ]


def test_revoked_assignment_is_not_satisfaction_and_is_assigned_again(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    add(
        session_factory,
        Assignment(user_id=ada, licence_id=figma, revoked_at=datetime.now(UTC)),
    )

    run_id = run(executor)

    assert get_run(session_factory, run_id).outcome_reason is (
        OutcomeReason.GOAL_SATISFIED
    )
    assert trace(session_factory, run_id) == FULL_TRACE
    assert [a.revoked_at is None for a in assignments(session_factory)] == [
        False,
        True,
    ]


def test_same_licence_held_by_another_user_is_not_satisfaction(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    bob = add(session_factory, User(email="bob@example.com", name="Bob"))
    add(session_factory, Assignment(user_id=bob, licence_id=figma))

    run_id = run(executor)

    assert get_run(session_factory, run_id).outcome_reason is (
        OutcomeReason.GOAL_SATISFIED
    )
    assert trace(session_factory, run_id) == FULL_TRACE


# --- blocked ------------------------------------------------------------------


def test_exhausted_capacity_blocks_the_run_and_the_service_decides(
    executor: AgentExecutor, session_factory: Sessions, ada: int
) -> None:
    bob = add(session_factory, User(email="bob@example.com", name="Bob"))
    zoom = add(session_factory, Licence(product="Zoom", seats_total=1))
    add(session_factory, Assignment(user_id=bob, licence_id=zoom))

    run_id = run(executor, product="Zoom")

    agent_run = get_run(session_factory, run_id)
    assert agent_run.status is AgentRunStatus.BLOCKED
    assert agent_run.outcome_reason is OutcomeReason.NO_SEATS_AVAILABLE
    # The capacity read saw no free seat, and the harness did not act on it:
    # the assignment was still attempted and refused by the service.
    calls = tool_calls(session_factory, run_id)
    assert [(c.tool_name, c.status) for c in calls] == [
        ("list_user_assignments", ToolCallStatus.SUCCEEDED),
        ("get_licence", ToolCallStatus.SUCCEEDED),
        ("assign_licence", ToolCallStatus.FAILED),
    ]
    assert calls[1].result is not None and calls[1].result["seats_available"] == 0
    assert [a.user_id for a in assignments(session_factory)] == [bob]
    assert audit_actors(session_factory) == []


def test_inactive_user_blocks_the_run(
    executor: AgentExecutor, session_factory: Sessions, figma: int
) -> None:
    add(
        session_factory,
        User(email="carol@example.com", name="Carol", status=UserStatus.INACTIVE),
    )

    run_id = run(executor, user_email="carol@example.com")

    agent_run = get_run(session_factory, run_id)
    assert (agent_run.status, agent_run.outcome_reason) == (
        AgentRunStatus.BLOCKED,
        OutcomeReason.USER_INACTIVE,
    )
    assert trace(session_factory, run_id)[-1] == (
        3,
        "assign_licence",
        ToolCallStatus.FAILED,
    )
    assert assignments(session_factory) == []
    assert audit_actors(session_factory) == []


# --- needs clarification ------------------------------------------------------


@pytest.mark.parametrize(
    ("user_email", "product", "reason"),
    [
        ("nobody@example.com", "Figma", OutcomeReason.USER_NOT_FOUND),
        ("ada@example.com", "Sketch", OutcomeReason.LICENCE_NOT_FOUND),
        ("ada@example.com", "figma", OutcomeReason.LICENCE_AMBIGUOUS),
        ("", "Figma", OutcomeReason.INVALID_INPUT),
    ],
)
def test_unresolved_goal_needs_clarification_and_makes_no_tool_calls(
    executor: AgentExecutor,
    session_factory: Sessions,
    ada: int,
    figma: int,
    user_email: str,
    product: str,
    reason: OutcomeReason,
) -> None:
    add(session_factory, Licence(product="FIGMA", seats_total=5))

    run_id = run(executor, user_email, product)

    agent_run = get_run(session_factory, run_id)
    assert (agent_run.status, agent_run.outcome_reason) == (
        AgentRunStatus.NEEDS_CLARIFICATION,
        reason,
    )
    assert trace(session_factory, run_id) == []
    assert assignments(session_factory) == []
    assert audit_actors(session_factory) == []


# --- tools that lie cannot complete a run -------------------------------------


def test_assign_tool_claiming_success_without_writing_ends_failed(
    executor: AgentExecutor,
    session_factory: Sessions,
    ada: int,
    figma: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lying_assign(
        context: ToolContext, args: AssignLicenceInput
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

    run_id = run(executor)

    agent_run = get_run(session_factory, run_id)
    assert agent_run.status is AgentRunStatus.FAILED
    assert agent_run.outcome_reason is OutcomeReason.VERIFICATION_FAILED
    # The trace records what the tool claimed; the verifier found otherwise.
    assert trace(session_factory, run_id) == FULL_TRACE
    assert assignments(session_factory) == []
    assert audit_actors(session_factory) == []


def test_list_tool_claiming_the_goal_already_holds_ends_failed(
    executor: AgentExecutor,
    session_factory: Sessions,
    ada: int,
    figma: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lying_list(
        context: ToolContext, args: ListUserAssignmentsInput
    ) -> UserAssignmentsSnapshot:
        return UserAssignmentsSnapshot(
            user_id=args.user_id,
            assignments=[
                AssignmentSnapshot(
                    assignment_id=999,
                    user_id=args.user_id,
                    licence_id=figma,
                    active=True,
                    assigned_at=datetime.now(UTC),
                    revoked_at=None,
                )
            ],
        )

    monkeypatch.setattr(tools, "list_user_assignments", lying_list)

    run_id = run(executor)

    agent_run = get_run(session_factory, run_id)
    assert agent_run.status is AgentRunStatus.FAILED
    assert agent_run.outcome_reason is OutcomeReason.VERIFICATION_FAILED
    assert trace(session_factory, run_id) == [
        (1, "list_user_assignments", ToolCallStatus.SUCCEEDED)
    ]
