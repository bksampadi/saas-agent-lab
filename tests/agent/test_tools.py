"""Tools return structured application facts, exactly the ones they read.

Called through the executor, as a planner will call them, and checked in
the persisted ToolCall.result.
"""

from dataclasses import fields
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agent import tools
from app.agent.executor import AgentExecutor
from app.models import AgentRunStatus, Assignment, AuditEvent, Licence, ToolCall, User
from app.schemas.agent import (
    AssignLicenceInput,
    ExtractedAssignmentIntent,
    GetLicenceInput,
    GetUserInput,
    ListUserAssignmentsInput,
    ToolInput,
)
from app.schemas.assignment import ID_MAX

Sessions = sessionmaker[Session]
BUSINESS_TABLES = ("users", "licences", "assignments", "audit_events")


def add(sessions: Sessions, row: User | Licence | Assignment) -> int:
    with sessions.begin() as session:
        session.add(row)
        session.flush()
        return row.id


def resolved_run(executor: AgentExecutor) -> int:
    run_id = executor.create_run(
        instruction="Give Ada a Figma seat.",
        requesting_actor="admin@example.com",
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product="Figma"),
    )
    assert executor.resolve_run(run_id) is AgentRunStatus.RESOLVED
    return run_id


def persisted_result(sessions: Sessions, tool_call_id: int) -> dict[str, Any]:
    with sessions() as session:
        call = session.get(ToolCall, tool_call_id)
        assert call is not None and call.result is not None
        return call.result


def count(sessions: Sessions, model: type[Assignment] | type[AuditEvent]) -> int:
    with sessions() as session:
        return len(session.scalars(select(model)).all())


@pytest.fixture
def ada(session_factory: Sessions) -> int:
    return add(session_factory, User(email="ada@example.com", name="Ada Lovelace"))


@pytest.fixture
def figma(session_factory: Sessions) -> int:
    return add(session_factory, Licence(product="Figma", seats_total=3))


def test_get_user_returns_exactly_the_user_facts(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    run_id = resolved_run(executor)

    outcome = executor.call_tool(run_id, GetUserInput(user_id=ada))

    assert persisted_result(session_factory, outcome.tool_call_id) == {
        "user_id": ada,
        "email": "ada@example.com",
        "name": "Ada Lovelace",
        "status": "active",
    }


def test_get_licence_returns_exactly_the_capacity_facts(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    bob = add(session_factory, User(email="bob@example.com", name="Bob"))
    carol = add(session_factory, User(email="carol@example.com", name="Carol"))
    slack = add(session_factory, Licence(product="Slack", seats_total=3))
    add(session_factory, Assignment(user_id=bob, licence_id=figma))
    add(
        session_factory,
        Assignment(user_id=carol, licence_id=figma, revoked_at=datetime.now(UTC)),
    )
    add(session_factory, Assignment(user_id=bob, licence_id=slack))
    run_id = resolved_run(executor)

    outcome = executor.call_tool(run_id, GetLicenceInput(licence_id=figma))

    assert persisted_result(session_factory, outcome.tool_call_id) == {
        "licence_id": figma,
        "product": "Figma",
        "seats_total": 3,
        "seats_active": 1,  # the revoked row and the other licence do not count
        "seats_available": 2,
    }


def test_list_user_assignments_returns_the_users_active_and_revoked_rows(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    bob = add(session_factory, User(email="bob@example.com", name="Bob"))
    slack = add(session_factory, Licence(product="Slack", seats_total=3))
    revoked_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    revoked = add(
        session_factory,
        Assignment(user_id=ada, licence_id=figma, revoked_at=revoked_at),
    )
    add(session_factory, Assignment(user_id=bob, licence_id=figma))
    active = add(session_factory, Assignment(user_id=ada, licence_id=slack))
    run_id = resolved_run(executor)

    outcome = executor.call_tool(run_id, ListUserAssignmentsInput(user_id=ada))

    result = persisted_result(session_factory, outcome.tool_call_id)
    assert set(result) == {"user_id", "assignments"}
    assert result["user_id"] == ada
    rows = result["assignments"]
    for row in rows:
        assert set(row) == {
            "assignment_id",
            "user_id",
            "licence_id",
            "active",
            "assigned_at",
            "revoked_at",
        }
    assert [
        (r["assignment_id"], r["user_id"], r["licence_id"], r["active"]) for r in rows
    ] == [(revoked, ada, figma, False), (active, ada, slack, True)]
    assert rows[0]["revoked_at"] == revoked_at.isoformat().replace("+00:00", "Z")
    assert rows[1]["revoked_at"] is None


def test_assign_licence_returns_exactly_the_new_assignment_facts(
    executor: AgentExecutor, session_factory: Sessions, ada: int, figma: int
) -> None:
    run_id = resolved_run(executor)

    outcome = executor.call_tool(
        run_id, AssignLicenceInput(user_id=ada, licence_id=figma)
    )

    result = persisted_result(session_factory, outcome.tool_call_id)
    assigned_at = result.pop("assigned_at")
    assert result == {
        "assignment_id": 1,
        "user_id": ada,
        "licence_id": figma,
        "active": True,
        "revoked_at": None,
    }
    assert datetime.fromisoformat(assigned_at).tzinfo is not None


def test_read_tools_change_nothing_and_write_no_audit_events(
    executor: AgentExecutor,
    session_factory: Sessions,
    statements: list[str],
    ada: int,
    figma: int,
) -> None:
    run_id = resolved_run(executor)
    statements.clear()

    for args in (
        GetUserInput(user_id=ada),
        GetLicenceInput(licence_id=figma),
        ListUserAssignmentsInput(user_id=ada),
    ):
        executor.call_tool(run_id, args)

    business = [s for s in statements if any(f" {t}" in s for t in BUSINESS_TABLES)]
    assert business
    assert all(s.startswith("SELECT") for s in business)
    assert count(session_factory, AuditEvent) == 0
    assert count(session_factory, Assignment) == 0


@pytest.mark.parametrize(
    "build",
    [
        lambda: GetUserInput.model_validate({"user_id": 1, "email": "x@example.com"}),
        lambda: AssignLicenceInput.model_validate(
            {"user_id": 1, "licence_id": 1, "actor": "admin@example.com"}
        ),
        lambda: GetUserInput(user_id=0),
        lambda: GetLicenceInput(licence_id=-1),
        lambda: AssignLicenceInput(user_id=1, licence_id=ID_MAX + 1),
    ],
    ids=["extra-field", "actor-field", "zero-id", "negative-id", "id-too-large"],
)
def test_tool_inputs_reject_unknown_fields_and_invalid_ids(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


def test_only_assign_licence_mutates() -> None:
    names = [tool_input.tool_name for tool_input in tools.TOOL_INPUTS]

    assert sorted(names) == [
        "assign_licence",
        "get_licence",
        "get_user",
        "list_user_assignments",
    ]
    assert tools.MUTATING_TOOL_NAMES == {"assign_licence"}


def test_tools_are_given_services_and_an_actor_never_a_session() -> None:
    assert [f.name for f in fields(tools.ToolContext)] == [
        "users",
        "licences",
        "assignments",
        "actor",
    ]


def test_every_tool_input_names_its_targets() -> None:
    # target_ids is what the goal-scope check compares against the goal.
    examples: list[ToolInput] = [
        GetUserInput(user_id=7),
        GetLicenceInput(licence_id=8),
        ListUserAssignmentsInput(user_id=7),
        AssignLicenceInput(user_id=7, licence_id=8),
    ]

    assert [tools.target_ids(args) for args in examples] == [
        (7, None),
        (None, 8),
        (7, None),
        (7, 8),
    ]
