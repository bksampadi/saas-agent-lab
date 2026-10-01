"""Tools return structured application facts, exactly the ones they read.

Called by a scripted model through the executor, and checked in the
persisted ToolCall.result.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agent.executor import AgentExecutor
from app.models import Assignment, AuditEvent, Licence, ToolCall, User
from support import (
    ASSIGN,
    ASSIGNMENTS,
    CAPACITY,
    GOAL_REACHED,
    NO_ACTION_NEEDED,
    USER,
    Script,
    call,
    resolved_run,
)

Sessions = sessionmaker[Session]
BUSINESS_TABLES = ("users", "licences", "assignments", "audit_events")


def add(sessions: Sessions, row: User | Licence | Assignment) -> int:
    with sessions.begin() as session:
        session.add(row)
        session.flush()
        return row.id


def result_of(executor: AgentExecutor, sessions: Sessions, tool: str) -> dict[str, Any]:
    """The persisted result of one call of the model-facing ``tool``."""
    run_id = resolved_run(executor)
    conclusion = GOAL_REACHED if tool == ASSIGN else NO_ACTION_NEEDED
    executor.decide(run_id, Script(call(tool), conclusion).planner())
    with sessions() as session:
        (tool_call,) = session.scalars(
            select(ToolCall).where(ToolCall.agent_run_id == run_id)
        ).all()
    assert tool_call.result is not None
    return tool_call.result


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
    result = result_of(executor, session_factory, USER)

    assert result == {
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

    result = result_of(executor, session_factory, CAPACITY)

    assert result == {
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

    result = result_of(executor, session_factory, ASSIGNMENTS)

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
    result = result_of(executor, session_factory, ASSIGN)

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

    executor.decide(
        run_id, Script(call(USER, CAPACITY, ASSIGNMENTS), NO_ACTION_NEEDED).planner()
    )

    business = [s for s in statements if any(f" {t}" in s for t in BUSINESS_TABLES)]
    assert business
    assert all(s.startswith("SELECT") for s in business)
    assert count(session_factory, AuditEvent) == 0
    assert count(session_factory, Assignment) == 0
