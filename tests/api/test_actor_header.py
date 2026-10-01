"""The X-Actor header on every mutating endpoint: required, within the actor
rules, and never in the "agent:" namespace, which is reserved so that HTTP
callers cannot write audit events that look like an agent run's. A request
refused for its actor writes nothing."""

from collections.abc import Callable

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Assignment, AuditEvent, Licence, User

RESERVED_ACTORS = ["agent:run-5", " Agent:run-5 ", "AGENT:run-1"]


def make_user(session: Session) -> User:
    user = User(email="ada@example.com", name="Ada")
    session.add(user)
    session.commit()
    return user


def make_licence(session: Session) -> Licence:
    licence = Licence(product="Figma", seats_total=5)
    session.add(licence)
    session.commit()
    return licence


def create_user(
    client: TestClient, session: Session, headers: dict[str, str]
) -> httpx2.Response:
    return client.post(
        "/users", json={"email": "bob@example.com", "name": "Bob"}, headers=headers
    )


def create_licence(
    client: TestClient, session: Session, headers: dict[str, str]
) -> httpx2.Response:
    return client.post(
        "/licences", json={"product": "Slack", "seats_total": 5}, headers=headers
    )


def create_assignment(
    client: TestClient, session: Session, headers: dict[str, str]
) -> httpx2.Response:
    user = make_user(session)
    licence = make_licence(session)
    return client.post(
        "/assignments",
        json={"user_id": user.id, "licence_id": licence.id},
        headers=headers,
    )


def revoke_assignment(
    client: TestClient, session: Session, headers: dict[str, str]
) -> httpx2.Response:
    assignment = Assignment(
        user_id=make_user(session).id, licence_id=make_licence(session).id
    )
    session.add(assignment)
    session.commit()
    return client.post(f"/assignments/{assignment.id}/revoke", headers=headers)


Request = Callable[[TestClient, Session, dict[str, str]], httpx2.Response]
MUTATING_REQUESTS: list[Request] = [
    create_user,
    create_licence,
    create_assignment,
    revoke_assignment,
]


def assert_nothing_written(session: Session) -> None:
    session.expire_all()
    assert session.scalars(select(AuditEvent)).all() == []
    assert (
        session.scalars(select(User).where(User.email == "bob@example.com")).all() == []
    )
    assert (
        session.scalars(select(Licence).where(Licence.product == "Slack")).all() == []
    )
    revoked = session.scalars(
        select(Assignment).where(Assignment.revoked_at.is_not(None))
    ).all()
    assert revoked == []


@pytest.mark.parametrize("request_", MUTATING_REQUESTS, ids=lambda r: r.__name__)
@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Actor": ""}, {"X-Actor": "   "}, {"X-Actor": "a" * 321}],
    ids=["missing", "empty", "whitespace", "too-long"],
)
def test_a_missing_or_invalid_actor_header_is_rejected_and_nothing_is_written(
    client: TestClient, session: Session, request_: Request, headers: dict[str, str]
) -> None:
    # "whitespace" passes the header's length check and is refused by the
    # service's actor rules, so each router's InvalidInput mapping is used.
    response = request_(client, session, headers)

    assert response.status_code == 422
    assert_nothing_written(session)


@pytest.mark.parametrize("request_", MUTATING_REQUESTS, ids=lambda r: r.__name__)
@pytest.mark.parametrize("actor", RESERVED_ACTORS)
def test_a_reserved_actor_header_is_rejected_and_nothing_is_written(
    client: TestClient, session: Session, request_: Request, actor: str
) -> None:
    response = request_(client, session, {"X-Actor": actor})

    assert response.status_code == 422
    assert "reserved for agent runs" in response.json()["detail"]
    assert_nothing_written(session)


@pytest.mark.parametrize("request_", MUTATING_REQUESTS, ids=lambda r: r.__name__)
@pytest.mark.parametrize("actor", ["admin@example.com", "agent-smith@example.com"])
def test_an_ordinary_actor_header_is_recorded_on_the_audit_event(
    client: TestClient, session: Session, request_: Request, actor: str
) -> None:
    response = request_(client, session, {"X-Actor": actor})
    session.expire_all()

    assert response.status_code in (200, 201)
    events = session.scalars(select(AuditEvent.actor).order_by(AuditEvent.id))
    assert list(events) == [actor]
