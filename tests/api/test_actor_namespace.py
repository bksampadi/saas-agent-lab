"""The "agent:" actor namespace is reserved: HTTP callers cannot write audit
events that look like an agent run's."""

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


def create_user(client: TestClient, session: Session, actor: str) -> httpx2.Response:
    return client.post(
        "/users",
        json={"email": "bob@example.com", "name": "Bob"},
        headers={"X-Actor": actor},
    )


def create_licence(client: TestClient, session: Session, actor: str) -> httpx2.Response:
    return client.post(
        "/licences",
        json={"product": "Slack", "seats_total": 5},
        headers={"X-Actor": actor},
    )


def create_assignment(
    client: TestClient, session: Session, actor: str
) -> httpx2.Response:
    user = make_user(session)
    licence = make_licence(session)
    return client.post(
        "/assignments",
        json={"user_id": user.id, "licence_id": licence.id},
        headers={"X-Actor": actor},
    )


def revoke_assignment(
    client: TestClient, session: Session, actor: str
) -> httpx2.Response:
    assignment = Assignment(
        user_id=make_user(session).id, licence_id=make_licence(session).id
    )
    session.add(assignment)
    session.commit()
    return client.post(
        f"/assignments/{assignment.id}/revoke", headers={"X-Actor": actor}
    )


Request = Callable[[TestClient, Session, str], httpx2.Response]
MUTATING_REQUESTS: list[Request] = [
    create_user,
    create_licence,
    create_assignment,
    revoke_assignment,
]


def audit_actors(session: Session) -> list[str]:
    return list(session.scalars(select(AuditEvent.actor).order_by(AuditEvent.id)))


@pytest.mark.parametrize("request_", MUTATING_REQUESTS, ids=lambda r: r.__name__)
@pytest.mark.parametrize("actor", RESERVED_ACTORS)
def test_reserved_actor_header_is_rejected_and_nothing_is_written(
    client: TestClient, session: Session, request_: Request, actor: str
) -> None:
    response = request_(client, session, actor)
    session.expire_all()

    assert response.status_code == 422
    assert "reserved for agent runs" in response.json()["detail"]
    assert audit_actors(session) == []
    revoked = session.scalars(
        select(Assignment).where(Assignment.revoked_at.is_not(None))
    ).all()
    assert revoked == []


@pytest.mark.parametrize("request_", MUTATING_REQUESTS, ids=lambda r: r.__name__)
@pytest.mark.parametrize("actor", ["admin@example.com", "agent-smith@example.com"])
def test_ordinary_actor_header_still_works(
    client: TestClient, session: Session, request_: Request, actor: str
) -> None:
    response = request_(client, session, actor)
    session.expire_all()

    assert response.status_code in (200, 201)
    assert audit_actors(session) == [actor]
