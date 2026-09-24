from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_session
from app.main import create_app
from app.models import AuditEvent, User
from app.repositories.audit_events import AuditEventRepository
from app.repositories.users import UserRepository

ACTOR_HEADERS = {"X-Actor": "admin@example.com"}


def post_user(
    client: TestClient, email: str = "ada@example.com", name: str = "Ada Lovelace"
) -> httpx2.Response:
    return client.post(
        "/users", json={"email": email, "name": name}, headers=ACTOR_HEADERS
    )


def count(session: Session, model: type[User] | type[AuditEvent]) -> int:
    return len(session.scalars(select(model)).all())


# --- POST /users --------------------------------------------------------------


def test_create_user_returns_201_with_normalized_email(client: TestClient) -> None:
    response = post_user(client, email="  Ada@Example.COM ", name="  Ada Lovelace ")

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "email", "name", "status", "created_at"}
    assert isinstance(body["id"], int)
    assert body["email"] == "ada@example.com"
    assert body["name"] == "Ada Lovelace"
    assert body["status"] == "active"


def test_create_user_records_actor_header_on_audit_event(
    client: TestClient, session: Session
) -> None:
    user_id = post_user(client).json()["id"]

    events = session.scalars(select(AuditEvent)).all()
    assert len(events) == 1
    assert events[0].actor == "admin@example.com"
    assert events[0].action == "user.create"
    assert events[0].entity_id == user_id


def test_create_user_accepts_internal_domain_without_dot(client: TestClient) -> None:
    response = post_user(client, email="ops@localhost")

    assert response.status_code == 201


def test_create_user_with_duplicate_normalized_email_returns_409(
    client: TestClient, session: Session
) -> None:
    post_user(client, email="ada@example.com")

    response = post_user(client, email="  ADA@Example.com ")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A user with email 'ada@example.com' already exists."
    }
    assert count(session, User) == 1
    assert count(session, AuditEvent) == 1


def test_create_user_is_committed_by_the_request_transaction(
    client: TestClient, session: Session
) -> None:
    post_user(client)

    # A rollback only discards uncommitted work, so both rows must survive it.
    session.rollback()

    assert count(session, User) == 1
    assert count(session, AuditEvent) == 1


def test_create_user_audit_failure_persists_neither_user_nor_audit_event(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_add = AuditEventRepository.add

    def add_then_fail(self: AuditEventRepository, event: AuditEvent) -> AuditEvent:
        real_add(self, event)
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(AuditEventRepository, "add", add_then_fail)

    # TestClient re-raises unhandled server errors instead of returning a 500.
    with pytest.raises(RuntimeError, match="simulated audit failure"):
        post_user(client)

    assert count(session, User) == 0
    assert count(session, AuditEvent) == 0


def test_create_user_duplicate_email_race_returns_409(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    post_user(client, email="ada@example.com")
    # The pre-check misses the existing user, so the INSERT hits the constraint.
    monkeypatch.setattr(UserRepository, "get_by_email", lambda self, email: None)

    response = post_user(client, email="ada@example.com")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A user with email 'ada@example.com' already exists."
    }
    assert count(session, User) == 1
    assert count(session, AuditEvent) == 1
    # The failed transaction was rolled back, so the next request works.
    assert client.get("/users").status_code == 200


def test_create_user_failed_commit_is_not_reported_as_created(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Own client, so the server error becomes a 500 response instead of being
    # re-raised: a 201 here would mean the commit ran after the response.
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session

    def fail_commit() -> None:
        raise RuntimeError("simulated commit failure")

    monkeypatch.setattr(session, "commit", fail_commit)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = post_user(client)

    assert response.status_code == 500
    assert count(session, User) == 0
    assert count(session, AuditEvent) == 0


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Actor": ""}, {"X-Actor": "   "}, {"X-Actor": "a" * 321}],
    ids=["missing", "empty", "whitespace", "too-long"],
)
def test_create_user_without_valid_actor_returns_422(
    client: TestClient, session: Session, headers: dict[str, str]
) -> None:
    response = client.post(
        "/users", json={"email": "ada@example.com", "name": "Ada"}, headers=headers
    )

    assert response.status_code == 422
    assert count(session, User) == 0
    assert count(session, AuditEvent) == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "Ada"},
        {"email": "ada@example.com"},
        {"email": "", "name": "Ada"},
        {"email": "   ", "name": "Ada"},
        {"email": "no-at-sign", "name": "Ada"},
        {"email": "@example.com", "name": "Ada"},
        {"email": "ada@", "name": "Ada"},
        {"email": "a@b@example.com", "name": "Ada"},
        {"email": "ada lovelace@example.com", "name": "Ada"},
        {"email": "a" * 309 + "@example.com", "name": "Ada"},  # 321 characters
        {"email": "ada@example.com", "name": ""},
        {"email": "ada@example.com", "name": "   "},
        {"email": "ada@example.com", "name": "x" * 201},
    ],
)
def test_create_user_with_invalid_body_returns_422(
    client: TestClient, session: Session, payload: dict[str, Any]
) -> None:
    response = client.post("/users", json=payload, headers=ACTOR_HEADERS)

    assert response.status_code == 422
    assert count(session, User) == 0
    assert count(session, AuditEvent) == 0


def test_create_user_rejected_by_service_after_normalization_returns_422(
    client: TestClient, session: Session
) -> None:
    # Passes the schema (165 characters) but is 325 characters once the
    # service lowercases it, because "İ" lowercases to two characters.
    response = post_user(client, email="İ" * 160 + "@x.io")

    assert response.status_code == 422
    assert response.json() == {"detail": "Email must be at most 320 characters."}
    assert count(session, User) == 0


# --- GET /users ---------------------------------------------------------------


def test_list_users_returns_empty_list(client: TestClient) -> None:
    response = client.get("/users")

    assert response.status_code == 200
    assert response.json() == []


def test_list_users_returns_users_ordered_by_id(client: TestClient) -> None:
    first = post_user(client, email="b@example.com", name="B").json()
    second = post_user(client, email="a@example.com", name="A").json()

    response = client.get("/users")  # no X-Actor needed for reads

    assert response.status_code == 200
    assert response.json() == [first, second]


# --- GET /users/{user_id} -----------------------------------------------------


def test_get_user_returns_user(client: TestClient) -> None:
    created = post_user(client).json()

    response = client.get(f"/users/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_user_returns_404_when_absent(client: TestClient) -> None:
    response = client.get("/users/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "User 999 not found."}


def test_get_user_with_non_integer_id_returns_422(client: TestClient) -> None:
    response = client.get("/users/not-a-number")

    assert response.status_code == 422
