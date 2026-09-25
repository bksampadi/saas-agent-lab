from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import get_session
from app.main import create_app
from app.models import Assignment, AuditEvent, Licence, User, UserStatus
from app.repositories.assignments import AssignmentRepository
from app.repositories.audit_events import AuditEventRepository

ACTOR_HEADERS = {"X-Actor": "admin@example.com"}


def make_user(
    session: Session,
    email: str = "ada@example.com",
    status: UserStatus = UserStatus.ACTIVE,
) -> User:
    user = User(email=email, name="Ada", status=status)
    session.add(user)
    session.commit()
    return user


def make_licence(session: Session, product: str = "Figma", seats: int = 5) -> Licence:
    licence = Licence(product=product, seats_total=seats)
    session.add(licence)
    session.commit()
    return licence


def make_assignment(
    session: Session, user: User, licence: Licence, *, revoked: bool = False
) -> Assignment:
    """Seed data written straight to the table, without an audit event."""
    assignment = Assignment(
        user_id=user.id,
        licence_id=licence.id,
        revoked_at=datetime.now(UTC) if revoked else None,
    )
    session.add(assignment)
    session.commit()
    return assignment


def post_assignment(
    client: TestClient, user_id: int, licence_id: int
) -> httpx2.Response:
    return client.post(
        "/assignments",
        json={"user_id": user_id, "licence_id": licence_id},
        headers=ACTOR_HEADERS,
    )


def count(session: Session, model: type[Assignment] | type[AuditEvent]) -> int:
    return len(session.scalars(select(model)).all())


# --- POST /assignments --------------------------------------------------------


def test_create_assignment_returns_201_with_assignment(
    client: TestClient, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "user_id", "licence_id", "assigned_at", "revoked_at"}
    assert isinstance(body["id"], int)
    assert body["user_id"] == user.id
    assert body["licence_id"] == licence.id
    assert datetime.fromisoformat(body["assigned_at"]).utcoffset() is not None
    assert body["revoked_at"] is None


def test_create_assignment_records_audit_event(
    client: TestClient, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    assignment_id = post_assignment(client, user.id, licence.id).json()["id"]

    events = session.scalars(select(AuditEvent)).all()
    assert len(events) == 1
    assert events[0].actor == "admin@example.com"
    assert events[0].action == "assignment.create"
    assert events[0].entity_type == "assignment"
    assert events[0].entity_id == assignment_id
    assert events[0].before is None
    assert events[0].after == {"user_id": user.id, "licence_id": licence.id}


def test_create_assignment_is_committed_by_the_request_transaction(
    client: TestClient, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    post_assignment(client, user.id, licence.id)

    # A rollback only discards uncommitted work, so both rows must survive it.
    session.rollback()

    assert count(session, Assignment) == 1
    assert count(session, AuditEvent) == 1


def test_create_assignment_audit_failure_persists_neither(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    real_add = AuditEventRepository.add

    def add_then_fail(self: AuditEventRepository, event: AuditEvent) -> AuditEvent:
        real_add(self, event)
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(AuditEventRepository, "add", add_then_fail)

    # TestClient re-raises unhandled server errors instead of returning a 500.
    with pytest.raises(RuntimeError, match="simulated audit failure"):
        post_assignment(client, user.id, licence.id)

    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


def test_create_assignment_failed_commit_is_not_reported_as_created(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    # Own client, so the server error becomes a 500 response instead of being
    # re-raised: a 201 here would mean the commit ran after the response.
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session

    def fail_commit() -> None:
        raise RuntimeError("simulated commit failure")

    monkeypatch.setattr(session, "commit", fail_commit)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 500
    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


def test_create_assignment_for_missing_user_returns_404(
    client: TestClient, session: Session
) -> None:
    licence = make_licence(session)

    response = post_assignment(client, 999, licence.id)

    assert response.status_code == 404
    assert response.json() == {"detail": "User 999 not found."}
    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


def test_create_assignment_for_inactive_user_returns_409(
    client: TestClient, session: Session
) -> None:
    user = make_user(session, status=UserStatus.INACTIVE)
    licence = make_licence(session)

    response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 409
    assert response.json() == {"detail": f"User {user.id} is inactive."}
    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


def test_create_assignment_for_missing_licence_returns_404(
    client: TestClient, session: Session
) -> None:
    user = make_user(session)

    response = post_assignment(client, user.id, 999)

    assert response.status_code == 404
    assert response.json() == {"detail": "Licence 999 not found."}
    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


def test_create_assignment_for_zero_seat_licence_returns_409(
    client: TestClient, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session, seats=0)

    response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"Licence {licence.id} has no seats available."
    }
    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


def test_create_assignment_for_full_licence_returns_409(
    client: TestClient, session: Session
) -> None:
    licence = make_licence(session, seats=2)
    make_assignment(session, make_user(session, "a@example.com"), licence)
    make_assignment(session, make_user(session, "b@example.com"), licence)
    user = make_user(session, "c@example.com")

    response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"Licence {licence.id} has no seats available."
    }
    assert count(session, Assignment) == 2
    assert count(session, AuditEvent) == 0


def test_create_assignment_takes_last_remaining_seat(
    client: TestClient, session: Session
) -> None:
    licence = make_licence(session, seats=2)
    make_assignment(session, make_user(session, "a@example.com"), licence)
    user = make_user(session, "b@example.com")

    response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 201
    assert count(session, Assignment) == 2
    assert count(session, AuditEvent) == 1


def test_create_duplicate_active_assignment_returns_409(
    client: TestClient, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    post_assignment(client, user.id, licence.id)

    response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            f"User {user.id} already has an active assignment for licence {licence.id}."
        )
    }
    assert count(session, Assignment) == 1
    assert count(session, AuditEvent) == 1


def test_create_duplicate_active_assignment_race_returns_409(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    post_assignment(client, user.id, licence.id)
    # The pre-check misses the existing assignment, so the INSERT hits the
    # real partial unique index.
    monkeypatch.setattr(
        AssignmentRepository, "get_active", lambda self, user_id, licence_id: None
    )

    response = post_assignment(client, user.id, licence.id)

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            f"User {user.id} already has an active assignment for licence {licence.id}."
        )
    }
    assert count(session, Assignment) == 1
    assert count(session, AuditEvent) == 1
    # The failed transaction was rolled back, so the next request works.
    assert client.get("/assignments").status_code == 200


def test_create_assignment_unrelated_integrity_error_is_not_a_409(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    real_add = AssignmentRepository.add

    def add_with_missing_user(
        self: AssignmentRepository, assignment: Assignment
    ) -> Assignment:
        assignment.user_id = 999  # a real FOREIGN KEY violation from SQLite
        return real_add(self, assignment)

    monkeypatch.setattr(AssignmentRepository, "add", add_with_missing_user)

    # Propagates as an unhandled server error, not a duplicate-assignment 409.
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
        post_assignment(client, user.id, licence.id)

    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Actor": ""}, {"X-Actor": "   "}, {"X-Actor": "a" * 321}],
    ids=["missing", "empty", "whitespace", "too-long"],
)
def test_create_assignment_without_valid_actor_returns_422(
    client: TestClient, session: Session, headers: dict[str, str]
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    response = client.post(
        "/assignments",
        json={"user_id": user.id, "licence_id": licence.id},
        headers=headers,
    )

    assert response.status_code == 422
    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"licence_id": 1},
        {"user_id": 1},
        {"user_id": 0, "licence_id": 1},
        {"user_id": 1, "licence_id": -1},
        {"user_id": 2_147_483_648, "licence_id": 1},
        {"user_id": 1.5, "licence_id": 1},
        {"user_id": "one", "licence_id": 1},
        {"user_id": 1, "licence_id": None},
    ],
)
def test_create_assignment_with_invalid_body_returns_422(
    client: TestClient, session: Session, payload: dict[str, Any]
) -> None:
    make_user(session)
    make_licence(session)

    response = client.post("/assignments", json=payload, headers=ACTOR_HEADERS)

    assert response.status_code == 422
    assert count(session, Assignment) == 0
    assert count(session, AuditEvent) == 0


# --- GET /assignments ---------------------------------------------------------


def test_list_assignments_returns_empty_list(client: TestClient) -> None:
    response = client.get("/assignments")

    assert response.status_code == 200
    assert response.json() == []


def test_list_assignments_returns_active_assignments_only(
    client: TestClient, session: Session
) -> None:
    licence = make_licence(session)
    ada = make_user(session, "ada@example.com")
    bob = make_user(session, "bob@example.com")
    make_assignment(session, ada, licence, revoked=True)
    active = post_assignment(client, bob.id, licence.id).json()

    response = client.get("/assignments")  # no X-Actor needed for reads

    assert response.status_code == 200
    assert response.json() == [active]


def test_list_assignments_is_ordered_by_id(
    client: TestClient, session: Session
) -> None:
    figma = make_licence(session, "Figma")
    slack = make_licence(session, "Slack")
    ada = make_user(session, "ada@example.com")
    bob = make_user(session, "bob@example.com")
    # Created in an order that differs from user and licence id order.
    first = post_assignment(client, bob.id, slack.id).json()
    second = post_assignment(client, ada.id, slack.id).json()
    third = post_assignment(client, ada.id, figma.id).json()

    response = client.get("/assignments")

    assert response.status_code == 200
    assert response.json() == [first, second, third]
    assert [a["id"] for a in response.json()] == sorted(
        [first["id"], second["id"], third["id"]]
    )
