from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import get_session
from app.main import create_app
from app.models import AuditEvent, Licence
from app.repositories.audit_events import AuditEventRepository
from app.repositories.licences import LicenceRepository

ACTOR_HEADERS = {"X-Actor": "admin@example.com"}


def post_licence(
    client: TestClient, product: str = "Figma", seats_total: int = 5
) -> httpx2.Response:
    return client.post(
        "/licences",
        json={"product": product, "seats_total": seats_total},
        headers=ACTOR_HEADERS,
    )


def count(session: Session, model: type[Licence] | type[AuditEvent]) -> int:
    return len(session.scalars(select(model)).all())


# --- POST /licences -----------------------------------------------------------


def test_create_licence_returns_201_with_trimmed_product(client: TestClient) -> None:
    response = post_licence(client, product="  Figma ", seats_total=5)

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "product", "seats_total"}
    assert isinstance(body["id"], int)
    assert body["product"] == "Figma"
    assert body["seats_total"] == 5


def test_create_licence_accepts_zero_seats(client: TestClient) -> None:
    response = post_licence(client, seats_total=0)

    assert response.status_code == 201
    assert response.json()["seats_total"] == 0


def test_create_licence_records_audit_event(
    client: TestClient, session: Session
) -> None:
    licence_id = post_licence(client).json()["id"]

    events = session.scalars(select(AuditEvent)).all()
    assert len(events) == 1
    assert events[0].actor == "admin@example.com"
    assert events[0].action == "licence.create"
    assert events[0].entity_type == "licence"
    assert events[0].entity_id == licence_id
    assert events[0].after == {"product": "Figma", "seats_total": 5}


def test_create_licence_is_committed_by_the_request_transaction(
    client: TestClient, session: Session
) -> None:
    post_licence(client)

    # A rollback only discards uncommitted work, so both rows must survive it.
    session.rollback()

    assert count(session, Licence) == 1
    assert count(session, AuditEvent) == 1


def test_create_licence_audit_failure_persists_neither_licence_nor_audit_event(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_add = AuditEventRepository.add

    def add_then_fail(self: AuditEventRepository, event: AuditEvent) -> AuditEvent:
        real_add(self, event)
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(AuditEventRepository, "add", add_then_fail)

    # TestClient re-raises unhandled server errors instead of returning a 500.
    with pytest.raises(RuntimeError, match="simulated audit failure"):
        post_licence(client)

    assert count(session, Licence) == 0
    assert count(session, AuditEvent) == 0


def test_create_licence_failed_commit_is_not_reported_as_created(
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
        response = post_licence(client)

    assert response.status_code == 500
    assert count(session, Licence) == 0
    assert count(session, AuditEvent) == 0


def test_create_licence_with_duplicate_product_returns_409(
    client: TestClient, session: Session
) -> None:
    post_licence(client, product="Figma")

    response = post_licence(client, product=" Figma ")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A licence for product 'Figma' already exists."
    }
    assert count(session, Licence) == 1
    assert count(session, AuditEvent) == 1


def test_create_licence_duplicate_product_race_returns_409(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    post_licence(client, product="Figma")
    # The pre-check misses the existing licence, so the INSERT hits the constraint.
    monkeypatch.setattr(LicenceRepository, "get_by_product", lambda self, product: None)

    response = post_licence(client, product="Figma")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A licence for product 'Figma' already exists."
    }
    assert count(session, Licence) == 1
    assert count(session, AuditEvent) == 1
    # The failed transaction was rolled back, so the next request works.
    assert client.get("/licences").status_code == 200


def test_create_licence_unrelated_integrity_error_is_not_a_409(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_add = LicenceRepository.add

    def add_with_negative_seats(self: LicenceRepository, licence: Licence) -> Licence:
        licence.seats_total = -1  # a real CHECK violation from SQLite
        return real_add(self, licence)

    monkeypatch.setattr(LicenceRepository, "add", add_with_negative_seats)

    # Propagates as an unhandled server error, not a duplicate-product 409.
    with pytest.raises(IntegrityError, match="CHECK constraint failed"):
        post_licence(client)

    assert count(session, Licence) == 0
    assert count(session, AuditEvent) == 0


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Actor": ""}, {"X-Actor": "   "}, {"X-Actor": "a" * 321}],
    ids=["missing", "empty", "whitespace", "too-long"],
)
def test_create_licence_without_valid_actor_returns_422(
    client: TestClient, session: Session, headers: dict[str, str]
) -> None:
    response = client.post(
        "/licences", json={"product": "Figma", "seats_total": 5}, headers=headers
    )

    assert response.status_code == 422
    assert count(session, Licence) == 0
    assert count(session, AuditEvent) == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"seats_total": 5},
        {"product": "Figma"},
        {"product": "", "seats_total": 5},
        {"product": "   ", "seats_total": 5},
        {"product": "x" * 201, "seats_total": 5},
        {"product": "Figma", "seats_total": -1},
        {"product": "Figma", "seats_total": 2_147_483_648},
        {"product": "Figma", "seats_total": 1.5},
        {"product": "Figma", "seats_total": "many"},
        {"product": "Figma", "seats_total": None},
    ],
)
def test_create_licence_with_invalid_body_returns_422(
    client: TestClient, session: Session, payload: dict[str, Any]
) -> None:
    response = client.post("/licences", json=payload, headers=ACTOR_HEADERS)

    assert response.status_code == 422
    assert count(session, Licence) == 0
    assert count(session, AuditEvent) == 0


# --- GET /licences ------------------------------------------------------------


def test_list_licences_returns_empty_list(client: TestClient) -> None:
    response = client.get("/licences")

    assert response.status_code == 200
    assert response.json() == []


def test_list_licences_returns_licences_ordered_by_id(client: TestClient) -> None:
    first = post_licence(client, product="Slack", seats_total=1).json()
    second = post_licence(client, product="Figma", seats_total=2).json()

    response = client.get("/licences")  # no X-Actor needed for reads

    assert response.status_code == 200
    assert response.json() == [first, second]


# --- GET /licences/{licence_id} -----------------------------------------------


def test_get_licence_returns_licence(client: TestClient) -> None:
    created = post_licence(client).json()

    response = client.get(f"/licences/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_licence_returns_404_when_absent(client: TestClient) -> None:
    response = client.get("/licences/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Licence 999 not found."}


def test_get_licence_with_non_integer_id_returns_422(client: TestClient) -> None:
    response = client.get("/licences/not-a-number")

    assert response.status_code == 422
