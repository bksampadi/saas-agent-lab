"""LicenceService called directly, without the HTTP schema in front of it.

Tests play the caller: they commit or roll back, because the service never does.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AuditEvent, Licence
from app.repositories.audit_events import AuditEventRepository
from app.repositories.licences import LicenceRepository
from app.services.errors import InvalidInput, LicenceNotFound, ProductAlreadyExists
from app.services.licences import SEATS_TOTAL_MAX, LicenceService

ACTOR = "admin@example.com"


@pytest.fixture
def service(session: Session) -> LicenceService:
    return LicenceService(session)


def all_licences(session: Session) -> list[Licence]:
    return list(session.scalars(select(Licence).order_by(Licence.id)))


def all_audit_events(session: Session) -> list[AuditEvent]:
    return list(session.scalars(select(AuditEvent).order_by(AuditEvent.id)))


# --- creation -----------------------------------------------------------------


def test_create_licence_stores_trimmed_product_and_seats(
    service: LicenceService, session: Session
) -> None:
    licence = service.create_licence(product="  Figma ", seats_total=5, actor=ACTOR)
    session.expire_all()  # force a reload from the database

    stored = session.get(Licence, licence.id)
    assert stored is not None
    assert stored.product == "Figma"
    assert stored.seats_total == 5


def test_create_licence_keeps_product_case(service: LicenceService) -> None:
    licence = service.create_licence(product="GitHub", seats_total=1, actor=ACTOR)

    assert licence.product == "GitHub"


@pytest.mark.parametrize("seats_total", [0, SEATS_TOTAL_MAX])
def test_create_licence_accepts_seat_bounds(
    service: LicenceService, seats_total: int
) -> None:
    licence = service.create_licence(
        product="Figma", seats_total=seats_total, actor=ACTOR
    )

    assert licence.seats_total == seats_total


# --- audit event --------------------------------------------------------------


def test_create_licence_writes_exactly_the_expected_audit_event(
    service: LicenceService, session: Session
) -> None:
    licence = service.create_licence(product=" Figma ", seats_total=5, actor=ACTOR)
    session.expire_all()

    events = all_audit_events(session)
    assert len(events) == 1
    event = events[0]
    assert event.actor == ACTOR
    assert event.action == "licence.create"
    assert event.entity_type == "licence"
    assert event.entity_id == licence.id
    assert event.before is None
    assert event.after == {"product": "Figma", "seats_total": 5}


# --- transactions: the caller owns them ---------------------------------------


def test_create_licence_neither_commits_nor_rolls_back(
    service: LicenceService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> None:
        raise AssertionError("the service must not end the caller's transaction")

    monkeypatch.setattr(session, "commit", forbidden)
    monkeypatch.setattr(session, "rollback", forbidden)

    service.create_licence(product="Figma", seats_total=5, actor=ACTOR)
    with pytest.raises(ProductAlreadyExists):
        service.create_licence(product="Figma", seats_total=1, actor=ACTOR)

    # The rejected duplicate did not undo the earlier, still uncommitted work.
    assert session.in_transaction()
    assert [licence.product for licence in all_licences(session)] == ["Figma"]


def test_caller_rollback_discards_licence_and_audit_event(
    service: LicenceService, session: Session
) -> None:
    service.create_licence(product="Figma", seats_total=5, actor=ACTOR)

    session.rollback()

    assert all_licences(session) == []
    assert all_audit_events(session) == []


def test_audit_failure_leaves_neither_licence_nor_audit_event_after_rollback(
    service: LicenceService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_add = AuditEventRepository.add
    rows_before_failure: dict[str, int] = {}

    def add_then_fail(self: AuditEventRepository, event: AuditEvent) -> AuditEvent:
        real_add(self, event)  # the event row is now flushed, not committed
        rows_before_failure["licences"] = len(all_licences(session))
        rows_before_failure["audit_events"] = len(all_audit_events(session))
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(AuditEventRepository, "add", add_then_fail)

    # session.begin() plays the caller: it rolls back when the block raises.
    with pytest.raises(RuntimeError, match="simulated audit failure"):
        with session.begin():
            service.create_licence(product="Figma", seats_total=5, actor=ACTOR)

    assert rows_before_failure == {"licences": 1, "audit_events": 1}
    assert all_licences(session) == []
    assert all_audit_events(session) == []


# --- duplicate products -------------------------------------------------------


def test_create_licence_rejects_duplicate_trimmed_product(
    service: LicenceService, session: Session
) -> None:
    service.create_licence(product="Figma", seats_total=5, actor=ACTOR)

    with pytest.raises(ProductAlreadyExists):
        service.create_licence(product="  Figma  ", seats_total=1, actor=ACTOR)

    assert [licence.seats_total for licence in all_licences(session)] == [5]
    assert len(all_audit_events(session)) == 1


def test_duplicate_product_race_is_reported_as_product_already_exists(
    service: LicenceService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    with session.begin():
        service.create_licence(product="Figma", seats_total=5, actor=ACTOR)

    # Simulate a race: the pre-check misses the existing licence (as if another
    # request committed it just after the check), so the INSERT hits the
    # real unique constraint.
    monkeypatch.setattr(LicenceRepository, "get_by_product", lambda self, product: None)

    with pytest.raises(ProductAlreadyExists) as excinfo:
        with session.begin():
            service.create_licence(product="Figma", seats_total=1, actor=ACTOR)

    assert isinstance(excinfo.value.__cause__, IntegrityError)
    assert [licence.seats_total for licence in all_licences(session)] == [5]
    assert len(all_audit_events(session)) == 1


def test_other_integrity_error_on_licence_insert_is_not_a_duplicate_product(
    service: LicenceService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_add = LicenceRepository.add

    def add_with_negative_seats(self: LicenceRepository, licence: Licence) -> Licence:
        # Bypasses the service check: a real CHECK violation from SQLite.
        licence.seats_total = -1
        return real_add(self, licence)

    monkeypatch.setattr(LicenceRepository, "add", add_with_negative_seats)

    with pytest.raises(IntegrityError, match="CHECK constraint failed"):
        with session.begin():
            service.create_licence(product="Figma", seats_total=5, actor=ACTOR)

    assert all_licences(session) == []
    assert all_audit_events(session) == []


def test_integrity_error_from_audit_insert_is_not_a_duplicate_product(
    service: LicenceService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_with_integrity_error(
        self: AuditEventRepository, event: AuditEvent
    ) -> AuditEvent:
        raise IntegrityError(
            "INSERT INTO audit_events ...", {}, Exception("simulated failure")
        )

    monkeypatch.setattr(AuditEventRepository, "add", fail_with_integrity_error)

    with pytest.raises(IntegrityError):
        with session.begin():
            service.create_licence(product="Figma", seats_total=5, actor=ACTOR)

    assert all_licences(session) == []
    assert all_audit_events(session) == []


# --- input validation (no HTTP schema involved) -------------------------------


@pytest.mark.parametrize("product", ["", "   ", "x" * 201])
def test_create_licence_rejects_invalid_product(
    service: LicenceService, session: Session, product: str
) -> None:
    with pytest.raises(InvalidInput):
        service.create_licence(product=product, seats_total=5, actor=ACTOR)

    assert all_licences(session) == []
    assert all_audit_events(session) == []


@pytest.mark.parametrize("seats_total", [-1, SEATS_TOTAL_MAX + 1])
def test_create_licence_rejects_out_of_range_seats(
    service: LicenceService, session: Session, seats_total: int
) -> None:
    with pytest.raises(InvalidInput):
        service.create_licence(product="Figma", seats_total=seats_total, actor=ACTOR)

    assert all_licences(session) == []
    assert all_audit_events(session) == []


@pytest.mark.parametrize("actor", ["", "   ", "a" * 321])
def test_create_licence_rejects_invalid_actor(
    service: LicenceService, session: Session, actor: str
) -> None:
    with pytest.raises(InvalidInput):
        service.create_licence(product="Figma", seats_total=5, actor=actor)

    assert all_licences(session) == []


# --- reads --------------------------------------------------------------------


def test_list_licences_returns_licences_ordered_by_id(
    service: LicenceService,
) -> None:
    service.create_licence(product="Slack", seats_total=1, actor=ACTOR)
    service.create_licence(product="Figma", seats_total=2, actor=ACTOR)

    assert [licence.product for licence in service.list_licences()] == [
        "Slack",
        "Figma",
    ]


def test_get_licence_returns_licence(service: LicenceService) -> None:
    created = service.create_licence(product="Figma", seats_total=5, actor=ACTOR)

    assert service.get_licence(created.id) is created


def test_get_licence_raises_licence_not_found(service: LicenceService) -> None:
    with pytest.raises(LicenceNotFound):
        service.get_licence(999)
