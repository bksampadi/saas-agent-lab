"""AssignmentService called directly, without the HTTP schema in front of it.

Tests play the caller: they commit or roll back, because the service never does.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Assignment, AuditEvent, Licence, User, UserStatus
from app.repositories.assignments import AssignmentRepository
from app.repositories.audit_events import AuditEventRepository
from app.services.assignments import AssignmentService
from app.services.errors import (
    AssignmentAlreadyExists,
    InvalidInput,
    LicenceNotFound,
    NoSeatsAvailable,
    UserInactive,
    UserNotFound,
)

ACTOR = "admin@example.com"


@pytest.fixture
def service(session: Session) -> AssignmentService:
    return AssignmentService(session)


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


def all_assignments(session: Session) -> list[Assignment]:
    return list(session.scalars(select(Assignment).order_by(Assignment.id)))


def all_audit_events(session: Session) -> list[AuditEvent]:
    return list(session.scalars(select(AuditEvent).order_by(AuditEvent.id)))


# --- creation -----------------------------------------------------------------


def test_assign_licence_stores_active_assignment(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    assignment = service.assign_licence(
        user_id=user.id, licence_id=licence.id, actor=ACTOR
    )
    session.expire_all()  # force a reload from the database

    stored = session.get(Assignment, assignment.id)
    assert stored is not None
    assert stored.user_id == user.id
    assert stored.licence_id == licence.id
    assert stored.assigned_at.tzinfo is UTC
    assert stored.revoked_at is None


# --- audit event --------------------------------------------------------------


def test_assign_licence_writes_exactly_the_expected_audit_event(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    assignment = service.assign_licence(
        user_id=user.id, licence_id=licence.id, actor=ACTOR
    )
    session.expire_all()

    events = all_audit_events(session)
    assert len(events) == 1
    event_row = events[0]
    assert event_row.actor == ACTOR
    assert event_row.action == "assignment.create"
    assert event_row.entity_type == "assignment"
    assert event_row.entity_id == assignment.id
    assert event_row.before is None
    assert event_row.after == {"user_id": user.id, "licence_id": licence.id}


# --- transactions: the caller owns them ---------------------------------------


def test_assign_licence_neither_commits_nor_rolls_back(
    service: AssignmentService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    def forbidden() -> None:
        raise AssertionError("the service must not end the caller's transaction")

    monkeypatch.setattr(session, "commit", forbidden)
    monkeypatch.setattr(session, "rollback", forbidden)

    service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)
    with pytest.raises(AssignmentAlreadyExists):
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    # The rejected duplicate did not undo the earlier, still uncommitted work.
    assert session.in_transaction()
    assert len(all_assignments(session)) == 1
    assert len(all_audit_events(session)) == 1


def test_caller_commit_persists_assignment_and_audit_event_together(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    with session.begin():
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    # A rollback only discards uncommitted work, so both rows must survive it.
    session.rollback()

    assert len(all_assignments(session)) == 1
    assert len(all_audit_events(session)) == 1


def test_caller_rollback_discards_assignment_and_audit_event(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    session.rollback()

    assert all_assignments(session) == []
    assert all_audit_events(session) == []


def test_audit_failure_leaves_neither_assignment_nor_audit_event_after_rollback(
    service: AssignmentService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    real_add = AuditEventRepository.add
    rows_before_failure: dict[str, int] = {}

    def add_then_fail(self: AuditEventRepository, event: AuditEvent) -> AuditEvent:
        real_add(self, event)  # the event row is now flushed, not committed
        rows_before_failure["assignments"] = len(all_assignments(session))
        rows_before_failure["audit_events"] = len(all_audit_events(session))
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(AuditEventRepository, "add", add_then_fail)

    # session.begin() plays the caller: it rolls back when the block raises.
    with pytest.raises(RuntimeError, match="simulated audit failure"):
        with session.begin():
            service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert rows_before_failure == {"assignments": 1, "audit_events": 1}
    assert all_assignments(session) == []
    assert all_audit_events(session) == []


# --- missing or inactive user, missing licence --------------------------------


def test_assign_licence_to_missing_user_raises_user_not_found(
    service: AssignmentService, session: Session
) -> None:
    licence = make_licence(session)

    with pytest.raises(UserNotFound):
        service.assign_licence(user_id=999, licence_id=licence.id, actor=ACTOR)

    assert all_assignments(session) == []
    assert all_audit_events(session) == []


def test_assign_licence_to_inactive_user_raises_user_inactive(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session, status=UserStatus.INACTIVE)
    licence = make_licence(session)

    with pytest.raises(UserInactive):
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert all_assignments(session) == []
    assert all_audit_events(session) == []


def test_assign_missing_licence_raises_licence_not_found(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)

    with pytest.raises(LicenceNotFound):
        service.assign_licence(user_id=user.id, licence_id=999, actor=ACTOR)

    assert all_assignments(session) == []
    assert all_audit_events(session) == []


# --- seat capacity ------------------------------------------------------------


def test_zero_seat_licence_cannot_be_assigned(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session, seats=0)

    with pytest.raises(NoSeatsAvailable):
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert all_assignments(session) == []
    assert all_audit_events(session) == []


def test_full_licence_cannot_be_assigned(
    service: AssignmentService, session: Session
) -> None:
    licence = make_licence(session, seats=2)
    make_assignment(session, make_user(session, "a@example.com"), licence)
    make_assignment(session, make_user(session, "b@example.com"), licence)
    user = make_user(session, "c@example.com")

    with pytest.raises(NoSeatsAvailable):
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert len(all_assignments(session)) == 2
    assert all_audit_events(session) == []


def test_last_remaining_seat_can_be_assigned(
    service: AssignmentService, session: Session
) -> None:
    licence = make_licence(session, seats=2)
    make_assignment(session, make_user(session, "a@example.com"), licence)
    user = make_user(session, "b@example.com")

    service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert len(all_assignments(session)) == 2
    assert len(all_audit_events(session)) == 1


def test_revoked_assignments_do_not_use_seats(
    service: AssignmentService, session: Session
) -> None:
    licence = make_licence(session, seats=1)
    make_assignment(session, make_user(session, "a@example.com"), licence, revoked=True)
    user = make_user(session, "b@example.com")

    service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert len(all_assignments(session)) == 2


def test_seats_are_counted_per_licence(
    service: AssignmentService, session: Session
) -> None:
    figma = make_licence(session, "Figma", seats=1)
    slack = make_licence(session, "Slack", seats=1)
    make_assignment(session, make_user(session, "a@example.com"), slack)
    user = make_user(session, "b@example.com")

    service.assign_licence(user_id=user.id, licence_id=figma.id, actor=ACTOR)

    assert len(all_assignments(session)) == 2


# --- duplicate active assignments ---------------------------------------------


def test_duplicate_active_assignment_is_rejected_by_the_pre_check(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    with pytest.raises(AssignmentAlreadyExists) as excinfo:
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert excinfo.value.__cause__ is None  # rejected before any INSERT
    assert len(all_assignments(session)) == 1
    assert len(all_audit_events(session)) == 1


def test_duplicate_is_checked_before_capacity(
    service: AssignmentService, session: Session
) -> None:
    # The user holds the only seat: they are told they have it, not that the
    # licence is full.
    user = make_user(session)
    licence = make_licence(session, seats=1)
    make_assignment(session, user, licence)

    with pytest.raises(AssignmentAlreadyExists):
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)


def test_revoked_assignment_does_not_block_reassignment(
    service: AssignmentService, session: Session
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    make_assignment(session, user, licence, revoked=True)

    service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert len(all_assignments(session)) == 2


def test_duplicate_active_assignment_race_is_reported_as_already_exists(
    service: AssignmentService,
    session: Session,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = make_user(session)
    licence = make_licence(session, seats=5)
    with session.begin():
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    # Simulate a race: the pre-check misses the existing assignment (as if
    # another request committed it just after the check), so the INSERT hits
    # the real partial unique index.
    monkeypatch.setattr(
        AssignmentRepository, "get_active", lambda self, user_id, licence_id: None
    )

    statements: list[str] = []

    def record(*args: Any) -> None:
        statements.append(args[2])  # (conn, cursor, statement, ...)

    def forbidden() -> None:
        raise AssertionError("the service must not end the caller's transaction")

    event.listen(engine, "before_cursor_execute", record)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(session, "commit", forbidden)
            patch.setattr(session, "rollback", forbidden)
            with pytest.raises(AssignmentAlreadyExists) as excinfo:
                service.assign_licence(
                    user_id=user.id, licence_id=licence.id, actor=ACTOR
                )
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert isinstance(excinfo.value.__cause__, IntegrityError)
    # The failed INSERT is the last statement: the session is not used again.
    assert statements[-1].startswith("INSERT INTO assignments")

    session.rollback()  # the caller's job
    assert len(all_assignments(session)) == 1
    assert len(all_audit_events(session)) == 1


def test_other_integrity_error_on_assignment_insert_is_not_a_duplicate(
    service: AssignmentService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)
    real_add = AssignmentRepository.add

    def add_with_missing_user(
        self: AssignmentRepository, assignment: Assignment
    ) -> Assignment:
        # Bypasses the service check: a real FOREIGN KEY violation from SQLite.
        assignment.user_id = 999
        return real_add(self, assignment)

    monkeypatch.setattr(AssignmentRepository, "add", add_with_missing_user)

    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
        with session.begin():
            service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert all_assignments(session) == []
    assert all_audit_events(session) == []


def test_integrity_error_from_audit_insert_is_not_a_duplicate(
    service: AssignmentService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    def fail_with_integrity_error(
        self: AuditEventRepository, event: AuditEvent
    ) -> AuditEvent:
        raise IntegrityError(
            "INSERT INTO audit_events ...", {}, Exception("simulated failure")
        )

    monkeypatch.setattr(AuditEventRepository, "add", fail_with_integrity_error)

    with pytest.raises(IntegrityError):
        with session.begin():
            service.assign_licence(user_id=user.id, licence_id=licence.id, actor=ACTOR)

    assert all_assignments(session) == []
    assert all_audit_events(session) == []


# --- input validation (no HTTP schema involved) -------------------------------


@pytest.mark.parametrize("actor", ["", "   ", "a" * 321])
def test_assign_licence_rejects_invalid_actor(
    service: AssignmentService, session: Session, actor: str
) -> None:
    user = make_user(session)
    licence = make_licence(session)

    with pytest.raises(InvalidInput):
        service.assign_licence(user_id=user.id, licence_id=licence.id, actor=actor)

    assert all_assignments(session) == []


# --- reads --------------------------------------------------------------------


def test_list_active_assignments_excludes_revoked_and_orders_by_id(
    service: AssignmentService, session: Session
) -> None:
    figma = make_licence(session, "Figma")
    slack = make_licence(session, "Slack")
    ada = make_user(session, "ada@example.com")
    bob = make_user(session, "bob@example.com")
    first = make_assignment(session, bob, slack)
    make_assignment(session, ada, figma, revoked=True)
    third = make_assignment(session, ada, slack)
    fourth = make_assignment(session, ada, figma)

    assert [a.id for a in service.list_active_assignments()] == [
        first.id,
        third.id,
        fourth.id,
    ]


def test_list_active_assignments_returns_empty_list(
    service: AssignmentService,
) -> None:
    assert service.list_active_assignments() == []
