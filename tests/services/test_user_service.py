"""UserService called directly, without the HTTP schema in front of it."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AuditEvent, User, UserStatus
from app.repositories.audit_events import AuditEventRepository
from app.repositories.users import UserRepository
from app.services.errors import EmailAlreadyExists, InvalidInput, UserNotFound
from app.services.users import UserService, normalize_email

ACTOR = "admin@example.com"


@pytest.fixture
def service(session: Session) -> UserService:
    return UserService(session)


def all_users(session: Session) -> list[User]:
    return list(session.scalars(select(User).order_by(User.id)))


def all_audit_events(session: Session) -> list[AuditEvent]:
    return list(session.scalars(select(AuditEvent).order_by(AuditEvent.id)))


# --- normalization ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ada@example.com", "ada@example.com"),
        ("Ada@Example.COM", "ada@example.com"),
        ("  ada@example.com\n", "ada@example.com"),
    ],
)
def test_normalize_email_strips_and_lowercases(raw: str, expected: str) -> None:
    assert normalize_email(raw) == expected


def test_create_user_stores_normalized_email_and_is_active(
    service: UserService, session: Session
) -> None:
    user = service.create_user(
        email="  Ada@Example.COM ", name="  Ada Lovelace ", actor=ACTOR
    )
    session.expire_all()  # force a reload from the database

    stored = session.get(User, user.id)
    assert stored is not None
    assert stored.email == "ada@example.com"
    assert stored.name == "Ada Lovelace"
    assert stored.status is UserStatus.ACTIVE


def test_create_user_accepts_internal_domain_without_dot(service: UserService) -> None:
    user = service.create_user(email="ops@localhost", name="Ops", actor=ACTOR)

    assert user.email == "ops@localhost"


# --- audit event --------------------------------------------------------------


def test_create_user_writes_exactly_the_expected_audit_event(
    service: UserService, session: Session
) -> None:
    user = service.create_user(
        email="Ada@Example.com", name="Ada Lovelace", actor=ACTOR
    )
    session.expire_all()

    events = all_audit_events(session)
    assert len(events) == 1
    event = events[0]
    assert event.actor == ACTOR
    assert event.action == "user.create"
    assert event.entity_type == "user"
    assert event.entity_id == user.id
    assert event.before is None
    assert event.after == {
        "email": "ada@example.com",
        "name": "Ada Lovelace",
        "status": "active",
    }


# --- transactions -------------------------------------------------------------


def test_create_user_commits_user_and_audit_event(
    service: UserService, session: Session
) -> None:
    service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    # A rollback discards anything not yet committed.
    session.rollback()

    assert [user.email for user in all_users(session)] == ["ada@example.com"]
    assert len(all_audit_events(session)) == 1


def test_audit_failure_rolls_back_user_and_leaves_no_partial_audit_event(
    service: UserService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_add = AuditEventRepository.add
    rows_before_failure: dict[str, int] = {}

    def add_then_fail(self: AuditEventRepository, event: AuditEvent) -> AuditEvent:
        real_add(self, event)  # the event row is now flushed, not committed
        rows_before_failure["users"] = len(all_users(session))
        rows_before_failure["audit_events"] = len(all_audit_events(session))
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(AuditEventRepository, "add", add_then_fail)

    with pytest.raises(RuntimeError, match="simulated audit failure"):
        service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    # Both rows existed inside the transaction...
    assert rows_before_failure == {"users": 1, "audit_events": 1}
    # ...and the rollback removed both.
    assert all_users(session) == []
    assert all_audit_events(session) == []


# --- duplicate emails ---------------------------------------------------------


def test_create_user_rejects_duplicate_normalized_email(
    service: UserService, session: Session
) -> None:
    service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    with pytest.raises(EmailAlreadyExists):
        service.create_user(email="  ADA@Example.com ", name="Other", actor=ACTOR)

    assert [user.name for user in all_users(session)] == ["Ada"]
    assert len(all_audit_events(session)) == 1


def test_duplicate_email_race_is_reported_as_email_already_exists(
    service: UserService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    # Simulate a race: the pre-check misses the existing user (as if another
    # request committed it just after the check), so the INSERT hits the
    # unique constraint. Later lookups see the real data again.
    real_get_by_email = UserRepository.get_by_email
    calls = 0

    def miss_first_lookup(self: UserRepository, email: str) -> User | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return real_get_by_email(self, email)

    monkeypatch.setattr(UserRepository, "get_by_email", miss_first_lookup)

    with pytest.raises(EmailAlreadyExists) as excinfo:
        service.create_user(email="Ada@example.com", name="Ada Again", actor=ACTOR)

    assert isinstance(excinfo.value.__cause__, IntegrityError)
    assert [user.name for user in all_users(session)] == ["Ada"]
    assert len(all_audit_events(session)) == 1


def test_unrelated_integrity_error_is_not_reported_as_duplicate_email(
    service: UserService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_with_integrity_error(
        self: AuditEventRepository, event: AuditEvent
    ) -> AuditEvent:
        raise IntegrityError(
            "INSERT INTO audit_events ...", {}, Exception("simulated failure")
        )

    monkeypatch.setattr(AuditEventRepository, "add", fail_with_integrity_error)

    with pytest.raises(IntegrityError):
        service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    assert all_users(session) == []
    assert all_audit_events(session) == []


# --- input validation (no HTTP schema involved) -------------------------------


@pytest.mark.parametrize(
    "email",
    [
        "",
        "   ",
        "no-at-sign",
        "@example.com",
        "ada@",
        "a@b@example.com",
        "ada lovelace@example.com",
        "a" * 309 + "@example.com",  # 321 characters
    ],
)
def test_create_user_rejects_invalid_email(
    service: UserService, session: Session, email: str
) -> None:
    with pytest.raises(InvalidInput):
        service.create_user(email=email, name="Ada", actor=ACTOR)

    assert all_users(session) == []
    assert all_audit_events(session) == []


def test_email_length_is_checked_after_normalization(service: UserService) -> None:
    # "İ" (capital I with dot) lowercases to two characters: 165
    # characters as given, 325 once normalized.
    email = "İ" * 160 + "@x.io"

    with pytest.raises(InvalidInput, match="at most 320 characters"):
        service.create_user(email=email, name="Ada", actor=ACTOR)


@pytest.mark.parametrize("name", ["", "   ", "x" * 201])
def test_create_user_rejects_invalid_name(
    service: UserService, session: Session, name: str
) -> None:
    with pytest.raises(InvalidInput):
        service.create_user(email="ada@example.com", name=name, actor=ACTOR)

    assert all_users(session) == []


@pytest.mark.parametrize("actor", ["", "   ", "a" * 321])
def test_create_user_rejects_invalid_actor(
    service: UserService, session: Session, actor: str
) -> None:
    with pytest.raises(InvalidInput):
        service.create_user(email="ada@example.com", name="Ada", actor=actor)

    assert all_users(session) == []


# --- reads --------------------------------------------------------------------


def test_list_users_returns_users_ordered_by_id(service: UserService) -> None:
    service.create_user(email="b@example.com", name="B", actor=ACTOR)
    service.create_user(email="a@example.com", name="A", actor=ACTOR)

    assert [user.email for user in service.list_users()] == [
        "b@example.com",
        "a@example.com",
    ]


def test_get_user_raises_user_not_found(service: UserService) -> None:
    with pytest.raises(UserNotFound):
        service.get_user(999)
