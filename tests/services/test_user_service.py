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


# --- transactions: the caller owns them ---------------------------------------


def forbid_ending_the_transaction(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> None:
        raise AssertionError("the service must not end the caller's transaction")

    monkeypatch.setattr(session, "commit", forbidden)
    monkeypatch.setattr(session, "rollback", forbidden)


def test_create_user_neither_commits_nor_rolls_back(
    service: UserService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_ending_the_transaction(session, monkeypatch)

    service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)
    with pytest.raises(EmailAlreadyExists):
        service.create_user(email="ADA@example.com", name="Ada", actor=ACTOR)

    # The rejected duplicate did not undo the earlier, still uncommitted work.
    assert session.in_transaction()
    assert [user.email for user in all_users(session)] == ["ada@example.com"]


def test_caller_rollback_discards_several_service_calls(
    service: UserService, session: Session
) -> None:
    service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)
    service.create_user(email="grace@example.com", name="Grace", actor=ACTOR)

    session.rollback()

    assert all_users(session) == []
    assert all_audit_events(session) == []


def test_caller_commit_persists_several_service_calls(
    service: UserService, session: Session
) -> None:
    service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)
    service.create_user(email="grace@example.com", name="Grace", actor=ACTOR)
    session.commit()

    # A rollback only discards uncommitted work, so both survive it.
    session.rollback()

    assert [user.email for user in all_users(session)] == [
        "ada@example.com",
        "grace@example.com",
    ]
    assert len(all_audit_events(session)) == 2


def test_audit_failure_leaves_neither_user_nor_audit_event_after_rollback(
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

    # session.begin() plays the caller: it rolls back when the block raises.
    with pytest.raises(RuntimeError, match="simulated audit failure"):
        with session.begin():
            service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    # Both rows existed inside the transaction...
    assert rows_before_failure == {"users": 1, "audit_events": 1}
    # ...and the caller's rollback removed both.
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
    with session.begin():
        service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    # Simulate a race: the pre-check misses the existing user (as if another
    # request committed it just after the check), so the INSERT hits the
    # real unique constraint.
    monkeypatch.setattr(UserRepository, "get_by_email", lambda self, email: None)

    with pytest.raises(EmailAlreadyExists) as excinfo:
        with session.begin():
            service.create_user(email="Ada@example.com", name="Ada Again", actor=ACTOR)

    assert isinstance(excinfo.value.__cause__, IntegrityError)
    assert [user.name for user in all_users(session)] == ["Ada"]
    assert len(all_audit_events(session)) == 1


def test_other_integrity_error_on_user_insert_is_not_a_duplicate_email(
    service: UserService, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_add = UserRepository.add

    def add_without_name(self: UserRepository, user: User) -> User:
        # Deliberately invalid: a real NOT NULL violation from SQLite.
        user.name = None  # type: ignore[assignment]
        return real_add(self, user)

    monkeypatch.setattr(UserRepository, "add", add_without_name)

    with pytest.raises(IntegrityError, match="NOT NULL constraint failed"):
        with session.begin():
            service.create_user(email="ada@example.com", name="Ada", actor=ACTOR)

    assert all_users(session) == []
    assert all_audit_events(session) == []


def test_integrity_error_from_audit_insert_is_not_a_duplicate_email(
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
        with session.begin():
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
