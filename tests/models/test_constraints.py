"""Database-level guarantees: these must hold even if a service has a bug."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from app.models import (
    Assignment,
    AuditEvent,
    Licence,
    PolicyDecision,
    User,
    UserStatus,
)


def make_user(session: Session, email: str = "ada@example.com") -> User:
    user = User(email=email, name="Ada Lovelace")
    session.add(user)
    session.flush()
    return user


def make_licence(session: Session, product: str = "Figma", seats: int = 5) -> Licence:
    licence = Licence(product=product, seats_total=seats)
    session.add(licence)
    session.flush()
    return licence


# --- User -------------------------------------------------------------------


def test_user_defaults_to_active_and_stores_enum_value(session: Session) -> None:
    user = make_user(session)

    stored = session.execute(
        text("SELECT status FROM users WHERE id = :id"), {"id": user.id}
    ).scalar_one()

    assert user.status is UserStatus.ACTIVE
    assert stored == "active"


def test_user_email_is_unique(session: Session) -> None:
    make_user(session, email="ada@example.com")

    with pytest.raises(IntegrityError):
        make_user(session, email="ada@example.com")


def test_user_status_rejects_unknown_value(session: Session) -> None:
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                "INSERT INTO users (email, name, status, created_at) "
                "VALUES ('x@example.com', 'X', 'suspended', '2026-01-01')"
            )
        )


def test_user_created_at_round_trips_as_utc(session: Session) -> None:
    user = make_user(session)
    session.expire_all()

    reloaded = session.get(User, user.id)

    assert reloaded is not None
    assert reloaded.created_at.tzinfo is UTC


def test_naive_datetime_is_rejected(session: Session) -> None:
    session.add(
        User(email="n@example.com", name="Naive", created_at=datetime(2026, 1, 1))
    )

    # SQLAlchemy wraps the ValueError raised by UTCDateTime in a StatementError.
    with pytest.raises(StatementError, match="Naive datetimes"):
        session.flush()


# --- Licence ----------------------------------------------------------------


def test_licence_product_is_unique(session: Session) -> None:
    make_licence(session, product="Figma")

    with pytest.raises(IntegrityError):
        make_licence(session, product="Figma")


def test_licence_seats_total_cannot_be_negative(session: Session) -> None:
    with pytest.raises(IntegrityError):
        make_licence(session, seats=-1)


def test_licence_agent_policy_defaults_to_allow_and_stores_enum_value(
    session: Session,
) -> None:
    licence = make_licence(session)

    stored = session.execute(
        text("SELECT agent_policy FROM licences WHERE id = :id"), {"id": licence.id}
    ).scalar_one()

    assert licence.agent_policy is PolicyDecision.ALLOW
    assert stored == "allow"


def test_licence_agent_policy_defaults_to_allow_in_the_database(
    session: Session,
) -> None:
    # A row written without the column, as the migration's existing rows are.
    session.execute(
        text("INSERT INTO licences (product, seats_total) VALUES ('Slack', 1)")
    )

    stored = session.execute(
        text("SELECT agent_policy FROM licences WHERE product = 'Slack'")
    ).scalar_one()

    assert stored == "allow"


@pytest.mark.parametrize("policy", ["sometimes", "ALLOW", None])
def test_licence_agent_policy_rejects_unknown_or_missing_value(
    session: Session, policy: str | None
) -> None:
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                "INSERT INTO licences (product, seats_total, agent_policy) "
                "VALUES ('Slack', 1, :policy)"
            ),
            {"policy": policy},
        )


# --- Assignment -------------------------------------------------------------


def test_assignment_requires_existing_user(session: Session) -> None:
    licence = make_licence(session)
    session.add(Assignment(user_id=999, licence_id=licence.id))

    with pytest.raises(IntegrityError):
        session.flush()


def test_only_one_active_assignment_per_user_and_licence(session: Session) -> None:
    user = make_user(session)
    licence = make_licence(session)
    session.add(Assignment(user_id=user.id, licence_id=licence.id))
    session.flush()

    session.add(Assignment(user_id=user.id, licence_id=licence.id))
    with pytest.raises(IntegrityError):
        session.flush()


def test_revoked_assignment_does_not_block_reassignment(session: Session) -> None:
    user = make_user(session)
    licence = make_licence(session)
    session.add(
        Assignment(user_id=user.id, licence_id=licence.id, revoked_at=datetime.now(UTC))
    )
    session.flush()

    session.add(Assignment(user_id=user.id, licence_id=licence.id))
    session.flush()  # must not raise


# --- AuditEvent -------------------------------------------------------------


def test_audit_event_round_trips_json_snapshots(session: Session) -> None:
    event = AuditEvent(
        actor="admin@example.com",
        action="user.deactivate",
        entity_type="user",
        entity_id=1,
        before={"status": "active"},
        after={"status": "inactive"},
    )
    session.add(event)
    session.flush()
    session.expire_all()

    reloaded = session.get(AuditEvent, event.id)

    assert reloaded is not None
    assert reloaded.before == {"status": "active"}
    assert reloaded.after == {"status": "inactive"}
    assert reloaded.created_at.tzinfo is UTC
