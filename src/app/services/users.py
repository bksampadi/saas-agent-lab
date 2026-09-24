"""User use cases and business rules.

Services never commit or roll back: the caller owns the transaction, so
several service calls can succeed or fail together.
"""

import re
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AuditEvent, User, UserStatus
from app.repositories.audit_events import AuditEventRepository
from app.repositories.users import UserRepository, is_duplicate_email
from app.services.errors import EmailAlreadyExists, InvalidInput, UserNotFound

# Limits match the column sizes: users.email, users.name, audit_events.actor.
EMAIL_MAX_LENGTH = 320
NAME_MAX_LENGTH = 200
ACTOR_MAX_LENGTH = 320

# Deliberately modest, not RFC validation: exactly one "@", something on each
# side, no whitespace. Internal domains such as "ops@localhost" are allowed.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+$"
_EMAIL_RE = re.compile(EMAIL_PATTERN)


def normalize_email(email: str) -> str:
    """The canonical form every email is stored and looked up in."""
    return email.strip().lower()


def _required_text(value: str, *, field: str, max_length: int) -> str:
    stripped = value.strip()
    if not stripped:
        raise InvalidInput(f"{field} must not be empty.")
    if len(stripped) > max_length:
        raise InvalidInput(f"{field} must be at most {max_length} characters.")
    return stripped


def _validated_email(email: str) -> str:
    # Length is checked after normalizing, because that is what gets stored
    # (lowercasing can change a string's length).
    normalized = _required_text(
        normalize_email(email), field="Email", max_length=EMAIL_MAX_LENGTH
    )
    if _EMAIL_RE.fullmatch(normalized) is None:
        raise InvalidInput("Email must have the form local@domain.")
    return normalized


def _audit_snapshot(user: User) -> dict[str, Any]:
    # Business fields only: the id and timestamps have their own columns.
    return {"email": user.email, "name": user.name, "status": user.status.value}


class UserService:
    def __init__(self, session: Session) -> None:
        self._users = UserRepository(session)
        self._audit_events = AuditEventRepository(session)

    def create_user(self, *, email: str, name: str, actor: str) -> User:
        """Create an active user and its audit event in the caller's transaction."""
        email = _validated_email(email)
        name = _required_text(name, field="Name", max_length=NAME_MAX_LENGTH)
        actor = _required_text(actor, field="Actor", max_length=ACTOR_MAX_LENGTH)

        if self._users.get_by_email(email) is not None:
            raise EmailAlreadyExists(email)

        try:
            user = self._users.add(
                User(email=email, name=name, status=UserStatus.ACTIVE)
            )
        except IntegrityError as error:
            # Another request may have created the same email between our check
            # and our insert. Only that constraint is reported as a duplicate;
            # any other failure is re-raised unchanged. The session cannot be
            # queried after a failed flush, so the error itself is inspected.
            if is_duplicate_email(error):
                raise EmailAlreadyExists(email) from error
            raise

        self._audit_events.add(
            AuditEvent(
                actor=actor,
                action="user.create",
                entity_type="user",
                entity_id=user.id,
                before=None,
                after=_audit_snapshot(user),
            )
        )
        return user

    def list_users(self) -> list[User]:
        return self._users.list_all()

    def get_user(self, user_id: int) -> User:
        user = self._users.get(user_id)
        if user is None:
            raise UserNotFound(user_id)
        return user
