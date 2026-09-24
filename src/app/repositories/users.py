from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import User


def is_duplicate_email(error: IntegrityError) -> bool:
    """True only if ``error`` is the unique constraint on users.email.

    SQLite reports the violated column, not the constraint name, so this
    matches SQLite's exact message. Postgres will need its own check.
    """
    return str(error.orig) == "UNIQUE constraint failed: users.email"


class UserRepository:
    """Persistence for users. Never commits or rolls back."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, user: User) -> User:
        self._session.add(user)
        self._session.flush()  # sends the INSERT so user.id is assigned
        return user

    def get(self, user_id: int) -> User | None:
        return self._session.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        """Expects an already-normalized email."""
        statement = select(User).where(User.email == email)
        return self._session.scalars(statement).one_or_none()

    def list_all(self) -> list[User]:
        statement = select(User).order_by(User.id)
        return list(self._session.scalars(statement))
