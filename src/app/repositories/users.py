from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User


class UserRepository:
    """Persistence for users. Never commits: the service owns the transaction."""

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
