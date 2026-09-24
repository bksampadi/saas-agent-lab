from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Licence


def is_duplicate_product(error: IntegrityError) -> bool:
    """True only if ``error`` is the unique constraint on licences.product.

    SQLite reports the violated column, not the constraint name, so this
    matches SQLite's exact message. Postgres will need its own check (v0.5).
    """
    return str(error.orig) == "UNIQUE constraint failed: licences.product"


class LicenceRepository:
    """Persistence for licences. Never commits or rolls back."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, licence: Licence) -> Licence:
        self._session.add(licence)
        self._session.flush()  # sends the INSERT so licence.id is assigned
        return licence

    def get(self, licence_id: int) -> Licence | None:
        return self._session.get(Licence, licence_id)

    def get_by_product(self, product: str) -> Licence | None:
        """Expects an already-normalized product name."""
        statement = select(Licence).where(Licence.product == product)
        return self._session.scalars(statement).one_or_none()

    def list_all(self) -> list[Licence]:
        statement = select(Licence).order_by(Licence.id)
        return list(self._session.scalars(statement))
