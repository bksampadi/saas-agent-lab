from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Assignment


def is_duplicate_active_assignment(error: IntegrityError) -> bool:
    """True only if ``error`` is the partial unique index
    ``uq_assignments_active_user_licence`` (one active assignment per user and
    licence).

    SQLite reports the violated columns, not the index name. That index is the
    only unique constraint on exactly these columns, so this exact message
    identifies it. Postgres reports the index name and will need its own check
    (v0.5).
    """
    return (
        str(error.orig)
        == "UNIQUE constraint failed: assignments.user_id, assignments.licence_id"
    )


class AssignmentRepository:
    """Persistence for assignments. Never commits or rolls back.

    "Active" means ``revoked_at IS NULL``.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, assignment: Assignment) -> Assignment:
        self._session.add(assignment)
        self._session.flush()  # sends the INSERT so assignment.id is assigned
        return assignment

    def get_active(self, user_id: int, licence_id: int) -> Assignment | None:
        statement = select(Assignment).where(
            Assignment.user_id == user_id,
            Assignment.licence_id == licence_id,
            Assignment.revoked_at.is_(None),
        )
        return self._session.scalars(statement).one_or_none()

    def count_active_for_licence(self, licence_id: int) -> int:
        statement = select(func.count(Assignment.id)).where(
            Assignment.licence_id == licence_id,
            Assignment.revoked_at.is_(None),
        )
        return self._session.scalar(statement) or 0

    def list_active(self) -> list[Assignment]:
        statement = (
            select(Assignment)
            .where(Assignment.revoked_at.is_(None))
            .order_by(Assignment.id)
        )
        return list(self._session.scalars(statement))
