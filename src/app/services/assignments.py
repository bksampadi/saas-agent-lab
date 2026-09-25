"""Assignment use cases and business rules.

Services never commit or roll back: the caller owns the transaction.
"""

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Assignment, AuditEvent, UserStatus
from app.models.base import utcnow
from app.repositories.assignments import (
    AssignmentRepository,
    is_duplicate_active_assignment,
)
from app.repositories.audit_events import AuditEventRepository
from app.repositories.licences import LicenceRepository
from app.repositories.users import UserRepository
from app.services.errors import (
    AssignmentAlreadyExists,
    AssignmentAlreadyRevoked,
    AssignmentNotFound,
    LicenceNotFound,
    NoSeatsAvailable,
    UserInactive,
    UserNotFound,
)
from app.services.validation import validated_actor


def _audit_snapshot(assignment: Assignment) -> dict[str, Any]:
    # Business fields only: the id and timestamps have their own columns.
    return {"user_id": assignment.user_id, "licence_id": assignment.licence_id}


def _revocation_snapshot(assignment: Assignment) -> dict[str, Any]:
    # revoked_at is the state being changed, so unlike other timestamps it is
    # part of the snapshot. The JSON column cannot hold a datetime, so it is
    # stored explicitly as ISO 8601 text, e.g. "2026-09-25T21:33:27.100145+00:00".
    revoked_at = assignment.revoked_at
    return {
        **_audit_snapshot(assignment),
        "revoked_at": None if revoked_at is None else revoked_at.isoformat(),
    }


class AssignmentService:
    def __init__(self, session: Session) -> None:
        self._assignments = AssignmentRepository(session)
        self._users = UserRepository(session)
        self._licences = LicenceRepository(session)
        self._audit_events = AuditEventRepository(session)

    def assign_licence(
        self, *, user_id: int, licence_id: int, actor: str
    ) -> Assignment:
        """Assign a licence to an active user, with its audit event, in the
        caller's transaction.

        The seat check is count-then-insert: two concurrent requests can both
        see one free seat and both insert, overfilling the licence. Nothing in
        the database prevents that yet; it needs real coordination (e.g. row
        locking) during production hardening (v0.5). Do not treat this as
        concurrency-safe.
        """
        actor = validated_actor(actor)

        user = self._users.get(user_id)
        if user is None:
            raise UserNotFound(user_id)
        if user.status is not UserStatus.ACTIVE:
            raise UserInactive(user_id)

        licence = self._licences.get(licence_id)
        if licence is None:
            raise LicenceNotFound(licence_id)

        # Checked before capacity: a user who already holds the last seat is
        # told they have it, not that the licence is full.
        if self._assignments.get_active(user_id, licence_id) is not None:
            raise AssignmentAlreadyExists(user_id, licence_id)

        seats_used = self._assignments.count_active_for_licence(licence_id)
        if seats_used >= licence.seats_total:
            raise NoSeatsAvailable(licence_id)

        try:
            assignment = self._assignments.add(
                Assignment(user_id=user_id, licence_id=licence_id)
            )
        except IntegrityError as error:
            # Another request may have assigned the same licence to the same
            # user between our check and our insert. Only that index is
            # reported as a duplicate; any other failure is re-raised
            # unchanged. The session cannot be queried after a failed flush,
            # so the error itself is inspected and the caller rolls back.
            if is_duplicate_active_assignment(error):
                raise AssignmentAlreadyExists(user_id, licence_id) from error
            raise

        self._audit_events.add(
            AuditEvent(
                actor=actor,
                action="assignment.create",
                entity_type="assignment",
                entity_id=assignment.id,
                before=None,
                after=_audit_snapshot(assignment),
            )
        )
        return assignment

    def revoke_assignment(self, *, assignment_id: int, actor: str) -> Assignment:
        """Revoke an active assignment, with its audit event, in the caller's
        transaction. The row is kept as history; once revoked it no longer
        uses a seat.

        Not safe against concurrent revokes: two requests can both read the
        assignment as active, and both would set revoked_at and write an
        audit event (the later timestamp wins on the row). Making the
        transition conditional in the database is deferred to production
        hardening (v0.5).
        """
        actor = validated_actor(actor)

        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise AssignmentNotFound(assignment_id)
        if assignment.revoked_at is not None:
            raise AssignmentAlreadyRevoked(assignment_id)

        before = _revocation_snapshot(assignment)
        self._assignments.mark_revoked(assignment, revoked_at=utcnow())

        self._audit_events.add(
            AuditEvent(
                actor=actor,
                action="assignment.revoke",
                entity_type="assignment",
                entity_id=assignment.id,
                before=before,
                # Built from the same value just written to the row.
                after=_revocation_snapshot(assignment),
            )
        )
        return assignment

    def list_active_assignments(self) -> list[Assignment]:
        return self._assignments.list_active()
