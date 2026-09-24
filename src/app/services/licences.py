"""Licence use cases and business rules.

Services never commit or roll back: the caller owns the transaction.
"""

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AuditEvent, Licence
from app.repositories.audit_events import AuditEventRepository
from app.repositories.licences import LicenceRepository, is_duplicate_product
from app.services.errors import InvalidInput, LicenceNotFound, ProductAlreadyExists
from app.services.validation import required_text, validated_actor

# Matches the licences.product column size.
PRODUCT_MAX_LENGTH = 200
# The largest value a 32-bit INTEGER column holds (Postgres "integer"). Larger
# numbers would fail inside the database driver instead of as a clean 422.
SEATS_TOTAL_MAX = 2_147_483_647


def _validated_seats_total(seats_total: int) -> int:
    if seats_total < 0:
        raise InvalidInput("Seats total must not be negative.")
    if seats_total > SEATS_TOTAL_MAX:
        raise InvalidInput(f"Seats total must be at most {SEATS_TOTAL_MAX}.")
    return seats_total


def _audit_snapshot(licence: Licence) -> dict[str, Any]:
    # Business fields only: the id has its own column.
    return {"product": licence.product, "seats_total": licence.seats_total}


class LicenceService:
    def __init__(self, session: Session) -> None:
        self._licences = LicenceRepository(session)
        self._audit_events = AuditEventRepository(session)

    def create_licence(self, *, product: str, seats_total: int, actor: str) -> Licence:
        """Create a licence and its audit event in the caller's transaction."""
        # Trimmed but case kept: product names are display names, and the
        # unique constraint in the database is case-sensitive.
        product = required_text(product, field="Product", max_length=PRODUCT_MAX_LENGTH)
        seats_total = _validated_seats_total(seats_total)
        actor = validated_actor(actor)

        if self._licences.get_by_product(product) is not None:
            raise ProductAlreadyExists(product)

        try:
            licence = self._licences.add(
                Licence(product=product, seats_total=seats_total)
            )
        except IntegrityError as error:
            # Another request may have created the same product between our
            # check and our insert. Only that constraint is reported as a
            # duplicate; any other failure is re-raised unchanged.
            if is_duplicate_product(error):
                raise ProductAlreadyExists(product) from error
            raise

        self._audit_events.add(
            AuditEvent(
                actor=actor,
                action="licence.create",
                entity_type="licence",
                entity_id=licence.id,
                before=None,
                after=_audit_snapshot(licence),
            )
        )
        return licence

    def list_licences(self) -> list[Licence]:
        return self._licences.list_all()

    def get_licence(self, licence_id: int) -> Licence:
        licence = self._licences.get(licence_id)
        if licence is None:
            raise LicenceNotFound(licence_id)
        return licence
