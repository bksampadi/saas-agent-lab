from datetime import datetime

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UTCDateTime, utcnow


class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (
        # At most one *active* assignment per user and licence. Revoked rows
        # are kept as history, so the uniqueness only applies while
        # revoked_at is NULL.
        Index(
            "uq_assignments_active_user_licence",
            "user_id",
            "licence_id",
            unique=True,
            sqlite_where=text("revoked_at IS NULL"),
            postgresql_where=text("revoked_at IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    licence_id: Mapped[int] = mapped_column(ForeignKey("licences.id"), index=True)
    assigned_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
