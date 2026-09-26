from datetime import datetime
from enum import StrEnum

from sqlalchemy import Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UTCDateTime, enum_values, utcnow


class UserStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Emails are lowercased by the service layer before they reach this column.
    email: Mapped[str] = mapped_column(String(320), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[UserStatus] = mapped_column(
        # Stored as a string plus a CHECK constraint rather than a native enum
        # type, so moving to Postgres needs no type migration.
        Enum(
            UserStatus,
            name="user_status",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=16,
        ),
        default=UserStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
