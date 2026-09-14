from sqlalchemy import CheckConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Licence(Base):
    __tablename__ = "licences"
    __table_args__ = (
        CheckConstraint("seats_total >= 0", name="seats_total_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    product: Mapped[str] = mapped_column(String(200), unique=True)
    seats_total: Mapped[int]
