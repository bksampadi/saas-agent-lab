from enum import StrEnum

from sqlalchemy import CheckConstraint, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, enum_values


class PolicyDecision(StrEnum):
    """What policy says about a mutation an agent run wants to make."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"  # wait for a person to approve it
    DENY = "deny"


class Licence(Base):
    __tablename__ = "licences"
    __table_args__ = (
        CheckConstraint("seats_total >= 0", name="seats_total_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    product: Mapped[str] = mapped_column(String(200), unique=True)
    seats_total: Mapped[int]
    # The decision for an agent run that assigns this licence. A person
    # assigning it through the API is not subject to it. The server default
    # gives rows written without it (existing ones, at migration) "allow".
    agent_policy: Mapped[PolicyDecision] = mapped_column(
        Enum(
            PolicyDecision,
            name="policy_decision",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
            length=32,
        ),
        default=PolicyDecision.ALLOW,
        server_default=PolicyDecision.ALLOW.value,
    )
