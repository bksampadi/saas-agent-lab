"""ORM models.

Every model is imported here so that importing ``app.models`` registers all
tables on ``Base.metadata`` (Alembic and the test fixtures rely on this).
"""

from app.models.assignment import Assignment
from app.models.audit_event import AuditEvent
from app.models.base import Base
from app.models.licence import Licence
from app.models.user import User, UserStatus

__all__ = ["Assignment", "AuditEvent", "Base", "Licence", "User", "UserStatus"]
