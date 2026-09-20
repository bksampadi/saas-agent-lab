from sqlalchemy.orm import Session

from app.models import AuditEvent


class AuditEventRepository:
    """Persistence for audit events. Never commits: the service commits each
    event in the same transaction as the change it describes."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, event: AuditEvent) -> AuditEvent:
        self._session.add(event)
        self._session.flush()
        return event
