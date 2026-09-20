"""Shared FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.core.database import get_session
from app.services.users import ACTOR_MAX_LENGTH, UserService


def get_user_service(session: Annotated[Session, Depends(get_session)]) -> UserService:
    return UserService(session)


def get_actor(
    x_actor: Annotated[str, Header(min_length=1, max_length=ACTOR_MAX_LENGTH)],
) -> str:
    """Trusted caller-supplied identity for the audit log, NOT authentication."""
    return x_actor
