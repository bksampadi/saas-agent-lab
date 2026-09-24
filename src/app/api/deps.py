"""Shared FastAPI dependencies."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.core.database import get_session
from app.services.users import ACTOR_MAX_LENGTH, UserService


def get_transaction(
    session: Annotated[Session, Depends(get_session)],
) -> Iterator[Session]:
    """The request's transaction: commit if the endpoint returns, roll back if
    it raises (including an HTTPException for a domain error)."""
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise


def get_user_service(
    # scope="function" ends the transaction before the response is sent. With
    # the default "request" scope the commit would run after the client had
    # already received a success response, so a failed commit would go unseen.
    session: Annotated[Session, Depends(get_transaction, scope="function")],
) -> UserService:
    return UserService(session)


def get_actor(
    x_actor: Annotated[str, Header(min_length=1, max_length=ACTOR_MAX_LENGTH)],
) -> str:
    """Trusted caller-supplied identity for the audit log, NOT authentication."""
    return x_actor
