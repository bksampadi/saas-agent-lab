"""Shared FastAPI dependencies."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session, sessionmaker

from app.agent.planner import DecisionPlanner, IntentPlanner
from app.agent.pydantic_ai_decision import PydanticAIDecisionPlanner
from app.agent.pydantic_ai_planner import PydanticAIIntentPlanner
from app.agent.runs import AgentRuns
from app.core.config import Settings, get_settings
from app.core.database import SessionLocal, get_session
from app.services.assignments import AssignmentService
from app.services.errors import InvalidInput
from app.services.licences import LicenceService
from app.services.users import UserService
from app.services.validation import ACTOR_MAX_LENGTH, reject_reserved_actor


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


def get_licence_service(
    session: Annotated[Session, Depends(get_transaction, scope="function")],
) -> LicenceService:
    return LicenceService(session)


def get_assignment_service(
    session: Annotated[Session, Depends(get_transaction, scope="function")],
) -> AssignmentService:
    return AssignmentService(session)


def get_session_factory() -> sessionmaker[Session]:
    """Where an agent run opens its own short sessions.

    Agent runs take no request session and no request transaction: none may
    be open while a model runs, and a run's log must commit on its own
    (app.agent.executor).
    """
    return SessionLocal


def get_agent_runs(
    session_factory: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> AgentRuns:
    return AgentRuns(session_factory)


def get_intent_planner(
    settings: Annotated[Settings, Depends(get_settings)],
) -> IntentPlanner:
    # The model is built at the first request, so a missing provider key
    # fails that run (planner_error), not the application's startup.
    return PydanticAIIntentPlanner(
        settings.planner_model, timeout_seconds=settings.planner_timeout_seconds
    )


def get_decision_planner(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DecisionPlanner:
    return PydanticAIDecisionPlanner(
        settings.planner_model, timeout_seconds=settings.planner_timeout_seconds
    )


def get_actor(
    x_actor: Annotated[str, Header(min_length=1, max_length=ACTOR_MAX_LENGTH)],
) -> str:
    """Trusted caller-supplied identity for the audit log, NOT authentication.

    The "agent:" namespace is refused here, so an HTTP caller cannot write
    audit events that look like an agent run's. The other actor rules stay
    with the services.
    """
    try:
        reject_reserved_actor(x_actor)
    except InvalidInput as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return x_actor
