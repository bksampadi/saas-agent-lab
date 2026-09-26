"""Fixtures for the agent layer.

The executor opens its own sessions from ``session_factory``. Tests seed and
inspect the database through short sessions of their own and never hold one
open across an executor call: with StaticPool every session shares one
connection, so an open transaction here would be committed or discarded by
the executor's sessions.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.agent.executor import AgentExecutor


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def executor(session_factory: sessionmaker[Session]) -> AgentExecutor:
    return AgentExecutor(session_factory)


@pytest.fixture
def statements(engine: Engine) -> Iterator[list[str]]:
    """Every SQL statement sent to ``engine`` during the test, in order."""
    sent: list[str] = []

    def record(*args: Any) -> None:
        sent.append(args[2])  # (conn, cursor, statement, ...)

    event.listen(engine, "before_cursor_execute", record)
    yield sent
    event.remove(engine, "before_cursor_execute", record)
