from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, event


@pytest.fixture
def statements(engine: Engine) -> Iterator[list[str]]:
    """Every SQL statement sent to ``engine`` during the test, in order."""
    sent: list[str] = []

    def record(*args: Any) -> None:
        sent.append(args[2])  # (conn, cursor, statement, ...)

    event.listen(engine, "before_cursor_execute", record)
    yield sent
    event.remove(engine, "before_cursor_execute", record)
