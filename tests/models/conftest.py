"""Fixtures for the database-constraint tests."""

import pytest
from sqlalchemy.orm import Session

from app.models import Licence, User


@pytest.fixture
def ids(session: Session) -> tuple[int, int]:
    """A user's id and a licence's id, for rows that must reference both."""
    user = User(email="ada@example.com", name="Ada")
    licence = Licence(product="Figma", seats_total=5)
    session.add_all([user, licence])
    session.flush()
    return user.id, licence.id
