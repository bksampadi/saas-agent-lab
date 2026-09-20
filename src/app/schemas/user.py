from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.models import UserStatus
from app.services.users import EMAIL_MAX_LENGTH, EMAIL_PATTERN, NAME_MAX_LENGTH


class UserCreate(BaseModel):
    """Early, field-level checks; UserService re-enforces them after normalizing."""

    email: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, max_length=EMAIL_MAX_LENGTH, pattern=EMAIL_PATTERN
        ),
    ]
    name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=NAME_MAX_LENGTH
        ),
    ]


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    status: UserStatus
    created_at: datetime
