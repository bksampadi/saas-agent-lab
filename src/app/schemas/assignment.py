from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Ids are positive and fit a 32-bit INTEGER column (Postgres "integer"). Larger
# numbers would fail inside the database driver instead of as a clean 422.
ID_MAX = 2_147_483_647

EntityId = Annotated[int, Field(ge=1, le=ID_MAX)]


class AssignmentCreate(BaseModel):
    user_id: EntityId
    licence_id: EntityId


class AssignmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    licence_id: int
    assigned_at: datetime
    revoked_at: datetime | None
