from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.services.licences import PRODUCT_MAX_LENGTH, SEATS_TOTAL_MAX


class LicenceCreate(BaseModel):
    """Early, field-level checks; LicenceService re-enforces them."""

    product: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=PRODUCT_MAX_LENGTH
        ),
    ]
    seats_total: Annotated[int, Field(ge=0, le=SEATS_TOTAL_MAX)]


class LicenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product: str
    seats_total: int
