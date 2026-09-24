from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_actor, get_licence_service
from app.schemas.licence import LicenceCreate, LicenceRead
from app.services.errors import InvalidInput, LicenceNotFound, ProductAlreadyExists
from app.services.licences import LicenceService

router = APIRouter(prefix="/licences", tags=["licences"])

LicenceServiceDep = Annotated[LicenceService, Depends(get_licence_service)]
ActorDep = Annotated[str, Depends(get_actor)]


@router.post("", response_model=LicenceRead, status_code=status.HTTP_201_CREATED)
def create_licence(
    body: LicenceCreate, service: LicenceServiceDep, actor: ActorDep
) -> LicenceRead:
    try:
        licence = service.create_licence(
            product=body.product, seats_total=body.seats_total, actor=actor
        )
    except ProductAlreadyExists as error:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InvalidInput as error:
        # Reachable only for a whitespace-only actor header.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return LicenceRead.model_validate(licence)


@router.get("", response_model=list[LicenceRead])
def list_licences(service: LicenceServiceDep) -> list[LicenceRead]:
    return [LicenceRead.model_validate(licence) for licence in service.list_licences()]


@router.get("/{licence_id}", response_model=LicenceRead)
def get_licence(licence_id: int, service: LicenceServiceDep) -> LicenceRead:
    try:
        licence = service.get_licence(licence_id)
    except LicenceNotFound as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return LicenceRead.model_validate(licence)
