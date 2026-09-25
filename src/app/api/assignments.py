from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_actor, get_assignment_service
from app.schemas.assignment import AssignmentCreate, AssignmentRead
from app.services.assignments import AssignmentService
from app.services.errors import (
    AssignmentAlreadyExists,
    InvalidInput,
    LicenceNotFound,
    NoSeatsAvailable,
    UserInactive,
    UserNotFound,
)

router = APIRouter(prefix="/assignments", tags=["assignments"])

AssignmentServiceDep = Annotated[AssignmentService, Depends(get_assignment_service)]
ActorDep = Annotated[str, Depends(get_actor)]


@router.post("", response_model=AssignmentRead, status_code=status.HTTP_201_CREATED)
def create_assignment(
    body: AssignmentCreate, service: AssignmentServiceDep, actor: ActorDep
) -> AssignmentRead:
    try:
        assignment = service.assign_licence(
            user_id=body.user_id, licence_id=body.licence_id, actor=actor
        )
    except (UserNotFound, LicenceNotFound) as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (UserInactive, AssignmentAlreadyExists, NoSeatsAvailable) as error:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InvalidInput as error:
        # Reachable only for a whitespace-only actor header.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return AssignmentRead.model_validate(assignment)


@router.get("", response_model=list[AssignmentRead])
def list_assignments(service: AssignmentServiceDep) -> list[AssignmentRead]:
    """Active assignments only (revoked_at IS NULL), ordered by id."""
    return [
        AssignmentRead.model_validate(assignment)
        for assignment in service.list_active_assignments()
    ]
