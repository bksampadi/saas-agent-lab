from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_actor, get_user_service
from app.schemas.user import UserCreate, UserRead
from app.services.errors import EmailAlreadyExists, InvalidInput, UserNotFound
from app.services.users import UserService

router = APIRouter(prefix="/users", tags=["users"])

UserServiceDep = Annotated[UserService, Depends(get_user_service)]
ActorDep = Annotated[str, Depends(get_actor)]


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreate, service: UserServiceDep, actor: ActorDep) -> UserRead:
    try:
        user = service.create_user(email=body.email, name=body.name, actor=actor)
    except EmailAlreadyExists as error:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(error)) from error
    except InvalidInput as error:
        # Reachable for input that only breaks a rule once normalized.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return UserRead.model_validate(user)


@router.get("", response_model=list[UserRead])
def list_users(service: UserServiceDep) -> list[UserRead]:
    return [UserRead.model_validate(user) for user in service.list_users()]


@router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: int, service: UserServiceDep) -> UserRead:
    try:
        user = service.get_user(user_id)
    except UserNotFound as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return UserRead.model_validate(user)
