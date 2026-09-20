"""Domain errors raised by services; the API layer maps them to status codes."""


class DomainError(Exception):
    """Base class for all domain errors."""


class InvalidInput(DomainError):
    """Input breaks a domain invariant, e.g. an empty name or malformed email."""


class EmailAlreadyExists(DomainError):
    def __init__(self, email: str) -> None:
        super().__init__(f"A user with email {email!r} already exists.")
        self.email = email


class UserNotFound(DomainError):
    def __init__(self, user_id: int) -> None:
        super().__init__(f"User {user_id} not found.")
        self.user_id = user_id
