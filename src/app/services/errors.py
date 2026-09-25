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


class ProductAlreadyExists(DomainError):
    def __init__(self, product: str) -> None:
        super().__init__(f"A licence for product {product!r} already exists.")
        self.product = product


class LicenceNotFound(DomainError):
    def __init__(self, licence_id: int) -> None:
        super().__init__(f"Licence {licence_id} not found.")
        self.licence_id = licence_id


class UserInactive(DomainError):
    def __init__(self, user_id: int) -> None:
        super().__init__(f"User {user_id} is inactive.")
        self.user_id = user_id


class AssignmentAlreadyExists(DomainError):
    def __init__(self, user_id: int, licence_id: int) -> None:
        super().__init__(
            f"User {user_id} already has an active assignment for licence {licence_id}."
        )
        self.user_id = user_id
        self.licence_id = licence_id


class NoSeatsAvailable(DomainError):
    def __init__(self, licence_id: int) -> None:
        super().__init__(f"Licence {licence_id} has no seats available.")
        self.licence_id = licence_id


class AssignmentNotFound(DomainError):
    def __init__(self, assignment_id: int) -> None:
        super().__init__(f"Assignment {assignment_id} not found.")
        self.assignment_id = assignment_id


class AssignmentAlreadyRevoked(DomainError):
    def __init__(self, assignment_id: int) -> None:
        super().__init__(f"Assignment {assignment_id} is already revoked.")
        self.assignment_id = assignment_id
