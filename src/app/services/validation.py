"""Input rules shared by more than one service."""

from app.services.errors import InvalidInput

# Matches the audit_events.actor column size.
ACTOR_MAX_LENGTH = 320


def required_text(value: str, *, field: str, max_length: int) -> str:
    """Strip ``value`` and check it is non-empty and within ``max_length``."""
    stripped = value.strip()
    if not stripped:
        raise InvalidInput(f"{field} must not be empty.")
    if len(stripped) > max_length:
        raise InvalidInput(f"{field} must be at most {max_length} characters.")
    return stripped


def validated_actor(actor: str) -> str:
    return required_text(actor, field="Actor", max_length=ACTOR_MAX_LENGTH)
