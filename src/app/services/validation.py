"""Input rules shared by more than one service."""

from app.services.errors import InvalidInput

# Matches the audit_events.actor column size.
ACTOR_MAX_LENGTH = 320

# Audit actors in this namespace are written only by agent runs
# ("agent:run-<id>"), so an audit event can be traced back to its run.
AGENT_ACTOR_PREFIX = "agent:"


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


def reject_reserved_actor(actor: str) -> None:
    """Raise InvalidInput if ``actor`` claims the agent namespace.

    Checked as the actor would be stored (stripped) and ignoring case, so
    " Agent:run-5" cannot pass for an agent either.
    """
    if actor.strip().casefold().startswith(AGENT_ACTOR_PREFIX):
        raise InvalidInput(
            f"Actors starting with {AGENT_ACTOR_PREFIX!r} are reserved for agent runs."
        )


def validated_external_actor(actor: str) -> str:
    """For identities supplied from outside the application: the usual actor
    rules, and the agent namespace is off limits."""
    reject_reserved_actor(actor)
    return validated_actor(actor)


def agent_run_actor(run_id: int) -> str:
    """The audit actor for changes made by agent run ``run_id``."""
    return validated_actor(f"{AGENT_ACTOR_PREFIX}run-{run_id}")
