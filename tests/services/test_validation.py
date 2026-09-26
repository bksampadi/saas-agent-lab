"""Actor rules: one base rule, plus a reserved namespace for agent runs."""

import pytest

from app.services.errors import InvalidInput
from app.services.validation import (
    agent_run_actor,
    reject_reserved_actor,
    validated_actor,
    validated_external_actor,
)

RESERVED = ["agent:run-5", "agent:", "Agent:run-5", "AGENT:anything", "  agent:run-5"]
ORDINARY = ["admin@example.com", "agent-smith@example.com", "agents:ops", "agent"]


@pytest.mark.parametrize("actor", RESERVED)
def test_external_actor_in_agent_namespace_is_rejected(actor: str) -> None:
    with pytest.raises(InvalidInput, match="reserved for agent runs"):
        validated_external_actor(actor)
    with pytest.raises(InvalidInput, match="reserved for agent runs"):
        reject_reserved_actor(actor)


@pytest.mark.parametrize("actor", ORDINARY)
def test_ordinary_external_actors_are_accepted(actor: str) -> None:
    reject_reserved_actor(actor)
    assert validated_external_actor(f"  {actor} ") == actor


@pytest.mark.parametrize("actor", ["", "   ", "a" * 321])
def test_external_actor_still_follows_the_base_rules(actor: str) -> None:
    with pytest.raises(InvalidInput):
        validated_external_actor(actor)


def test_base_actor_rule_is_unchanged_and_allows_the_agent_namespace() -> None:
    # Services validate every actor, agent or human, with the base rule.
    assert validated_actor("agent:run-5") == "agent:run-5"


def test_agent_run_actor_is_valid_and_names_the_run() -> None:
    actor = agent_run_actor(42)

    assert actor == "agent:run-42"
    assert validated_actor(actor) == actor
