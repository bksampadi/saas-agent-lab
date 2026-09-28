"""One real model, end to end through HTTP. Opt-in only.

Deselected by default (pyproject addopts: -m 'not live'). Run it with

    uv run pytest -m live

and the configured provider's API key in the environment (ANTHROPIC_API_KEY
for the default anthropic model); without the key it is skipped. It sends
real, billable requests to the application's configured planner model.

A smoke test, not an evaluation: it checks that a real model drives the
bounded loop to a verified result, not which tools it chose or in what
order. It is never retried.
"""

import os

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models import override_allow_model_requests
from sqlalchemy import Engine, select
from sqlalchemy.orm import sessionmaker

from app.api.deps import get_session_factory
from app.core.config import get_settings
from app.main import create_app
from app.models import AgentRun, Assignment, AuditEvent, Licence, User

pytestmark = pytest.mark.live

# The environment variable each supported provider's SDK reads its key from.
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY"}

HUMAN = "requesting-user@example.com"
INSTRUCTION = "Ensure ada@example.com has a GitHub Enterprise licence"


def require_provider_key(planner_model: str) -> None:
    provider = planner_model.split(":", 1)[0]
    key = PROVIDER_KEYS.get(provider)
    if key is None:
        pytest.skip(f"No live test support for provider {provider!r}.")
    if not os.environ.get(key):
        pytest.skip(f"{key} is not set.")


def test_a_real_model_drives_a_run_to_a_verified_assignment(engine: Engine) -> None:
    planner_model = get_settings().planner_model  # the configured default
    require_provider_key(planner_model)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with sessions.begin() as session:
        ada = User(email="ada@example.com", name="Ada")
        licence = Licence(product="GitHub Enterprise", seats_total=5)
        session.add_all([ada, licence])
        session.flush()
        ada_id, licence_id = ada.id, licence.id
    app = create_app()
    # Only the database is the test's: the planners are the production ones.
    app.dependency_overrides[get_session_factory] = lambda: sessions

    # tests/conftest.py disallows model requests for the whole suite; this
    # test alone lifts that, and it is restored on the way out.
    with override_allow_model_requests(True), TestClient(app) as client:
        response = client.post(
            "/agent-runs", json={"instruction": INSTRUCTION}, headers={"X-Actor": HUMAN}
        )
        assert response.status_code == 201, response.text
        detail = client.get(f"/agent-runs/{response.json()['id']}").json()

    run_id = detail["id"]
    trace = detail["trace"]
    model_calls = [entry for entry in trace if entry["kind"] == "model_call"]
    tool_calls = [entry for entry in trace if entry["kind"] == "tool_call"]
    # Extraction: the real model's accepted intent became the run's goal.
    extraction = [call for call in model_calls if call["stage"] == "extraction"]
    assert extraction and extraction[-1]["status"] == "succeeded", trace
    assert extraction[-1]["output"]["kind"] == "ensure_assignment"
    assert detail["goal"]["user_email"].casefold() == "ada@example.com"
    assert detail["goal"]["product"].casefold() == "github enterprise"
    # Resolution: only a resolved run reaches the decision stage.
    assert detail["decision_context"] is not None
    # Decision: the real model worked through the goal-bound tools, each
    # call answered with a persisted observation, and it made the change.
    decision = [call for call in model_calls if call["stage"] == "decision"]
    assert decision and all(call["model"] for call in decision)
    assert all(
        call["input_tokens"] is not None
        for call in decision
        if call["status"] == "succeeded"
    )
    assert tool_calls and all(call["observation"] for call in tool_calls), trace
    assert (
        "assign_target_licence",
        "succeeded",
        '{"outcome":"assigned","reason_code":null}',
    ) in [(call["tool"], call["status"], call["observation"]) for call in tool_calls]
    # Verification: the application, not the model, completed the run.
    assert (detail["status"], detail["outcome_reason"]) == (
        "completed",
        "goal_satisfied",
    ), trace
    assert detail["verification"]["satisfied"] is True
    with sessions() as session:
        run = session.get(AgentRun, run_id)
        (assignment,) = session.scalars(select(Assignment)).all()
        (audit,) = session.scalars(select(AuditEvent)).all()
    assert run is not None and run.resolved_user_id == ada_id
    assert (assignment.user_id, assignment.licence_id) == (ada_id, licence_id)
    assert assignment.revoked_at is None
    assert audit.actor == f"agent:run-{run_id}"
