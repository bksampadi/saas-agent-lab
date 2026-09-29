"""POST /agent-runs and GET /agent-runs/{run_id}: the whole natural-language
flow through HTTP. The production PydanticAI planners run with scripted
models (FunctionModel) in place of a provider, injected through the API's
planner dependencies; no test sends a real model request.

Rows are seeded with distinctive seven-digit ids, longer than any run of
digits in a timestamp, so an id leaked into a response would show in its
text.
"""

import inspect
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx2
import pytest
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic_ai import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.agent.decision import (
    MAX_DECISION_MODEL_REQUESTS,
    MAX_DECISION_READ_CALLS,
    decision_context,
)
from app.agent.executor import INSTRUCTION_MAX_LENGTH
from app.agent.pydantic_ai_decision import PydanticAIDecisionPlanner
from app.agent.pydantic_ai_planner import PydanticAIIntentPlanner
from app.api import agent_runs
from app.api.deps import (
    get_decision_planner,
    get_intent_planner,
    get_session_factory,
    get_transaction,
)
from app.core.config import Settings, get_settings
from app.core.database import create_db_engine, get_session
from app.main import create_app
from app.models import (
    AgentRun,
    Assignment,
    AuditEvent,
    Base,
    GoalType,
    Licence,
    PolicyDecision,
    ToolCall,
    User,
    UserStatus,
)
from app.repositories.tool_calls import ToolCallRepository
from app.schemas.agent import DecisionTask

HUMAN = "requesting-user@example.com"
MODEL = "scripted-model"
EMAIL = "ada@example.com"
PRODUCT = "GitHub Enterprise"
INSTRUCTION = "Ensure ada@example.com has a GitHub Enterprise licence"
Sessions = sessionmaker[Session]

# Distinctive ids: none of them may appear in a response.
ADA = 4821357
GHE = 9753124
HELD = 8642097  # Ada's assignment, when she already holds a seat
OTHERS = 3640011  # other users, and their assignments, from here up

USER = "get_target_user"
CAPACITY = "get_target_licence_capacity"
ASSIGNMENTS = "list_target_user_assignments"
ASSIGN = "assign_target_licence"

Step = ModelResponse | Exception | Callable[[], ModelResponse]


# --- scripted models ------------------------------------------------------------


@dataclass
class Script:
    """A FunctionModel that answers each request with the next step, and keeps
    what it was sent. A step may be an exception to raise, or a callable run
    when the request arrives."""

    steps: list[Step] = field(default_factory=list)
    requests: list[list[ModelMessage]] = field(default_factory=list)
    infos: list[AgentInfo] = field(default_factory=list)

    def will(self, *steps: Step) -> None:
        self.steps.extend(steps)

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(list(messages))
        self.infos.append(info)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ModelResponse):
            return step
        return step()

    def model(self) -> FunctionModel:
        return FunctionModel(self.respond, model_name=MODEL)

    def sent_parts(self) -> list[Any]:
        """Every request part the model was sent: the last request carries
        the whole conversation."""
        last = self.requests[-1] if self.requests else []
        return [p for m in last if isinstance(m, ModelRequest) for p in m.parts]


def usage() -> RequestUsage:
    return RequestUsage(input_tokens=120, output_tokens=15)


def answer(tool_name: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name, args)], usage=usage())


def extracted(user_email: str = EMAIL, product: str = PRODUCT) -> ModelResponse:
    return answer("ensure_assignment", user_email=user_email, product=product)


def call(*tool_names: str) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(name, {}) for name in tool_names], usage=usage()
    )


GOAL_REACHED = answer("goal_reached")
NO_ACTION_NEEDED = answer("no_action_needed")


def cannot_proceed(reason: str) -> ModelResponse:
    return answer("cannot_proceed", reason_code=reason)


# --- the API --------------------------------------------------------------------


@dataclass
class AgentApi:
    client: TestClient
    sessions: Sessions
    extraction: Script
    decision: Script

    def post(
        self, instruction: str = INSTRUCTION, *, actor: str = HUMAN
    ) -> httpx2.Response:
        return self.client.post(
            "/agent-runs", json={"instruction": instruction}, headers={"X-Actor": actor}
        )

    def run(self, instruction: str = INSTRUCTION) -> dict[str, Any]:
        """POST, check that a run was created, and return its summary."""
        response = self.post(instruction)
        assert response.status_code == 201
        return response.json()

    def get(self, run_id: int) -> httpx2.Response:
        return self.client.get(f"/agent-runs/{run_id}")

    def detail(self, run_id: int) -> dict[str, Any]:
        response = self.get(run_id)
        assert response.status_code == 200
        return response.json()


def agent_api(sessions: Sessions) -> Iterator[AgentApi]:
    """The application with its agent dependencies pointed at ``sessions``
    and at scripted models. Planners are built per request, as in
    production; only their model is replaced."""
    extraction, decision = Script(), Script()
    app = create_app()

    # The agent endpoints take no request session; this only keeps any that
    # did off the configured database.
    def request_session() -> Iterator[Session]:
        with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = request_session
    app.dependency_overrides[get_session_factory] = lambda: sessions
    app.dependency_overrides[get_intent_planner] = lambda: PydanticAIIntentPlanner(
        extraction.model(), timeout_seconds=5
    )
    app.dependency_overrides[get_decision_planner] = lambda: PydanticAIDecisionPlanner(
        decision.model(), timeout_seconds=5
    )
    with TestClient(app) as client:
        yield AgentApi(client, sessions, extraction, decision)


@pytest.fixture
def sessions(engine: Engine) -> Sessions:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def api(sessions: Sessions) -> Iterator[AgentApi]:
    yield from agent_api(sessions)


# --- database -------------------------------------------------------------------


def seed(
    sessions: Sessions,
    *,
    seats: int = 5,
    held_by_others: int = 0,
    ada_holds: bool = False,
    agent_policy: PolicyDecision = PolicyDecision.ALLOW,
) -> None:
    with sessions.begin() as session:
        session.add(User(id=ADA, email=EMAIL, name="Ada", status=UserStatus.ACTIVE))
        session.add(
            Licence(
                id=GHE, product=PRODUCT, seats_total=seats, agent_policy=agent_policy
            )
        )
        session.flush()
        if ada_holds:
            session.add(Assignment(id=HELD, user_id=ADA, licence_id=GHE))
        for n in range(held_by_others):
            other = OTHERS + n
            session.add(User(id=other, email=f"u{n}@example.com", name=f"U{n}"))
            session.flush()
            session.add(Assignment(id=other, user_id=other, licence_id=GHE))


def count(sessions: Sessions, model: type[Any]) -> int:
    with sessions() as session:
        return len(session.scalars(select(model)).all())


def outcome(body: dict[str, Any]) -> tuple[str, str | None, str | None]:
    return body["status"], body["outcome_reason"], body["outcome_code"]


def tool_entries(detail: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    return [
        (entry["tool"], entry["status"], entry["observation"])
        for entry in detail["trace"]
        if entry["kind"] == "tool_call"
    ]


# --- a run end to end -------------------------------------------------------------


def test_a_run_reads_assigns_and_is_completed_by_the_verifier(api: AgentApi) -> None:
    seed(api.sessions)
    api.extraction.will(extracted())
    api.decision.will(call(ASSIGNMENTS), call(CAPACITY), call(ASSIGN), GOAL_REACHED)

    response = api.post()

    assert response.status_code == 201
    body = response.json()
    run_id = body["id"]
    assert body == {
        "id": run_id,
        "instruction": INSTRUCTION,
        "requesting_actor": HUMAN,
        "status": "completed",
        "outcome_reason": "goal_satisfied",
        "outcome_code": None,
        "goal": {"kind": "ensure_assignment", "user_email": EMAIL, "product": PRODUCT},
        "decision_proposal": {"kind": "goal_reached"},
        "created_at": body["created_at"],
        "completed_at": body["completed_at"],
    }
    assert body["completed_at"] is not None
    # The change is audited as the run's, and the person who asked for it is
    # recorded on the run, not on the audit event.
    with api.sessions() as session:
        (assignment,) = session.scalars(select(Assignment)).all()
        (audit,) = session.scalars(select(AuditEvent)).all()
        run = session.get(AgentRun, run_id)
    assert (assignment.user_id, assignment.licence_id) == (ADA, GHE)
    assert audit.actor == f"agent:run-{run_id}"
    assert run is not None and run.requesting_actor == HUMAN


def test_a_goal_already_held_completes_without_a_change(api: AgentApi) -> None:
    seed(api.sessions, ada_holds=True)
    api.extraction.will(extracted())
    api.decision.will(call(ASSIGNMENTS), NO_ACTION_NEEDED)

    body = api.run()

    assert outcome(body) == ("completed", "already_satisfied", None)
    assert body["decision_proposal"] == {"kind": "no_action_needed"}
    assert tool_entries(api.detail(body["id"])) == [
        (ASSIGNMENTS, "succeeded", '{"holds_active_seat":true}')
    ]
    assert count(api.sessions, Assignment) == 1
    assert count(api.sessions, AuditEvent) == 0


def test_a_confirmed_lack_of_seats_blocks_the_run_without_a_change(
    api: AgentApi,
) -> None:
    seed(api.sessions, seats=1, held_by_others=1)
    api.extraction.will(extracted())
    api.decision.will(call(CAPACITY), cannot_proceed("no_seats_available"))

    body = api.run()

    assert outcome(body) == ("blocked", "no_seats_available", None)
    assert body["decision_proposal"] == {
        "kind": "cannot_proceed",
        "reason_code": "no_seats_available",
    }
    detail = api.detail(body["id"])
    assert tool_entries(detail) == [
        (
            CAPACITY,
            "succeeded",
            '{"seats_active":1,"seats_available":0,"seats_total":1}',
        )
    ]
    # Blocked because the application confirmed the claim, not because the
    # model made it.
    assert detail["verification"] == {
        "satisfied": False,
        "blocking_rejection": None,
        "block_check": {"reason": "no_seats_available", "confirmed": True},
    }
    assert count(api.sessions, Assignment) == 1  # only the other user's
    assert count(api.sessions, AuditEvent) == 0


def test_a_mutation_policy_holds_pauses_the_run_for_approval(api: AgentApi) -> None:
    seed(api.sessions, agent_policy=PolicyDecision.REQUIRE_APPROVAL)
    api.extraction.will(extracted())
    api.decision.will(call(ASSIGNMENTS), call(ASSIGN), GOAL_REACHED)

    response = api.post()

    # An expected outcome: 201, with the pause in the body.
    assert response.status_code == 201
    body = response.json()
    assert outcome(body) == ("awaiting_approval", None, None)
    assert (body["decision_proposal"], body["completed_at"]) == (None, None)
    detail = api.detail(body["id"])
    assert detail["verification"] is None
    assert [e for e in detail["trace"] if e["kind"] == "tool_call"][-1] == {
        "kind": "tool_call",
        "sequence_no": 5,
        "tool": ASSIGN,
        "status": "awaiting_approval",
        "policy": "require_approval",
        "error_code": None,
        "observation": None,
    }
    assert api.decision.steps == [GOAL_REACHED]  # the model was asked no more
    assert (count(api.sessions, Assignment), count(api.sessions, AuditEvent)) == (0, 0)


def test_a_mutation_policy_denies_blocks_the_run(api: AgentApi) -> None:
    seed(api.sessions, agent_policy=PolicyDecision.DENY)
    api.extraction.will(extracted())
    api.decision.will(call(ASSIGN), GOAL_REACHED)

    body = api.run()

    assert outcome(body) == ("blocked", "policy_denied", None)
    assert body["decision_proposal"] is None
    detail = api.detail(body["id"])
    assert detail["verification"] is None  # ended at admission, not verified
    assert [e for e in detail["trace"] if e["kind"] == "tool_call"] == [
        {
            "kind": "tool_call",
            "sequence_no": 3,
            "tool": ASSIGN,
            "status": "failed",
            "policy": "deny",
            "error_code": "policy_denied",
            "observation": None,
        }
    ]
    assert api.decision.steps == [GOAL_REACHED]
    assert (count(api.sessions, Assignment), count(api.sessions, AuditEvent)) == (0, 0)


@pytest.mark.parametrize(
    "agent_policy", [PolicyDecision.DENY, PolicyDecision.REQUIRE_APPROVAL]
)
def test_a_policy_stop_shows_no_internal_id_arguments_or_message(
    api: AgentApi, agent_policy: PolicyDecision
) -> None:
    seed(api.sessions, agent_policy=agent_policy)
    api.extraction.will(extracted())
    api.decision.will(call(ASSIGN))
    created = api.post()
    fetched = api.get(created.json()["id"])

    with api.sessions() as session:
        (held_or_denied,) = session.scalars(select(ToolCall)).all()
        run = session.get(AgentRun, created.json()["id"])
    # The internals hold the ids (and, when denied, a message and detail).
    assert held_or_denied.arguments == {"user_id": ADA, "licence_id": GHE}
    assert run is not None and run.resolved_licence_id == GHE
    for response in (created, fetched):
        assert response.status_code in (200, 201)
        for internal in (str(ADA), str(GHE), "user_id", "licence_id", "message"):
            assert internal not in response.text
        # No field of that name (the instructions may use the word).
        assert '"arguments"' not in response.text


def test_a_false_success_claim_fails_verification(api: AgentApi) -> None:
    seed(api.sessions)
    api.extraction.will(extracted())
    api.decision.will(GOAL_REACHED)

    body = api.run()

    assert outcome(body) == ("failed", "verification_failed", None)
    assert body["decision_proposal"] == {"kind": "goal_reached"}
    assert api.detail(body["id"])["verification"] == {
        "satisfied": False,
        "blocking_rejection": None,
        "block_check": None,
    }
    assert count(api.sessions, Assignment) == 0


# --- runs that end before the decision stage ---------------------------------------


@pytest.mark.parametrize(
    ("kind", "code", "reason"),
    [
        ("needs_clarification", "multiple_users", "instruction_unclear"),
        ("unsupported", "additional_request", "unsupported_request"),
    ],
    ids=["ambiguous", "unsupported"],
)
def test_an_unclear_instruction_needs_clarification_and_nothing_is_decided(
    api: AgentApi, kind: str, code: str, reason: str
) -> None:
    seed(api.sessions)
    api.extraction.will(answer(kind, reason_code=code))

    body = api.run("Give ada@example.com and bob@example.com GitHub Enterprise")

    assert outcome(body) == ("needs_clarification", reason, code)
    assert (body["goal"], body["decision_proposal"]) == (None, None)
    assert api.decision.requests == []
    detail = api.detail(body["id"])
    assert (detail["decision_context"], detail["verification"]) == (None, None)
    (extraction,) = detail["trace"]
    assert (extraction["stage"], extraction["output"]) == (
        "extraction",
        {"kind": kind, "reason_code": code},
    )


@pytest.mark.parametrize(
    ("instruction", "user_email", "product", "reason"),
    [
        (
            "Ensure grace@example.com has a GitHub Enterprise licence",
            "grace@example.com",
            PRODUCT,
            "user_not_found",
        ),
        (
            "Ensure ada@example.com has a Jira licence",
            EMAIL,
            "Jira",
            "licence_not_found",
        ),
    ],
    ids=["unknown-user", "unknown-product"],
)
def test_an_unknown_user_or_product_is_a_run_outcome_not_a_server_error(
    api: AgentApi, instruction: str, user_email: str, product: str, reason: str
) -> None:
    seed(api.sessions)
    api.extraction.will(extracted(user_email, product))

    body = api.run(instruction)

    assert outcome(body) == ("needs_clarification", reason, None)
    assert body["goal"] == {
        "kind": "ensure_assignment",
        "user_email": user_email,
        "product": product,
    }
    assert body["decision_proposal"] is None
    assert api.decision.requests == []
    assert api.detail(body["id"])["decision_context"] is None


# --- runs that fail -----------------------------------------------------------------


def test_a_run_over_its_read_limit_fails_and_stays_retrievable(api: AgentApi) -> None:
    seed(api.sessions)
    api.extraction.will(extracted())
    api.decision.will(call(*[USER] * (MAX_DECISION_READ_CALLS + 1)))

    body = api.run()

    assert outcome(body) == ("failed", "step_limit", "read_calls")
    assert body["decision_proposal"] is None
    detail = api.detail(body["id"])
    # The refused call is in the trace; the model was shown nothing for it.
    assert tool_entries(detail) == [
        *[(USER, "succeeded", '{"status":"active"}')] * MAX_DECISION_READ_CALLS,
        (USER, "failed", None),
    ]
    assert detail["trace"][-1]["error_code"] == "step_limit"
    assert detail["verification"] is None


def test_a_run_over_its_model_request_limit_fails_and_stays_retrievable(
    api: AgentApi,
) -> None:
    seed(api.sessions)
    api.extraction.will(extracted())
    api.decision.will(*[call(USER)] * (MAX_DECISION_MODEL_REQUESTS + 1))

    body = api.run()

    assert outcome(body) == ("failed", "step_limit", "model_requests")
    detail = api.detail(body["id"])
    # The request over the limit was never made, so it is not in the trace.
    decision_requests = [e for e in detail["trace"] if e.get("stage") == "decision"]
    assert len(decision_requests) == MAX_DECISION_MODEL_REQUESTS
    assert len(api.decision.requests) == MAX_DECISION_MODEL_REQUESTS
    assert detail["verification"] is None


@pytest.mark.parametrize("stage", ["extraction", "decision"])
def test_a_provider_failure_fails_the_run_which_stays_retrievable(
    api: AgentApi, stage: str
) -> None:
    seed(api.sessions)
    failure = ModelHTTPError(503, MODEL, body={"secret": "provider-internal-detail"})
    if stage == "extraction":
        api.extraction.will(failure)
    else:
        api.extraction.will(extracted())
        api.decision.will(failure)

    body = api.run()

    assert outcome(body) == ("failed", "planner_error", "provider_error")
    response = api.get(body["id"])
    assert response.status_code == 200
    failed = response.json()["trace"][-1]
    assert failed == {
        "kind": "model_call",
        "sequence_no": failed["sequence_no"],
        "stage": stage,
        "model": MODEL,
        "status": "failed",
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": failed["latency_ms"],
        "output": None,
        "error_code": "provider_error",
    }
    for internal in ("provider-internal-detail", "HTTP 503", "ModelHTTPError"):
        assert internal not in response.text


def test_without_a_provider_key_the_real_planner_fails_the_run_not_the_app(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production planner dependencies, unreplaced: only the database and
    # the settings are the test's. No model request can be made (no key, and
    # conftest disallows them anyway).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seed(sessions)
    app = create_app()
    app.dependency_overrides[get_session_factory] = lambda: sessions
    app.dependency_overrides[get_settings] = lambda: Settings(
        planner_model="anthropic:claude-sonnet-5"
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post(
            "/agent-runs", json={"instruction": INSTRUCTION}, headers={"X-Actor": HUMAN}
        )
        assert response.status_code == 201
        detail = client.get(f"/agent-runs/{response.json()['id']}").json()

    assert outcome(detail) == ("failed", "planner_error", "configuration_error")
    assert detail["trace"] == []  # no request was made


# --- GET: the run as its models saw it ------------------------------------------------


def test_get_returns_the_persisted_trace_in_order_as_the_model_saw_it(
    api: AgentApi,
) -> None:
    # The model tries first, is rejected, looks, and concludes.
    seed(api.sessions, seats=1, held_by_others=1)
    api.extraction.will(extracted())
    api.decision.will(
        call(ASSIGN), call(CAPACITY), cannot_proceed("no_seats_available")
    )
    created = api.run()

    detail = api.detail(created["id"])

    # The summary is the one POST returned.
    assert {key: detail[key] for key in created} == created
    assert outcome(detail) == ("blocked", "no_seats_available", None)
    trace = detail["trace"]
    assert [entry["sequence_no"] for entry in trace] == [1, 2, 3, 4, 5, 6]
    assert [
        (entry["kind"], entry.get("stage") or entry.get("tool")) for entry in trace
    ] == [
        ("model_call", "extraction"),
        ("model_call", "decision"),
        ("tool_call", ASSIGN),
        ("model_call", "decision"),
        ("tool_call", CAPACITY),
        ("model_call", "decision"),
    ]
    assert trace[0] == {
        "kind": "model_call",
        "sequence_no": 1,
        "stage": "extraction",
        "model": MODEL,
        "status": "succeeded",
        "input_tokens": 120,
        "output_tokens": 15,
        "latency_ms": trace[0]["latency_ms"],
        "output": {
            "kind": "ensure_assignment",
            "user_email": EMAIL,
            "product": PRODUCT,
        },
        "error_code": None,
    }
    assert trace[1]["output"] == {"kind": "tool_calls", "tool_names": [ASSIGN]}
    assert trace[2] == {
        "kind": "tool_call",
        "sequence_no": 3,
        "tool": ASSIGN,
        "status": "failed",
        # Admitted by policy, then rejected by the domain rule.
        "policy": "allow",
        "error_code": "no_seats_available",
        "observation": '{"outcome":"rejected","reason_code":"no_seats_available"}',
    }
    assert trace[4]["policy"] is None  # a read: policy does not govern it
    assert trace[5]["output"] == {
        "kind": "cannot_proceed",
        "reason_code": "no_seats_available",
    }

    with api.sessions() as session:
        run = session.get(AgentRun, created["id"])
        calls = ToolCallRepository(session).list_for_run(created["id"])
    assert run is not None
    # Observations: exactly as persisted, and exactly what the model was given.
    observations = [entry["observation"] for entry in trace if "tool" in entry]
    assert observations == [call.observation for call in calls]
    assert observations == [
        part.model_response_str()
        for part in api.decision.sent_parts()
        if isinstance(part, ToolReturnPart) and part.tool_name != "cannot_proceed"
    ]
    # The initial context: as persisted, and exactly what the model was sent.
    first_request = api.decision.requests[0][-1]
    assert isinstance(first_request, ModelRequest)
    (prompt,) = [p for p in first_request.parts if isinstance(p, UserPromptPart)]
    assert (
        detail["decision_context"]
        == run.decision_context
        == {
            "instructions": api.decision.infos[0].instructions,
            "prompt": prompt.content,
        }
        == decision_context(
            DecisionTask(
                goal_type=GoalType.ENSURE_ASSIGNMENT, user_email=EMAIL, product=PRODUCT
            )
        ).model_dump()
    )
    # The rejection blocked the run, and the claim was confirmed as well.
    assert detail["verification"] == {
        "satisfied": False,
        "blocking_rejection": "no_seats_available",
        "block_check": {"reason": "no_seats_available", "confirmed": True},
    }


def test_no_internal_id_or_exception_message_reaches_a_response(api: AgentApi) -> None:
    # A run that makes a change, and one whose attempt is rejected: between
    # them, every internal field that holds an id or a domain message.
    seed(api.sessions, seats=2, held_by_others=1)
    api.extraction.will(extracted())
    api.decision.will(call(USER, ASSIGNMENTS, CAPACITY), call(ASSIGN), GOAL_REACHED)
    completed = api.post()
    # Ada took the last seat, so the next user's attempt is rejected.
    with api.sessions.begin() as session:
        session.add(User(id=OTHERS + 1, email="u1@example.com", name="U1"))
    api.extraction.will(extracted("u1@example.com"))
    api.decision.will(call(ASSIGN), cannot_proceed("no_seats_available"))
    blocked = api.post("Ensure u1@example.com has a GitHub Enterprise licence")
    responses = [
        completed,
        blocked,
        api.get(completed.json()["id"]),
        api.get(blocked.json()["id"]),
    ]
    assert [r.status_code for r in responses] == [201, 201, 200, 200]
    assert outcome(responses[2].json())[:2] == ("completed", "goal_satisfied")
    assert outcome(responses[3].json())[:2] == ("blocked", "no_seats_available")

    with api.sessions() as session:
        runs = session.scalars(select(AgentRun)).all()
        calls = session.scalars(select(ToolCall)).all()
        assignment_ids = session.scalars(select(Assignment.id)).all()
    internal_ids = {ADA, GHE, OTHERS + 1, *assignment_ids}
    assert min(internal_ids) > 1_000_000  # distinctive, so the check means something
    # The internals really do hold them, and a domain message.
    assert {run.resolved_licence_id for run in runs} == {GHE}
    assert all(set(call.arguments.values()) <= internal_ids for call in calls)
    assert any(call.result is not None for call in calls)
    assert any(
        call.error is not None and "has no seats available" in call.error["message"]
        for call in calls
    )
    assert all(str(GHE) in str(run.outcome_detail) for run in runs)

    for response in responses:
        for internal_id in internal_ids:
            assert str(internal_id) not in response.text
        for internal in ("has no seats available", "user_id", "licence_id", "message"):
            assert internal not in response.text


def test_an_unknown_run_is_not_found(api: AgentApi) -> None:
    response = api.get(999)

    assert response.status_code == 404
    assert response.json() == {"detail": "Agent run 999 not found."}


@pytest.mark.parametrize("run_id", ["0", "-1", "abc", "2147483648"])
def test_an_invalid_run_id_is_rejected(api: AgentApi, run_id: str) -> None:
    assert api.client.get(f"/agent-runs/{run_id}").status_code == 422


# --- request validation: no run is created ------------------------------------------


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        ({}, None),
        ({"X-Actor": ""}, None),
        ({"X-Actor": "x" * 321}, None),
        ({"X-Actor": "agent:run-1"}, "reserved for agent runs"),
        ({"X-Actor": " Agent:run-1 "}, "reserved for agent runs"),
        ({"X-Actor": "   "}, None),
    ],
    ids=["missing", "empty", "too-long", "reserved", "reserved-disguised", "blank"],
)
def test_a_missing_or_invalid_actor_is_rejected_and_no_run_is_created(
    api: AgentApi, headers: dict[str, str], detail: str | None
) -> None:
    response = api.client.post(
        "/agent-runs", json={"instruction": INSTRUCTION}, headers=headers
    )

    assert response.status_code == 422
    if detail is not None:
        assert detail in response.json()["detail"]
    assert count(api.sessions, AgentRun) == 0
    assert api.extraction.requests == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"instruction": ""},
        {"instruction": 42},
        {"instruction": "x" * (INSTRUCTION_MAX_LENGTH + 1)},
        # Nothing can be smuggled in beside the instruction.
        {"instruction": INSTRUCTION, "user_id": ADA},
        {"instruction": "   "},  # refused by the executor, as for any caller
    ],
    ids=["missing", "empty", "not-text", "too-long", "extra-field", "blank"],
)
def test_a_malformed_body_is_rejected_and_no_run_is_created(
    api: AgentApi, body: dict[str, Any]
) -> None:
    response = api.client.post("/agent-runs", json=body, headers={"X-Actor": HUMAN})

    assert response.status_code == 422
    assert count(api.sessions, AgentRun) == 0
    assert api.extraction.requests == []


def test_a_body_that_is_not_json_is_rejected(api: AgentApi) -> None:
    response = api.client.post(
        "/agent-runs",
        content=b"Ensure ada@example.com has a licence",
        headers={"X-Actor": HUMAN, "Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert count(api.sessions, AgentRun) == 0


# --- no transaction is open while a model runs -------------------------------------


def dependency_calls(dependant: Dependant) -> set[Any]:
    calls: set[Any] = set()
    for dependency in dependant.dependencies:
        calls.add(dependency.call)
        calls |= dependency_calls(dependency)
    return calls


def test_the_agent_endpoints_are_synchronous_and_take_no_request_transaction() -> None:
    # A plain def runs in a worker thread, so a run's synchronous model
    # requests never block the event loop.
    assert not inspect.iscoroutinefunction(agent_runs.create_agent_run)
    assert not inspect.iscoroutinefunction(agent_runs.get_agent_run)
    for route in agent_runs.router.routes:
        assert isinstance(route, APIRoute)
        calls = dependency_calls(route.dependant)
        assert get_transaction not in calls
        assert get_session not in calls


@dataclass
class TransactionCounter:
    open_now: int = 0
    peak: int = 0

    def began(self, *_: Any) -> None:
        self.open_now += 1
        self.peak = max(self.peak, self.open_now)

    def ended(self, *_: Any) -> None:
        self.open_now -= 1


@pytest.fixture
def locking_engine() -> Iterator[Engine]:
    # As in tests/agent/test_isolation.py: every session has its own
    # connection, so a transaction left open would make another writer fail
    # at once instead of silently sharing it.
    name = f"agentlab-{uuid4().hex}"
    keeper = sqlite3.connect(f"file:{name}?mode=memory&cache=shared", uri=True)
    engine = create_db_engine(
        f"sqlite:///file:{name}?mode=memory&cache=shared&uri=true",
        poolclass=QueuePool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    keeper.close()


@pytest.fixture
def locking_api(locking_engine: Engine) -> Iterator[AgentApi]:
    yield from agent_api(
        sessionmaker(bind=locking_engine, autoflush=False, expire_on_commit=False)
    )


def test_no_transaction_is_open_during_any_model_request_of_an_api_run(
    locking_engine: Engine, locking_api: AgentApi
) -> None:
    api = locking_api
    seed(api.sessions)
    counter = TransactionCounter()
    event.listen(locking_engine, "begin", counter.began)
    event.listen(locking_engine, "commit", counter.ended)
    event.listen(locking_engine, "rollback", counter.ended)
    open_at_request: list[int] = []

    def probe(response: ModelResponse) -> Callable[[], ModelResponse]:
        def step() -> ModelResponse:
            open_at_request.append(counter.open_now)
            # A write here fails at once if any transaction holds a lock.
            with api.sessions.begin() as session:
                session.add(
                    User(email=f"probe{len(open_at_request)}@example.com", name="P")
                )
            return response

        return step

    api.extraction.will(probe(extracted()))
    api.decision.will(probe(call(CAPACITY)), probe(call(ASSIGN)), probe(GOAL_REACHED))

    body = api.run()

    assert outcome(body) == ("completed", "goal_satisfied", None)
    assert open_at_request == [0, 0, 0, 0]
    assert (counter.peak, counter.open_now) == (1, 0)
