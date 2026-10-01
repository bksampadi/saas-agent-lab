"""The decision stage end to end: a scripted model (FunctionModel) chooses
goal-bound tool calls through the PydanticAI decision planner, and the
application decides the outcome.

Rows are seeded with distinctive ids, so any id that crossed the model
boundary would be visible in what the model was sent.
"""

import json
from typing import Any

import pydantic_ai.models
import pytest
from pydantic_ai import ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agent import tools
from app.agent.decision import (
    MAX_DECISION_MODEL_REQUESTS,
    MAX_DECISION_MUTATION_CALLS,
    MAX_DECISION_READ_CALLS,
    decision_context,
)
from app.agent.executor import AgentExecutor
from app.agent.pydantic_ai_decision import OUTPUT_RETRIES, PydanticAIDecisionPlanner
from app.core.config import Settings
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    AuditEvent,
    CannotProceedReason,
    DecisionProposalKind,
    Licence,
    ModelCall,
    ModelCallStage,
    ModelCallStatus,
    OutcomeReason,
    ToolCall,
    ToolCallStatus,
    User,
    UserStatus,
)
from app.repositories.agent_runs import AgentRunRepository
from app.repositories.tool_calls import ToolCallRepository
from support import (
    ASSIGN,
    ASSIGNMENTS,
    CAPACITY,
    GOAL_REACHED,
    MODEL,
    NO_ACTION_NEEDED,
    PROPOSALS,
    USER,
    Script,
    Step,
    call,
    cannot_proceed,
    conclude,
    resolved_run,
    usage,
)

Sessions = sessionmaker[Session]
S = AgentRunStatus
R = OutcomeReason

# Distinctive ids: none of them may appear in anything the model is sent.
ADA = 48213
FIGMA = 97531
HELD = 86420  # Ada's assignment, when she already holds a seat
OTHERS = 36400  # other users, and their assignments, from here up
DISTINCTIVE_IDS = [str(n) for n in (ADA, FIGMA, HELD, OTHERS)]


def returned_to_model(script: Script) -> list[str]:
    """The tool results the model was given, in order, as sent."""
    return [
        part.model_response_str()
        for part in script.sent_parts()
        if isinstance(part, ToolReturnPart) and part.tool_name not in PROPOSALS
    ]


# --- database -----------------------------------------------------------------


def seed(
    sessions: Sessions,
    *,
    seats: int = 5,
    held_by_others: int = 0,
    ada_holds: bool = False,
    ada_status: UserStatus = UserStatus.ACTIVE,
) -> None:
    with sessions.begin() as session:
        session.add(
            User(id=ADA, email="ada@example.com", name="Ada", status=ada_status)
        )
        session.add(Licence(id=FIGMA, product="Figma", seats_total=seats))
        session.flush()
        if ada_holds:
            session.add(Assignment(id=HELD, user_id=ADA, licence_id=FIGMA))
        for n in range(held_by_others):
            other = OTHERS + n
            session.add(User(id=other, email=f"u{n}@example.com", name=f"U{n}"))
            session.flush()
            session.add(Assignment(id=other, user_id=other, licence_id=FIGMA))


def decide(executor: AgentExecutor, script: Script) -> tuple[int, AgentRunStatus]:
    run_id = resolved_run(executor)
    return run_id, executor.decide(run_id, script.planner())


def get_run(sessions: Sessions, run_id: int) -> AgentRun:
    with sessions() as session:
        run = session.get(AgentRun, run_id)
        assert run is not None
        return run


def outcome(
    sessions: Sessions, run_id: int
) -> tuple[AgentRunStatus, OutcomeReason | None]:
    run = get_run(sessions, run_id)
    return run.status, run.outcome_reason


def trace(sessions: Sessions, run_id: int) -> list[tuple[str, str]]:
    """The run's trace in order: model calls by status, tool calls by name and
    status."""
    with sessions() as session:
        entries = AgentRunRepository(session).list_trace(run_id)
    return [
        ("model", entry.status.value)
        if isinstance(entry, ModelCall)
        else (entry.tool_name, entry.status.value)
        for entry in entries
    ]


def tool_calls(sessions: Sessions, run_id: int) -> list[ToolCall]:
    with sessions() as session:
        return ToolCallRepository(session).list_for_run(run_id)


def model_calls(sessions: Sessions, run_id: int) -> list[ModelCall]:
    with sessions() as session:
        return list(
            session.scalars(
                select(ModelCall)
                .where(ModelCall.agent_run_id == run_id)
                .order_by(ModelCall.sequence_no)
            )
        )


def count(sessions: Sessions, model: type[Any]) -> int:
    with sessions() as session:
        return len(session.scalars(select(model)).all())


def detail(sessions: Sessions, run_id: int) -> dict[str, Any]:
    outcome_detail = get_run(sessions, run_id).outcome_detail
    assert outcome_detail is not None
    return outcome_detail


OK = ToolCallStatus.SUCCEEDED.value
FAILED = ToolCallStatus.FAILED.value
MODEL_OK = ("model", ModelCallStatus.SUCCEEDED.value)
MODEL_FAILED = ("model", ModelCallStatus.FAILED.value)


# --- successful model-directed execution ----------------------------------------


def test_the_model_reads_assigns_and_the_verifier_completes_the_run(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory)
    script = Script(call(ASSIGNMENTS), call(CAPACITY), call(ASSIGN), GOAL_REACHED)

    run_id, status = decide(executor, script)

    assert status is S.COMPLETED
    assert outcome(session_factory, run_id) == (S.COMPLETED, R.GOAL_SATISFIED)
    assert trace(session_factory, run_id) == [
        MODEL_OK,
        ("list_user_assignments", OK),
        MODEL_OK,
        ("get_licence", OK),
        MODEL_OK,
        ("assign_licence", OK),
        MODEL_OK,
    ]
    run = get_run(session_factory, run_id)
    assert (run.decision_proposal, run.decision_reason_code) == (
        DecisionProposalKind.GOAL_REACHED,
        None,
    )
    with session_factory() as session:
        (assignment,) = session.scalars(select(Assignment)).all()
        (event,) = session.scalars(select(AuditEvent)).all()
    assert (assignment.user_id, assignment.licence_id) == (ADA, FIGMA)
    assert event.actor == f"agent:run-{run_id}"
    assert [c.observation for c in tool_calls(session_factory, run_id)] == [
        '{"holds_active_seat":false}',
        '{"seats_active":0,"seats_available":5,"seats_total":5}',
        '{"outcome":"assigned","reason_code":null}',
    ]
    assert {c.stage for c in model_calls(session_factory, run_id)} == {
        ModelCallStage.DECISION
    }


# --- no forced preflight: the model may act without looking ----------------------


@pytest.mark.parametrize(
    "conclusion",
    [cannot_proceed("no_seats_available"), GOAL_REACHED],
    ids=["then-cannot-proceed", "then-false-success-claim"],
)
def test_an_immediate_assignment_without_reading_capacity_is_rejected_and_blocks(
    executor: AgentExecutor, session_factory: Sessions, conclusion: ModelResponse
) -> None:
    seed(session_factory, seats=1, held_by_others=1)
    script = Script(call(ASSIGN), conclusion)

    run_id, _ = decide(executor, script)

    # The rejection is a deterministic domain block, whatever the model says.
    assert outcome(session_factory, run_id) == (S.BLOCKED, R.NO_SEATS_AVAILABLE)
    # Nothing was read before the attempt, by the model or by the application.
    assert trace(session_factory, run_id) == [
        MODEL_OK,
        ("assign_licence", FAILED),
        MODEL_OK,
    ]
    (attempt,) = tool_calls(session_factory, run_id)
    assert attempt.observation == (
        '{"outcome":"rejected","reason_code":"no_seats_available"}'
    )
    # The model saw the rejection and nothing about capacity.
    assert returned_to_model(script) == [attempt.observation]
    assert count(session_factory, Assignment) == 1  # only the other user's
    assert detail(session_factory, run_id)["rejection"]["tool_call_id"] == attempt.id


def test_a_rejected_attempt_is_shown_to_the_model_which_may_carry_on(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, seats=0)
    script = Script(call(ASSIGN), call(CAPACITY), cannot_proceed("no_seats_available"))

    run_id, _ = decide(executor, script)

    assert outcome(session_factory, run_id) == (S.BLOCKED, R.NO_SEATS_AVAILABLE)
    assert trace(session_factory, run_id) == [
        MODEL_OK,
        ("assign_licence", FAILED),
        MODEL_OK,
        ("get_licence", OK),
        MODEL_OK,
    ]


# --- already satisfied ------------------------------------------------------------


def test_an_already_satisfied_goal_needs_no_action(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, ada_holds=True)
    script = Script(call(ASSIGNMENTS), NO_ACTION_NEEDED)

    run_id, _ = decide(executor, script)

    assert outcome(session_factory, run_id) == (S.COMPLETED, R.ALREADY_SATISFIED)
    assert trace(session_factory, run_id) == [
        MODEL_OK,
        ("list_user_assignments", OK),
        MODEL_OK,
    ]
    assert returned_to_model(script) == ['{"holds_active_seat":true}']
    assert count(session_factory, Assignment) == 1
    assert count(session_factory, AuditEvent) == 0
    assert get_run(session_factory, run_id).decision_proposal is (
        DecisionProposalKind.NO_ACTION_NEEDED
    )


def test_assigning_a_seat_already_held_is_shown_as_such_and_completes(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, ada_holds=True)
    script = Script(call(ASSIGN), GOAL_REACHED)

    run_id, _ = decide(executor, script)

    # Not a block: the verifier finds the goal holding, and this run changed
    # nothing.
    assert outcome(session_factory, run_id) == (S.COMPLETED, R.ALREADY_SATISFIED)
    assert returned_to_model(script) == [
        '{"outcome":"rejected","reason_code":"already_assigned"}'
    ]


@pytest.mark.parametrize(
    "conclusion",
    [cannot_proceed("no_seats_available"), GOAL_REACHED, NO_ACTION_NEEDED],
    ids=["cannot-proceed", "goal-reached", "no-action-needed"],
)
def test_a_satisfied_goal_completes_whatever_the_model_proposes(
    executor: AgentExecutor, session_factory: Sessions, conclusion: ModelResponse
) -> None:
    # Ada holds the only seat, so "no seats available" is even literally true.
    seed(session_factory, seats=1, ada_holds=True)

    run_id, _ = decide(executor, Script(call(CAPACITY), conclusion))

    assert outcome(session_factory, run_id) == (S.COMPLETED, R.ALREADY_SATISFIED)


# --- the model's claims are not evidence -------------------------------------------


@pytest.mark.parametrize(
    "steps",
    [
        (GOAL_REACHED,),
        (call(CAPACITY), GOAL_REACHED),
        (call(ASSIGNMENTS), NO_ACTION_NEEDED),
    ],
    ids=["no-tools", "after-a-read", "no-action-claim"],
)
def test_a_false_success_claim_fails_verification(
    executor: AgentExecutor, session_factory: Sessions, steps: tuple[Step, ...]
) -> None:
    seed(session_factory)

    run_id, _ = decide(executor, Script(*steps))

    assert outcome(session_factory, run_id) == (S.FAILED, R.VERIFICATION_FAILED)
    assert count(session_factory, Assignment) == 0
    outcome_detail = detail(session_factory, run_id)
    assert outcome_detail["satisfied"] is False
    assert (outcome_detail["rejection"], outcome_detail["block_check"]) == (None, None)


@pytest.mark.parametrize(
    ("seed_args", "read", "reason", "blocked_reason"),
    [
        ({"seats": 0}, CAPACITY, "no_seats_available", R.NO_SEATS_AVAILABLE),
        ({"ada_status": UserStatus.INACTIVE}, USER, "user_inactive", R.USER_INACTIVE),
    ],
    ids=["no-seats", "user-inactive"],
)
def test_a_correctly_observed_block_is_blocked_without_attempting_the_mutation(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed_args: dict[str, Any],
    read: str,
    reason: str,
    blocked_reason: OutcomeReason,
) -> None:
    # The invariant: responding correctly to real evidence is never worse
    # than ignoring it and attempting a doomed mutation.
    seed(session_factory, **seed_args)

    run_id, _ = decide(executor, Script(call(read), cannot_proceed(reason)))

    assert outcome(session_factory, run_id) == (S.BLOCKED, blocked_reason)
    assert [c.tool_name for c in tool_calls(session_factory, run_id)] == [
        "get_licence" if read == CAPACITY else "get_user"
    ]
    run = get_run(session_factory, run_id)
    assert (run.decision_proposal, run.decision_reason_code) == (
        DecisionProposalKind.CANNOT_PROCEED,
        CannotProceedReason(reason),
    )
    assert detail(session_factory, run_id)["block_check"]["confirmed"] is True
    assert detail(session_factory, run_id)["rejection"] is None


@pytest.mark.parametrize(
    ("seed_args", "reason"),
    [
        ({}, "no_seats_available"),
        ({}, "user_inactive"),
        # A genuine block exists, but not the one claimed: a claim does not
        # create BLOCKED by itself.
        ({"seats": 0}, "user_inactive"),
        ({"ada_status": UserStatus.INACTIVE}, "no_seats_available"),
    ],
    ids=["free-seat", "active-user", "wrong-block-no-seats", "wrong-block-inactive"],
)
def test_an_unconfirmed_cannot_proceed_claim_fails_verification(
    executor: AgentExecutor,
    session_factory: Sessions,
    seed_args: dict[str, Any],
    reason: str,
) -> None:
    seed(session_factory, **seed_args)

    run_id, _ = decide(executor, Script(cannot_proceed(reason)))

    assert outcome(session_factory, run_id) == (S.FAILED, R.VERIFICATION_FAILED)
    block_check = detail(session_factory, run_id)["block_check"]
    assert (block_check["reason"], block_check["confirmed"]) == (reason, False)


def test_a_claimed_block_is_checked_against_current_state_not_the_observation(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, seats=1, held_by_others=1)

    def seat_freed_then_claim() -> ModelResponse:
        # After the model saw zero free seats, before it concludes.
        with session_factory.begin() as session:
            other = session.get(Assignment, OTHERS)
            assert other is not None
            session.delete(other)
        return cannot_proceed("no_seats_available")

    run_id, _ = decide(executor, Script(call(CAPACITY), seat_freed_then_claim))

    assert outcome(session_factory, run_id) == (S.FAILED, R.VERIFICATION_FAILED)
    (read,) = tool_calls(session_factory, run_id)
    assert read.observation == '{"seats_active":1,"seats_available":0,"seats_total":1}'


# --- limits ---------------------------------------------------------------------


def spy(monkeypatch: pytest.MonkeyPatch, tool: str) -> list[int]:
    """Count the calls of ``tool`` that reached a business transaction:
    tools.run is called only inside one."""
    calls: list[int] = []
    real = tools.run

    def counting(called: Any, goal: Any, **services: Any) -> Any:
        if called == tool:
            calls.append(1)
        return real(called, goal, **services)

    monkeypatch.setattr(tools, "run", counting)
    return calls


@pytest.mark.parametrize(
    "steps",
    [(call(ASSIGN), call(ASSIGN)), (call(ASSIGN, ASSIGN),)],
    ids=["two-responses", "one-response"],
)
def test_a_second_mutation_never_reaches_the_business_transaction(
    executor: AgentExecutor,
    session_factory: Sessions,
    monkeypatch: pytest.MonkeyPatch,
    steps: tuple[Step, ...],
) -> None:
    seed(session_factory)
    assigned = spy(monkeypatch, ASSIGN)
    script = Script(*steps, GOAL_REACHED)

    run_id, _ = decide(executor, script)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.FAILED, R.STEP_LIMIT)
    assert run.outcome_detail == {
        "limit": "mutation_calls",
        "maximum": MAX_DECISION_MUTATION_CALLS,
    }
    assert len(assigned) == 1
    assert count(session_factory, Assignment) == 1
    first, refused = tool_calls(session_factory, run_id)
    assert first.status is ToolCallStatus.SUCCEEDED
    assert (refused.tool_name, refused.status) == (
        "assign_licence",
        ToolCallStatus.FAILED,
    )
    assert refused.error is not None
    assert (refused.error["code"], refused.result, refused.observation) == (
        "step_limit",
        None,
        None,
    )
    # The loop stopped at once: the model was asked nothing afterwards.
    assert len(script.requests) == len(steps)
    assert run.decision_proposal is None


def test_a_rejected_mutation_attempt_still_uses_the_mutation_budget(
    executor: AgentExecutor, session_factory: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(session_factory, seats=0)
    assigned = spy(monkeypatch, ASSIGN)

    run_id, _ = decide(executor, Script(call(ASSIGN), call(ASSIGN), GOAL_REACHED))

    assert outcome(session_factory, run_id) == (S.FAILED, R.STEP_LIMIT)
    assert len(assigned) == 1


def test_reads_over_the_limit_are_refused_before_any_business_read(
    executor: AgentExecutor, session_factory: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(session_factory)
    reads = spy(monkeypatch, CAPACITY)
    script = Script(call(*[CAPACITY] * (MAX_DECISION_READ_CALLS + 1)), GOAL_REACHED)

    run_id, _ = decide(executor, script)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.FAILED, R.STEP_LIMIT)
    assert run.outcome_detail == {
        "limit": "read_calls",
        "maximum": MAX_DECISION_READ_CALLS,
    }
    assert len(reads) == MAX_DECISION_READ_CALLS
    calls = tool_calls(session_factory, run_id)
    assert [c.status for c in calls] == [
        *[ToolCallStatus.SUCCEEDED] * MAX_DECISION_READ_CALLS,
        ToolCallStatus.FAILED,
    ]
    assert calls[-1].error is not None
    assert calls[-1].error["code"] == "step_limit"
    assert len(script.requests) == 1


def test_model_requests_over_the_limit_are_never_made(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory)
    script = Script(*[call(USER)] * (MAX_DECISION_MODEL_REQUESTS + 1))

    run_id, _ = decide(executor, script)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.FAILED, R.STEP_LIMIT)
    assert run.outcome_detail == {
        "limit": "model_requests",
        "maximum": MAX_DECISION_MODEL_REQUESTS,
    }
    assert len(script.requests) == MAX_DECISION_MODEL_REQUESTS
    assert len(model_calls(session_factory, run_id)) == MAX_DECISION_MODEL_REQUESTS
    # Everything up to the refused request is kept.
    assert len(tool_calls(session_factory, run_id)) == MAX_DECISION_MODEL_REQUESTS


def test_rejected_responses_count_against_the_request_limit(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory)
    invalid = conclude("goal_reached", user_id=ADA)
    script = Script(
        invalid, call(USER), invalid, call(USER), call(USER), call(USER), GOAL_REACHED
    )

    run_id, _ = decide(executor, script)

    assert outcome(session_factory, run_id) == (S.FAILED, R.STEP_LIMIT)
    assert [c.status for c in model_calls(session_factory, run_id)] == [
        ModelCallStatus.FAILED,
        ModelCallStatus.SUCCEEDED,
        ModelCallStatus.FAILED,
        *[ModelCallStatus.SUCCEEDED] * 3,
    ]
    assert len(script.requests) == MAX_DECISION_MODEL_REQUESTS


# --- decision-stage model failures -----------------------------------------------


def test_a_provider_failure_after_an_observation_fails_the_run_and_keeps_the_trace(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory)
    script = Script(
        call(CAPACITY), ModelHTTPError(status_code=500, model_name=MODEL, body=None)
    )

    run_id, status = decide(executor, script)

    assert status is S.FAILED
    run = get_run(session_factory, run_id)
    assert (run.outcome_reason, run.outcome_detail) == (
        R.PLANNER_ERROR,
        {"stage": "decision", "code": "provider_error", "error_type": "ModelHTTPError"},
    )
    assert trace(session_factory, run_id) == [
        MODEL_OK,
        ("get_licence", OK),
        MODEL_FAILED,
    ]
    assert model_calls(session_factory, run_id)[-1].error == {
        "code": "provider_error",
        "message": "The model provider returned HTTP 500.",
        "error_type": "ModelHTTPError",
    }


def test_invalid_output_until_retries_run_out_fails_the_run(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory)
    text = ModelResponse(parts=[TextPart("I have assigned the seat.")], usage=usage())
    script = Script(call(ASSIGNMENTS), *[text] * (OUTPUT_RETRIES + 1))

    run_id, _ = decide(executor, script)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.FAILED, R.PLANNER_ERROR)
    assert run.outcome_detail == {
        "stage": "decision",
        "code": "output_retries_exhausted",
        "error_type": "UnexpectedModelBehavior",
    }
    assert trace(session_factory, run_id) == [
        MODEL_OK,
        ("list_user_assignments", OK),
        *[MODEL_FAILED] * (OUTPUT_RETRIES + 1),
    ]


REJECTED = {
    "argument-to-a-tool": (
        ModelResponse(parts=[ToolCallPart(ASSIGN, {"user_id": 7})], usage=usage()),
        "assign_target_licence takes no arguments",
    ),
    "unknown-tool": (call("revoke_assignment"), "'revoke_assignment' is not a tool."),
    "conclusion-with-a-tool": (
        ModelResponse(
            parts=[ToolCallPart(ASSIGN, {}), ToolCallPart("goal_reached", {})],
            usage=usage(),
        ),
        "A conclusion must be the only tool call in its response",
    ),
    "two-conclusions": (
        ModelResponse(
            parts=[
                ToolCallPart("goal_reached", {}),
                ToolCallPart("no_action_needed", {}),
            ],
            usage=usage(),
        ),
        "A conclusion must be the only tool call in its response",
    ),
    "text-only": (
        ModelResponse(parts=[TextPart("Done.")], usage=usage()),
        "Expected a tool call, got none.",
    ),
    "unknown-reason-code": (
        cannot_proceed("licence_expired"),
        "Invalid arguments: reason_code: enum",
    ),
    "free-form-field": (
        conclude("goal_reached", explanation="I assigned it."),
        "Invalid arguments: explanation: extra_forbidden",
    ),
    "id-field": (
        conclude("cannot_proceed", reason_code="user_inactive", user_id=ADA),
        "Invalid arguments: user_id: extra_forbidden",
    ),
}


@pytest.mark.parametrize(("response", "message"), REJECTED.values(), ids=REJECTED)
def test_a_rejected_response_is_recorded_retried_and_runs_no_tool(
    executor: AgentExecutor,
    session_factory: Sessions,
    response: ModelResponse,
    message: str,
) -> None:
    seed(session_factory, ada_holds=True)
    script = Script(response, call(ASSIGNMENTS), NO_ACTION_NEEDED)

    run_id, _ = decide(executor, script)

    assert outcome(session_factory, run_id) == (S.COMPLETED, R.ALREADY_SATISFIED)
    rejected = model_calls(session_factory, run_id)[0]
    assert rejected.status is ModelCallStatus.FAILED
    assert rejected.error is not None
    assert rejected.error["code"] == "invalid_output"
    assert rejected.error["message"].startswith(message)
    # Nothing in the rejected response ran; the model was told why.
    assert [c.tool_name for c in tool_calls(session_factory, run_id)] == [
        "list_user_assignments"
    ]
    retry = script.requests[1][-1]
    assert isinstance(retry, ModelRequest)
    (feedback,) = [p for p in retry.parts if isinstance(p, RetryPromptPart)]
    assert message.split(":")[0] in str(feedback.content)


def test_accepted_responses_are_recorded_as_closed_shapes(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, seats=0)
    script = Script(call(CAPACITY, USER), cannot_proceed("no_seats_available"))

    run_id, _ = decide(executor, script)

    assert [c.output for c in model_calls(session_factory, run_id)] == [
        {"kind": "tool_calls", "tool_names": [CAPACITY, USER]},
        {"kind": "cannot_proceed", "reason_code": "no_seats_available"},
    ]


# --- what crosses the model boundary ---------------------------------------------


def test_the_model_facing_tools_take_no_arguments(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory)
    script = Script(GOAL_REACHED)

    decide(executor, script)

    (info,) = script.infos
    assert {tool.name: tool.parameters_json_schema for tool in info.function_tools} == {
        name: {"type": "object", "properties": {}, "additionalProperties": False}
        for name in (USER, CAPACITY, ASSIGNMENTS, ASSIGN)
    }
    assert all(tool.sequential for tool in info.function_tools)
    assert {
        tool.name: sorted(tool.parameters_json_schema["properties"])
        for tool in info.output_tools
    } == {
        "goal_reached": ["kind"],
        "no_action_needed": ["kind"],
        "cannot_proceed": ["kind", "reason_code"],
    }
    assert all(
        tool.parameters_json_schema["additionalProperties"] is False
        for tool in info.output_tools
    )
    assert info.allow_text_output is False


def model_visible_text(script: Script) -> str:
    """Everything the model was given as content over the whole run: the
    instructions, the tools' definitions, prompts, tool results and retry
    feedback."""
    texts: list[str] = []
    for info in script.infos:
        texts.append(info.instructions or "")
        for tool in [*info.function_tools, *info.output_tools]:
            texts.append(
                json.dumps([tool.name, tool.description, tool.parameters_json_schema])
            )
    for part in script.sent_parts():
        if isinstance(part, UserPromptPart | ToolReturnPart | RetryPromptPart):
            texts.append(str(part.content))
    return "\n".join(texts)


@pytest.mark.parametrize(
    "seed_args", [{}, {"seats": 1, "held_by_others": 1}], ids=["assigned", "rejected"]
)
def test_observations_are_exactly_what_the_model_is_given_and_carry_no_ids(
    executor: AgentExecutor, session_factory: Sessions, seed_args: dict[str, Any]
) -> None:
    seed(session_factory, **seed_args)
    script = Script(call(USER, CAPACITY, ASSIGNMENTS), call(ASSIGN), GOAL_REACHED)

    run_id, _ = decide(executor, script)

    calls = tool_calls(session_factory, run_id)
    assert len(calls) == 4
    # Returned to the model: the persisted text, exactly, in order.
    assert returned_to_model(script) == [c.observation for c in calls]
    # The internal arguments, results and errors do carry ids...
    internal = json.dumps([[c.arguments, c.result, c.error] for c in calls])
    assert str(ADA) in internal and str(FIGMA) in internal
    # ...and nothing the model was given does.
    visible = model_visible_text(script)
    for row_id in DISTINCTIVE_IDS:
        assert row_id not in visible
    assert "ada@example.com" in visible and "Figma" in visible


def test_a_rejection_reaches_the_model_as_a_code_never_as_the_error_message(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory, seats=0)
    script = Script(call(ASSIGN), cannot_proceed("no_seats_available"))

    run_id, _ = decide(executor, script)

    (attempt,) = tool_calls(session_factory, run_id)
    assert attempt.error is not None
    assert str(FIGMA) in attempt.error["message"]  # "Licence 97531 has no ..."
    assert returned_to_model(script) == [
        '{"outcome":"rejected","reason_code":"no_seats_available"}'
    ]


def test_the_initial_context_is_persisted_first_id_free_and_exactly_as_sent(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    seed(session_factory)
    run_id = resolved_run(executor)
    seen: dict[str, Any] = {}

    def check_persisted_first() -> ModelResponse:
        run = get_run(session_factory, run_id)
        seen["status"], seen["context"] = run.status, run.decision_context
        return GOAL_REACHED

    script = Script(check_persisted_first)

    executor.decide(run_id, script.planner())

    run = get_run(session_factory, run_id)
    assert run.decision_context is not None
    assert seen == {"status": S.EXECUTING, "context": run.decision_context}
    # Exactly what the model was sent.
    ((request,),) = script.requests
    assert isinstance(request, ModelRequest)
    (prompt,) = [p for p in request.parts if isinstance(p, UserPromptPart)]
    (info,) = script.infos
    assert run.decision_context == {
        "instructions": info.instructions,
        "prompt": prompt.content,
    }
    # Semantic values in, row ids out.
    assert "ada@example.com" in run.decision_context["prompt"]
    assert "Figma" in run.decision_context["prompt"]
    for row_id in DISTINCTIVE_IDS:
        assert row_id not in json.dumps(run.decision_context)
    # It can be rebuilt from the run's persisted semantic goal alone.
    assert run.goal_type is not None
    assert run.extracted_user_email is not None
    assert run.extracted_product is not None
    rebuilt = decision_context(run.extracted_user_email, run.extracted_product)
    assert rebuilt.model_dump(mode="json") == run.decision_context


# --- the real default model is never called in tests -----------------------------

DEFAULT_MODEL = Settings.model_fields["planner_model"].default


def test_real_decision_requests_are_blocked_even_with_a_key(
    executor: AgentExecutor,
    session_factory: Sessions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    seed(session_factory)
    run_id = resolved_run(executor)

    assert pydantic_ai.models.ALLOW_MODEL_REQUESTS is False
    executor.decide(run_id, PydanticAIDecisionPlanner(DEFAULT_MODEL, timeout_seconds=5))

    # Refused before any network access, recorded, and the run ended.
    assert outcome(session_factory, run_id) == (S.FAILED, R.UNEXPECTED_ERROR)
    (call_record,) = model_calls(session_factory, run_id)
    assert call_record.model_name == "claude-sonnet-5"
    assert call_record.error is not None
    assert call_record.error["error_type"] == "RuntimeError"
    assert tool_calls(session_factory, run_id) == []


def test_the_default_decision_model_needs_no_key_until_it_is_used(
    executor: AgentExecutor,
    session_factory: Sessions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    seed(session_factory)
    run_id = resolved_run(executor)

    planner = PydanticAIDecisionPlanner(DEFAULT_MODEL, timeout_seconds=5)
    executor.decide(run_id, planner)

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason, run.outcome_detail) == (
        S.FAILED,
        R.PLANNER_ERROR,
        {"stage": "decision", "code": "configuration_error", "error_type": "UserError"},
    )
    assert model_calls(session_factory, run_id) == []


# --- failures the model is not shown -----------------------------------------------


def test_a_tool_failure_the_model_is_not_shown_ends_the_run_at_once(
    executor: AgentExecutor, session_factory: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(session_factory)

    real = tools.run

    def broken(tool: Any, goal: Any, **services: Any) -> Any:
        if tool == CAPACITY:
            raise RuntimeError(f"database said something about licence {FIGMA}")
        return real(tool, goal, **services)

    monkeypatch.setattr(tools, "run", broken)
    script = Script(call(CAPACITY), GOAL_REACHED)

    run_id, _ = decide(executor, script)

    assert outcome(session_factory, run_id) == (S.FAILED, R.TOOL_FAILED)
    (failed,) = tool_calls(session_factory, run_id)
    assert (failed.status, failed.observation) == (ToolCallStatus.FAILED, None)
    assert len(script.requests) == 1  # never asked to react to it
    assert get_run(session_factory, run_id).decision_proposal is None
