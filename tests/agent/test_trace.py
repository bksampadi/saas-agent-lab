"""The trace substrate: a run received before any model call, persisted model
calls, and one ordered trace per run, shared by model calls and tool calls."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.agent.executor import AgentExecutor, RunNotExecutable
from app.agent.pydantic_ai_planner import PydanticAIIntentPlanner
from app.models import (
    AgentRun,
    AgentRunStatus,
    Licence,
    ModelCall,
    ModelCallStage,
    ModelCallStatus,
    ToolCall,
    User,
)
from app.repositories.agent_runs import AgentRunRepository
from app.repositories.model_calls import ModelCallRepository
from app.services.errors import InvalidInput
from support import ASSIGN, ASSIGNMENTS, GOAL_REACHED, Script, call, conclude

HUMAN = "admin@example.com"
INSTRUCTION = "Give ada@example.com a Figma seat."
Sessions = sessionmaker[Session]


# --- helpers ------------------------------------------------------------------


def add(sessions: Sessions, row: User | Licence) -> int:
    with sessions.begin() as session:
        session.add(row)
        session.flush()
        return row.id


def get_run(sessions: Sessions, run_id: int) -> AgentRun:
    with sessions() as session:
        run = session.get(AgentRun, run_id)
        assert run is not None
        return run


def trace(sessions: Sessions, run_id: int) -> list[ModelCall | ToolCall]:
    with sessions() as session:
        return AgentRunRepository(session).list_trace(run_id)


def record_model_call(sessions: Sessions, run_id: int) -> int:
    """Persist a model call the way any model stage does: in its own log
    transaction, taking the run's next sequence_no."""
    with sessions.begin() as session:
        calls = ModelCallRepository(session)
        call = calls.add(
            ModelCall(
                agent_run_id=run_id,
                sequence_no=calls.next_sequence_no(run_id),
                stage=ModelCallStage.DECISION,
                model_name="test-model",
                status=ModelCallStatus.SUCCEEDED,
                input_tokens=10,
                output_tokens=2,
                latency_ms=7,
                output={"chosen": "get_user"},
            )
        )
        return call.sequence_no


@pytest.fixture
def seed(session_factory: Sessions) -> tuple[int, int]:
    return (
        add(session_factory, User(email="ada@example.com", name="Ada")),
        add(session_factory, Licence(product="Figma", seats_total=5)),
    )


# --- receive_run --------------------------------------------------------------


def test_receive_run_persists_a_run_with_no_goal_before_any_model_call(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = executor.receive_run(
        instruction="  give ADA a figma seat ", requesting_actor="  admin@example.com "
    )

    run = get_run(session_factory, run_id)
    assert run.status is AgentRunStatus.RECEIVED
    assert run.instruction == "  give ADA a figma seat "
    assert run.requesting_actor == HUMAN
    assert (
        run.goal_type,
        run.desired_state,
        run.extracted_user_email,
        run.extracted_product,
    ) == (None, None, None, None)
    assert run.last_sequence_no == 0
    assert trace(session_factory, run_id) == []


@pytest.mark.parametrize(
    ("instruction", "actor"),
    [
        *[(instruction, HUMAN) for instruction in ("", "   ", "x" * 2001)],
        *[
            (INSTRUCTION, actor)
            for actor in (
                "",
                "   ",
                "a" * 321,
                "agent:run-1",
                " Agent:run-1",
                "AGENT:x",
            )
        ],
    ],
)
def test_receive_run_rejects_invalid_input_and_records_nothing(
    executor: AgentExecutor, session_factory: Sessions, instruction: str, actor: str
) -> None:
    with pytest.raises(InvalidInput):
        executor.receive_run(instruction=instruction, requesting_actor=actor)

    with session_factory() as session:
        assert session.scalars(select(AgentRun)).all() == []


def test_a_received_run_is_not_resolved_before_its_goal_is_extracted(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = executor.receive_run(instruction=INSTRUCTION, requesting_actor=HUMAN)

    with pytest.raises(RunNotExecutable):
        executor.resolve_run(run_id)

    assert get_run(session_factory, run_id).status is AgentRunStatus.RECEIVED


# --- the shared trace counter -------------------------------------------------


def test_the_counter_hands_out_increasing_numbers_and_persists_them(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = executor.receive_run(instruction=INSTRUCTION, requesting_actor=HUMAN)

    with session_factory.begin() as session:
        runs = AgentRunRepository(session)
        numbers = [runs.next_sequence_no(run_id) for _ in range(3)]

    assert numbers == [1, 2, 3]
    assert get_run(session_factory, run_id).last_sequence_no == 3


def test_a_whole_run_has_one_order_that_ignores_timestamps(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    # Extraction's model call, then each decision request and the tool call
    # it led to, all on the run's one counter.
    extraction = Script(
        conclude("ensure_assignment", user_email="ada@example.com", product="Figma")
    )
    decision = Script(call(ASSIGNMENTS), call(ASSIGN), GOAL_REACHED)
    run_id = executor.run(
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
        intent_planner=PydanticAIIntentPlanner(extraction.model(), timeout_seconds=5),
        decision_planner=decision.planner(),
    )
    tie = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    with session_factory.begin() as session:
        for model in (ModelCall, ToolCall):
            session.execute(
                update(model).where(model.agent_run_id == run_id).values(created_at=tie)
            )

    entries = trace(session_factory, run_id)

    assert [
        (
            entry.sequence_no,
            entry.stage.value if isinstance(entry, ModelCall) else entry.tool_name,
        )
        for entry in entries
    ] == [
        (1, "extraction"),
        (2, "decision"),
        (3, "list_user_assignments"),
        (4, "decision"),
        (5, "assign_licence"),
        (6, "decision"),
    ]
    assert {entry.created_at for entry in entries} == {tie}
    assert get_run(session_factory, run_id).last_sequence_no == 6


def test_each_run_has_its_own_counter(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    first = executor.receive_run(instruction=INSTRUCTION, requesting_actor=HUMAN)
    second = executor.receive_run(instruction=INSTRUCTION, requesting_actor=HUMAN)

    record_model_call(session_factory, first)
    record_model_call(session_factory, second)
    record_model_call(session_factory, first)

    assert [entry.sequence_no for entry in trace(session_factory, first)] == [1, 2]
    assert [entry.sequence_no for entry in trace(session_factory, second)] == [1]
