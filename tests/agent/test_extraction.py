"""Extraction through the executor, with a fake planner in place of a model:
the run's lifecycle, persisted model calls, and the whole run's ordered
trace. The trace substrate itself is tested in test_trace.py."""

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.agent.executor import AgentExecutor, RunNotExecutable
from app.agent.harness import run_instruction
from app.agent.planner import ModelCallRecorder, PlannerError
from app.core.database import create_db_engine
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    Base,
    DesiredState,
    GoalType,
    Licence,
    ModelCall,
    ModelCallStatus,
    OutcomeReason,
    ToolCall,
    User,
)
from app.repositories.agent_runs import AgentRunRepository
from app.schemas.agent import (
    EnsureAssignmentIntent,
    ExtractedAssignmentIntent,
    ExtractedIntent,
    ModelCallError,
    ModelCallRecord,
    NeedsClarification,
    Unsupported,
)
from app.services.assignments import AssignmentService

HUMAN = "admin@example.com"
INSTRUCTION = "Give ada@example.com a Figma seat."
Sessions = sessionmaker[Session]


class FakePlanner:
    """An IntentPlanner with a canned answer. Like a real one, it records a
    model call for its answer, failed or not, before returning or raising."""

    def __init__(
        self,
        answer: ExtractedIntent | PlannerError | Exception,
        *,
        during: Callable[[], None] | None = None,
    ) -> None:
        self.answer = answer
        self.during = during  # run while planning, e.g. to inspect the database
        self.instructions: list[str] = []

    def plan(self, instruction: str, calls: ModelCallRecorder) -> ExtractedIntent:
        self.instructions.append(instruction)
        if self.during is not None:
            self.during()
        if isinstance(self.answer, Exception):
            calls.record(
                ModelCallRecord(
                    model_name="fake-model",
                    input_tokens=None,
                    output_tokens=None,
                    latency_ms=7,
                    output=None,
                    error=ModelCallError(
                        code="provider_error", message="Fake.", error_type="Fake"
                    ),
                )
            )
            raise self.answer
        calls.record(
            ModelCallRecord(
                model_name="fake-model",
                input_tokens=10,
                output_tokens=2,
                latency_ms=7,
                output=self.answer.model_dump(mode="json"),
                error=None,
            )
        )
        return self.answer


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


def rows(sessions: Sessions, model: type[ModelCall | ToolCall | Assignment]) -> int:
    with sessions() as session:
        return len(session.scalars(select(model)).all())


def received(executor: AgentExecutor, instruction: str = INSTRUCTION) -> int:
    return executor.receive_run(instruction=instruction, requesting_actor=HUMAN)


def ada_figma() -> EnsureAssignmentIntent:
    return EnsureAssignmentIntent(user_email="ada@example.com", product="Figma")


@pytest.fixture
def seed(session_factory: Sessions) -> tuple[int, int]:
    return (
        add(session_factory, User(email="ada@example.com", name="Ada")),
        add(session_factory, Licence(product="Figma", seats_total=5)),
    )


# --- extract_intent -----------------------------------------------------------


def test_the_run_exists_and_is_received_while_the_planner_runs(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = received(executor)
    seen: list[AgentRunStatus] = []
    planner = FakePlanner(
        ada_figma(), during=lambda: seen.append(get_run(session_factory, run_id).status)
    )

    executor.extract_intent(run_id, planner)

    assert seen == [AgentRunStatus.RECEIVED]
    assert planner.instructions == [INSTRUCTION]


def test_an_assignment_intent_becomes_the_goal_exactly_as_extracted(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    run_id = received(executor, "Give ADA@Example.com a figma seat.")
    intent = EnsureAssignmentIntent(user_email=" ADA@Example.com", product="figma")

    status = executor.extract_intent(run_id, FakePlanner(intent))

    assert status is AgentRunStatus.RECEIVED
    run = get_run(session_factory, run_id)
    assert (run.goal_type, run.desired_state) == (
        GoalType.ENSURE_ASSIGNMENT,
        DesiredState.ASSIGNED,
    )
    assert (run.extracted_user_email, run.extracted_product) == (
        " ADA@Example.com",
        "figma",
    )
    assert run.resolved_user_id is None  # extraction resolves nothing
    # The Day 1 resolver takes it from here, unchanged.
    assert executor.resolve_run(run_id) is AgentRunStatus.RESOLVED
    assert (
        executor.get_goal(run_id).user_id,
        executor.get_goal(run_id).licence_id,
    ) == (seed)


@pytest.mark.parametrize(
    ("answer", "reason", "detail"),
    [
        (
            NeedsClarification(reason_code="missing_user_email"),
            OutcomeReason.INSTRUCTION_UNCLEAR,
            {"reason_code": "missing_user_email"},
        ),
        (
            NeedsClarification(reason_code="multiple_products"),
            OutcomeReason.INSTRUCTION_UNCLEAR,
            {"reason_code": "multiple_products"},
        ),
        (
            Unsupported(reason_code="unsupported_action"),
            OutcomeReason.UNSUPPORTED_REQUEST,
            {"reason_code": "unsupported_action"},
        ),
        (
            Unsupported(reason_code="additional_request"),
            OutcomeReason.UNSUPPORTED_REQUEST,
            {"reason_code": "additional_request"},
        ),
    ],
)
def test_other_intents_end_the_run_needing_clarification_with_no_goal(
    executor: AgentExecutor,
    session_factory: Sessions,
    answer: ExtractedIntent,
    reason: OutcomeReason,
    detail: dict[str, str],
) -> None:
    run_id = received(executor)

    status = executor.extract_intent(run_id, FakePlanner(answer))

    assert status is AgentRunStatus.NEEDS_CLARIFICATION
    run = get_run(session_factory, run_id)
    assert (run.outcome_reason, run.outcome_detail) == (reason, detail)
    assert run.completed_at is not None
    assert (run.goal_type, run.extracted_user_email) == (None, None)
    with pytest.raises(RunNotExecutable):
        executor.resolve_run(run_id)


def test_a_planner_error_fails_the_run_and_keeps_the_failed_call(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = received(executor)

    status = executor.extract_intent(
        run_id, FakePlanner(PlannerError("timeout", "ModelAPIError"))
    )

    assert status is AgentRunStatus.FAILED
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.PLANNER_ERROR
    assert run.outcome_detail == {
        "stage": "extraction",
        "code": "timeout",
        "error_type": "ModelAPIError",
    }
    (call,) = trace(session_factory, run_id)
    assert isinstance(call, ModelCall)
    assert call.status is ModelCallStatus.FAILED


def test_an_unexpected_planner_exception_fails_the_run(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = received(executor)

    status = executor.extract_intent(run_id, FakePlanner(RuntimeError("boom")))

    assert status is AgentRunStatus.FAILED
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.UNEXPECTED_ERROR
    assert run.outcome_detail == {"stage": "extraction", "error_type": "RuntimeError"}


@pytest.mark.parametrize(
    ("intent", "fields"),
    [
        (
            EnsureAssignmentIntent(user_email="alice@example.com", product="Figma"),
            ["user_email"],
        ),
        (
            EnsureAssignmentIntent(user_email="ada@example.com", product="GitHub"),
            ["product"],
        ),
        (EnsureAssignmentIntent(user_email=" ", product=""), ["user_email", "product"]),
    ],
)
def test_text_that_is_not_in_the_instruction_is_never_resolved(
    executor: AgentExecutor,
    session_factory: Sessions,
    intent: EnsureAssignmentIntent,
    fields: list[str],
) -> None:
    # Checked by the executor for any planner, not only by the PydanticAI one.
    run_id = received(executor)

    status = executor.extract_intent(run_id, FakePlanner(intent))

    assert status is AgentRunStatus.FAILED
    run = get_run(session_factory, run_id)
    assert run.outcome_reason is OutcomeReason.PLANNER_ERROR
    assert run.outcome_detail == {
        "stage": "extraction",
        "code": "ungrounded_output",
        "error_type": None,
        "fields": fields,
    }
    assert run.goal_type is None


def test_a_run_is_extracted_only_once(
    executor: AgentExecutor, session_factory: Sessions
) -> None:
    run_id = received(executor)
    executor.extract_intent(run_id, FakePlanner(ada_figma()))
    second = FakePlanner(ada_figma())

    with pytest.raises(RunNotExecutable):
        executor.extract_intent(run_id, second)

    assert second.instructions == []  # the planner was never called
    assert len(trace(session_factory, run_id)) == 1


def test_a_run_created_with_an_intent_is_not_extracted(executor: AgentExecutor) -> None:
    run_id = executor.create_run(
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
        intent=ExtractedAssignmentIntent(user_email="ada@example.com", product="Figma"),
    )
    planner = FakePlanner(ada_figma())

    with pytest.raises(RunNotExecutable):
        executor.extract_intent(run_id, planner)

    assert planner.instructions == []


# --- the whole run, and its trace ---------------------------------------------


def test_an_instruction_runs_to_completion_through_a_fake_planner(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    run_id = run_instruction(
        executor,
        FakePlanner(ada_figma()),
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
    )

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (
        AgentRunStatus.COMPLETED,
        OutcomeReason.GOAL_SATISFIED,
    )
    assert rows(session_factory, Assignment) == 1


def test_trace_order_follows_sequence_numbers_when_timestamps_tie(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    run_id = run_instruction(
        executor,
        FakePlanner(ada_figma()),
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
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
            "model" if isinstance(entry, ModelCall) else entry.tool_name,
        )
        for entry in entries
    ] == [
        (1, "model"),
        (2, "list_user_assignments"),
        (3, "get_licence"),
        (4, "assign_licence"),
    ]
    assert {entry.created_at for entry in entries} == {tie}
    assert get_run(session_factory, run_id).last_sequence_no == 4


def test_model_calls_and_tool_calls_share_one_counter_per_run(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    first = run_instruction(
        executor,
        FakePlanner(ada_figma()),
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
    )
    second = received(executor)
    executor.extract_intent(
        second, FakePlanner(Unsupported(reason_code="not_a_request"))
    )

    assert [entry.sequence_no for entry in trace(session_factory, first)] == [
        1,
        2,
        3,
        4,
    ]
    assert [entry.sequence_no for entry in trace(session_factory, second)] == [1]


# --- no lock is held while the planner runs ----------------------------------


@pytest.fixture
def locking_sessions() -> Iterator[Sessions]:
    # As in test_isolation: every session has its own connection, so a
    # transaction left open would make another writer fail at once.
    name = f"agentlab-{uuid4().hex}"
    keeper = sqlite3.connect(f"file:{name}?mode=memory&cache=shared", uri=True)
    engine: Engine = create_db_engine(
        f"sqlite:///file:{name}?mode=memory&cache=shared&uri=true",
        poolclass=QueuePool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()
    keeper.close()


def test_no_transaction_is_open_while_the_planner_runs(
    locking_sessions: Sessions,
) -> None:
    ada = add(locking_sessions, User(email="ada@example.com", name="Ada"))
    figma = add(locking_sessions, Licence(product="Figma", seats_total=5))

    def business_write() -> None:
        with locking_sessions.begin() as session:
            AssignmentService(session).assign_licence(
                user_id=ada, licence_id=figma, actor=HUMAN
            )

    executor = AgentExecutor(locking_sessions)
    run_id = run_instruction(
        executor,
        FakePlanner(ada_figma(), during=business_write),
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
    )

    run = get_run(locking_sessions, run_id)
    assert (run.status, run.outcome_reason) == (
        AgentRunStatus.COMPLETED,
        OutcomeReason.ALREADY_SATISFIED,
    )
