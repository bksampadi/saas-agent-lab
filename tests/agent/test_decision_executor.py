"""The decision stage at the executor boundary, with deterministic fake
planners where no model behaviour is needed: which runs may decide, what a
planner is handed, how its failures end the run, the whole run
(AgentExecutor.run), and that no transaction is open while a model is asked
anything."""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.agent.decision import DecisionStopped
from app.agent.executor import AgentExecutor, RunNotExecutable
from app.agent.planner import (
    CallTool,
    DecisionModelCallRecorder,
    ModelCallRecorder,
    PlannerError,
)
from app.agent.pydantic_ai_decision import PydanticAIDecisionPlanner
from app.core.database import create_db_engine
from app.models import (
    AgentRun,
    AgentRunStatus,
    Assignment,
    Base,
    Licence,
    ModelCall,
    ModelCallStage,
    OutcomeReason,
    User,
)
from app.repositories.agent_runs import AgentRunRepository
from app.schemas.agent import (
    DecisionContext,
    DecisionProposal,
    EnsureAssignmentIntent,
    ExtractedIntent,
    GoalReached,
    ModelCallRecord,
    NeedsClarification,
    NoActionNeeded,
)
from support import (
    ASSIGN,
    CAPACITY,
    GOAL_REACHED,
    HUMAN,
    Act,
    FakeDecisionPlanner,
    Script,
    call,
    extracted_run,
    resolved_run,
)

INSTRUCTION = "Give ada@example.com a Figma seat."
Sessions = sessionmaker[Session]
S = AgentRunStatus
R = OutcomeReason


class FakeIntentPlanner:
    def __init__(self, answer: ExtractedIntent) -> None:
        self.answer = answer

    def plan(self, instruction: str, calls: ModelCallRecorder) -> ExtractedIntent:
        calls.record(record(self.answer.model_dump(mode="json")))
        return self.answer


def record(output: dict[str, Any]) -> ModelCallRecord:
    return ModelCallRecord(
        model_name="fake-model",
        input_tokens=10,
        output_tokens=2,
        latency_ms=7,
        output=output,
        error=None,
    )


def answer(proposal: DecisionProposal) -> Act:
    def act(
        context: DecisionContext, call_tool: CallTool, calls: DecisionModelCallRecorder
    ) -> DecisionProposal:
        calls.before_request()
        calls.record(record(proposal.model_dump(mode="json")))
        return proposal

    return act


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


def rows(sessions: Sessions, model: type[Any]) -> int:
    with sessions() as session:
        return len(session.scalars(select(model)).all())


@pytest.fixture
def seed(session_factory: Sessions) -> tuple[int, int]:
    return (
        add(session_factory, User(email="ada@example.com", name="Ada")),
        add(session_factory, Licence(product="Figma", seats_total=5)),
    )


# --- which runs may decide ------------------------------------------------------


def test_only_a_resolved_run_can_decide(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    planner = FakeDecisionPlanner(answer(GoalReached()))
    run_id = extracted_run(executor)

    with pytest.raises(RunNotExecutable):
        executor.decide(run_id, planner)

    assert planner.contexts == []
    run = get_run(session_factory, run_id)
    assert (run.status, run.decision_context) == (S.RECEIVED, None)


def test_a_run_decides_only_once(
    executor: AgentExecutor, seed: tuple[int, int]
) -> None:
    run_id = resolved_run(executor)
    executor.decide(run_id, FakeDecisionPlanner(answer(NoActionNeeded())))

    with pytest.raises(RunNotExecutable):
        executor.decide(run_id, FakeDecisionPlanner(answer(NoActionNeeded())))


def test_a_planner_is_handed_the_context_already_persisted(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    run_id = resolved_run(executor)
    planner = FakeDecisionPlanner(answer(NoActionNeeded()))

    executor.decide(run_id, planner)

    (context,) = planner.contexts
    assert get_run(session_factory, run_id).decision_context == context.model_dump()


# --- planner failures -----------------------------------------------------------


def test_a_planner_error_fails_the_run(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    def act(*_: Any) -> DecisionProposal:
        raise PlannerError("timeout", "ModelAPIError")

    run_id = resolved_run(executor)
    executor.decide(run_id, FakeDecisionPlanner(act))

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason, run.outcome_detail) == (
        S.FAILED,
        R.PLANNER_ERROR,
        {"stage": "decision", "code": "timeout", "error_type": "ModelAPIError"},
    )


def test_an_unexpected_planner_exception_fails_the_run_without_its_message(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    def act(*_: Any) -> DecisionProposal:
        raise RuntimeError("secret detail 48213")

    run_id = resolved_run(executor)
    executor.decide(run_id, FakeDecisionPlanner(act))

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason, run.outcome_detail) == (
        S.FAILED,
        R.UNEXPECTED_ERROR,
        {"stage": "decision", "error_type": "RuntimeError"},
    )


def test_before_request_refuses_the_request_over_the_limit(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    # A real planner calls before_request before every request.
    allowed: list[int] = []

    def act(
        context: DecisionContext, call_tool: CallTool, calls: DecisionModelCallRecorder
    ) -> DecisionProposal:
        while True:
            calls.before_request()
            allowed.append(1)
            calls.record(record({"kind": "tool_calls", "tool_names": []}))

    run_id = resolved_run(executor)
    executor.decide(run_id, FakeDecisionPlanner(act))

    assert len(allowed) == rows(session_factory, ModelCall) == 6
    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason, run.outcome_detail) == (
        S.FAILED,
        R.STEP_LIMIT,
        {"limit": "model_requests", "maximum": 6},
    )


def test_a_limit_is_an_exception_through_the_planner_not_an_observation(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    ended: list[tuple[AgentRunStatus, OutcomeReason | None]] = []

    def act(
        context: DecisionContext, call_tool: CallTool, calls: DecisionModelCallRecorder
    ) -> DecisionProposal:
        call_tool(ASSIGN)
        try:
            call_tool(ASSIGN)
        except DecisionStopped:
            # The run has already ended where the limit was counted.
            run = get_run(session_factory, run_id)
            ended.append((run.status, run.outcome_reason))
            raise
        return GoalReached()

    run_id = resolved_run(executor)
    status = executor.decide(run_id, FakeDecisionPlanner(act))

    assert status is S.FAILED
    assert ended == [(S.FAILED, R.STEP_LIMIT)]


# --- the whole run ---------------------------------------------------------------


def test_an_instruction_runs_to_completion_with_model_chosen_tool_calls(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    run_id = executor.run(
        instruction=INSTRUCTION,
        requesting_actor=HUMAN,
        intent_planner=FakeIntentPlanner(
            EnsureAssignmentIntent(user_email="ada@example.com", product="Figma")
        ),
        decision_planner=Script(call(ASSIGN), GOAL_REACHED).planner(),
    )

    run = get_run(session_factory, run_id)
    assert (run.status, run.outcome_reason) == (S.COMPLETED, R.GOAL_SATISFIED)
    with session_factory() as session:
        entries = AgentRunRepository(session).list_trace(run_id)
    # One trace, one counter: extraction, then decision, in order.
    assert [entry.sequence_no for entry in entries] == [1, 2, 3, 4]
    assert [
        entry.stage if isinstance(entry, ModelCall) else entry.tool_name
        for entry in entries
    ] == [
        ModelCallStage.EXTRACTION,
        ModelCallStage.DECISION,
        "assign_licence",
        ModelCallStage.DECISION,
    ]


def test_an_instruction_that_needs_clarification_never_reaches_a_decision(
    executor: AgentExecutor, session_factory: Sessions, seed: tuple[int, int]
) -> None:
    decider = FakeDecisionPlanner(answer(GoalReached()))

    run_id = executor.run(
        instruction="Give Ada Figma.",
        requesting_actor=HUMAN,
        intent_planner=FakeIntentPlanner(
            NeedsClarification(reason_code="missing_user_email")
        ),
        decision_planner=decider,
    )

    assert get_run(session_factory, run_id).status is S.NEEDS_CLARIFICATION
    assert decider.contexts == []


# --- no transaction is open while the model is asked anything ------------------


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
    # As in test_isolation: every session has its own connection, so a
    # transaction left open would make another writer fail at once.
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


def test_no_transaction_is_open_during_any_decision_model_request(
    locking_engine: Engine,
) -> None:
    sessions = sessionmaker(
        bind=locking_engine, autoflush=False, expire_on_commit=False
    )
    add(sessions, User(email="ada@example.com", name="Ada"))
    add(sessions, Licence(product="Figma", seats_total=5))
    executor = AgentExecutor(sessions)
    run_id = resolved_run(executor)

    counter = TransactionCounter()
    event.listen(locking_engine, "begin", counter.began)
    event.listen(locking_engine, "commit", counter.ended)
    event.listen(locking_engine, "rollback", counter.ended)
    open_at_request: list[int] = []
    responses = [call(CAPACITY), call(ASSIGN), GOAL_REACHED]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        open_at_request.append(counter.open_now)
        # A write here would fail at once if any transaction held a lock.
        add(sessions, User(email=f"probe{len(open_at_request)}@example.com", name="P"))
        return responses.pop(0)

    planner = PydanticAIDecisionPlanner(
        FunctionModel(respond, model_name="scripted-model"), timeout_seconds=5
    )
    status = executor.decide(run_id, planner)

    assert status is S.COMPLETED
    assert open_at_request == [0, 0, 0]
    assert (counter.peak, counter.open_now) == (1, 0)
    assert rows(sessions, Assignment) == 1
