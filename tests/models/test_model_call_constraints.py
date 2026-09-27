"""Database-level guarantees for model calls, and for agent runs whose goal is
extracted from natural language: these must hold even if the executor has a
bug."""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    AgentRun,
    AgentRunStatus,
    DesiredState,
    GoalType,
    Licence,
    ModelCall,
    ModelCallStage,
    ModelCallStatus,
    OutcomeReason,
    User,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
S = AgentRunStatus
R = OutcomeReason
GOAL = {
    "goal_type": GoalType.ENSURE_ASSIGNMENT,
    "desired_state": DesiredState.ASSIGNED,
    "extracted_user_email": "ada@example.com",
    "extracted_product": "Figma",
}


@pytest.fixture
def ids(session: Session) -> tuple[int, int]:
    user = User(email="ada@example.com", name="Ada")
    licence = Licence(product="Figma", seats_total=5)
    session.add_all([user, licence])
    session.flush()
    return user.id, licence.id


def add_run(session: Session, **values: Any) -> AgentRun:
    run = AgentRun(
        instruction="Give ada@example.com Figma.",
        requesting_actor="admin@example.com",
        **{"status": S.RECEIVED, **values},
    )
    session.add(run)
    session.flush()
    return run


def model_call(run_id: int, **overrides: Any) -> ModelCall:
    values: dict[str, Any] = {
        "agent_run_id": run_id,
        "sequence_no": 1,
        "stage": ModelCallStage.EXTRACTION,
        "model_name": "claude-sonnet-5",
        "status": ModelCallStatus.SUCCEEDED,
        "input_tokens": 120,
        "output_tokens": 15,
        "latency_ms": 250,
        "output": {"kind": "unsupported", "reason_code": "not_a_request"},
    }
    values.update(overrides)
    return ModelCall(**values)


def terminal(reason: OutcomeReason) -> dict[str, Any]:
    return {"outcome_reason": reason, "completed_at": NOW}


# --- agent_runs: the goal and extraction outcomes ------------------------------


VALID_RUNS = {
    "received-awaiting-extraction": lambda ids: {},
    "received-extracted": lambda ids: GOAL,
    "unsupported-without-goal": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **terminal(R.UNSUPPORTED_REQUEST),
    },
    "unclear-without-goal": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **terminal(R.INSTRUCTION_UNCLEAR),
    },
    "planner-error-without-goal": lambda ids: {
        "status": S.FAILED,
        **terminal(R.PLANNER_ERROR),
    },
    # A later model stage (decision) may fail after resolution.
    "planner-error-after-resolution": lambda ids: {
        "status": S.FAILED,
        **GOAL,
        "resolved_user_id": ids[0],
        "resolved_licence_id": ids[1],
        **terminal(R.PLANNER_ERROR),
    },
    "resolution-outcome-with-goal": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **GOAL,
        **terminal(R.USER_NOT_FOUND),
    },
}

INVALID_RUNS = {
    "goal-type-alone": lambda ids: {"goal_type": GoalType.ENSURE_ASSIGNMENT},
    "goal-without-product": lambda ids: {**GOAL, "extracted_product": None},
    "goal-without-desired-state": lambda ids: {**GOAL, "desired_state": None},
    "resolved-without-goal": lambda ids: {
        "status": S.RESOLVED,
        "resolved_user_id": ids[0],
        "resolved_licence_id": ids[1],
    },
    "unsupported-with-goal": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **GOAL,
        **terminal(R.UNSUPPORTED_REQUEST),
    },
    "unclear-with-goal": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **GOAL,
        **terminal(R.INSTRUCTION_UNCLEAR),
    },
    "planner-error-needing-clarification": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **terminal(R.PLANNER_ERROR),
    },
    "unsupported-failed": lambda ids: {
        "status": S.FAILED,
        **terminal(R.UNSUPPORTED_REQUEST),
    },
    "negative-counter": lambda ids: {"last_sequence_no": -1},
}


@pytest.mark.parametrize("make", VALID_RUNS.values(), ids=VALID_RUNS)
def test_consistent_runs_are_accepted(
    session: Session, ids: tuple[int, int], make: Any
) -> None:
    add_run(session, **make(ids))


@pytest.mark.parametrize("make", INVALID_RUNS.values(), ids=INVALID_RUNS)
def test_inconsistent_runs_are_rejected(
    session: Session, ids: tuple[int, int], make: Any
) -> None:
    with pytest.raises(IntegrityError):
        add_run(session, **make(ids))


def test_the_counter_starts_at_zero_in_the_database(session: Session) -> None:
    # Raw SQL, so the ORM's Python-side default (0) plays no part.
    session.execute(
        text(
            "INSERT INTO agent_runs (instruction, requesting_actor, status, "
            "created_at, updated_at) "
            "VALUES ('i', 'a', 'received', '2026-09-27', '2026-09-27')"
        )
    )

    counter = session.execute(text("SELECT last_sequence_no FROM agent_runs")).scalar()

    assert counter == 0


# --- model_calls --------------------------------------------------------------


FAILED = ModelCallStatus.FAILED
ERROR = {
    "code": "timeout",
    "message": "The model request timed out.",
    "error_type": "X",
}

VALID_CALLS = {
    "succeeded": {},
    "failed-without-a-response": {
        "status": FAILED,
        "output": None,
        "error": ERROR,
        "input_tokens": None,
        "output_tokens": None,
    },
    "failed-with-a-rejected-response": {
        "status": FAILED,
        "output": None,
        "error": ERROR,
    },
    "decision-stage": {"stage": ModelCallStage.DECISION},
}

INVALID_CALLS = {
    "succeeded-without-output": {"output": None},
    "succeeded-with-error": {"error": ERROR},
    "succeeded-without-tokens": {"input_tokens": None, "output_tokens": None},
    "failed-without-error": {"status": FAILED, "output": None},
    "failed-with-output": {"status": FAILED, "error": ERROR},
    "only-input-tokens": {"output_tokens": None},
    "negative-tokens": {"input_tokens": -1},
    "negative-latency": {"latency_ms": -1},
    "sequence-zero": {"sequence_no": 0},
}


@pytest.mark.parametrize("overrides", VALID_CALLS.values(), ids=VALID_CALLS)
def test_consistent_model_calls_are_accepted(
    session: Session, overrides: dict[str, Any]
) -> None:
    run = add_run(session)
    session.add(model_call(run.id, **overrides))
    session.flush()


@pytest.mark.parametrize("overrides", INVALID_CALLS.values(), ids=INVALID_CALLS)
def test_inconsistent_model_calls_are_rejected(
    session: Session, overrides: dict[str, Any]
) -> None:
    run = add_run(session)
    session.add(model_call(run.id, **overrides))

    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize(("column", "value"), [("stage", "planning"), ("status", "ok")])
def test_model_call_enums_reject_unknown_values(
    session: Session, column: str, value: str
) -> None:
    run = add_run(session)
    call = model_call(run.id)
    session.add(call)
    session.flush()

    with pytest.raises(IntegrityError):
        session.execute(
            text(f"UPDATE model_calls SET {column} = :value WHERE id = :id"),
            {"value": value, "id": call.id},
        )


def test_sequence_number_is_unique_within_a_run(session: Session) -> None:
    run = add_run(session)
    session.add(model_call(run.id, sequence_no=1))
    session.flush()

    session.add(model_call(run.id, sequence_no=1))
    with pytest.raises(IntegrityError, match="UNIQUE"):
        session.flush()


def test_model_call_requires_an_existing_run(session: Session) -> None:
    session.add(model_call(999))

    with pytest.raises(IntegrityError):
        session.flush()


def test_a_run_with_model_calls_cannot_be_deleted(session: Session) -> None:
    run = add_run(session)
    session.add(model_call(run.id))
    session.flush()

    session.delete(run)
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        session.flush()
