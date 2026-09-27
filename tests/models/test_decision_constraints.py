"""Database-level guarantees for the decision stage: the run's decision
columns and the step_limit outcome. These must hold even if the executor has
a bug. Tool-call observations are covered with tool calls, in
test_agent_constraints.py."""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    AgentRun,
    AgentRunStatus,
    CannotProceedReason,
    DecisionProposalKind,
    DesiredState,
    GoalType,
    Licence,
    OutcomeReason,
    User,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
S = AgentRunStatus
R = OutcomeReason
P = DecisionProposalKind
CONTEXT = {"instructions": "You work for...", "prompt": "The goal: ..."}


@pytest.fixture
def ids(session: Session) -> tuple[int, int]:
    user = User(email="ada@example.com", name="Ada")
    licence = Licence(product="Figma", seats_total=5)
    session.add_all([user, licence])
    session.flush()
    return user.id, licence.id


def add_run(session: Session, ids: tuple[int, int], **values: Any) -> AgentRun:
    run = AgentRun(
        instruction="Give ada@example.com Figma.",
        requesting_actor="admin@example.com",
        goal_type=GoalType.ENSURE_ASSIGNMENT,
        desired_state=DesiredState.ASSIGNED,
        extracted_user_email="ada@example.com",
        extracted_product="Figma",
        **{
            "status": S.EXECUTING,
            "resolved_user_id": ids[0],
            "resolved_licence_id": ids[1],
            **values,
        },
    )
    session.add(run)
    session.flush()
    return run


def terminal(status: AgentRunStatus, reason: OutcomeReason) -> dict[str, Any]:
    return {"status": status, "outcome_reason": reason, "completed_at": NOW}


VALID_RUNS = {
    "deterministic": {},
    "deciding": {"decision_context": CONTEXT},
    "verifying-a-proposal": {
        "status": S.VERIFYING,
        "decision_context": CONTEXT,
        "decision_proposal": P.GOAL_REACHED,
    },
    "blocked-by-a-confirmed-claim": {
        **terminal(S.BLOCKED, R.NO_SEATS_AVAILABLE),
        "decision_context": CONTEXT,
        "decision_proposal": P.CANNOT_PROCEED,
        "decision_reason_code": CannotProceedReason.NO_SEATS_AVAILABLE,
    },
    "step-limit": {
        **terminal(S.FAILED, R.STEP_LIMIT),
        "decision_context": CONTEXT,
    },
}

INVALID_RUNS = {
    "context-before-resolution": {
        "status": S.RECEIVED,
        "resolved_user_id": None,
        "resolved_licence_id": None,
        "decision_context": CONTEXT,
    },
    "proposal-without-context": {"decision_proposal": P.NO_ACTION_NEEDED},
    "cannot-proceed-without-reason": {
        "decision_context": CONTEXT,
        "decision_proposal": P.CANNOT_PROCEED,
    },
    "reason-without-proposal": {
        "decision_context": CONTEXT,
        "decision_reason_code": CannotProceedReason.USER_INACTIVE,
    },
    "reason-with-goal-reached": {
        "decision_context": CONTEXT,
        "decision_proposal": P.GOAL_REACHED,
        "decision_reason_code": CannotProceedReason.USER_INACTIVE,
    },
    "step-limit-blocked": terminal(S.BLOCKED, R.STEP_LIMIT),
    "step-limit-completed": terminal(S.COMPLETED, R.STEP_LIMIT),
}


@pytest.mark.parametrize("values", VALID_RUNS.values(), ids=VALID_RUNS)
def test_consistent_decision_runs_are_accepted(
    session: Session, ids: tuple[int, int], values: dict[str, Any]
) -> None:
    add_run(session, ids, **values)


@pytest.mark.parametrize("values", INVALID_RUNS.values(), ids=INVALID_RUNS)
def test_inconsistent_decision_runs_are_rejected(
    session: Session, ids: tuple[int, int], values: dict[str, Any]
) -> None:
    with pytest.raises(IntegrityError):
        add_run(session, ids, **values)


@pytest.mark.parametrize(
    ("column", "value"),
    [("decision_proposal", "goal_maybe_reached"), ("decision_reason_code", "tired")],
)
def test_decision_enums_reject_unknown_values(
    session: Session, ids: tuple[int, int], column: str, value: str
) -> None:
    run = add_run(
        session,
        ids,
        decision_context=CONTEXT,
        decision_proposal=P.CANNOT_PROCEED,
        decision_reason_code=CannotProceedReason.USER_INACTIVE,
    )

    with pytest.raises(IntegrityError):
        session.execute(
            text(f"UPDATE agent_runs SET {column} = :value WHERE id = :id"),
            {"value": value, "id": run.id},
        )
