"""Database-level guarantees for agent runs and tool calls: these must hold
even if the executor has a bug."""

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
    OutcomeReason,
    PolicyDecision,
    ToolCall,
    ToolCallStatus,
)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
S = AgentRunStatus
R = OutcomeReason


def agent_run(**overrides: Any) -> AgentRun:
    values: dict[str, Any] = {
        "instruction": "Give Ada a Figma seat.",
        "requesting_actor": "admin@example.com",
        "status": S.RECEIVED,
        "goal_type": GoalType.ENSURE_ASSIGNMENT,
        "desired_state": DesiredState.ASSIGNED,
        "extracted_user_email": "ada@example.com",
        "extracted_product": "Figma",
    }
    values.update(overrides)
    return AgentRun(**values)


def add_run(session: Session, **overrides: Any) -> AgentRun:
    run = agent_run(**overrides)
    session.add(run)
    session.flush()
    return run


def tool_call(run_id: int, **overrides: Any) -> ToolCall:
    values: dict[str, Any] = {
        "agent_run_id": run_id,
        "sequence_no": 1,
        "tool_name": "get_user",
        "arguments": {"user_id": 1},
        "status": ToolCallStatus.STARTED,
    }
    values.update(overrides)
    return ToolCall(**values)


# --- agent_runs ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("status", "paused"),
        ("goal_type", "ensure_revocation"),
        ("desired_state", "revoked"),
    ],
)
def test_agent_run_enums_reject_unknown_values(
    session: Session, column: str, value: str
) -> None:
    run = add_run(session)

    with pytest.raises(IntegrityError):
        session.execute(
            text(f"UPDATE agent_runs SET {column} = :value WHERE id = :id"),
            {"value": value, "id": run.id},
        )


def resolved(ids: tuple[int, int]) -> dict[str, Any]:
    return {"resolved_user_id": ids[0], "resolved_licence_id": ids[1]}


def terminal(reason: OutcomeReason) -> dict[str, Any]:
    return {"outcome_reason": reason, "completed_at": NOW}


VALID_ROWS = {
    "received": lambda ids: {},
    "resolved": lambda ids: {"status": S.RESOLVED, **resolved(ids)},
    "executing": lambda ids: {"status": S.EXECUTING, **resolved(ids)},
    "verifying": lambda ids: {"status": S.VERIFYING, **resolved(ids)},
    "completed": lambda ids: {
        "status": S.COMPLETED,
        **resolved(ids),
        **terminal(R.GOAL_SATISFIED),
    },
    "needs-clarification": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **terminal(R.LICENCE_AMBIGUOUS),
    },
    "blocked": lambda ids: {
        "status": S.BLOCKED,
        **resolved(ids),
        **terminal(R.NO_SEATS_AVAILABLE),
    },
    "awaiting-approval": lambda ids: {"status": S.AWAITING_APPROVAL, **resolved(ids)},
    "blocked-by-policy": lambda ids: {
        "status": S.BLOCKED,
        **resolved(ids),
        **terminal(R.POLICY_DENIED),
    },
    "failed-before-resolution": lambda ids: {
        "status": S.FAILED,
        **terminal(R.UNEXPECTED_ERROR),
    },
    "failed-after-resolution": lambda ids: {
        "status": S.FAILED,
        **resolved(ids),
        **terminal(R.VERIFICATION_FAILED),
    },
}

INVALID_ROWS = {
    "only-user-resolved": lambda ids: {
        "status": S.FAILED,
        "resolved_user_id": ids[0],
        **terminal(R.TOOL_FAILED),
    },
    "resolved-without-ids": lambda ids: {"status": S.RESOLVED},
    "executing-without-ids": lambda ids: {"status": S.EXECUTING},
    "completed-without-ids": lambda ids: {
        "status": S.COMPLETED,
        **terminal(R.GOAL_SATISFIED),
    },
    "received-with-ids": lambda ids: resolved(ids),
    "clarification-with-ids": lambda ids: {
        "status": S.NEEDS_CLARIFICATION,
        **resolved(ids),
        **terminal(R.USER_NOT_FOUND),
    },
    "non-terminal-with-completed-at": lambda ids: {
        "status": S.EXECUTING,
        **resolved(ids),
        "completed_at": NOW,
    },
    "terminal-without-completed-at": lambda ids: {
        "status": S.BLOCKED,
        **resolved(ids),
        "outcome_reason": R.NO_SEATS_AVAILABLE,
    },
    "non-terminal-with-reason": lambda ids: {
        "status": S.VERIFYING,
        **resolved(ids),
        "outcome_reason": R.GOAL_SATISFIED,
    },
    "terminal-without-reason": lambda ids: {
        "status": S.FAILED,
        "completed_at": NOW,
    },
    "completed-with-a-blocked-reason": lambda ids: {
        "status": S.COMPLETED,
        **resolved(ids),
        **terminal(R.NO_SEATS_AVAILABLE),
    },
    "blocked-with-a-completed-reason": lambda ids: {
        "status": S.BLOCKED,
        **resolved(ids),
        **terminal(R.GOAL_SATISFIED),
    },
    "completed-with-a-failed-reason": lambda ids: {
        "status": S.COMPLETED,
        **resolved(ids),
        **terminal(R.VERIFICATION_FAILED),
    },
    "failed-with-a-clarification-reason": lambda ids: {
        "status": S.FAILED,
        **terminal(R.USER_NOT_FOUND),
    },
    "unknown-resolved-user": lambda ids: {
        "status": S.RESOLVED,
        "resolved_user_id": 999,
        "resolved_licence_id": ids[1],
    },
    # A paused run: resolved, not terminal.
    "awaiting-approval-without-ids": lambda ids: {"status": S.AWAITING_APPROVAL},
    "awaiting-approval-with-completed-at": lambda ids: {
        "status": S.AWAITING_APPROVAL,
        **resolved(ids),
        "completed_at": NOW,
    },
    "awaiting-approval-with-reason": lambda ids: {
        "status": S.AWAITING_APPROVAL,
        **resolved(ids),
        **terminal(R.POLICY_DENIED),
    },
    # policy_denied blocks a run; it never completes or fails one.
    "completed-with-policy-denied": lambda ids: {
        "status": S.COMPLETED,
        **resolved(ids),
        **terminal(R.POLICY_DENIED),
    },
    "failed-with-policy-denied": lambda ids: {
        "status": S.FAILED,
        **resolved(ids),
        **terminal(R.POLICY_DENIED),
    },
}


@pytest.mark.parametrize("make", VALID_ROWS.values(), ids=VALID_ROWS)
def test_consistent_agent_run_rows_are_accepted(
    session: Session, ids: tuple[int, int], make: Any
) -> None:
    add_run(session, **make(ids))


@pytest.mark.parametrize("make", INVALID_ROWS.values(), ids=INVALID_ROWS)
def test_inconsistent_agent_run_rows_are_rejected(
    session: Session, ids: tuple[int, int], make: Any
) -> None:
    with pytest.raises(IntegrityError):
        add_run(session, **make(ids))


def test_agent_run_json_none_is_stored_as_sql_null(session: Session) -> None:
    run = add_run(session, outcome_detail=None)

    stored = session.execute(
        text("SELECT outcome_detail IS NULL FROM agent_runs WHERE id = :id"),
        {"id": run.id},
    ).scalar_one()

    assert stored == 1


# --- tool_calls ---------------------------------------------------------------


def test_sequence_number_is_unique_within_a_run(session: Session) -> None:
    run = add_run(session)
    session.add(tool_call(run.id, sequence_no=1))
    session.flush()

    session.add(tool_call(run.id, sequence_no=1, tool_name="get_licence"))
    with pytest.raises(IntegrityError, match="UNIQUE"):
        session.flush()


def test_the_same_sequence_number_may_appear_in_different_runs(
    session: Session,
) -> None:
    first = add_run(session)
    second = add_run(session)

    session.add_all([tool_call(first.id), tool_call(second.id)])
    session.flush()  # must not raise


@pytest.mark.parametrize("sequence_no", [0, -1])
def test_sequence_number_starts_at_one(session: Session, sequence_no: int) -> None:
    run = add_run(session)
    session.add(tool_call(run.id, sequence_no=sequence_no))

    with pytest.raises(IntegrityError):
        session.flush()


FAILED = ToolCallStatus.FAILED
SUCCEEDED = ToolCallStatus.SUCCEEDED
ERROR = {"code": "x", "message": "y", "error_type": None}

INVALID_CALLS = {
    "started-with-completed-at": {"completed_at": NOW},
    "started-with-result": {"result": {"a": 1}},
    "started-with-error": {"error": ERROR},
    "succeeded-without-result": {"status": SUCCEEDED, "completed_at": NOW},
    "succeeded-without-completed-at": {"status": SUCCEEDED, "result": {"a": 1}},
    "succeeded-with-error": {
        "status": SUCCEEDED,
        "completed_at": NOW,
        "result": {"a": 1},
        "error": ERROR,
    },
    "failed-without-error": {"status": FAILED, "completed_at": NOW},
    "failed-with-result": {
        "status": FAILED,
        "completed_at": NOW,
        "error": ERROR,
        "result": {"a": 1},
    },
    "failed-without-completed-at": {"status": FAILED, "error": ERROR},
}

VALID_CALLS = {
    "started": {},
    "succeeded": {"status": SUCCEEDED, "completed_at": NOW, "result": {"a": 1}},
    "failed": {"status": FAILED, "completed_at": NOW, "error": ERROR},
}


@pytest.mark.parametrize("overrides", VALID_CALLS.values(), ids=VALID_CALLS)
def test_consistent_tool_calls_are_accepted(
    session: Session, overrides: dict[str, Any]
) -> None:
    run = add_run(session)
    session.add(tool_call(run.id, **overrides))
    session.flush()


@pytest.mark.parametrize("overrides", INVALID_CALLS.values(), ids=INVALID_CALLS)
def test_inconsistent_tool_calls_are_rejected(
    session: Session, overrides: dict[str, Any]
) -> None:
    run = add_run(session)
    session.add(tool_call(run.id, **overrides))

    with pytest.raises(IntegrityError):
        session.flush()


def test_tool_call_requires_an_existing_run(session: Session) -> None:
    session.add(tool_call(999))

    with pytest.raises(IntegrityError):
        session.flush()


def test_a_run_with_tool_calls_cannot_be_deleted(session: Session) -> None:
    # No cascade: history is never removed silently.
    run = add_run(session)
    session.add(tool_call(run.id))
    session.flush()

    session.delete(run)
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        session.flush()


@pytest.mark.parametrize(
    "overrides", [VALID_CALLS["succeeded"], VALID_CALLS["failed"]], ids=["ok", "failed"]
)
def test_a_finished_tool_call_may_carry_an_observation(
    session: Session, overrides: dict[str, Any]
) -> None:
    run = add_run(session)
    session.add(tool_call(run.id, observation='{"x":1}', **overrides))
    session.flush()


def test_a_started_tool_call_has_no_observation(session: Session) -> None:
    # A model sees a call's result only once the call has one.
    run = add_run(session)
    session.add(tool_call(run.id, observation="{}"))

    with pytest.raises(IntegrityError):
        session.flush()


def test_an_observation_is_stored_byte_for_byte(session: Session) -> None:
    run = add_run(session)
    observation = '{"seats_active":2,"seats_available":0,"seats_total":2}'
    call = tool_call(run.id, observation=observation, **VALID_CALLS["succeeded"])
    session.add(call)
    session.flush()

    stored = session.execute(
        text("SELECT observation FROM tool_calls WHERE id = :id"), {"id": call.id}
    ).scalar_one()

    assert stored == observation


# --- tool_calls: policy decisions and held calls ------------------------------

AWAITING = ToolCallStatus.AWAITING_APPROVAL
STARTED = ToolCallStatus.STARTED
ALLOW = PolicyDecision.ALLOW
REQUIRE = PolicyDecision.REQUIRE_APPROVAL
DENY = PolicyDecision.DENY
ASSIGN = {"tool_name": "assign_licence", "arguments": {"user_id": 1, "licence_id": 2}}

VALID_POLICY_CALLS: dict[str, dict[str, Any]] = {
    "read-without-a-decision": {},
    "allowed-and-started": {**ASSIGN, "policy_decision": ALLOW},
    "denied-and-failed": {
        **ASSIGN,
        "policy_decision": DENY,
        **VALID_CALLS["failed"],
    },
    "held-for-approval": {**ASSIGN, "status": AWAITING, "policy_decision": REQUIRE},
    # Once approved (later), a held call runs like any other.
    "required-approval-then-started": {**ASSIGN, "policy_decision": REQUIRE},
    "required-approval-then-succeeded": {
        **ASSIGN,
        "policy_decision": REQUIRE,
        **VALID_CALLS["succeeded"],
    },
}

INVALID_POLICY_CALLS: dict[str, dict[str, Any]] = {
    # The NULL case explicitly: NULL = 'require_approval' is NULL, which a
    # CHECK would let through without its IS NOT NULL.
    "held-without-a-decision": {**ASSIGN, "status": AWAITING},
    "held-though-allowed": {**ASSIGN, "status": AWAITING, "policy_decision": ALLOW},
    "held-though-denied": {**ASSIGN, "status": AWAITING, "policy_decision": DENY},
    "held-with-completed-at": {
        **ASSIGN,
        "status": AWAITING,
        "policy_decision": REQUIRE,
        "completed_at": NOW,
    },
    "held-with-result": {
        **ASSIGN,
        "status": AWAITING,
        "policy_decision": REQUIRE,
        "result": {"a": 1},
    },
    "held-with-error": {
        **ASSIGN,
        "status": AWAITING,
        "policy_decision": REQUIRE,
        "error": ERROR,
    },
    "held-with-observation": {
        **ASSIGN,
        "status": AWAITING,
        "policy_decision": REQUIRE,
        "observation": "{}",
    },
    "denied-but-started": {**ASSIGN, "status": STARTED, "policy_decision": DENY},
    "denied-but-succeeded": {
        **ASSIGN,
        "policy_decision": DENY,
        **VALID_CALLS["succeeded"],
    },
}


@pytest.mark.parametrize(
    "overrides", VALID_POLICY_CALLS.values(), ids=VALID_POLICY_CALLS
)
def test_consistent_policy_decisions_are_accepted(
    session: Session, overrides: dict[str, Any]
) -> None:
    run = add_run(session)
    session.add(tool_call(run.id, **overrides))
    session.flush()


@pytest.mark.parametrize(
    "overrides", INVALID_POLICY_CALLS.values(), ids=INVALID_POLICY_CALLS
)
def test_inconsistent_policy_decisions_are_rejected(
    session: Session, overrides: dict[str, Any]
) -> None:
    run = add_run(session)
    session.add(tool_call(run.id, **overrides))

    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize(
    ("column", "value"),
    [("policy_decision", "sometimes"), ("status", "paused")],
)
def test_tool_call_enums_reject_unknown_values(
    session: Session, column: str, value: str
) -> None:
    run = add_run(session)
    call = tool_call(run.id)
    session.add(call)
    session.flush()

    with pytest.raises(IntegrityError):
        session.execute(
            text(f"UPDATE tool_calls SET {column} = :value WHERE id = :id"),
            {"value": value, "id": call.id},
        )


def test_tool_call_json_none_is_stored_as_sql_null(session: Session) -> None:
    run = add_run(session)
    call = tool_call(run.id, result=None, error=None)
    session.add(call)
    session.flush()

    stored = session.execute(
        text("SELECT result IS NULL, error IS NULL FROM tool_calls WHERE id = :id"),
        {"id": call.id},
    ).one()

    assert tuple(stored) == (1, 1)
