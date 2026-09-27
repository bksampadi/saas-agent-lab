"""The AgentRun status machine, on in-memory runs (no database)."""

import pytest

from app.agent.status import (
    ALLOWED_TRANSITIONS,
    REASONS_BY_STATUS,
    IllegalTransition,
    transition,
)
from app.models import AgentRun, AgentRunStatus, OutcomeReason
from app.models.agent_run import TERMINAL_STATUSES

S = AgentRunStatus
ALL = list(AgentRunStatus)
ALLOWED = [(a, b) for a in ALL for b in ALL if b in ALLOWED_TRANSITIONS[a]]
DISALLOWED = [(a, b) for a in ALL for b in ALL if b not in ALLOWED_TRANSITIONS[a]]


def pair_id(pair: object) -> str:
    return str(pair)


def a_reason_for(status: AgentRunStatus) -> OutcomeReason | None:
    if status not in TERMINAL_STATUSES:
        return None
    return sorted(REASONS_BY_STATUS[status])[0]


def test_transition_table_is_exactly_the_approved_one() -> None:
    assert ALLOWED_TRANSITIONS == {
        S.RECEIVED: {S.RESOLVED, S.NEEDS_CLARIFICATION, S.FAILED},
        S.RESOLVED: {S.EXECUTING, S.FAILED},
        S.EXECUTING: {S.VERIFYING, S.BLOCKED, S.FAILED},
        # BLOCKED from VERIFYING: a model-directed run's block is established
        # by the application only after verification finds the goal unmet.
        S.VERIFYING: {S.COMPLETED, S.BLOCKED, S.FAILED},
        S.COMPLETED: set(),
        S.NEEDS_CLARIFICATION: set(),
        S.BLOCKED: set(),
        S.FAILED: set(),
    }


def test_completed_is_reachable_only_from_verifying() -> None:
    assert [s for s in ALL if S.COMPLETED in ALLOWED_TRANSITIONS[s]] == [S.VERIFYING]


@pytest.mark.parametrize(("current", "to"), ALLOWED, ids=pair_id)
def test_every_allowed_transition_works(
    current: AgentRunStatus, to: AgentRunStatus
) -> None:
    run = AgentRun(status=current)
    terminal = to in TERMINAL_STATUSES

    transition(
        run, to, reason=a_reason_for(to), detail={"why": "test"} if terminal else None
    )

    assert run.status is to
    if terminal:
        assert run.outcome_reason is a_reason_for(to)
        assert run.outcome_detail == {"why": "test"}
        assert run.completed_at is not None
    else:
        assert run.outcome_reason is None
        assert run.completed_at is None


@pytest.mark.parametrize(("current", "to"), DISALLOWED, ids=pair_id)
def test_every_other_transition_is_rejected_without_change(
    current: AgentRunStatus, to: AgentRunStatus
) -> None:
    run = AgentRun(status=current)

    with pytest.raises(IllegalTransition):
        transition(run, to, reason=a_reason_for(to))

    assert run.status is current
    assert run.outcome_reason is None
    assert run.completed_at is None


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STATUSES))
@pytest.mark.parametrize("to", ALL)
def test_terminal_statuses_are_final(
    terminal: AgentRunStatus, to: AgentRunStatus
) -> None:
    run = AgentRun(status=terminal)

    with pytest.raises(IllegalTransition):
        transition(run, to, reason=a_reason_for(to))


@pytest.mark.parametrize(
    ("current", "to", "reason"),
    [
        (current, to, reason)
        for current, to in ALLOWED
        if to in TERMINAL_STATUSES
        for reason in OutcomeReason
        if reason not in REASONS_BY_STATUS[to]
    ],
    ids=pair_id,
)
def test_terminal_status_rejects_a_reason_belonging_to_another_status(
    current: AgentRunStatus, to: AgentRunStatus, reason: OutcomeReason
) -> None:
    run = AgentRun(status=current)

    with pytest.raises(ValueError, match="not a valid reason"):
        transition(run, to, reason=reason)

    assert run.status is current


@pytest.mark.parametrize(
    ("current", "to"), [p for p in ALLOWED if p[1] in TERMINAL_STATUSES], ids=pair_id
)
def test_terminal_status_requires_a_reason(
    current: AgentRunStatus, to: AgentRunStatus
) -> None:
    run = AgentRun(status=current)

    with pytest.raises(ValueError):
        transition(run, to)

    assert run.status is current


@pytest.mark.parametrize(
    ("current", "to"),
    [p for p in ALLOWED if p[1] not in TERMINAL_STATUSES],
    ids=pair_id,
)
def test_non_terminal_status_takes_no_reason_or_detail(
    current: AgentRunStatus, to: AgentRunStatus
) -> None:
    with pytest.raises(ValueError):
        transition(AgentRun(status=current), to, reason=OutcomeReason.TOOL_FAILED)
    with pytest.raises(ValueError):
        transition(AgentRun(status=current), to, detail={"x": 1})
