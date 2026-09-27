"""The AgentRun status machine: which status may follow which.

Every status change goes through ``transition``. Terminal statuses are final.
COMPLETED is reachable only from VERIFYING; the executor enters it only when
the verifier reports the goal satisfied.

Extraction happens within RECEIVED and adds no status: it either fills in
the run's goal columns, leaving it RECEIVED and ready for resolution, or
ends the run (NEEDS_CLARIFICATION or FAILED, with an extraction reason).
"""

from typing import Any

from app.models import AgentRun, AgentRunStatus, OutcomeReason
from app.models.agent_run import TERMINAL_STATUSES
from app.models.base import utcnow

Status = AgentRunStatus
Reason = OutcomeReason

ALLOWED_TRANSITIONS: dict[AgentRunStatus, frozenset[AgentRunStatus]] = {
    Status.RECEIVED: frozenset(
        {Status.RESOLVED, Status.NEEDS_CLARIFICATION, Status.FAILED}
    ),
    Status.RESOLVED: frozenset({Status.EXECUTING, Status.FAILED}),
    Status.EXECUTING: frozenset({Status.VERIFYING, Status.BLOCKED, Status.FAILED}),
    Status.VERIFYING: frozenset({Status.COMPLETED, Status.FAILED}),
    Status.COMPLETED: frozenset(),
    Status.NEEDS_CLARIFICATION: frozenset(),
    Status.BLOCKED: frozenset(),
    Status.FAILED: frozenset(),
}

# The reasons each terminal status may carry. The agent_runs
# outcome_matches_status CHECK constraint enforces the same pairs.
REASONS_BY_STATUS: dict[AgentRunStatus, frozenset[OutcomeReason]] = {
    Status.COMPLETED: frozenset({Reason.GOAL_SATISFIED, Reason.ALREADY_SATISFIED}),
    Status.NEEDS_CLARIFICATION: frozenset(
        {
            Reason.UNSUPPORTED_REQUEST,
            Reason.INSTRUCTION_UNCLEAR,
            Reason.INVALID_INPUT,
            Reason.USER_NOT_FOUND,
            Reason.LICENCE_NOT_FOUND,
            Reason.LICENCE_AMBIGUOUS,
        }
    ),
    Status.BLOCKED: frozenset({Reason.NO_SEATS_AVAILABLE, Reason.USER_INACTIVE}),
    Status.FAILED: frozenset(
        {
            Reason.PLANNER_ERROR,
            Reason.GOAL_SCOPE_VIOLATION,
            Reason.TOOL_FAILED,
            Reason.VERIFICATION_FAILED,
            Reason.UNEXPECTED_ERROR,
        }
    ),
}


class IllegalTransition(Exception):
    def __init__(self, current: AgentRunStatus, requested: AgentRunStatus) -> None:
        super().__init__(f"An agent run cannot move from {current} to {requested}.")
        self.current = current
        self.requested = requested


def transition(
    run: AgentRun,
    to: AgentRunStatus,
    *,
    reason: OutcomeReason | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Move ``run`` to ``to``, or raise without changing anything.

    A terminal status needs a reason allowed for it and sets completed_at; a
    non-terminal status takes no reason or detail.
    """
    if to not in ALLOWED_TRANSITIONS[run.status]:
        raise IllegalTransition(run.status, to)
    if to in TERMINAL_STATUSES:
        if reason not in REASONS_BY_STATUS[to]:
            raise ValueError(f"{reason!r} is not a valid reason for {to}.")
        run.outcome_reason = reason
        run.outcome_detail = detail
        run.completed_at = utcnow()
    elif reason is not None or detail is not None:
        raise ValueError(f"{to} is not terminal and takes no reason or detail.")
    run.status = to
