"""Policy: whether an agent run may make a change, decided by the application.

A tool call is admitted by AgentExecutor._start_call, in the log transaction
that records it. Its goal scope and the run's limits are checked first; only
a call that passes them is evaluated here. For a mutation the decision is:

- allow: the call runs as any other call does;
- deny: the call never runs and the run ends BLOCKED (policy_denied);
- require_approval: the call is held, unrun, and the run waits for a
  person's approval (AWAITING_APPROVAL).

Policy is decided at admission, from committed state, and the decision is
recorded with the call. The read and the recorded decision are atomic
together; they are not atomic with the business transaction that runs an
admitted call later, which is a separate transaction. Whatever a paused call
needs when it resumes is decided then, not here.

A model never sees a decision: a denied or held call gives it no
observation, and its loop stops.
"""

from typing import assert_never

from app.models import PolicyDecision
from app.schemas.agent import (
    AssignLicenceInput,
    GetLicenceInput,
    GetUserInput,
    ListUserAssignmentsInput,
    ToolInput,
)
from app.services.licences import LicenceService


def evaluate(args: ToolInput, licences: LicenceService) -> PolicyDecision | None:
    """The policy decision for the call ``args``, read from committed state;
    None for a read, which policy does not govern.

    Every tool is listed, so a new one cannot be added without deciding here
    whether, and how, policy governs it.
    """
    match args:
        case AssignLicenceInput():
            # The licence says whether an agent may give out its seats.
            return licences.get_licence(args.licence_id).agent_policy
        case GetUserInput() | GetLicenceInput() | ListUserAssignmentsInput():
            return None
        case _:
            assert_never(args)
