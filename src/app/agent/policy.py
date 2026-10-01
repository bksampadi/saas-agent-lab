"""Policy: whether an agent run may make a change, decided by the application.

A tool call is admitted by AgentExecutor._start_call, in the log transaction
that records it. The run's limits are checked first; only a call within them
is evaluated here. For the mutation the decision is:

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

from app.agent import tools
from app.models import PolicyDecision
from app.schemas.agent import ResolvedAssignmentGoal, TargetToolName
from app.services.licences import LicenceService


def evaluate(
    tool: TargetToolName, goal: ResolvedAssignmentGoal, licences: LicenceService
) -> PolicyDecision | None:
    """The policy decision for a call of ``tool``, read from committed state;
    None for a read, which policy does not govern."""
    if tool != tools.MUTATION:
        return None
    # The licence says whether an agent may give out its seats.
    return licences.get_licence(goal.licence_id).agent_policy
