"""The decision model's tools, bound to one run's resolved goal.

No tool takes an argument. The model chooses the capability; the
application chooses the target. Each call is built here from the run's
persisted goal and goes through AgentExecutor.call_decision_tool (goal-scope
check, decision limits, policy, ToolCall trace, business transaction, audit
actor "agent:run-<id>", observation boundary). The model gets back only the
call's persisted observation, never the internal result. A call policy
denies or holds for approval gives it nothing, and stops its loop.
"""

from typing import TYPE_CHECKING

from app.agent.decision import RunAwaitingApproval, RunEndedDuringDecision
from app.models import AgentRunStatus
from app.schemas.agent import (
    AssignLicenceInput,
    GetLicenceInput,
    GetUserInput,
    ListUserAssignmentsInput,
    ResolvedAssignmentGoal,
    TargetToolName,
    ToolInput,
)

if TYPE_CHECKING:
    # Only for the annotation: the executor imports this module.
    from app.agent.executor import AgentExecutor

# Each tool call is recorded under the internal tool's name; this is the
# model-facing tool that makes it (see GoalBoundTools below), for showing a
# trace in the model's own terms.
MODEL_TOOL_NAMES: dict[str, TargetToolName] = {
    GetUserInput.tool_name: "get_target_user",
    GetLicenceInput.tool_name: "get_target_licence_capacity",
    ListUserAssignmentsInput.tool_name: "list_target_user_assignments",
    AssignLicenceInput.tool_name: "assign_target_licence",
}


class GoalBoundTools:
    """TargetTools for one run, holding its persisted goal. The goal's ids
    stay here: a planner is handed this object only as TargetTools, and a
    model sees tool names and observations."""

    def __init__(
        self, executor: "AgentExecutor", run_id: int, goal: ResolvedAssignmentGoal
    ) -> None:
        self._executor = executor
        self._run_id = run_id
        self._goal = goal

    def get_target_user(self) -> str:
        return self._call(GetUserInput(user_id=self._goal.user_id))

    def get_target_licence_capacity(self) -> str:
        return self._call(GetLicenceInput(licence_id=self._goal.licence_id))

    def list_target_user_assignments(self) -> str:
        return self._call(ListUserAssignmentsInput(user_id=self._goal.user_id))

    def assign_target_licence(self) -> str:
        return self._call(
            AssignLicenceInput(
                user_id=self._goal.user_id, licence_id=self._goal.licence_id
            )
        )

    def _call(self, args: ToolInput) -> str:
        # May raise DecisionLimitExceeded: the call was refused and recorded.
        outcome = self._executor.call_decision_tool(self._run_id, args)
        if outcome.run_status is AgentRunStatus.AWAITING_APPROVAL:
            # Held for approval, unrun: there is no result to show.
            raise RunAwaitingApproval(self._run_id)
        if (
            outcome.observation is None
            or outcome.run_status is not AgentRunStatus.EXECUTING
        ):
            # A failure the model is not shown, or a denied mutation, which
            # ended the run.
            raise RunEndedDuringDecision(self._run_id, outcome.run_status)
        return outcome.observation
