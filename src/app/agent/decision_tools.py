"""The decision model's tools, bound to one run's resolved goal.

No tool takes an argument. The model chooses the capability; the
application chooses the target. Each call is built here from the run's
persisted goal and goes through AgentExecutor.call_decision_tool: the same
path as a deterministic call (goal-scope check, ToolCall trace, business
transaction, audit actor "agent:run-<id>"), plus the decision limits and the
observation boundary. The model gets back only the call's persisted
observation, never the internal result.
"""

from typing import TYPE_CHECKING

from app.agent.decision import RunEndedDuringDecision
from app.models import AgentRunStatus
from app.schemas.agent import (
    AssignLicenceInput,
    GetLicenceInput,
    GetUserInput,
    ListUserAssignmentsInput,
    ResolvedAssignmentGoal,
    ToolInput,
)

if TYPE_CHECKING:
    # Only for the annotation: the executor imports this module.
    from app.agent.executor import AgentExecutor


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
        if (
            outcome.observation is None
            or outcome.run_status is not AgentRunStatus.EXECUTING
        ):
            # A failure the model is not shown, which ended the run.
            raise RunEndedDuringDecision(self._run_id, outcome.run_status)
        return outcome.observation
