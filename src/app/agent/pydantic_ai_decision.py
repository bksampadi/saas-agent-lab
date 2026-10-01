"""DecisionPlanner backed by PydanticAI: after resolution, the model chooses
which goal-bound tools to call, and when to conclude.

The model sees four tools, none with parameters, and three result tools,
one per member of DecisionProposal. Every response is checked by the
application (parse_decision) before PydanticAI processes it: it either asks
for tools, each by a known name with no arguments, or concludes, with
exactly one result tool call alone. Before every request the run's request
limit is checked, and every request is recorded as soon as it has an
outcome (app.agent.model_requests). Tool calls run one at a time, in the
order the model asked for them.

A run the application ends or pauses (a limit, policy, a failure the model
is not shown) stops the loop at once: DecisionStopped passes through
unchanged, and is never turned into an observation for the model to answer.
"""

import time
from collections.abc import Callable
from typing import Any, get_args

from pydantic import ValidationError
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings

from app.agent.decision import DecisionStopped
from app.agent.model_requests import (
    CheckAndRecordModelRequests,
    InvalidOutput,
    as_planner_error,
    describe_validation_error,
)
from app.agent.planner import CallTool, DecisionModelCallRecorder
from app.schemas.agent import (
    CannotProceed,
    DecisionContext,
    DecisionProposal,
    DecisionToolRequest,
    GoalReached,
    NoActionNeeded,
    TargetToolName,
)

# Rejected responses answered with feedback before the stage gives up. Each
# retry is another request, counted against MAX_DECISION_MODEL_REQUESTS.
OUTPUT_RETRIES = 2
MAX_OUTPUT_TOKENS = 4096

# The result tools the model concludes with, by the name it sees.
PROPOSAL_TOOLS: dict[
    str, type[GoalReached] | type[NoActionNeeded] | type[CannotProceed]
] = {
    "goal_reached": GoalReached,
    "no_action_needed": NoActionNeeded,
    "cannot_proceed": CannotProceed,
}

TARGET_TOOL_NAMES: frozenset[str] = frozenset(get_args(TargetToolName))


# The model-facing tools. Each takes only PydanticAI's run context, which is
# not part of the tool's schema, so the model supplies nothing. The context
# carries the run's CallTool, which binds the run's goal in.


def get_target_user(ctx: RunContext[CallTool]) -> str:
    """The user's account status."""
    return ctx.deps("get_target_user")


def get_target_licence_capacity(ctx: RunContext[CallTool]) -> str:
    """The product's seats: total, in use and available."""
    return ctx.deps("get_target_licence_capacity")


def list_target_user_assignments(ctx: RunContext[CallTool]) -> str:
    """Whether the user already holds an active seat of the product."""
    return ctx.deps("list_target_user_assignments")


def assign_target_licence(ctx: RunContext[CallTool]) -> str:
    """Try to give the user a seat of the product: assigned, or rejected with
    a reason_code."""
    return ctx.deps("assign_target_licence")


TARGET_TOOLS = (
    get_target_user,
    get_target_licence_capacity,
    list_target_user_assignments,
    assign_target_licence,
)


def parse_decision(response: ModelResponse) -> DecisionProposal | DecisionToolRequest:
    """The response's conclusion or tool requests, or raise InvalidOutput.

    Checked before any of the response's tools run. PydanticAI on its own
    would run the tools of a response that also concludes, and would reject
    arguments only as each tool is validated.
    """
    calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
    if not calls:
        raise InvalidOutput(
            "Expected a tool call, got none. Call tools, or conclude with "
            "exactly one of goal_reached, no_action_needed or cannot_proceed."
        )
    if any(call.tool_name in PROPOSAL_TOOLS for call in calls):
        if len(calls) != 1:
            raise InvalidOutput(
                "A conclusion must be the only tool call in its response, got "
                f"{len(calls)} tool calls."
            )
        (call,) = calls
        try:
            return PROPOSAL_TOOLS[call.tool_name].model_validate(_arguments(call))
        except ValidationError as error:
            raise InvalidOutput(describe_validation_error(error)) from None
    for call in calls:
        if call.tool_name not in TARGET_TOOL_NAMES:
            raise InvalidOutput(f"{call.tool_name[:64]!r} is not a tool.")
        if _arguments(call):
            raise InvalidOutput(
                f"{call.tool_name} takes no arguments: the application chooses "
                "the user and the product."
            )
    return DecisionToolRequest.model_validate(
        {"tool_names": [call.tool_name for call in calls]}
    )


def _arguments(call: ToolCallPart) -> dict[str, Any]:
    try:
        return call.args_as_dict(raise_if_invalid=True)
    except (ValueError, AssertionError):
        raise InvalidOutput(
            f"The arguments of {call.tool_name[:64]!r} are not a JSON object."
        ) from None


class PydanticAIDecisionPlanner:
    """The decision stage with a PydanticAI agent: goal-bound function tools,
    and tool-based structured output for the proposal."""

    def __init__(
        self,
        model: Model | str,
        *,
        timeout_seconds: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._clock = clock
        self._agent = Agent(
            model,
            deps_type=CallTool,
            output_type=[
                ToolOutput(output_type, name=name)
                for name, output_type in PROPOSAL_TOOLS.items()
            ],
            # sequential: each call is a barrier, so calls run one at a time;
            # the executor's tool calls within a run must be serial.
            tools=[
                Tool(tool, takes_ctx=True, sequential=True) for tool in TARGET_TOOLS
            ],
            retries={"tools": 0, "output": OUTPUT_RETRIES},
            model_settings=ModelSettings(
                max_tokens=MAX_OUTPUT_TOKENS, timeout=timeout_seconds
            ),
            # Build the model at the first request, not here: the application
            # and the tests start without provider credentials.
            defer_model_check=True,
        )

    def decide(
        self,
        context: DecisionContext,
        call_tool: CallTool,
        calls: DecisionModelCallRecorder,
    ) -> DecisionProposal:
        checker = CheckAndRecordModelRequests(
            parse=parse_decision,
            calls=calls,
            clock=self._clock,
            before_request=calls.before_request,
        )
        try:
            result = self._agent.run_sync(
                context.prompt,
                instructions=context.instructions,
                deps=call_tool,
                capabilities=[checker],
            )
        except DecisionStopped:
            raise
        except Exception as error:
            planner_error = as_planner_error(error)
            if planner_error is None:
                raise
            raise planner_error from None
        return result.output
