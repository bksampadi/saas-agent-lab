"""Support for tests that drive agent runs the way the API does.

Runs go through the executor's real steps: receive, extract, resolve,
decide. Only the models are stood in for: extraction answered without a
model request (FixedIntent), a scripted model behind the real PydanticAI
planners (Script), or a planner that acts in place of a model loop
(FakeDecisionPlanner).
"""

from collections.abc import Callable
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from app.agent.executor import AgentExecutor
from app.agent.planner import CallTool, DecisionModelCallRecorder, ModelCallRecorder
from app.agent.pydantic_ai_decision import PydanticAIDecisionPlanner
from app.models import AgentRunStatus
from app.schemas.agent import (
    DecisionContext,
    DecisionProposal,
    EnsureAssignmentIntent,
    ExtractedIntent,
    TargetToolName,
)

HUMAN = "admin@example.com"
MODEL = "scripted-model"

# The model-facing tools and conclusions, by the names the model sees.
USER: TargetToolName = "get_target_user"
CAPACITY: TargetToolName = "get_target_licence_capacity"
ASSIGNMENTS: TargetToolName = "list_target_user_assignments"
ASSIGN: TargetToolName = "assign_target_licence"
PROPOSALS = frozenset({"goal_reached", "no_action_needed", "cannot_proceed"})


# --- extraction without a model -----------------------------------------------


class FixedIntent:
    """An IntentPlanner that answers ``intent`` without a model request, so
    it records no ModelCall. For tests about what happens after extraction."""

    def __init__(self, intent: ExtractedIntent) -> None:
        self.intent = intent

    def plan(self, instruction: str, calls: ModelCallRecorder) -> ExtractedIntent:
        return self.intent


def instruction_for(user_email: str, product: str) -> str:
    """An instruction that contains both texts, as extraction requires."""
    return f"Give {user_email} a {product} seat."


def extracted_run(
    executor: AgentExecutor, user_email: str = "ada@example.com", product: str = "Figma"
) -> int:
    """A RECEIVED run whose goal text has been extracted, not yet resolved."""
    run_id = executor.receive_run(
        instruction=instruction_for(user_email, product), requesting_actor=HUMAN
    )
    intent = EnsureAssignmentIntent(user_email=user_email, product=product)
    assert executor.extract_intent(run_id, FixedIntent(intent)) is (
        AgentRunStatus.RECEIVED
    )
    return run_id


def resolved_run(
    executor: AgentExecutor, user_email: str = "ada@example.com", product: str = "Figma"
) -> int:
    """A run taken through receive, extract and resolve, ready to decide."""
    run_id = extracted_run(executor, user_email, product)
    assert executor.resolve_run(run_id) is AgentRunStatus.RESOLVED
    return run_id


# --- a scripted model ---------------------------------------------------------

Step = ModelResponse | Exception | Callable[[], ModelResponse]


class Script:
    """A FunctionModel that answers each request with the next of ``steps``,
    and keeps every request it was sent. A step may be an exception to
    raise, or a callable run when the request arrives (to change state
    between requests). Steps left over were never requested."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[list[ModelMessage]] = []
        self.infos: list[AgentInfo] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(list(messages))
        self.infos.append(info)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ModelResponse):
            return step
        return step()

    def model(self) -> FunctionModel:
        return FunctionModel(self.respond, model_name=MODEL)

    def planner(self) -> PydanticAIDecisionPlanner:
        """The production decision planner, with this script as its model."""
        return PydanticAIDecisionPlanner(self.model(), timeout_seconds=5)

    def sent_parts(self) -> list[Any]:
        """Every request part the model was sent: the last request carries
        the whole conversation."""
        last = self.requests[-1] if self.requests else []
        return [part for m in last if isinstance(m, ModelRequest) for part in m.parts]


def usage() -> RequestUsage:
    return RequestUsage(input_tokens=100, output_tokens=10)


def call(*tool_names: str) -> ModelResponse:
    """A response asking for these tools, in order, with no arguments."""
    return ModelResponse(
        parts=[ToolCallPart(name, {}) for name in tool_names], usage=usage()
    )


def conclude(tool_name: str, **args: Any) -> ModelResponse:
    """A response calling one result tool."""
    return ModelResponse(parts=[ToolCallPart(tool_name, args)], usage=usage())


GOAL_REACHED = conclude("goal_reached")
NO_ACTION_NEEDED = conclude("no_action_needed")


def cannot_proceed(reason: str) -> ModelResponse:
    return conclude("cannot_proceed", reason_code=reason)


def directed_run(
    executor: AgentExecutor,
    script: Script,
    user_email: str = "ada@example.com",
    product: str = "Figma",
) -> int:
    """A whole run (AgentExecutor.run), with extraction answered without a
    model and ``script`` as the decision model."""
    return executor.run(
        instruction=instruction_for(user_email, product),
        requesting_actor=HUMAN,
        intent_planner=FixedIntent(
            EnsureAssignmentIntent(user_email=user_email, product=product)
        ),
        decision_planner=script.planner(),
    )


# --- a planner without a model ------------------------------------------------

Act = Callable[[DecisionContext, CallTool, DecisionModelCallRecorder], DecisionProposal]


class FakeDecisionPlanner:
    """A DecisionPlanner that runs ``act`` in place of a model loop."""

    def __init__(self, act: Act) -> None:
        self.act = act
        self.contexts: list[DecisionContext] = []

    def decide(
        self,
        context: DecisionContext,
        call_tool: CallTool,
        calls: DecisionModelCallRecorder,
    ) -> DecisionProposal:
        self.contexts.append(context)
        return self.act(context, call_tool, calls)
