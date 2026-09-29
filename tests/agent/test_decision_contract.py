"""The decision stage's contract on its own, no database: the terminal
outcome table, what the model is told, and the tool names every layer
agrees on."""

import inspect
import json
from typing import cast, get_args

import pytest

from app.agent import tools
from app.agent.decision import (
    DECISION_INSTRUCTIONS,
    MAX_DECISION_MODEL_REQUESTS,
    MAX_DECISION_MUTATION_CALLS,
    MAX_DECISION_READ_CALLS,
    decision_context,
    decision_outcome,
    decision_task,
)
from app.agent.decision_tools import MODEL_TOOL_NAMES, GoalBoundTools
from app.agent.executor import AgentExecutor, ToolCallOutcome
from app.agent.planner import TargetTools
from app.agent.pydantic_ai_decision import PROPOSAL_TOOLS, TARGET_TOOLS
from app.models import (
    AgentRunStatus,
    CannotProceedReason,
    DecisionProposalKind,
    DesiredState,
    GoalType,
    OutcomeReason,
    UserStatus,
)
from app.schemas.agent import (
    CannotProceed,
    DecisionProposal,
    DecisionTask,
    GoalReached,
    NoActionNeeded,
    ResolvedAssignmentGoal,
    TargetToolName,
    ToolInput,
    UserSnapshot,
)

S = AgentRunStatus
R = OutcomeReason

GOAL_REACHED = GoalReached()
NO_ACTION = NoActionNeeded()
NO_SEATS = CannotProceed(reason_code=CannotProceedReason.NO_SEATS_AVAILABLE)
INACTIVE = CannotProceed(reason_code=CannotProceedReason.USER_INACTIVE)
PROPOSALS: list[DecisionProposal] = [GOAL_REACHED, NO_ACTION, NO_SEATS, INACTIVE]
REJECTIONS = [None, R.NO_SEATS_AVAILABLE, R.USER_INACTIVE]


def name(value: object) -> str:
    """A readable test id for a proposal or a rejection."""
    if isinstance(value, CannotProceed):
        return f"cannot_proceed:{value.reason_code.value}"
    if isinstance(value, GoalReached | NoActionNeeded):
        return value.kind
    return str(value)


# --- the outcome table ----------------------------------------------------------
#
#   satisfied  rejection   proposal              claim confirmed  outcome
#   yes        any         any                   any              COMPLETED
#   no         blocking r  any                   any              BLOCKED (r)
#   no         none        CannotProceed(r)      yes              BLOCKED (r)
#   no         none        CannotProceed(r)      no               FAILED (verif.)
#   no         none        GoalReached/NoAction  -                FAILED (verif.)
#
# COMPLETED is goal_satisfied if this run's mutation succeeded, else
# already_satisfied; "verif." is verification_failed.

TABLE = [
    # (proposal, satisfied, changed, rejection, claim_confirmed) -> outcome
    (GOAL_REACHED, True, True, None, False, (S.COMPLETED, R.GOAL_SATISFIED)),
    (NO_ACTION, True, False, None, False, (S.COMPLETED, R.ALREADY_SATISFIED)),
    (NO_SEATS, True, False, None, True, (S.COMPLETED, R.ALREADY_SATISFIED)),
    (
        GOAL_REACHED,
        False,
        False,
        R.NO_SEATS_AVAILABLE,
        False,
        (S.BLOCKED, R.NO_SEATS_AVAILABLE),
    ),
    (
        INACTIVE,
        False,
        False,
        R.NO_SEATS_AVAILABLE,
        True,
        (S.BLOCKED, R.NO_SEATS_AVAILABLE),
    ),
    (NO_SEATS, False, False, None, True, (S.BLOCKED, R.NO_SEATS_AVAILABLE)),
    (INACTIVE, False, False, None, True, (S.BLOCKED, R.USER_INACTIVE)),
    (NO_SEATS, False, False, None, False, (S.FAILED, R.VERIFICATION_FAILED)),
    (GOAL_REACHED, False, False, None, False, (S.FAILED, R.VERIFICATION_FAILED)),
    (NO_ACTION, False, False, None, False, (S.FAILED, R.VERIFICATION_FAILED)),
]


@pytest.mark.parametrize(
    ("proposal", "satisfied", "changed", "rejection", "confirmed", "expected"), TABLE
)
def test_the_outcome_table(
    proposal: DecisionProposal,
    satisfied: bool,
    changed: bool,
    rejection: OutcomeReason | None,
    confirmed: bool,
    expected: tuple[AgentRunStatus, OutcomeReason],
) -> None:
    assert (
        decision_outcome(
            proposal=proposal,
            satisfied=satisfied,
            changed=changed,
            rejection=rejection,
            claim_confirmed=confirmed,
        )
        == expected
    )


@pytest.mark.parametrize("proposal", PROPOSALS, ids=name)
@pytest.mark.parametrize("rejection", REJECTIONS, ids=name)
@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("changed", [True, False])
def test_a_satisfied_goal_completes_whatever_else_is_true(
    proposal: DecisionProposal,
    rejection: OutcomeReason | None,
    confirmed: bool,
    changed: bool,
) -> None:
    assert decision_outcome(
        proposal=proposal,
        satisfied=True,
        changed=changed,
        rejection=rejection,
        claim_confirmed=confirmed,
    ) == (S.COMPLETED, R.GOAL_SATISFIED if changed else R.ALREADY_SATISFIED)


@pytest.mark.parametrize("proposal", PROPOSALS, ids=name)
@pytest.mark.parametrize("confirmed", [True, False])
def test_an_unsatisfied_goal_after_a_blocking_rejection_is_blocked_by_it(
    proposal: DecisionProposal, confirmed: bool
) -> None:
    for rejection in (R.NO_SEATS_AVAILABLE, R.USER_INACTIVE):
        assert decision_outcome(
            proposal=proposal,
            satisfied=False,
            changed=False,
            rejection=rejection,
            claim_confirmed=confirmed,
        ) == (S.BLOCKED, rejection)


@pytest.mark.parametrize("proposal", [GOAL_REACHED, NO_ACTION], ids=name)
def test_only_a_cannot_proceed_claim_can_be_confirmed_into_a_block(
    proposal: DecisionProposal,
) -> None:
    # Even if a caller passed claim_confirmed=True by mistake.
    assert decision_outcome(
        proposal=proposal,
        satisfied=False,
        changed=False,
        rejection=None,
        claim_confirmed=True,
    ) == (S.FAILED, R.VERIFICATION_FAILED)


def test_no_proposal_ever_makes_an_unsatisfied_goal_completed() -> None:
    for proposal in PROPOSALS:
        for rejection in REJECTIONS:
            for confirmed in (True, False):
                status, _ = decision_outcome(
                    proposal=proposal,
                    satisfied=False,
                    changed=True,
                    rejection=rejection,
                    claim_confirmed=confirmed,
                )
                assert status is not S.COMPLETED


# --- what the model is told -----------------------------------------------------

GOAL = ResolvedAssignmentGoal(
    goal_type=GoalType.ENSURE_ASSIGNMENT,
    desired_state=DesiredState.ASSIGNED,
    user_id=48213,
    licence_id=97531,
    extracted_user_email="Ada@Example.com",
    extracted_product="figma",
)


def test_the_task_holds_the_extracted_text_and_no_ids() -> None:
    task = decision_task(GOAL)

    assert task == DecisionTask(
        goal_type=GoalType.ENSURE_ASSIGNMENT,
        user_email="Ada@Example.com",
        product="figma",
    )
    assert set(DecisionTask.model_fields) == {"goal_type", "user_email", "product"}


def test_the_context_is_deterministic_and_id_free() -> None:
    context = decision_context(decision_task(GOAL))

    assert context == decision_context(decision_task(GOAL))
    assert context.instructions == DECISION_INSTRUCTIONS
    assert context.prompt == (
        "The goal:\n"
        + json.dumps(
            {
                "goal_type": "ensure_assignment",
                "product": "figma",
                "user_email": "Ada@Example.com",
            }
        )
    )
    for row_id in ("48213", "97531"):
        assert row_id not in context.instructions + context.prompt


def test_the_instructions_are_sent_as_written() -> None:
    # PydanticAI strips instructions; stripped, these are unchanged.
    assert DECISION_INSTRUCTIONS == DECISION_INSTRUCTIONS.strip()


def test_the_instructions_name_every_tool_and_conclusion() -> None:
    for tool_name in [*get_args(TargetToolName), *PROPOSAL_TOOLS]:
        assert tool_name in DECISION_INSTRUCTIONS
    for reason in CannotProceedReason:
        assert reason.value in DECISION_INSTRUCTIONS


# --- names every layer agrees on ------------------------------------------------


def public_methods(cls: type) -> set[str]:
    return {
        n
        for n, _ in inspect.getmembers(cls, inspect.isfunction)
        if not n.startswith("_")
    }


def test_every_layer_offers_the_same_four_tools() -> None:
    names = set(get_args(TargetToolName))

    assert len(names) == 4
    assert {tool.__name__ for tool in TARGET_TOOLS} == names
    assert public_methods(TargetTools) == names
    assert public_methods(GoalBoundTools) == names


class RecordingExecutor:
    """Stands in for AgentExecutor: records the call a goal-bound tool builds."""

    def __init__(self) -> None:
        self.tool_names: list[str] = []

    def call_decision_tool(self, run_id: int, args: ToolInput) -> ToolCallOutcome:
        self.tool_names.append(args.tool_name)
        # Any successful outcome will do: only the tool name is checked.
        return ToolCallOutcome(
            tool_call_id=1,
            sequence_no=1,
            output=UserSnapshot(
                user_id=1, email="ada@example.com", name="Ada", status=UserStatus.ACTIVE
            ),
            error=None,
            run_status=AgentRunStatus.EXECUTING,
            observation="{}",
        )


def test_each_tool_call_is_shown_under_the_model_facing_tool_that_made_it() -> None:
    goal = ResolvedAssignmentGoal(
        goal_type=GoalType.ENSURE_ASSIGNMENT,
        desired_state=DesiredState.ASSIGNED,
        user_id=48213,
        licence_id=97531,
        extracted_user_email="ada@example.com",
        extracted_product="Figma",
    )
    executor = RecordingExecutor()
    target_tools = GoalBoundTools(cast(AgentExecutor, executor), 7, goal)

    for model_name in get_args(TargetToolName):
        getattr(target_tools, model_name)()
        assert MODEL_TOOL_NAMES[executor.tool_names[-1]] == model_name
    assert set(MODEL_TOOL_NAMES) == {tool.tool_name for tool in tools.TOOL_INPUTS}


def test_the_tools_take_nothing_but_the_run_context() -> None:
    for tool in TARGET_TOOLS:
        assert list(inspect.signature(tool).parameters) == ["ctx"]
    for tool_name in get_args(TargetToolName):
        method = getattr(GoalBoundTools, tool_name)
        assert list(inspect.signature(method).parameters) == ["self"]


def test_proposal_tools_match_the_persisted_proposal_kinds() -> None:
    assert set(PROPOSAL_TOOLS) == {kind.value for kind in DecisionProposalKind}
    for tool_name, proposal in PROPOSAL_TOOLS.items():
        assert proposal.model_fields["kind"].default == tool_name


def test_the_limits_are_the_approved_conservative_ones() -> None:
    assert (
        MAX_DECISION_MODEL_REQUESTS,
        MAX_DECISION_READ_CALLS,
        MAX_DECISION_MUTATION_CALLS,
    ) == (6, 6, 1)
