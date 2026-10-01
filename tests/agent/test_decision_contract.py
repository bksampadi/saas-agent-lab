"""The decision stage's contract on its own, no database: the terminal
outcome table, what the model is told, and each tool's names."""

import inspect
import json
from typing import get_args

import pytest

from app.agent import tools
from app.agent.decision import (
    DECISION_INSTRUCTIONS,
    MAX_DECISION_MODEL_REQUESTS,
    MAX_DECISION_MUTATION_CALLS,
    MAX_DECISION_READ_CALLS,
    decision_context,
    decision_outcome,
)
from app.agent.pydantic_ai_decision import PROPOSAL_TOOLS, TARGET_TOOLS
from app.models import (
    AgentRunStatus,
    CannotProceedReason,
    DecisionProposalKind,
    OutcomeReason,
)
from app.schemas.agent import (
    CannotProceed,
    DecisionProposal,
    GoalReached,
    NoActionNeeded,
    ResolvedAssignmentGoal,
    TargetToolName,
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
    user_id=48213,
    licence_id=97531,
    extracted_user_email="Ada@Example.com",
    extracted_product="figma",
)


def test_the_context_is_built_from_the_goals_text_alone() -> None:
    # It is never handed an id, so it cannot pass one on.
    assert list(inspect.signature(decision_context).parameters) == [
        "user_email",
        "product",
    ]


def test_the_context_is_deterministic() -> None:
    context = decision_context("Ada@Example.com", "figma")

    assert context == decision_context("Ada@Example.com", "figma")
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


def test_the_instructions_are_sent_as_written() -> None:
    # PydanticAI strips instructions; stripped, these are unchanged.
    assert DECISION_INSTRUCTIONS == DECISION_INSTRUCTIONS.strip()


def test_the_instructions_name_every_tool_and_conclusion() -> None:
    for tool_name in [*get_args(TargetToolName), *PROPOSAL_TOOLS]:
        assert tool_name in DECISION_INSTRUCTIONS
    for reason in CannotProceedReason:
        assert reason.value in DECISION_INSTRUCTIONS


# --- each tool's names ------------------------------------------------------


def test_each_tool_has_one_name_for_the_model_and_one_on_record() -> None:
    names = set(get_args(TargetToolName))

    assert len(names) == 4
    assert {tool.__name__ for tool in TARGET_TOOLS} == names
    assert set(tools.RECORDED_NAMES) == names
    # Distinct, so the public trace can show each recorded call by its tool.
    assert len(set(tools.RECORDED_NAMES.values())) == len(names)


def test_the_tools_take_nothing_but_the_run_context() -> None:
    for tool in TARGET_TOOLS:
        assert list(inspect.signature(tool).parameters) == ["ctx"]


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
