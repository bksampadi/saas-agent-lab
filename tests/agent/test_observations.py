"""The observation boundary on its own: internal results (which carry row
ids) in, closed id-free observations and their exact text out. No database.

Snapshots are built with distinctive ids, so a leak would be visible.
"""

import json
from datetime import UTC, datetime
from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

from app.agent.executor import tool_error
from app.agent.observations import REJECTION_CODES, observe, serialize
from app.models import DesiredState, GoalType, UserStatus
from app.schemas.agent import (
    AssignLicenceInput,
    AssignmentAttemptObservation,
    AssignmentSnapshot,
    GetLicenceInput,
    GetUserInput,
    LicenceCapacityObservation,
    LicenceSnapshot,
    ListUserAssignmentsInput,
    ModelObservation,
    ResolvedAssignmentGoal,
    TargetAssignmentsObservation,
    TargetUserObservation,
    ToolError,
    ToolInput,
    ToolOutput,
    UserAssignmentsSnapshot,
    UserSnapshot,
)
from app.services.errors import (
    AssignmentAlreadyExists,
    LicenceNotFound,
    NoSeatsAvailable,
    UserInactive,
    UserNotFound,
)

ADA = 48213
FIGMA = 97531
SLACK = 97532
ASSIGNMENT = 86420
IDS = [str(n) for n in (ADA, FIGMA, SLACK, ASSIGNMENT)]
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)

GOAL = ResolvedAssignmentGoal(
    goal_type=GoalType.ENSURE_ASSIGNMENT,
    desired_state=DesiredState.ASSIGNED,
    user_id=ADA,
    licence_id=FIGMA,
    extracted_user_email="ada@example.com",
    extracted_product="Figma",
)
ASSIGN = AssignLicenceInput(user_id=ADA, licence_id=FIGMA)
OBSERVATION_TYPES: list[type[BaseModel]] = list(get_args(ModelObservation))


def assignment(
    licence_id: int = FIGMA, *, revoked: bool = False, assignment_id: int = ASSIGNMENT
) -> AssignmentSnapshot:
    return AssignmentSnapshot(
        assignment_id=assignment_id,
        user_id=ADA,
        licence_id=licence_id,
        active=not revoked,
        assigned_at=NOW,
        revoked_at=NOW if revoked else None,
    )


def assert_id_free(text: str) -> None:
    for row_id in IDS:
        assert row_id not in text


# --- each read tool -------------------------------------------------------------

READS: dict[str, tuple[ToolInput, ToolOutput, ModelObservation, str]] = {
    "user": (
        GetUserInput(user_id=ADA),
        UserSnapshot(
            user_id=ADA, email="ada@example.com", name="Ada", status=UserStatus.ACTIVE
        ),
        TargetUserObservation(status=UserStatus.ACTIVE),
        '{"status":"active"}',
    ),
    "capacity": (
        GetLicenceInput(licence_id=FIGMA),
        LicenceSnapshot(
            licence_id=FIGMA,
            product="Figma",
            seats_total=3,
            seats_active=2,
            seats_available=1,
        ),
        LicenceCapacityObservation(seats_total=3, seats_active=2, seats_available=1),
        '{"seats_active":2,"seats_available":1,"seats_total":3}',
    ),
    "assignments": (
        ListUserAssignmentsInput(user_id=ADA),
        UserAssignmentsSnapshot(user_id=ADA, assignments=[assignment()]),
        TargetAssignmentsObservation(holds_active_seat=True),
        '{"holds_active_seat":true}',
    ),
}


@pytest.mark.parametrize(
    ("args", "internal", "expected", "text"), READS.values(), ids=READS
)
def test_each_read_result_becomes_its_id_free_observation(
    args: ToolInput, internal: ToolOutput, expected: ModelObservation, text: str
) -> None:
    observation = observe(GOAL, args, internal)

    assert observation == expected
    assert serialize(expected) == text
    # The internal result carries the ids; the observation's text does not.
    assert any(row_id in internal.model_dump_json() for row_id in IDS)
    assert_id_free(text)


@pytest.mark.parametrize(
    ("assignments", "holds"),
    [
        ([], False),
        ([assignment(revoked=True)], False),
        ([assignment(SLACK)], False),
        ([assignment(revoked=True), assignment(assignment_id=ASSIGNMENT + 1)], True),
    ],
    ids=["none", "revoked", "other-product", "revoked-then-reassigned"],
)
def test_only_an_active_seat_of_the_target_licence_counts_as_held(
    assignments: list[AssignmentSnapshot], holds: bool
) -> None:
    listing = UserAssignmentsSnapshot(user_id=ADA, assignments=assignments)

    assert observe(GOAL, ListUserAssignmentsInput(user_id=ADA), listing) == (
        TargetAssignmentsObservation(holds_active_seat=holds)
    )


# --- the mutation -----------------------------------------------------------------


def test_a_successful_assignment_is_observed_as_assigned() -> None:
    observation = observe(GOAL, ASSIGN, assignment())

    assert observation == AssignmentAttemptObservation(
        outcome="assigned", reason_code=None
    )
    assert observation is not None
    assert serialize(observation) == '{"outcome":"assigned","reason_code":null}'


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (NoSeatsAvailable(FIGMA), "no_seats_available"),
        (UserInactive(ADA), "user_inactive"),
        (AssignmentAlreadyExists(ADA, FIGMA), "already_assigned"),
    ],
    ids=["no-seats", "inactive", "already-assigned"],
)
def test_a_domain_rejection_is_observed_by_code_never_by_message(
    error: Exception, code: str
) -> None:
    internal = tool_error(error, "assign_licence")
    assert any(row_id in internal.message for row_id in IDS)  # ids in the message

    observation = observe(GOAL, ASSIGN, internal)

    assert observation == AssignmentAttemptObservation.model_validate(
        {"outcome": "rejected", "reason_code": code}
    )
    assert observation is not None
    text = serialize(observation)
    assert text == f'{{"outcome":"rejected","reason_code":"{code}"}}'
    assert_id_free(text)


def test_the_rejections_a_model_may_see_are_exactly_the_domain_rules() -> None:
    assert REJECTION_CODES == {
        "no_seats_available": "no_seats_available",
        "user_inactive": "user_inactive",
        "assignment_already_exists": "already_assigned",
    }


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (ASSIGN, tool_error(RuntimeError(f"SQL mentions {ADA}"), "assign_licence")),
        (ASSIGN, tool_error(UserNotFound(ADA), "assign_licence")),
        (ASSIGN, tool_error(LicenceNotFound(FIGMA), "assign_licence")),
        (
            ASSIGN,
            ToolError(code="goal_scope_violation", message="x", error_type=None),
        ),
        (GetUserInput(user_id=ADA), tool_error(UserNotFound(ADA), "get_user")),
        # A blocking code from anything but the assignment is not a rejection.
        (
            GetLicenceInput(licence_id=FIGMA),
            ToolError(code="no_seats_available", message="x", error_type=None),
        ),
    ],
    ids=[
        "unexpected",
        "user-not-found",
        "licence-not-found",
        "scope",
        "read-failure",
        "read-with-blocking-code",
    ],
)
def test_any_other_failure_is_shown_nothing(args: Any, error: ToolError) -> None:
    assert observe(GOAL, args, error) is None


# --- the DTOs themselves ----------------------------------------------------------


@pytest.mark.parametrize("model", OBSERVATION_TYPES, ids=lambda m: m.__name__)
def test_observations_are_closed_and_have_no_id_fields(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()

    assert schema["additionalProperties"] is False
    assert not [name for name in schema["properties"] if "id" in name.split("_")]
    assert all(
        prop.get("type") != "object" for prop in schema["properties"].values()
    ), "no nested free-form objects"


def test_an_observation_rejects_an_extra_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TargetUserObservation.model_validate({"status": "active", "user_id": ADA})


@pytest.mark.parametrize(
    "values",
    [
        {"outcome": "assigned", "reason_code": "no_seats_available"},
        {"outcome": "rejected", "reason_code": None},
        {"outcome": "rejected", "reason_code": "database_error"},
    ],
    ids=["assigned-with-reason", "rejected-without-reason", "unknown-reason"],
)
def test_an_attempt_observation_is_consistent(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AssignmentAttemptObservation.model_validate(values)


def test_serialization_is_deterministic_compact_and_sorted() -> None:
    observation = LicenceCapacityObservation(
        seats_total=3, seats_active=2, seats_available=1
    )

    text = serialize(observation)

    assert text == serialize(observation.model_copy())
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    assert " " not in text
