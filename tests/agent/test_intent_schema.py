"""The extraction output contract: a closed union of three shapes, with no id
fields, no free-form fields and no action outside the union."""

import re
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from app.agent.pydantic_ai_planner import OUTPUT_TOOLS
from app.schemas.agent import (
    EnsureAssignmentIntent,
    ExtractedIntent,
    NeedsClarification,
    Unsupported,
)

INTENT = TypeAdapter[ExtractedIntent](ExtractedIntent)
ID_LIKE = re.compile(r"(^|_)ids?($|_)", re.IGNORECASE)


def properties(schema: dict[str, Any]) -> dict[str, Any]:
    return schema.get("properties", {})


def test_the_union_is_exactly_three_shapes() -> None:
    assert set(OUTPUT_TOOLS.values()) == {
        EnsureAssignmentIntent,
        NeedsClarification,
        Unsupported,
    }
    assert {
        variant["$ref"].rsplit("/", 1)[-1] for variant in INTENT.json_schema()["oneOf"]
    } == {"EnsureAssignmentIntent", "NeedsClarification", "Unsupported"}


@pytest.mark.parametrize("shape", OUTPUT_TOOLS.values(), ids=OUTPUT_TOOLS)
def test_no_shape_has_an_id_or_free_form_field(shape: type[Any]) -> None:
    schema = shape.model_json_schema()

    assert schema["additionalProperties"] is False
    for name, field in properties(schema).items():
        assert not ID_LIKE.search(name), name
        # Strings and closed sets only: nothing that could carry a structure.
        assert field.get("type") == "string", (name, field)


def test_the_fields_are_exactly_the_contract() -> None:
    assert {
        name: set(properties(shape.model_json_schema()))
        for name, shape in OUTPUT_TOOLS.items()
    } == {
        "ensure_assignment": {"kind", "user_email", "product"},
        "needs_clarification": {"kind", "reason_code"},
        "unsupported": {"kind", "reason_code"},
    }


@pytest.mark.parametrize(
    "data",
    [
        {"user_email": "ada@example.com", "product": "Figma", "user_id": 1},
        {"user_email": "ada@example.com", "product": "Figma", "licence_id": 3},
        {"user_email": "ada@example.com", "product": "Figma", "metadata": {"id": 1}},
        {"user_email": "ada@example.com"},
    ],
    ids=["user-id", "licence-id", "metadata", "missing-product"],
)
def test_an_assignment_rejects_extra_or_missing_fields(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EnsureAssignmentIntent.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "needs_clarification", "reason_code": "which_alice"},
        {"kind": "needs_clarification", "reason_code": "multiple_users", "note": "x"},
        {"kind": "unsupported", "reason_code": "delete_user"},
        {"kind": "unsupported", "reason_code": "not_a_request", "detail": {}},
        {"kind": "revoke_licence", "user_email": "ada@example.com"},
        {"kind": "make_admin", "user_email": "bob@example.com"},
        {"user_email": "ada@example.com", "product": "Figma"},  # no kind
    ],
)
def test_the_union_rejects_anything_outside_it(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        INTENT.validate_python(data)


@pytest.mark.parametrize(
    "intent",
    [
        EnsureAssignmentIntent(user_email="ada@example.com", product="Figma"),
        NeedsClarification(reason_code="conflicting_request"),
        Unsupported(reason_code="additional_request"),
    ],
)
def test_a_persisted_intent_reads_back_as_the_same_shape(intent: Any) -> None:
    assert INTENT.validate_python(intent.model_dump(mode="json")) == intent
