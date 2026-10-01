"""The resolver: exact, deterministic identity resolution; never a guess."""

from typing import get_args

import pytest
from sqlalchemy.orm import Session

from app.agent.resolver import resolve_assignment_goal
from app.models import Licence, OutcomeReason, User
from app.schemas.agent import (
    ResolutionFailure,
    ResolvedAssignmentGoal,
)
from app.services.licences import LicenceService
from app.services.users import UserService


def make_user(session: Session, email: str = "ada@example.com") -> User:
    user = User(email=email, name="Ada")
    session.add(user)
    session.commit()
    return user


def make_licence(session: Session, product: str = "Figma") -> Licence:
    licence = Licence(product=product, seats_total=5)
    session.add(licence)
    session.commit()
    return licence


def resolve(
    session: Session, user_email: str = "ada@example.com", product: str = "Figma"
) -> ResolvedAssignmentGoal | ResolutionFailure:
    return resolve_assignment_goal(
        UserService(session), LicenceService(session), user_email, product
    )


def resolved(session: Session, user_email: str, product: str) -> ResolvedAssignmentGoal:
    result = resolve(session, user_email, product)
    assert isinstance(result, ResolvedAssignmentGoal), result
    return result


def failure(session: Session, user_email: str, product: str) -> ResolutionFailure:
    result = resolve(session, user_email, product)
    assert isinstance(result, ResolutionFailure), result
    return result


# --- users: exact normalized email ----------------------------------------------


def test_exact_email_and_product_resolve_to_a_goal(session: Session) -> None:
    user = make_user(session)
    licence = make_licence(session)

    goal = resolved(session, "ada@example.com", "Figma")

    assert goal == ResolvedAssignmentGoal(
        user_id=user.id,
        licence_id=licence.id,
        extracted_user_email="ada@example.com",
        extracted_product="Figma",
    )


@pytest.mark.parametrize(
    "email",
    ["ADA@EXAMPLE.COM", "Ada@Example.Com", "  ada@example.com ", "ada@example.com\t"],
)
def test_email_case_and_surrounding_whitespace_are_normalized(
    session: Session, email: str
) -> None:
    user = make_user(session)
    make_licence(session)

    goal = resolved(session, email, "Figma")

    assert goal.user_id == user.id
    assert goal.extracted_user_email == email  # the original text is kept


@pytest.mark.parametrize(
    "email",
    ["bob@example.com", "ada@example.co", "da@example.com", "ada@example.com.au"],
)
def test_unknown_or_near_miss_email_is_user_not_found(
    session: Session, email: str
) -> None:
    make_user(session)
    make_licence(session)

    result = failure(session, email, "Figma")

    assert result.code is OutcomeReason.USER_NOT_FOUND
    assert result.detail == {"user_email": email}


# --- licences: exact name, case-insensitive ---------------------------------------


@pytest.mark.parametrize("product", ["Figma", "figma", "FIGMA", "  Figma "])
def test_licence_name_matches_exactly_ignoring_case(
    session: Session, product: str
) -> None:
    make_user(session)
    licence = make_licence(session, "Figma")
    make_licence(session, "Slack")

    goal = resolved(session, "ada@example.com", product)

    assert goal.licence_id == licence.id
    assert goal.extracted_product == product


def test_non_ascii_licence_name_matches_ignoring_case(session: Session) -> None:
    make_user(session)
    licence = make_licence(session, "Ångström Suite")

    assert (
        resolved(session, "ada@example.com", "ÅNGSTRÖM SUITE").licence_id == licence.id
    )


@pytest.mark.parametrize("product", ["Fig", "Figma Pro", "Fig ma", "Figmа"])
def test_unknown_or_near_miss_licence_is_licence_not_found(
    session: Session, product: str
) -> None:
    # "Figmа" ends in a Cyrillic "а": it looks the same but is not.
    make_user(session)
    make_licence(session, "Figma")

    result = failure(session, "ada@example.com", product)

    assert result.code is OutcomeReason.LICENCE_NOT_FOUND
    assert result.detail == {"product": product}


@pytest.mark.parametrize("product", ["Figma", "FIGMA", "figma"])
def test_two_case_insensitive_matches_are_ambiguous_even_for_an_exact_case_match(
    session: Session, product: str
) -> None:
    # The schema allows this: licences.product is unique only case-sensitively.
    make_user(session)
    make_licence(session, "Figma")
    make_licence(session, "FIGMA")

    result = failure(session, "ada@example.com", product)

    assert result.code is OutcomeReason.LICENCE_AMBIGUOUS
    assert result.detail == {"product": product, "candidates": ["Figma", "FIGMA"]}


# --- invalid input ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("email", "product"),
    [
        ("", "Figma"),
        ("   ", "Figma"),
        ("not-an-email", "Figma"),
        ("ada lovelace@example.com", "Figma"),
        ("x" * 321 + "@e.com", "Figma"),
        ("ada@example.com", ""),
        ("ada@example.com", "   "),
        ("ada@example.com", "x" * 201),
        # Invalid input is reported before "not found" for either field.
        ("nobody@example.com", ""),
        ("", "Unknown"),
    ],
)
def test_empty_or_invalid_input_is_invalid_input(
    session: Session, email: str, product: str
) -> None:
    make_user(session)
    make_licence(session)

    result = failure(session, email, product)

    assert result.code is OutcomeReason.INVALID_INPUT
    assert set(result.detail) == {"message"}


def test_failure_codes_are_exactly_the_clarification_reasons() -> None:
    codes = set(get_args(ResolutionFailure.model_fields["code"].annotation))

    assert codes == {
        OutcomeReason.INVALID_INPUT,
        OutcomeReason.USER_NOT_FOUND,
        OutcomeReason.LICENCE_NOT_FOUND,
        OutcomeReason.LICENCE_AMBIGUOUS,
    }


# --- reads only -------------------------------------------------------------------


def test_resolver_only_reads(session: Session, statements: list[str]) -> None:
    make_user(session)
    make_licence(session)
    statements.clear()

    resolve(session)
    resolve(session, "nobody@example.com")

    assert statements
    assert [s for s in statements if not s.startswith("SELECT")] == []
    assert not session.new and not session.dirty and not session.deleted
