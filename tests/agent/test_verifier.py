"""The verifier: the goal holds only if an active assignment of exactly this
licence to exactly this user exists right now."""

import inspect
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.agent.verifier import verify
from app.models import Assignment, DesiredState, GoalType, Licence, User
from app.schemas.agent import ResolvedAssignmentGoal
from app.services.assignments import AssignmentService


def make_user(session: Session, email: str) -> User:
    user = User(email=email, name="Someone")
    session.add(user)
    session.commit()
    return user


def make_licence(session: Session, product: str) -> Licence:
    licence = Licence(product=product, seats_total=5)
    session.add(licence)
    session.commit()
    return licence


def make_assignment(
    session: Session, user: User, licence: Licence, *, revoked: bool = False
) -> Assignment:
    assignment = Assignment(
        user_id=user.id,
        licence_id=licence.id,
        revoked_at=datetime.now(UTC) if revoked else None,
    )
    session.add(assignment)
    session.commit()
    return assignment


def goal_for(user: User, licence: Licence) -> ResolvedAssignmentGoal:
    return ResolvedAssignmentGoal(
        goal_type=GoalType.ENSURE_ASSIGNMENT,
        desired_state=DesiredState.ASSIGNED,
        user_id=user.id,
        licence_id=licence.id,
        extracted_user_email=user.email,
        extracted_product=licence.product,
    )


def test_active_assignment_satisfies_the_goal(session: Session) -> None:
    ada = make_user(session, "ada@example.com")
    figma = make_licence(session, "Figma")
    assignment = make_assignment(session, ada, figma)

    result = verify(AssignmentService(session), goal_for(ada, figma))

    assert result.satisfied is True
    evidence = result.evidence
    assert (evidence.user_id, evidence.licence_id) == (ada.id, figma.id)
    assert evidence.desired_state is DesiredState.ASSIGNED
    assert evidence.active_assignment is not None
    assert evidence.active_assignment.assignment_id == assignment.id
    assert evidence.active_assignment.active is True


def test_no_assignment_does_not_satisfy_the_goal(session: Session) -> None:
    ada = make_user(session, "ada@example.com")
    figma = make_licence(session, "Figma")

    result = verify(AssignmentService(session), goal_for(ada, figma))

    assert result.satisfied is False
    assert result.evidence.active_assignment is None


def test_revoked_assignment_does_not_satisfy_the_goal(session: Session) -> None:
    ada = make_user(session, "ada@example.com")
    figma = make_licence(session, "Figma")
    make_assignment(session, ada, figma, revoked=True)

    assert verify(AssignmentService(session), goal_for(ada, figma)).satisfied is False


def test_same_licence_for_a_different_user_does_not_satisfy_the_goal(
    session: Session,
) -> None:
    ada = make_user(session, "ada@example.com")
    bob = make_user(session, "bob@example.com")
    figma = make_licence(session, "Figma")
    make_assignment(session, bob, figma)

    assert verify(AssignmentService(session), goal_for(ada, figma)).satisfied is False


def test_same_user_with_a_different_licence_does_not_satisfy_the_goal(
    session: Session,
) -> None:
    ada = make_user(session, "ada@example.com")
    figma = make_licence(session, "Figma")
    slack = make_licence(session, "Slack")
    make_assignment(session, ada, slack)

    assert verify(AssignmentService(session), goal_for(ada, figma)).satisfied is False


def test_reassignment_after_revocation_satisfies_with_the_active_row(
    session: Session,
) -> None:
    ada = make_user(session, "ada@example.com")
    figma = make_licence(session, "Figma")
    make_assignment(session, ada, figma, revoked=True)
    current = make_assignment(session, ada, figma)

    result = verify(AssignmentService(session), goal_for(ada, figma))

    assert result.satisfied is True
    assert result.evidence.active_assignment is not None
    assert result.evidence.active_assignment.assignment_id == current.id


def test_verifier_only_reads(session: Session, statements: list[str]) -> None:
    ada = make_user(session, "ada@example.com")
    figma = make_licence(session, "Figma")
    make_assignment(session, ada, figma)
    statements.clear()

    verify(AssignmentService(session), goal_for(ada, figma))

    assert statements
    assert [s for s in statements if not s.startswith("SELECT")] == []


def test_verifier_takes_only_the_goal_and_a_way_to_read_state() -> None:
    # No parameter through which a tool result or planner output could flow.
    assert list(inspect.signature(verify).parameters) == ["assignments", "goal"]
