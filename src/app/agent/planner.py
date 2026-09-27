"""The planner boundary: where a model reaches the application.

An intent planner turns an instruction into an unresolved intent. A
decision planner, after resolution, chooses which goal-bound tools to call
and concludes with a proposal.

A planner interprets and chooses. It is never authoritative for which rows
exist or are meant (the resolver decides), what a tool acts on (the run's
persisted goal decides), whether a change is allowed (the services decide),
whether it happened (the business transaction decides) or whether the goal
holds (the verifier decides).

The executor hands a planner a ModelCallRecorder bound to the run and
stage. The planner reports every model request through it as soon as the
request has an outcome, so each one is persisted, in trace order, even when
planning then fails.
"""

from typing import Protocol

from app.schemas.agent import (
    DecisionContext,
    DecisionProposal,
    EnsureAssignmentIntent,
    ExtractedIntent,
    ModelCallRecord,
)


class ModelCallRecorder(Protocol):
    def record(self, call: ModelCallRecord) -> None:
        """Persist one model request's outcome as the run's next trace entry."""
        ...


class IntentPlanner(Protocol):
    def plan(self, instruction: str, calls: ModelCallRecorder) -> ExtractedIntent:
        """What ``instruction`` asks for.

        Raises PlannerError when no acceptable answer was obtained within the
        planner's bounds.
        """
        ...


class DecisionModelCallRecorder(ModelCallRecorder, Protocol):
    def before_request(self) -> None:
        """Call before every model request. Raises DecisionLimitExceeded,
        and the request must not be made, if it would exceed the stage's
        request limit."""
        ...


class TargetTools(Protocol):
    """The decision model's tools, bound by the application to the run's
    resolved goal. None takes an argument, so nothing a model says can name
    a user, licence or assignment.

    Each returns the call's observation: the exact text persisted with the
    call. Each may raise DecisionStopped (a limit, or a failure that ended
    the run), which must end the planner's loop at once.
    """

    def get_target_user(self) -> str: ...

    def get_target_licence_capacity(self) -> str: ...

    def list_target_user_assignments(self) -> str: ...

    def assign_target_licence(self) -> str: ...


class DecisionPlanner(Protocol):
    def decide(
        self,
        context: DecisionContext,
        tools: TargetTools,
        calls: DecisionModelCallRecorder,
    ) -> DecisionProposal:
        """Call ``tools`` as the model chooses, then return its proposal.

        ``context`` is exactly what the model is to be sent first; it is
        already persisted. Raises PlannerError when no acceptable answer was
        obtained within the planner's bounds, and lets DecisionStopped
        through unchanged.
        """
        ...


class PlannerError(Exception):
    """Planning failed: timeout, provider error, or no acceptable output within
    the retry budget. ``code`` is a stable, application-chosen value."""

    def __init__(self, code: str, error_type: str | None) -> None:
        super().__init__(f"Planning failed: {code}.")
        self.code = code
        self.error_type = error_type


def ungrounded_fields(intent: EnsureAssignmentIntent, instruction: str) -> list[str]:
    """The fields of ``intent`` whose value is blank or not written in the
    instruction (ignoring case and surrounding space).

    Extracted text must be copied, never guessed: "give Alice GitHub" has no
    email address, so any email a planner returns for it was invented.
    """
    text = instruction.casefold()
    return [
        field
        for field, value in (
            ("user_email", intent.user_email),
            ("product", intent.product),
        )
        if not value.strip() or value.strip().casefold() not in text
    ]
