"""The planner boundary: an instruction in, an unresolved intent out.

A planner interprets text. It is never authoritative for which rows exist
or are meant (the resolver decides), whether a change is allowed (the
services decide), whether it happened (the business transaction decides)
or whether the goal holds (the verifier decides).

The executor hands a planner a ModelCallRecorder bound to the run and
stage. The planner reports every model request through it as soon as the
request has an outcome, so each one is persisted, in trace order, even when
planning then fails.
"""

from typing import Protocol

from app.schemas.agent import EnsureAssignmentIntent, ExtractedIntent, ModelCallRecord


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
