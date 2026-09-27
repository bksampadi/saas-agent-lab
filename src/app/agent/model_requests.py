"""What every PydanticAI planner does around each model request.

Each response is checked by the application (a ``parse`` function) before
PydanticAI processes it, and each request, accepted, rejected or failed, is
recorded through the run's ModelCallRecorder as soon as it has an outcome. A
rejected response is sent back to the model with the reason, within the
planner's output retry budget. Exceptions are normalized here too: never
str(error), because provider messages may echo request content.

Shared by every PydanticAI planner, so every model stage is recorded the
same way.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_ai import (
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
    RunContext,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext

from app.agent.planner import ModelCallRecorder, PlannerError
from app.schemas.agent import ModelCallError, ModelCallRecord

INVALID_OUTPUT_MESSAGE_MAX_LENGTH = 500


class InvalidOutput(Exception):
    """A response that does not satisfy the output contract. The message is
    written by the application and is sent back to the model as feedback."""


def describe_validation_error(error: ValidationError) -> str:
    # Locations and error types only, never the offending input values.
    problems = "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'arguments'}: {item['type']}"
        for item in error.errors()
    )
    return f"Invalid arguments: {problems}"[:INVALID_OUTPUT_MESSAGE_MAX_LENGTH]


def is_timeout(error: BaseException) -> bool:
    # Provider SDKs raise their own timeout classes (anthropic.APITimeoutError
    # is not a TimeoutError), and PydanticAI wraps them in ModelAPIError, so
    # look along the cause chain, by type and by class name.
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, TimeoutError) or "Timeout" in type(current).__name__:
            return True
        current = current.__cause__
    return False


def error_code(error: BaseException) -> str:
    if is_timeout(error):
        return "timeout"
    if isinstance(error, ModelAPIError):
        return "provider_error"
    return "unexpected_error"


def model_call_error(error: BaseException) -> ModelCallError:
    """Normalize an exception raised by a model request."""
    code = error_code(error)
    if code == "timeout":
        message = "The model request timed out."
    elif isinstance(error, ModelHTTPError):
        message = f"The model provider returned HTTP {error.status_code}."
    elif code == "provider_error":
        message = "The model provider could not be reached or returned an error."
    else:
        message = "Unexpected error during the model request."
    return ModelCallError(code=code, message=message, error_type=type(error).__name__)


def as_planner_error(error: Exception) -> PlannerError | None:
    """The PlannerError for an exception from a PydanticAI run, or None if it
    is not a planning failure (the caller re-raises it unchanged)."""
    if isinstance(error, UnexpectedModelBehavior):
        # The retry budget ran out without an acceptable response.
        return PlannerError("output_retries_exhausted", type(error).__name__)
    if isinstance(error, UsageLimitExceeded):
        return PlannerError("request_limit_exceeded", type(error).__name__)
    if isinstance(error, UserError):
        # e.g. no API key for the configured provider; no request was made.
        return PlannerError("configuration_error", type(error).__name__)
    if isinstance(error, ModelAPIError) or is_timeout(error):
        return PlannerError(error_code(error), type(error).__name__)
    return None


@dataclass
class CheckAndRecordModelRequests(AbstractCapability[Any]):
    """Checks each response of one planner call and records each request.

    ``parse`` returns a response's accepted output, or raises InvalidOutput.
    ``before_request``, if set, runs before every request and may refuse it
    by raising; the refused request is not made and not recorded.
    """

    parse: Callable[[ModelResponse], BaseModel]
    calls: ModelCallRecorder
    clock: Callable[[], float]
    before_request: Callable[[], None] | None = None

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        if self.before_request is not None:
            self.before_request()
        model_name = request_context.model.model_name
        started = self.clock()
        try:
            response = await handler(request_context)
        except Exception as error:
            self.calls.record(
                ModelCallRecord(
                    model_name=model_name,
                    input_tokens=None,
                    output_tokens=None,
                    latency_ms=self._elapsed_ms(started),
                    output=None,
                    error=model_call_error(error),
                )
            )
            raise
        latency_ms = self._elapsed_ms(started)

        try:
            output = self.parse(response)
        except InvalidOutput as invalid:
            self.calls.record(
                ModelCallRecord(
                    model_name=model_name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=latency_ms,
                    output=None,
                    error=ModelCallError(
                        code="invalid_output",
                        message=str(invalid),
                        error_type=type(invalid).__name__,
                    ),
                )
            )
            # Counts against the output retry budget; PydanticAI sends the
            # message back to the model and never processes this response.
            raise ModelRetry(str(invalid)) from None

        self.calls.record(
            ModelCallRecord(
                model_name=model_name,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                output=output.model_dump(mode="json"),
                error=None,
            )
        )
        return response

    def _elapsed_ms(self, started: float) -> int:
        return max(0, round((self.clock() - started) * 1000))
