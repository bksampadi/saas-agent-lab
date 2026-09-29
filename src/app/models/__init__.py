"""ORM models.

Every model is imported here so that importing ``app.models`` registers all
tables on ``Base.metadata`` (Alembic and the test fixtures rely on this).
"""

from app.models.agent_run import (
    AgentRun,
    AgentRunStatus,
    CannotProceedReason,
    DecisionProposalKind,
    DesiredState,
    GoalType,
    OutcomeReason,
)
from app.models.assignment import Assignment
from app.models.audit_event import AuditEvent
from app.models.base import Base
from app.models.licence import Licence, PolicyDecision
from app.models.model_call import ModelCall, ModelCallStage, ModelCallStatus
from app.models.tool_call import ToolCall, ToolCallStatus
from app.models.user import User, UserStatus

__all__ = [
    "AgentRun",
    "AgentRunStatus",
    "Assignment",
    "AuditEvent",
    "Base",
    "CannotProceedReason",
    "DecisionProposalKind",
    "DesiredState",
    "GoalType",
    "Licence",
    "ModelCall",
    "ModelCallStage",
    "ModelCallStatus",
    "OutcomeReason",
    "PolicyDecision",
    "ToolCall",
    "ToolCallStatus",
    "User",
    "UserStatus",
]
