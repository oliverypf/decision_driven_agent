"""Shared data contracts for the decision layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


TASK_DOMAINS = (
    "implementation",
    "documentation",
    "information",
    "investigation",
    "configuration",
    "external_action",
    "unknown",
)

_DOMAIN_ALIASES = {
    "code": "implementation",
    "coding": "implementation",
    "implementation": "implementation",
    "software": "implementation",
    "docs": "documentation",
    "documentation": "documentation",
    "writing": "documentation",
    "question": "information",
    "information": "information",
    "qa": "information",
    "research": "investigation",
    "investigation": "investigation",
    "diagnosis": "investigation",
    "config": "configuration",
    "configuration": "configuration",
    "external": "external_action",
    "external_action": "external_action",
    "action": "external_action",
}


def normalize_task_domain(value: Any) -> str:
    """Normalize a caller or model domain hint to a bounded domain set."""

    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _DOMAIN_ALIASES.get(normalized, normalized if normalized in TASK_DOMAINS else "unknown")


DOMAIN_STOP_CONTRACTS = {
    "implementation": (
        "The requested behavior or code change must be complete, the relevant change must be evidenced, "
        "and applicable validation must support the result."
    ),
    "documentation": (
        "The requested document or text outcome must be complete and the requested document change must be evidenced. "
        "Tests are not required unless the change also affects executable behavior."
    ),
    "information": (
        "A clear answer to the user's question is sufficient. Code diff and test evidence are not required unless "
        "the user explicitly asked for an implementation or executable result."
    ),
    "investigation": (
        "The requested investigation or diagnosis must have a supported finding or explanation. Code diff and tests "
        "are not required unless the requested outcome includes a fix."
    ),
    "configuration": (
        "The requested configuration outcome must be complete and the relevant configuration change must be evidenced "
        "with applicable validation when the configuration affects runtime behavior."
    ),
    "external_action": (
        "The requested external action must be completed and its result or confirmation must be evidenced."
    ),
    "unknown": (
        "Only stop when the user's goal is clearly complete and the evidence is sufficient for the actual task domain; "
        "do not assume code or test requirements without support from the goal."
    ),
}


def utc_now() -> str:
    """Return a stable, timezone-aware timestamp for persisted records."""

    return datetime.now(timezone.utc).isoformat()


class EvidenceKind(str, Enum):
    REQUIREMENT = "requirement"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    GIT_DIFF = "git_diff"
    TEST_RESULT = "test_result"
    RUNTIME = "runtime"
    LOG = "log"
    DECISION = "decision"
    USER_REQUEST = "user_request"
    SECURITY_RISK = "security_risk"
    BUILD_FAILURE = "build_failure"
    FINAL_ACCEPTANCE = "final_acceptance"


class RetentionPolicy(str, Enum):
    KEEP = "keep"
    COMPRESS = "compress"
    DROP = "drop"


class StopStatus(str, Enum):
    STOP = "stop"
    CONTINUE = "continue"
    ESCALATE = "escalate"


class FailureType(str, Enum):
    CODE_ERROR = "CODE_ERROR"
    TEST_ERROR = "TEST_ERROR"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    FLAKY = "FLAKY"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    UNKNOWN = "UNKNOWN"


FAILURE_TYPES = tuple(item.value for item in FailureType)

TOOL_RISK_LEVELS = ("low", "medium", "high")
TOOL_RECOMMENDATIONS = ("proceed", "confirm", "block")


@dataclass(frozen=True)
class EvidenceRecord:
    """A single piece of evidence collected during an agent task."""

    kind: EvidenceKind | str
    content: Any
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    severity: str = "info"
    retention: RetentionPolicy | str = RetentionPolicy.KEEP
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = str(self.kind.value if isinstance(self.kind, EvidenceKind) else self.kind)
        value["retention"] = str(
            self.retention.value if isinstance(self.retention, RetentionPolicy) else self.retention
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            kind=value.get("kind", EvidenceKind.LOG),
            content=value.get("content"),
            id=str(value.get("id") or uuid4().hex),
            created_at=str(value.get("created_at") or utc_now()),
            severity=str(value.get("severity") or "info"),
            retention=value.get("retention", RetentionPolicy.KEEP),
            metadata=dict(value.get("metadata") or {}),
        )

    @property
    def kind_value(self) -> str:
        return self.kind.value if isinstance(self.kind, EvidenceKind) else str(self.kind)

    @property
    def retention_value(self) -> str:
        return self.retention.value if isinstance(self.retention, RetentionPolicy) else str(self.retention)


@dataclass(frozen=True)
class DecisionRequest:
    state: Mapping[str, Any]
    question: str
    choices: list[str] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionResult:
    choice: str
    confidence: float
    reasoning_required: bool = False
    missing_information: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSufficiencyResult:
    implemented: bool
    evidence_sufficient: bool
    missing: list[str]
    criterion_results: dict[str, bool]
    confidence: float
    reasoning_required: bool = False
    reasons: list[str] = field(default_factory=list)
    source: str = "local"
    fallback: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StopDecision:
    status: StopStatus | str
    reason: str
    confidence: float
    missing: list[str] = field(default_factory=list)
    iteration: int = 0
    goal_completed: bool | None = None
    goal_confidence: float | None = None
    domain: str = "unknown"
    domain_confidence: float | None = None
    source: str = "local"
    fallback: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, StopStatus) else str(self.status),
            "reason": self.reason,
            "confidence": self.confidence,
            "missing": list(self.missing),
            "iteration": self.iteration,
            "goal_completed": self.goal_completed,
            "goal_confidence": self.goal_confidence,
            "domain": self.domain,
            "domain_confidence": self.domain_confidence,
            "source": self.source,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ToolRiskDecision:
    """PreToolUse risk and tool-appropriateness judgment."""

    tool_name: str
    risk: str
    recommendation: str
    reason: str
    confidence: float = 0.0
    appropriate: bool | None = None
    signals: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    source: str = "local"
    fallback: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "risk": self.risk,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "confidence": self.confidence,
            "appropriate": self.appropriate,
            "signals": list(self.signals),
            "scores": {str(key): float(value) for key, value in self.scores.items()},
            "source": self.source,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class FailureRoute:
    type: FailureType | str
    confidence: float
    reason: str
    signals: list[str] = field(default_factory=list)
    reasoning_required: bool = False
    source: str = "local"
    fallback: bool = False
    fallback_reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, FailureType) else str(self.type),
            "confidence": self.confidence,
            "reason": self.reason,
            "signals": list(self.signals),
            "reasoning_required": self.reasoning_required,
            "source": self.source,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "scores": {str(key): float(value) for key, value in self.scores.items()},
        }
