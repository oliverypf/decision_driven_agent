"""Failure classification for PostToolUse results.

JEV is the decision authority: the router submits the failure evidence and
uses the returned failure type. The local rules only run as an explicitly
marked safety fallback when JEV is unavailable or its answer is invalid.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, replace
from typing import Any

from ..models import FAILURE_TYPES, FailureRoute, FailureType


@dataclass(frozen=True)
class FailureInput:
    message: str = ""
    stack_trace: str = ""
    environment: str = ""
    recent_diff: str = ""
    exit_code: int | None = None
    test_name: str = ""

    def combined_text(self) -> str:
        return "\n".join(
            value
            for value in (
                self.message,
                self.stack_trace,
                self.environment,
                self.recent_diff,
                self.test_name,
            )
            if value
        ).lower()

    def summary(self, *, limit: int = 600) -> str:
        """Return a bounded digest recorded with the JEV call audit."""

        fields = (
            ("message", self.message),
            ("stack_trace", self.stack_trace),
            ("environment", self.environment),
            ("recent_diff", self.recent_diff),
            ("test_name", self.test_name),
        )
        parts = [f"{name}={' '.join(str(value).split())}" for name, value in fields if value]
        if self.exit_code is not None:
            parts.append(f"exit_code={self.exit_code}")
        text = " | ".join(parts)
        return text if len(text) <= limit else text[:limit] + "...[truncated]"


class FailureRouter:
    """Route failures with JEV; local rules are only a marked safety fallback."""

    _rules: tuple[tuple[FailureType, tuple[str, ...], float, str], ...] = (
        (
            FailureType.MISSING_EVIDENCE,
            (
                "missing evidence",
                "evidence insufficient",
                "not covered",
                "no assertion",
                "no test",
                "acceptance criteria",
                "proof is missing",
            ),
            0.94,
            "The output indicates that implementation evidence or acceptance coverage is missing.",
        ),
        (
            FailureType.FLAKY,
            (
                "flaky",
                "intermittent",
                "nondeterministic",
                "non-deterministic",
                "passes on retry",
                "sometimes fails",
                "race condition",
            ),
            0.93,
            "The failure is described as intermittent or retry-sensitive.",
        ),
        (
            FailureType.ENVIRONMENT_ERROR,
            (
                "command not found",
                "no such file or directory",
                "permission denied",
                "address already in use",
                "connection refused",
                "network is unreachable",
                "could not resolve host",
                "environment variable",
                "modulenotfounderror",
                "docker daemon",
                "cannot connect to the docker daemon",
            ),
            0.91,
            "The failure points to the execution environment or a missing dependency.",
        ),
        (
            FailureType.TEST_ERROR,
            (
                "assertionerror",
                "assertion failed",
                "expected .* but got",
                "fixture",
                "test_",
                "tests/",
                "pytest",
                "unittest",
            ),
            0.88,
            "The failure is associated with a test assertion, fixture, or test harness.",
        ),
        (
            FailureType.CODE_ERROR,
            (
                "traceback",
                "syntaxerror",
                "typeerror",
                "attributeerror",
                "nameerror",
                "referenceerror",
                "nullpointer",
                "undefined is not",
                "cannot read properties",
                "compile error",
            ),
            0.86,
            "The failure points to an implementation or compilation defect.",
        ),
    )

    def classify(
        self,
        failure: FailureInput,
        *,
        use_model: bool = False,
        model_client: Any | None = None,
        model_timeout: float | None = None,
    ) -> FailureRoute:
        """Classify a failure, preferring the JEV decision.

        The local rules are a safety fallback only: when they run they are
        marked with ``source="local_fallback"``, ``fallback=True`` and a
        ``fallback_reason`` so the result is never presented as a JEV
        conclusion. The caller (DecisionController) records the JEV call
        status, input digest, output and error alongside the adopted route.
        """

        if use_model and model_client is None:
            return self._local_fallback(
                failure,
                "JEV failure classification was requested but no model client was provided.",
            )
        if use_model and model_client is not None:
            if not getattr(model_client, "available", False):
                note = getattr(model_client, "note_unavailable", None)
                if callable(note):
                    note("failure_type")
                return self._local_fallback(
                    failure,
                    "JEV is unavailable (OPENROUTER_API_KEY is not configured); "
                    "the local failure router executed as the safety fallback.",
                )
            try:
                answer = model_client.judge_failure_type(
                    message=failure.message,
                    stack_trace=failure.stack_trace,
                    environment=failure.environment,
                    recent_diff=failure.recent_diff,
                    exit_code=failure.exit_code,
                    test_name=failure.test_name,
                    timeout=model_timeout,
                )
                failure_type = answer["failure_type"]
                if not isinstance(failure_type, str) or failure_type not in FAILURE_TYPES:
                    raise ValueError(f"JEV returned an invalid failure type: {failure_type}")
                confidence = answer["confidence"]
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not math.isfinite(float(confidence))
                    or not 0 <= confidence <= 1
                ):
                    raise ValueError("JEV returned an invalid confidence")
                reasoning_required = answer.get("reasoning_required", False)
                if not isinstance(reasoning_required, bool):
                    raise ValueError("JEV returned an invalid reasoning_required flag")
                raw_scores = answer.get("scores")
                if raw_scores is not None and not isinstance(raw_scores, dict):
                    raise ValueError("JEV returned invalid failure scores")
                scores = dict(raw_scores or {})
                if any(str(key) not in FAILURE_TYPES for key in scores):
                    raise ValueError("JEV returned invalid failure score labels")
                if any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0 <= value <= 1
                    for value in scores.values()
                ):
                    raise ValueError("JEV returned invalid failure scores")
                reason = answer.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("JEV returned an invalid reason")
                return FailureRoute(
                    type=failure_type,
                    confidence=float(confidence),
                    reason=reason,
                    signals=["jev:failure_type"],
                    reasoning_required=reasoning_required,
                    source="JEV",
                    fallback=False,
                    scores={
                        str(key): float(value)
                        for key, value in scores.items()
                    },
                )
            except Exception as exc:  # Any JEV failure must fall back explicitly.
                return self._local_fallback(failure, f"JEV failure classification failed: {exc}"[:300])
        return self._classify_locally(failure)

    def _local_fallback(self, failure: FailureInput, reason: str) -> FailureRoute:
        return replace(
            self._classify_locally(failure),
            source="local_fallback",
            fallback=True,
            fallback_reason=reason,
        )

    def _classify_locally(self, failure: FailureInput) -> FailureRoute:
        text = failure.combined_text()
        for failure_type, patterns, confidence, reason in self._rules:
            matches = [pattern for pattern in patterns if self._matches(pattern, text)]
            if matches:
                return FailureRoute(
                    type=failure_type,
                    confidence=confidence,
                    reason=reason,
                    signals=matches,
                    reasoning_required=False,
                )

        if failure.exit_code not in (None, 0) and not text:
            return FailureRoute(
                type=FailureType.UNKNOWN,
                confidence=0.55,
                reason="The command failed but supplied no diagnostic text.",
                signals=[f"exit_code:{failure.exit_code}"],
                reasoning_required=True,
            )
        return FailureRoute(
            type=FailureType.UNKNOWN,
            confidence=0.35,
            reason="No deterministic failure signal matched; strong-model analysis may be needed.",
            signals=[],
            reasoning_required=True,
        )

    @staticmethod
    def _matches(pattern: str, text: str) -> bool:
        if ".*" in pattern:
            return re.search(pattern, text) is not None
        return pattern in text
