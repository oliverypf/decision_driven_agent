"""High-level Phase 1 controller used by hook adapters and the CLI."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .decision.evidence_store import EvidenceStore
from .decision.failure_router import FailureInput, FailureRouter
from .decision.metrics import MetricsAggregator, TaskMetrics
from .decision.prescreen import extract_tool_command, prescreen_tool_use
from .decision.stop_judge import StopJudge
from .decision.sufficiency import EvidenceSufficiencyJudge
from .models import (
    FAILURE_TYPES,
    EvidenceKind,
    EvidenceRecord,
    FailureRoute,
    ToolRiskDecision,
    TOOL_RISK_LEVELS,
)
from .jev_client import JevClient
from .decision.routers import MemoryDecision, ModelRouter, TestSelector, ToolRouter, RoutingDecision


class DecisionController:
    """Compose the four Phase 1 components behind one stable interface."""

    def __init__(self, store: EvidenceStore, *, jev_client: JevClient | None = None):
        self.store = store
        self.jev_client = jev_client or JevClient()
        self.sufficiency_judge = EvidenceSufficiencyJudge()
        self.stop_judge = StopJudge(sufficiency_judge=self.sufficiency_judge)
        self.failure_router = FailureRouter()
        self.model_router = ModelRouter()
        self.tool_router = ToolRouter()
        self.test_selector = TestSelector()
        self.memory_decision = MemoryDecision()
        self._audited_jev_calls = 0

    @classmethod
    def from_directory(
        cls,
        root_dir: str | Path,
        *,
        jev_timeout: float | None = None,
    ) -> "DecisionController":
        client = JevClient(timeout=jev_timeout) if jev_timeout else None
        return cls(EvidenceStore(root_dir), jev_client=client)

    def record_evidence(
        self,
        kind: EvidenceKind | str,
        content: Any,
        *,
        severity: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        return self.store.append(kind, content, severity=severity, metadata=metadata)

    def evidence(self) -> list[EvidenceRecord]:
        return self.store.read_all()

    def judge_sufficiency(
        self,
        *,
        requirement: str,
        acceptance_criteria: Iterable[str] | None = None,
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]] | None = None,
        use_model: bool = True,
        domain: str | None = None,
        timeout: float | None = None,
    ):
        records = list(evidence) if evidence is not None else self.evidence()
        result = self.sufficiency_judge.evaluate(
            requirement,
            acceptance_criteria,
            records,
            domain=domain,
            use_model=use_model,
            model_client=self.jev_client,
            timeout=timeout,
        )
        self._record_jev_audit(
            operation="evidence_sufficiency",
            context={
                "requirement": requirement[:600],
                "evidence_count": len(records),
                "adopted": result.to_dict(),
            },
        )
        return result

    def judge_stop(
        self,
        *,
        requirement: str,
        acceptance_criteria: Iterable[str] | None = None,
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]] | None = None,
        iteration: int = 0,
        agent_requested_stop: bool = True,
        use_model: bool = True,
        domain: str | None = None,
        timeout: float | None = None,
    ):
        records = list(evidence) if evidence is not None else self.evidence()
        result = self.stop_judge.evaluate(
            requirement=requirement,
            acceptance_criteria=acceptance_criteria,
            evidence=records,
            iteration=iteration,
            agent_requested_stop=agent_requested_stop,
            domain=domain,
            use_model=use_model,
            model_client=self.jev_client,
            timeout=timeout,
        )
        self._record_jev_audit(
            operation="stop",
            context={
                "requirement": requirement[:600],
                "evidence_count": len(records),
                "iteration": iteration,
                "adopted": result.to_dict(),
            },
        )
        return result

    def _record_jev_audit(
        self,
        *,
        operation: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist one evidence record per new JEV call.

        The record carries the call status, the submission digest, the adopted
        output and any error, so a safety fallback can be told apart from a
        JEV decision.
        """

        calls = getattr(self.jev_client, "call_history", None)
        if isinstance(calls, list):
            pending = [
                dict(call)
                for call in calls[self._audited_jev_calls :]
                if isinstance(call, Mapping)
            ]
            self._audited_jev_calls = len(calls)
        else:
            call = getattr(self.jev_client, "last_call", None)
            pending = [dict(call)] if isinstance(call, Mapping) else []
        final_adopted = (
            dict(context.get("adopted"))
            if context and isinstance(context.get("adopted"), Mapping)
            else {}
        )
        for call in pending:
            call_context = dict(context or {})
            if len(pending) > 1:
                fallback = bool(not call.get("ok") or final_adopted.get("fallback"))
                call_context["adopted"] = {
                    **final_adopted,
                    "source": (
                        "local_fallback"
                        if fallback
                        else str(final_adopted.get("source") or "JEV")
                    ),
                    "fallback": fallback,
                    "fallback_reason": (
                        str(final_adopted.get("fallback_reason") or "")
                        if fallback
                        else ""
                    ),
                    "call_operation": call.get("operation") or operation,
                }
            content: dict[str, Any] = {"jev": call}
            if call_context:
                content["context"] = call_context
            self.record_evidence(
                EvidenceKind.DECISION,
                content,
                metadata={
                    "source": "JevClient",
                    "decision": "jev_call",
                    "operation": call.get("operation") or operation,
                },
            )

    def route_failure(
        self,
        failure: FailureInput,
        *,
        use_model: bool = True,
        timeout: float | None = None,
    ) -> FailureRoute:
        """Submit failure evidence to JEV and record the call audit.

        Local classification only runs as a marked safety fallback when JEV is
        unavailable or its answer is invalid.
        """

        route = self.failure_router.classify(
            failure,
            use_model=use_model,
            model_client=self.jev_client,
            model_timeout=timeout,
        )
        self._record_jev_audit(
            operation="failure_route",
            context={"input_summary": failure.summary(), "adopted": route.to_dict()},
        )
        return route

    def route_direction(
        self,
        requirement: str,
        space: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Route the current task through the controller's audited JEV client."""

        from .direction import route

        result = route(requirement, space, client=self.jev_client, timeout=timeout)
        self._record_jev_audit(
            operation="direction_route",
            context={"requirement": requirement[:600], "adopted": result},
        )
        return result

    def classify_tool_evidence(
        self,
        *,
        tool_name: str,
        command: str,
        response_text: str,
        exit_code: int | None,
        prescreen: Mapping[str, Any],
        use_model: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Resolve a low-confidence PostToolUse result with at most one JEV call."""

        if use_model and self.jev_client.available:
            try:
                answer = self.jev_client.classify_tool_evidence(
                    tool_name=tool_name,
                    command=command,
                    response_summary=response_text,
                    exit_code=exit_code,
                    prescreen=prescreen,
                    timeout=timeout,
                )
            except Exception as exc:
                fallback_reason = f"JEV tool-evidence classification failed: {exc}"[:300]
                adopted = self._with_provenance(
                    prescreen,
                    source="local_fallback",
                    fallback=True,
                    fallback_reason=fallback_reason,
                )
                self._record_jev_audit(
                    operation="tool_evidence",
                    context={
                        "prescreen": dict(prescreen),
                        "error": str(exc)[:240],
                        "adopted": adopted,
                    },
                )
                return adopted
            try:
                self._validate_tool_evidence_answer(answer)
            except ValueError as exc:
                fallback_reason = f"Invalid JEV tool-evidence response: {exc}"[:300]
                adopted = self._with_provenance(
                    prescreen,
                    source="local_fallback",
                    fallback=True,
                    fallback_reason=fallback_reason,
                )
                self._record_jev_audit(
                    operation="tool_evidence",
                    context={"prescreen": dict(prescreen), "error": str(exc), "adopted": adopted},
                )
                return adopted
            adopted = self._with_provenance(
                answer,
                source="JEV",
                fallback=False,
                fallback_reason="",
            )
            self._record_jev_audit(
                operation="tool_evidence",
                context={"prescreen": dict(prescreen), "adopted": adopted},
            )
            return adopted
        if use_model:
            note = getattr(self.jev_client, "note_unavailable", None)
            if callable(note):
                note("tool_evidence")
        fallback_reason = (
            "JEV is unavailable; the local tool-evidence pre-screen was adopted."
            if use_model
            else ""
        )
        adopted = self._with_provenance(
            prescreen,
            source="local_fallback" if use_model else "local",
            fallback=bool(use_model),
            fallback_reason=fallback_reason,
        )
        self._record_jev_audit(
            operation="tool_evidence",
            context={"prescreen": dict(prescreen), "adopted": adopted},
        )
        return adopted

    def judge_tool_use(
        self,
        *,
        tool_name: str,
        tool_input: Any = None,
        requirement: str = "",
        domain: str | None = None,
        use_model: bool = True,
        timeout: float | None = None,
    ) -> ToolRiskDecision:
        """Judge a planned tool call before it runs.

        The local pre-screen decides read-only calls, validation commands,
        dedicated file-edit tools and unambiguously destructive commands.
        Everything else is submitted to JEV. When JEV cannot answer, the local
        pre-screen result is returned with explicit ``source``/``fallback``
        markers, so a safety fallback is never presented as a JEV conclusion.
        """

        prescreen = prescreen_tool_use(tool_name=tool_name, tool_input=tool_input)
        command = extract_tool_command(tool_input)
        summary = ""
        if tool_input is not None and not command:
            try:
                summary = json.dumps(tool_input, ensure_ascii=False, default=str, sort_keys=True)[:1200]
            except (TypeError, ValueError):
                summary = str(tool_input)[:1200]

        base = ToolRiskDecision(
            tool_name=tool_name,
            risk=prescreen.risk,
            recommendation=prescreen.recommendation,
            reason=prescreen.reason,
            confidence=prescreen.confidence,
            appropriate=prescreen.appropriate,
            signals=list(prescreen.signals),
            source="local_prescreen",
            fallback=False,
        )
        if not prescreen.needs_model:
            return base
        if not use_model or not self.jev_client.available:
            if use_model:
                note = getattr(self.jev_client, "note_unavailable", None)
                if callable(note):
                    note("tool_risk")
            fallback_reason = (
                "JEV is unavailable; the local pre-screen judgment was adopted."
                if use_model
                else ""
            )
            decision = replace(
                base,
                source="local_fallback" if use_model else "local",
                fallback=bool(use_model),
                fallback_reason=fallback_reason,
            )
            self._record_jev_audit(
                operation="tool_risk",
                context={"prescreen": prescreen.to_dict(), "adopted": decision.to_dict()},
            )
            return decision
        try:
            answer = self.jev_client.judge_tool_risk(
                tool_name=tool_name,
                command=command,
                tool_input_summary=summary,
                requirement=requirement,
                domain=domain or "unknown",
                prescreen=prescreen.to_dict(),
                timeout=timeout,
            )
            self._validate_tool_risk_answer(answer)
        except Exception as exc:
            fallback_reason = f"JEV tool-risk judgment failed: {exc}"[:300]
            decision = replace(
                base,
                source="local_fallback",
                fallback=True,
                fallback_reason=fallback_reason,
            )
            self._record_jev_audit(
                operation="tool_risk",
                context={
                    "prescreen": prescreen.to_dict(),
                    "error": str(exc)[:240],
                    "adopted": decision.to_dict(),
                },
            )
            return decision
        decision = ToolRiskDecision(
            tool_name=tool_name,
            risk=str(answer["risk"]),
            recommendation=str(answer["recommendation"]),
            reason=str(answer["reason"]),
            confidence=float(answer.get("confidence") or 0.0),
            appropriate=bool(answer.get("appropriate")),
            signals=[*prescreen.signals, "jev:tool_risk"],
            scores={
                str(key): float(value)
                for key, value in dict(answer.get("risk_scores") or {}).items()
            },
            source="JEV",
            fallback=False,
            fallback_reason="",
        )
        self._record_jev_audit(
            operation="tool_risk",
            context={"prescreen": prescreen.to_dict(), "adopted": decision.to_dict()},
        )
        return decision

    @staticmethod
    def _with_provenance(
        value: Mapping[str, Any],
        *,
        source: str,
        fallback: bool,
        fallback_reason: str,
    ) -> dict[str, Any]:
        result = dict(value)
        result.update(
            source=source,
            fallback=fallback,
            fallback_reason=fallback_reason,
        )
        return result

    @staticmethod
    def _validate_tool_evidence_answer(answer: Any) -> None:
        if not isinstance(answer, Mapping):
            raise ValueError("response must be an object")
        if answer.get("kind") not in {"test_result", "build_failure", "runtime", "other"}:
            raise ValueError("invalid evidence kind")

        def validate_probability(key: str, value: Any) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid {key}")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid {key}") from exc
            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ValueError(f"invalid {key}")

        for key in ("kind_confidence", "failed_confidence", "failure_type_confidence"):
            validate_probability(key, answer.get(key))
        if not isinstance(answer.get("failed"), bool):
            raise ValueError("invalid failed flag")

        for key, allowed in (
            ("kind_scores", {"test_result", "build_failure", "runtime", "other"}),
            ("failure_type_scores", set(FAILURE_TYPES)),
        ):
            scores = answer.get(key)
            if scores is None:
                continue
            if not isinstance(scores, Mapping):
                raise ValueError(f"invalid {key}")
            for label, value in scores.items():
                if str(label) not in allowed:
                    raise ValueError(f"invalid {key} label")
                validate_probability(f"{key}.{label}", value)

        failure_type = answer.get("failure_type")
        if answer["failed"] and failure_type not in FAILURE_TYPES:
            raise ValueError("invalid failure_type")
        if failure_type is not None and failure_type not in FAILURE_TYPES:
            raise ValueError("invalid failure_type")

        for key in ("kind_certain", "failure_type_certain", "failure_type_reasoning_required"):
            value = answer.get(key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"invalid {key}")

    @staticmethod
    def _validate_tool_risk_answer(answer: Any) -> None:
        if not isinstance(answer, Mapping):
            raise ValueError("response must be an object")
        if answer.get("risk") not in TOOL_RISK_LEVELS:
            raise ValueError("invalid risk level")
        if answer.get("recommendation") not in {"proceed", "confirm", "block"}:
            raise ValueError("invalid recommendation")
        if not isinstance(answer.get("reason"), str) or not answer["reason"].strip():
            raise ValueError("invalid reason")
        def validate_probability(key: str, value: Any) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid {key}")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid {key}") from exc
            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ValueError(f"invalid {key}")

        for key in ("confidence", "risk_confidence", "appropriate_confidence"):
            validate_probability(key, answer.get(key))
        if not isinstance(answer.get("appropriate"), bool):
            raise ValueError("invalid appropriate flag")
        for key in ("risk_certain", "reasoning_required"):
            value = answer.get(key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"invalid {key}")
        scores = answer.get("risk_scores")
        if not isinstance(scores, Mapping):
            raise ValueError("invalid risk_scores")
        if set(str(key) for key in scores) != set(TOOL_RISK_LEVELS):
            raise ValueError("invalid risk_scores label")
        for label, value in scores.items():
            validate_probability(f"risk_scores.{label}", value)

    def metrics(self) -> TaskMetrics:
        """Aggregate the observation metrics for this task's evidence store."""

        records = self.evidence()
        return MetricsAggregator().aggregate(
            records,
            task_id=self.store.root_dir.name,
            session_id=self.store.root_dir.parent.name,
            evidence_path=str(self.store.path),
            unreadable=self.store.unreadable_records,
        )

    def route_model(self, candidates, *, state=None, timeout=None) -> RoutingDecision:
        return self._route(self.model_router, candidates, state=state, timeout=timeout)

    def route_tool(self, candidates, *, state=None, timeout=None) -> RoutingDecision:
        return self._route(self.tool_router, candidates, state=state, timeout=timeout)

    def select_tests(self, candidates, *, state=None, timeout=None) -> RoutingDecision:
        return self._route(self.test_selector, candidates, state=state, timeout=timeout)

    def decide_memory(self, candidates, *, state=None, timeout=None) -> RoutingDecision:
        return self._route(self.memory_decision, candidates, state=state, timeout=timeout)

    def _route(self, router, candidates, *, state, timeout):
        decision = router.decide(candidates=candidates, client=self.jev_client, state=state, timeout=timeout)
        self._record_jev_audit(
            operation=router.operation,
            context={"adopted": decision.to_dict()},
        )
        self.record_evidence(EvidenceKind.DECISION, {"operation": router.operation, "decision": decision.to_dict()}, metadata={"decision": router.operation, "source": decision.source})
        return decision
