"""Stop/continue decisions for the Stop hook."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from ..models import TASK_DOMAINS, EvidenceRecord, StopDecision, StopStatus, normalize_task_domain
from ..jev_client import JevClient
from .sufficiency import EvidenceSufficiencyJudge


class StopJudge:
    """Gate an agent's completion claim with evidence and loop protection."""

    def __init__(self, *, max_iterations: int = 3, sufficiency_judge: EvidenceSufficiencyJudge | None = None):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.max_iterations = max_iterations
        self.sufficiency_judge = sufficiency_judge or EvidenceSufficiencyJudge()

    def evaluate(
        self,
        *,
        requirement: str,
        acceptance_criteria: Iterable[str] | None,
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]],
        iteration: int = 0,
        agent_requested_stop: bool = True,
        use_model: bool = False,
        model_client: JevClient | None = None,
        domain: str | None = None,
        timeout: float | None = None,
    ) -> StopDecision:
        records = list(evidence)
        fallback_reason = ""
        if use_model and model_client is None:
            fallback_reason = "JEV stop decision was requested but no model client was provided."
        if use_model and model_client:
            try:
                answer = model_client.judge_stop(
                    requirement=requirement,
                    acceptance_criteria=[str(item) for item in (acceptance_criteria or [])],
                    evidence=[record.to_dict() if isinstance(record, EvidenceRecord) else dict(record) for record in records],
                    iteration=iteration, max_iterations=self.max_iterations, agent_requested_stop=agent_requested_stop,
                    domain=domain,
                    timeout=timeout,
                )
                validated = self._validate_jev_answer(answer)
                status = validated["status"]
                missing = validated["missing"]
                resolved_domain = validated["domain"]
                return StopDecision(
                    status=status,
                    reason=validated["reason"] or "JEV returned a domain stop decision.",
                    confidence=validated["confidence"],
                    missing=missing,
                    iteration=iteration,
                    goal_completed=validated["goal_completed"],
                    goal_confidence=validated["goal_confidence"],
                    domain=resolved_domain,
                    domain_confidence=validated["domain_confidence"],
                    source="JEV",
                    fallback=False,
                )
            except Exception as exc:
                fallback_reason = f"JEV stop decision failed: {exc}"[:300]
        sufficiency = self.sufficiency_judge.evaluate(
            requirement,
            acceptance_criteria,
            records,
            domain=domain,
            use_model=False,
            model_client=None,
            fallback_reason=fallback_reason,
            timeout=timeout,
        )
        resolved_domain = normalize_task_domain(domain)
        local_source = "local_fallback" if fallback_reason else "local"
        if not agent_requested_stop:
            return StopDecision(
                status=StopStatus.CONTINUE,
                reason="The agent has not requested completion.",
                confidence=0.99,
                missing=sufficiency.missing,
                iteration=iteration,
                domain=resolved_domain,
                source=local_source,
                fallback=bool(fallback_reason),
                fallback_reason=fallback_reason,
            )

        if sufficiency.evidence_sufficient:
            return StopDecision(
                status=StopStatus.STOP,
                reason="Requirement and validation evidence are sufficient.",
                confidence=sufficiency.confidence,
                missing=[],
                iteration=iteration,
                goal_completed=True,
                goal_confidence=sufficiency.confidence,
                domain=resolved_domain,
                domain_confidence=1.0 if resolved_domain != "unknown" else None,
                source=local_source,
                fallback=bool(fallback_reason),
                fallback_reason=fallback_reason,
            )

        if iteration >= self.max_iterations:
            return StopDecision(
                status=StopStatus.ESCALATE,
                reason=(
                    f"Evidence is still insufficient after {iteration} decision cycles; "
                    "escalate to strong-model review."
                ),
                confidence=max(0.6, sufficiency.confidence),
                missing=sufficiency.missing,
                iteration=iteration,
                goal_completed=False,
                goal_confidence=sufficiency.confidence,
                domain=resolved_domain,
                source=local_source,
                fallback=bool(fallback_reason),
                fallback_reason=fallback_reason,
            )

        return StopDecision(
            status=StopStatus.CONTINUE,
            reason=self._continue_reason(sufficiency.missing),
            confidence=max(0.5, sufficiency.confidence),
            missing=sufficiency.missing,
            iteration=iteration,
            goal_completed=False,
            goal_confidence=sufficiency.confidence,
            domain=resolved_domain,
            source=local_source,
            fallback=bool(fallback_reason),
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _validate_jev_answer(answer: Any) -> dict[str, Any]:
        if not isinstance(answer, Mapping):
            raise ValueError("JEV Stop response must be an object")

        status = answer.get("status")
        if not isinstance(status, str) or status not in {item.value for item in StopStatus}:
            raise ValueError("JEV returned an invalid Stop status")

        def probability(key: str, *, required: bool = False) -> float | None:
            value = answer.get(key)
            if value is None and not required:
                return None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"JEV returned an invalid {key}")
            return float(value)

        confidence = probability("confidence", required=True)
        raw_missing = answer.get("missing", [])
        if isinstance(raw_missing, str) or not isinstance(raw_missing, (list, tuple)):
            raise ValueError("JEV returned an invalid missing list")
        if any(not isinstance(item, str) or not item.strip() for item in raw_missing):
            raise ValueError("JEV returned an invalid missing list")

        goal_completed = answer.get("goal_completed")
        if not isinstance(goal_completed, bool):
            raise ValueError("JEV returned an invalid goal_completed flag")
        if status == StopStatus.STOP and (not goal_completed or raw_missing):
            raise ValueError(
                "JEV stop decision contradicts goal completion or missing evidence"
            )
        goal_confidence = probability("goal_confidence", required=True)
        domain_confidence = probability("domain_confidence", required=True)

        raw_domain = answer.get("domain")
        if not isinstance(raw_domain, str) or raw_domain not in TASK_DOMAINS:
            raise ValueError("JEV returned an invalid domain")
        resolved_domain = normalize_task_domain(raw_domain)

        domain_scores = answer.get("domain_scores")
        if domain_scores is not None:
            if not isinstance(domain_scores, Mapping):
                raise ValueError("JEV returned invalid domain scores")
            for key, value in domain_scores.items():
                if str(key) not in TASK_DOMAINS:
                    raise ValueError("JEV returned invalid domain scores")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0 <= value <= 1
                ):
                    raise ValueError("JEV returned invalid domain scores")

        reason = answer.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("JEV returned an invalid reason")
        return {
            "status": status,
            "confidence": confidence,
            "missing": list(raw_missing),
            "goal_completed": goal_completed,
            "goal_confidence": goal_confidence,
            "domain": resolved_domain,
            "domain_confidence": domain_confidence,
            "reason": reason,
        }

    @staticmethod
    def _continue_reason(missing: list[str]) -> str:
        if not missing:
            return "Additional verification is required before stopping."
        return "Continue: missing " + ", ".join(missing) + "."
