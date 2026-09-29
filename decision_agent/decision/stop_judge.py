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
                    require_certificate=True,
                )
                certificate = answer.get("completion_certificate") if isinstance(answer, Mapping) else None
                if certificate is not None:
                    validated_certificate = self._validate_completion_certificate(certificate, records)
                    return StopDecision(
                        status=validated_certificate["status"],
                        reason=validated_certificate["reason"],
                        confidence=validated_certificate["confidence"],
                        missing=validated_certificate["missing"],
                        iteration=iteration,
                        goal_completed=validated_certificate["status"] == StopStatus.STOP,
                        goal_confidence=validated_certificate["confidence"],
                        domain=validated_certificate["domain"],
                        domain_confidence=validated_certificate["domain_confidence"],
                        source="JEV",
                        fallback=False,
                        completion_standards=validated_certificate["standards"],
                        standard_results=validated_certificate["claims"],
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
        certificate_failure = "completion certificate" in fallback_reason.lower() or "completion_certificate" in fallback_reason.lower()
        if certificate_failure and use_model and isinstance(model_client, JevClient):
            # A certificate failure must not silently become the legacy local
            # git_diff/validation gate. That was the source of misleading Stop
            # messages after the certificate path was introduced.
            resolved_domain = normalize_task_domain(domain)
            return StopDecision(
                status=StopStatus.CONTINUE,
                reason="Completion certificate unavailable or invalid; evidence could not be verified.",
                confidence=0.0,
                missing=["completion_certificate"],
                iteration=iteration,
                goal_completed=False,
                goal_confidence=0.0,
                domain=resolved_domain,
                source="local_fallback",
                fallback=True,
                fallback_reason=fallback_reason,
            )
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
    def _validate_completion_certificate(certificate: Any, records: list[EvidenceRecord | Mapping[str, Any]]) -> dict[str, Any]:
        if not isinstance(certificate, Mapping):
            raise ValueError("JEV completion certificate must be an object")
        status = certificate.get("status")
        if status not in {item.value for item in StopStatus}:
            raise ValueError("JEV completion certificate has invalid status")
        domain = normalize_task_domain(certificate.get("domain"))
        if domain == "unknown":
            raise ValueError("JEV completion certificate requires a concrete domain")
        claims = certificate.get("claims")
        standards = certificate.get("standards")
        if not isinstance(claims, list) or not isinstance(standards, list):
            raise ValueError("JEV completion certificate requires standard and claim lists")
        if not claims or not standards:
            if status not in {StopStatus.CONTINUE.value, StopStatus.ESCALATE.value}:
                raise ValueError("JEV cannot stop without defined completion standards")
            reason_code = certificate.get("reason_code")
            if reason_code not in {
                "completion_standards_need_clarification",
                "no_verifiable_completion_standard",
                "iteration_limit",
            }:
                raise ValueError("empty standard set requires an explicit JEV clarification outcome")
            clarification_cause = certificate.get("clarification_cause")
            if clarification_cause not in {
                "none", "input_unreadable", "goal_ambiguous", "scope_unclear",
                "conflicting_requirements", "missing_success_condition", "no_verifiable_outcome",
            }:
                raise ValueError("JEV clarification outcome has invalid clarification_cause")
            confidence = certificate.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
                raise ValueError("JEV completion certificate has invalid confidence")
            reason = certificate.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("JEV clarification outcome requires a reason")
            if any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("id"), str)
                or not isinstance(item.get("text"), str)
                for item in standards
            ):
                raise ValueError("JEV clarification outcome has invalid proposed standards")
            return {
                "status": StopStatus(status),
                "domain": domain,
                "confidence": float(confidence),
                "domain_confidence": float(certificate.get("domain_confidence", confidence)),
                "missing": [reason_code],
                "reason": reason,
                "reason_code": reason_code,
                "clarification_cause": clarification_cause,
                "reason_detail": reason,
                "standards": [dict(item) for item in standards],
                "claims": [],
            }
        if not standards:
            raise ValueError("JEV completion certificate requires JEV-defined standards")
        standard_ids = {
            item.get("id") for item in standards
            if isinstance(item, Mapping)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("text"), str)
        }
        if len(standard_ids) != len(standards):
            raise ValueError("JEV completion certificate has invalid standards")
        if "goal" not in standard_ids:
            raise ValueError("JEV completion certificate must include the user's goal as a standard")
        evidence_ids = {
            str(item.id)
            for item in records
            if isinstance(item, EvidenceRecord) and item.id
        }
        evidence_ids.update(
            str(item.get("id"))
            for item in records
            if isinstance(item, Mapping) and item.get("id")
        )
        missing: list[str] = []
        for claim in claims:
            if not isinstance(claim, Mapping) or not isinstance(claim.get("id"), str):
                raise ValueError("JEV completion certificate has invalid claim")
            if claim.get("id") not in standard_ids:
                raise ValueError("JEV completion certificate claim is not a defined standard")
            claim_status = claim.get("status")
            if claim_status not in {"satisfied", "unsatisfied", "not_applicable", "unverifiable"}:
                raise ValueError("JEV completion certificate has invalid claim status")
            refs = claim.get("evidence_ids", [])
            if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in evidence_ids for ref in refs):
                raise ValueError("JEV completion certificate has invalid evidence_ids")
            if claim_status == "satisfied" and not refs:
                raise ValueError("satisfied claim must cite evidence_ids")
            evidence_status = claim.get("evidence_status")
            if evidence_status not in {"none", "sufficient", "missing", "weak", "irrelevant", "contradictory"}:
                raise ValueError("JEV completion certificate has invalid evidence_status")
            if claim.get("next_evidence") not in {
                "none", "requirement_confirmation", "implementation_diff", "targeted_test",
                "full_test_suite", "build_or_lint", "runtime_validation", "tool_output", "user_confirmation",
            }:
                raise ValueError("JEV completion certificate has invalid next_evidence")
            if claim_status in {"unsatisfied", "unverifiable"}:
                missing.append(str(claim.get("id")))
            if claim_status == "not_applicable":
                raise ValueError("a JEV-defined required standard cannot be not_applicable")
            if evidence_status in {"missing", "weak", "irrelevant", "contradictory"}:
                missing.append(str(claim.get("id")))
            if status == StopStatus.STOP.value and evidence_status != "sufficient":
                raise ValueError("JEV cannot stop unless every required standard has sufficient evidence")
        if {claim.get("id") for claim in claims if isinstance(claim, Mapping)} != standard_ids:
            raise ValueError("JEV completion certificate must evaluate every defined standard")
        contradictions = certificate.get("contradictions", [])
        if not isinstance(contradictions, list) or any(not isinstance(item, str) for item in contradictions):
            raise ValueError("JEV completion certificate has invalid contradictions")
        if status == StopStatus.STOP.value and (missing or contradictions):
            raise ValueError("JEV completion certificate contradicts stop status")
        confidence = certificate.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
            raise ValueError("JEV completion certificate has invalid confidence")
        domain_confidence = certificate.get("domain_confidence", confidence)
        if isinstance(domain_confidence, bool) or not isinstance(domain_confidence, (int, float)) or not math.isfinite(float(domain_confidence)) or not 0 <= domain_confidence <= 1:
            raise ValueError("JEV completion certificate has invalid domain_confidence")
        reason = certificate.get("reason") or "JEV returned a completion certificate."
        if not isinstance(reason, str):
            raise ValueError("JEV completion certificate has invalid reason")
        reason_code = certificate.get("reason_code")
        valid_reason_codes = {
            "none", "goal_incomplete", "missing_claim", "insufficient_evidence",
            "contradiction", "uncertain", "iteration_limit", "no_verifiable_standard",
            "completion_standards_need_clarification", "no_verifiable_completion_standard",
        }
        if reason_code not in valid_reason_codes:
            raise ValueError("JEV completion certificate has invalid reason_code")
        if status != StopStatus.STOP.value and reason_code == "none":
            raise ValueError("non-stop completion certificate requires a blocking reason_code")
        reason_detail = certificate.get("reason_detail") or reason
        if not isinstance(reason_detail, str):
            raise ValueError("JEV completion certificate has invalid reason_detail")
        return {
            "status": StopStatus(status),
            "domain": domain,
            "confidence": float(confidence),
            "domain_confidence": float(domain_confidence),
            "missing": missing,
            "reason": reason,
            "reason_code": reason_code,
            "reason_detail": reason_detail,
            "standards": [dict(item) for item in standards],
            "claims": [dict(item) for item in claims],
        }

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
