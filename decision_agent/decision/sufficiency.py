"""Evidence sufficiency checks for Phase 1 stop decisions."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..models import EvidenceKind, EvidenceRecord, EvidenceSufficiencyResult, normalize_task_domain
from ..jev_client import JevClient, JevDecisionError


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_SUCCESS_RE = re.compile(r"\b(?:passed|pass|success|succeeded|ok)\b", re.IGNORECASE)
_FAILURE_RE = re.compile(r"\b(?:failed|failure|error|exception)\b", re.IGNORECASE)
_NONZERO_FAILURE_COUNT_RE = re.compile(
    r"\b(?!0\b)\d+\s+(?:tests?\s+)?failed\b", re.IGNORECASE
)
_ZERO_FAILURE_RE = re.compile(
    r"\b(?:0|no)\s+(?:tests?\s+)?(?:failed|failures?|errors?)\b", re.IGNORECASE
)
_GOAL_PAYLOAD_KEYS = ("answer", "response", "finding", "completion", "final_response")


def _as_record(value: EvidenceRecord | Mapping[str, Any]) -> EvidenceRecord:
    return value if isinstance(value, EvidenceRecord) else EvidenceRecord.from_dict(value)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _status(record: EvidenceRecord) -> str:
    metadata_status = record.metadata.get("status")
    if metadata_status is not None:
        return str(metadata_status).lower()
    if isinstance(record.content, Mapping):
        content_status = record.content.get("status")
        if content_status is not None:
            return str(content_status).lower()
        structured_status = [
            record.content[key]
            for key in ("passed", "success")
            if key in record.content
        ]
        if structured_status:
            if not all(isinstance(item, bool) for item in structured_status):
                return "unknown"
            return "failed" if any(item is False for item in structured_status) else "passed"
    text = _text(record.content)
    has_success = bool(_SUCCESS_RE.search(text))
    has_failure = bool(_FAILURE_RE.search(text))
    if has_failure and has_success:
        # Mixed summaries such as "1 failed, 1 passed" must never be treated
        # as a passing validation. A zero-failure summary is the exception.
        if _ZERO_FAILURE_RE.search(text) and not _NONZERO_FAILURE_COUNT_RE.search(text):
            return "passed"
        return "failed"
    if has_failure:
        return "failed"
    if has_success:
        return "passed"
    return "unknown"


def _is_positive(record: EvidenceRecord) -> bool:
    if record.kind_value == EvidenceKind.FINAL_ACCEPTANCE.value:
        if isinstance(record.content, Mapping):
            return record.content.get("approved", record.content.get("implemented", True)) is not False
        return record.metadata.get("approved", True) is not False
    return _status(record) in {"passed", "pass", "success", "succeeded", "ok", "true"}


def _nonempty_payload(content: Any, keys: tuple[str, ...]) -> bool:
    if not isinstance(content, Mapping):
        return False
    return any(_has_payload_value(content.get(key)) for key in keys)


def _has_payload_value(value: Any) -> bool:
    """Return whether a decoded evidence value carries meaningful content."""

    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return math.isfinite(float(value)) and value != 0
    if isinstance(value, Mapping):
        return any(_has_payload_value(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return any(_has_payload_value(item) for item in value)
    return False


def _is_goal_evidence(record: EvidenceRecord) -> bool:
    """Return whether a record contains an answer/finding, not a gate claim."""

    if record.kind_value != EvidenceKind.FINAL_ACCEPTANCE.value or not _is_positive(record):
        return False
    return _nonempty_payload(
        record.content,
        _GOAL_PAYLOAD_KEYS,
    )


def _is_explicit_implementation_evidence(record: EvidenceRecord) -> bool:
    """Accept an explicit integration marker, but never an assistant claim."""

    if record.kind_value != EvidenceKind.FINAL_ACCEPTANCE.value:
        return False
    if record.metadata.get("role") in {"assistant_response", "answer", "finding", "completion"}:
        return False
    if record.metadata.get("source") == "Stop":
        return False
    if isinstance(record.content, Mapping) and record.content.get("implemented") is True:
        return True
    return record.metadata.get("implemented") is True


def _is_unverified_final_claim(record: EvidenceRecord) -> bool:
    if record.kind_value != EvidenceKind.FINAL_ACCEPTANCE.value:
        return False
    if _is_explicit_implementation_evidence(record):
        return False
    return not _nonempty_payload(record.content, ("validated", "validation"))


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value) if token.lower() not in _STOP_WORDS}


def _has_goal_evidence(records: list[EvidenceRecord]) -> bool:
    """Recognize an answer/finding without requiring implementation evidence."""

    for record in records:
        if not _nonempty_payload(record.content, _GOAL_PAYLOAD_KEYS) or not _is_positive(record):
            continue
        if _is_goal_evidence(record) or record.metadata.get("role") in {
            "assistant_response",
            "answer",
            "finding",
            "completion",
        }:
            return True
    return False


def _validation_correlation(record: EvidenceRecord) -> str:
    validation_id = record.metadata.get("validation_id", record.metadata.get("validation_group"))
    if isinstance(validation_id, str) and validation_id.strip():
        return f"id:{validation_id.strip().casefold()}"
    command = ""
    if isinstance(record.content, Mapping):
        raw_command = record.content.get("command")
        if isinstance(raw_command, str):
            command = raw_command
    if not command:
        raw_command = record.metadata.get("command")
        if isinstance(raw_command, str):
            command = raw_command
    normalized = " ".join(command.split()).casefold()
    return f"command:{normalized}" if normalized else ""


def _is_failed_record(record: EvidenceRecord) -> bool:
    return record.kind_value == EvidenceKind.BUILD_FAILURE.value or (
        record.kind_value in {EvidenceKind.TEST_RESULT.value, EvidenceKind.RUNTIME.value}
        and _status(record) in {"failed", "failure", "error"}
    )


# Shared evidence semantics used by the observation metrics. Keeping one
# correlation definition prevents metrics from declaring a failure resolved
# under weaker rules than the sufficiency gate.
is_positive_evidence = _is_positive
is_failed_evidence = _is_failed_record
validation_correlation = _validation_correlation


def _has_external_action_evidence(records: list[EvidenceRecord]) -> bool:
    """Recognize a result or confirmation without treating a final claim as proof."""

    result_keys = ("result", "confirmation", "receipt", "external_action_result", "completed")
    for record in records:
        if record.metadata.get("external_action_result") is True and _is_positive(record):
            return True
        if record.kind_value == EvidenceKind.RUNTIME.value:
            if _is_positive(record) or _nonempty_payload(record.content, result_keys):
                return True
            continue
        if record.kind_value not in {
            EvidenceKind.FINAL_ACCEPTANCE.value,
            EvidenceKind.LOG.value,
        }:
            continue
        if record.metadata.get("role") in {"assistant_response", "answer", "finding", "completion"}:
            continue
        if record.metadata.get("source") == "Stop" or not _is_positive(record):
            continue
        if _nonempty_payload(record.content, result_keys):
            return True
    return False


class EvidenceSufficiencyJudge:
    """Determine whether implementation and validation evidence is adequate.

    Integrations can provide exact ``criterion_ids`` in evidence metadata. A
    conservative text-match fallback is retained for hook payloads that only
    have human-readable test output.
    """

    def evaluate(
        self,
        requirement: str,
        acceptance_criteria: Iterable[str] | None,
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]],
        *,
        use_model: bool = False,
        model_client: JevClient | None = None,
        domain: str | None = None,
        timeout: float | None = None,
        fallback_reason: str = "",
    ) -> EvidenceSufficiencyResult:
        records = [_as_record(item) for item in evidence]
        if use_model and model_client is None:
            fallback_reason = fallback_reason or (
                "JEV evidence sufficiency was requested but no model client was provided."
            )
        validation_fallback_reason = ""
        resolved_domain = normalize_task_domain(domain)
        criteria = [str(item).strip() for item in (acceptance_criteria or []) if str(item).strip()]
        diff_records = [record for record in records if record.kind_value == EvidenceKind.GIT_DIFF.value]
        validation_records = [
            record
            for record in records
            if record.kind_value
            in {
                EvidenceKind.TEST_RESULT.value,
                EvidenceKind.RUNTIME.value,
            }
            and _is_positive(record)
        ]
        # A later success closes a failure only when both records identify the
        # same command or explicitly share a validation correlation id.
        unresolved_failures: list[EvidenceRecord] = []
        for failure_index, record in enumerate(records):
            if not _is_failed_record(record):
                continue
            correlation = _validation_correlation(record)
            resolved = bool(correlation) and any(
                later_index > failure_index
                and later.kind_value in {
                    EvidenceKind.TEST_RESULT.value,
                    EvidenceKind.RUNTIME.value,
                }
                and _is_positive(later)
                and _validation_correlation(later) == correlation
                for later_index, later in enumerate(records)
            )
            if not resolved:
                unresolved_failures.append(record)

        if resolved_domain in {"information", "investigation"}:
            has_implementation = _has_goal_evidence(records)
        elif resolved_domain == "external_action":
            has_implementation = _has_external_action_evidence(records)
        else:
            has_implementation = bool(diff_records) or any(
                _is_explicit_implementation_evidence(record)
                for record in records
            )
        criterion_results: dict[str, bool] = {}
        missing: list[str] = []
        reasons: list[str] = []
        validation_required = resolved_domain not in {
            "documentation",
            "information",
            "investigation",
            "external_action",
        }

        if use_model and model_client:
            try:
                validation_answer = model_client.decide_validation_required(
                    requirement=requirement,
                    evidence=[record.to_dict() for record in records],
                    timeout=timeout,
                )
                if not isinstance(validation_answer, bool):
                    raise ValueError("JEV validation_required result must be boolean")
                validation_required = validation_answer
            except Exception as exc:
                validation_fallback_reason = f"JEV validation decision failed: {exc}"[:300]
                reasons.append(f"JEV validation decision unavailable: {exc}")

        for index, criterion in enumerate(criteria):
            criterion_records = records
            if resolved_domain not in {"information", "investigation", "external_action"}:
                criterion_records = [
                    record for record in records if not _is_unverified_final_claim(record)
                ]
            covered = self._criterion_is_covered(criterion, index, criterion_records)
            criterion_results[criterion] = covered
            if not covered:
                missing.append(f"acceptance_criterion:{criterion}")

        if not has_implementation:
            if resolved_domain in {"information", "investigation"}:
                missing.append("goal_completion")
            elif resolved_domain == "external_action":
                missing.append("external_action_result")
            else:
                missing.append("git_diff")
            reasons.append(
                "No answer or finding evidence was found."
                if resolved_domain in {"information", "investigation"}
                else "No external action result or confirmation was found."
                if resolved_domain == "external_action"
                else "No implementation diff or final acceptance record was found."
            )
        if validation_required and not validation_records:
            missing.append("validation")
            reasons.append("No positive test, runtime, or final acceptance evidence was found.")
        elif not validation_required:
            reasons.append("JEV determined that this change does not require test or runtime validation.")
        if unresolved_failures:
            missing.append("unresolved_failure")
            reasons.append("At least one test or build failure remains unresolved.")

        # A requirement without explicit criteria still needs a validation trail.
        if not criteria and not requirement.strip():
            missing.append("requirement")
            reasons.append("The requirement text is empty.")

        # Preserve order while removing duplicate missing signals.
        missing = list(dict.fromkeys(missing))
        implemented = has_implementation and not unresolved_failures
        sufficient = implemented and not missing
        confidence = self._confidence(
            has_implementation=has_implementation,
            has_validation=bool(validation_records),
            criteria=criteria,
            missing=missing,
            failed=bool(unresolved_failures),
        )
        result = EvidenceSufficiencyResult(
            implemented=implemented,
            evidence_sufficient=sufficient,
            missing=missing,
            criterion_results=criterion_results,
            confidence=confidence,
            reasoning_required=not sufficient and confidence < 0.75,
            reasons=reasons,
            source="local_fallback" if fallback_reason or validation_fallback_reason else "local",
            fallback=bool(fallback_reason or validation_fallback_reason),
            fallback_reason="; ".join(item for item in (fallback_reason, validation_fallback_reason) if item),
        )
        # JEV is the decision authority whenever model use is requested.  The
        # local result above is only the explicitly marked safety fallback.
        if use_model and model_client:
            try:
                answer = model_client.evaluate_sufficiency(
                    requirement=requirement,
                    acceptance_criteria=criteria,
                    evidence=[record.to_dict() for record in records],
                    timeout=timeout,
                )
                result = self._merge_model_result(result, answer)
                combined_fallback_reason = "; ".join(
                    item for item in (fallback_reason, validation_fallback_reason) if item
                )
                if combined_fallback_reason:
                    result = EvidenceSufficiencyResult(
                        implemented=result.implemented,
                        evidence_sufficient=result.evidence_sufficient,
                        missing=result.missing,
                        criterion_results=result.criterion_results,
                        confidence=result.confidence,
                        reasoning_required=result.reasoning_required,
                        reasons=result.reasons,
                        source="local_fallback",
                        fallback=True,
                        fallback_reason=combined_fallback_reason,
                    )
            except Exception as exc:
                combined_fallback_reason = "; ".join(
                    item
                    for item in (
                        fallback_reason,
                        validation_fallback_reason,
                        f"JEV evidence sufficiency failed: {exc}"[:300],
                    )
                    if item
                )
                result = EvidenceSufficiencyResult(
                    implemented=result.implemented,
                    evidence_sufficient=result.evidence_sufficient,
                    missing=result.missing,
                    criterion_results=result.criterion_results,
                    confidence=result.confidence,
                    reasoning_required=True,
                    reasons=[*result.reasons, f"JEV unavailable: {exc}"],
                    source="local_fallback",
                    fallback=True,
                    fallback_reason=combined_fallback_reason,
                )
        return result

    @staticmethod
    def _merge_model_result(result: EvidenceSufficiencyResult, answer: Mapping[str, Any]) -> EvidenceSufficiencyResult:
        if not isinstance(answer, Mapping):
            raise JevDecisionError("JEV evidence sufficiency answer was invalid")
        probability = answer.get("noul", answer.get("probability", answer.get("confidence")))
        if isinstance(probability, bool):
            raise JevDecisionError("JEV evidence sufficiency probability was invalid")
        try:
            if probability is None:
                raise ValueError("missing sufficiency probability")
            confidence = float(probability)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("probability outside finite 0 to 1 range")
            confidence = min(0.99, confidence)
        except (TypeError, ValueError, OverflowError):
            raise JevDecisionError("JEV evidence sufficiency probability was invalid")
        if "missing" in answer or "missing_information" in answer:
            model_missing = answer.get("missing", answer.get("missing_information"))
            if not isinstance(model_missing, (list, tuple)) or any(
                not isinstance(item, str) or not item.strip() for item in model_missing
            ):
                raise JevDecisionError("JEV evidence sufficiency missing list was invalid")
            # JEV owns the final result: its missing list replaces the local
            # signals instead of being merged with them. The local list is kept
            # only when the answer omits the field entirely.
            missing = list(dict.fromkeys(str(item) for item in model_missing))
        else:
            missing = list(result.missing)
        reasons = [*result.reasons]
        reason = answer.get("reason")
        explanation = answer.get("explanation")
        if reason is not None and not isinstance(reason, str):
            raise JevDecisionError("JEV evidence sufficiency reason was invalid")
        if explanation is not None and not isinstance(explanation, str):
            raise JevDecisionError("JEV evidence sufficiency explanation was invalid")
        if reason or explanation:
            reasons.append(f"JEV: {reason or explanation}")
        implemented = answer.get("implemented")
        if implemented is None:
            implemented = answer.get("implemented_bool")
        if not isinstance(implemented, bool):
            raise JevDecisionError("JEV implemented result was invalid")
        evidence_sufficient = answer.get("evidence_sufficient")
        if evidence_sufficient is None:
            evidence_sufficient = answer.get("evidence_sufficient_bool")
        if not isinstance(evidence_sufficient, bool):
            raise JevDecisionError("JEV evidence_sufficient result was invalid")
        if evidence_sufficient and (not implemented or missing):
            raise JevDecisionError(
                "JEV evidence sufficiency contradicts implemented or missing fields"
            )
        return EvidenceSufficiencyResult(
            implemented=bool(implemented),
            evidence_sufficient=bool(evidence_sufficient),
            missing=missing,
            criterion_results=result.criterion_results,
            confidence=round(confidence, 2),
            reasoning_required=result.reasoning_required,
            reasons=reasons,
            source="JEV",
            fallback=False,
            fallback_reason="",
        )

    def _criterion_is_covered(
        self,
        criterion: str,
        index: int,
        records: list[EvidenceRecord],
    ) -> bool:
        criterion_tokens = _tokens(criterion)
        for record in records:
            metadata_ids = record.metadata.get("criterion_ids", record.metadata.get("criteria", []))
            if isinstance(metadata_ids, str):
                metadata_ids = [metadata_ids]
            if index in metadata_ids or str(index) in metadata_ids or criterion in metadata_ids:
                return _is_positive(record)
            content_text = _text(record.content)
            if criterion.lower() in content_text.lower() and _is_positive(record):
                return True
            record_tokens = _tokens(content_text)
            if criterion_tokens and len(criterion_tokens & record_tokens) / len(criterion_tokens) >= 0.6:
                if _is_positive(record):
                    return True
        return False

    @staticmethod
    def _confidence(
        *,
        has_implementation: bool,
        has_validation: bool,
        criteria: list[str],
        missing: list[str],
        failed: bool,
    ) -> float:
        score = 0.25
        if has_implementation:
            score += 0.3
        if has_validation:
            score += 0.3
        if criteria and not any(item.startswith("acceptance_criterion:") for item in missing):
            score += 0.1
        if failed:
            score -= 0.25
        if missing:
            score -= min(0.15, 0.03 * len(missing))
        return round(max(0.0, min(0.99, score)), 2)
