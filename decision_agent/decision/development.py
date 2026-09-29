"""Evidence-bound development planning decisions.

The planner does not edit files or run commands. It narrows the next Codex
action to a finite, evidence-linked choice; execution and evidence collection
remain program/tool responsibilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class DevelopmentPlanner:
    OPERATIONS = {
        "next_action": "next_development_action",
        "files": "modification_file_selection",
        "direction": "modification_direction",
        "verification": "modification_verification",
    }

    def __init__(self, client: Any):
        self.client = client

    def choose_next_action(self, *, problem: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]] = (), timeout: float | None = None) -> dict[str, Any]:
        return self._choose("next_action", problem, candidates, evidence, timeout)

    def choose_files(self, *, problem: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]] = (), timeout: float | None = None) -> dict[str, Any]:
        return self._choose("files", problem, candidates, evidence, timeout)

    def choose_direction(self, *, problem: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]] = (), timeout: float | None = None) -> dict[str, Any]:
        return self._choose("direction", problem, candidates, evidence, timeout)

    def choose_verification(self, *, problem: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]] = (), timeout: float | None = None) -> dict[str, Any]:
        return self._choose("verification", problem, candidates, evidence, timeout)

    @staticmethod
    def verify_execution(*, problem: Mapping[str, Any], selected_files: Sequence[str], changed_files: Sequence[str], validation_evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Deterministically verify the executed step stayed within its JEV scope.

        JEV chooses the scope and verification action; this check only validates
        observable execution facts and therefore cannot silently replace a JEV
        decision.
        """
        expected = {str(path).replace("\\", "/") for path in selected_files if str(path).strip()}
        actual = {str(path).replace("\\", "/") for path in changed_files if str(path).strip()}
        problem_id = str(problem.get("problem_id") or "")
        linked = [item for item in validation_evidence if str(item.get("problem_id") or item.get("metadata", {}).get("problem_id") or "") == problem_id]
        if actual - expected:
            status = "scope_violation"
            reason = "changed_files_outside_selected_scope"
        elif not linked:
            status = "missing_validation"
            reason = "no_validation_evidence_bound_to_problem"
        else:
            status = "verified"
            reason = "execution_scope_and_validation_evidence_verified"
        return {
            "problem": dict(problem),
            "selected_files": sorted(expected),
            "changed_files": sorted(actual),
            "validation_evidence_ids": [str(item.get("id")) for item in linked if item.get("id")],
            "status": status,
            "reason": reason,
            "source": "local_validation",
            "fallback": False,
            "fallback_reason": "",
        }

    def _choose(self, stage: str, problem: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]], timeout: float | None) -> dict[str, Any]:
        normalized = [self._candidate(item) for item in candidates]
        if not normalized or len(normalized) > 8:
            raise ValueError("development planner requires 1-8 candidates")
        if not isinstance(problem.get("problem_id"), str) or not problem["problem_id"].strip():
            raise ValueError("problem_id is required")
        if not isinstance(problem.get("problem_statement"), str) or not problem["problem_statement"].strip():
            raise ValueError("problem_statement is required")
        evidence_ids = {str(item.get("id")) for item in evidence if item.get("id")}
        for item in normalized:
            if not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids):
                raise ValueError(f"candidate {item['id']} must cite existing evidence_ids")
        operation = self.OPERATIONS[stage]
        questions = {
            f"candidate_{index}": {
                "type": "noul",
                "instructions": (
                    f"Score whether candidate {item['id']!r} is the best {stage} choice for the stated problem. "
                    "Use only problem relevance, evidence_ids, risk and expected outcome; do not invent evidence."
                ),
            }
            for index, item in enumerate(normalized)
        }
        state = {
            "problem": dict(problem),
            "candidates": normalized,
            "evidence": list(evidence),
        }
        provenance = {"source": "JEV", "fallback": False, "fallback_reason": ""}
        try:
            response = self.client.decide(state=state, questions=questions, operation=operation, timeout=timeout)
            answers = response.get("answers", {})
            scores: dict[str, float] = {}
            for index, item in enumerate(normalized):
                value = (answers.get(f"candidate_{index}") or {}).get("noul")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= value <= 1:
                    raise ValueError("invalid JEV candidate score")
                scores[item["id"]] = float(value)
            selected = max(scores, key=scores.get)
        except Exception as exc:
            reason = f"JEV {operation} failed: {exc}"[:300]
            provenance = {"source": "local_fallback", "fallback": True, "fallback_reason": reason}
            selected = normalized[0]["id"]
            scores = {selected: 0.0}
        return {
            "stage": stage,
            "operation": operation,
            "problem": dict(problem),
            "selected_id": selected,
            "selected": next(item for item in normalized if item["id"] == selected),
            "scores": scores,
            **provenance,
        }

    @staticmethod
    def _candidate(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("development candidate must be an object")
        candidate_id = str(value.get("id") or "").strip()
        action = str(value.get("action") or "").strip()
        evidence_ids = value.get("evidence_ids")
        if not candidate_id or not action or not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("candidate requires id, action and evidence_ids")
        return {**dict(value), "id": candidate_id, "action": action, "evidence_ids": [str(item) for item in evidence_ids]}
