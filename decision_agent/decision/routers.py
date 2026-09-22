"""Efficiency-oriented routing decisions backed by JEV.

Each router returns a structured decision and marks local heuristics as an
explicit fallback.  The classes are intentionally small so hook integrations
can supply their own candidate lists without coupling to execution code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RoutingDecision:
    choice: str
    confidence: float
    reason: str
    candidates: list[str] = field(default_factory=list)
    source: str = "JEV"
    fallback: bool = False
    fallback_reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice, "confidence": self.confidence,
            "reason": self.reason, "candidates": list(self.candidates),
            "source": self.source, "fallback": self.fallback,
            "fallback_reason": self.fallback_reason, "scores": dict(self.scores),
        }


class _Router:
    operation = "router"

    def _jev(self, client: Any, state: Mapping[str, Any], candidates: Sequence[str], *, timeout: float | None) -> RoutingDecision:
        questions = {
            f"candidate_{index}": {"type": "noul", "instructions": f"Score 0 to 1 for whether candidate {candidate!r} is the best choice."}
            for index, candidate in enumerate(candidates)
        }
        response = client.decide(
            state={**dict(state), "candidates": list(candidates)},
            questions=questions,
            operation=self.operation,
            timeout=timeout,
        )
        answers = response.get("answers", {})
        scores = {}
        for index, candidate in enumerate(candidates):
            answer = answers.get(f"candidate_{index}")
            value = answer.get("noul") if isinstance(answer, Mapping) else None
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError("invalid JEV routing response")
            scores[candidate] = float(value)
        choice = max(scores, key=scores.get)
        return RoutingDecision(str(choice), scores[choice], "JEV selected the highest-scoring candidate.", list(candidates), "JEV", False, "", scores)

    def decide(self, *, candidates: Sequence[str], client: Any | None = None, state: Mapping[str, Any] | None = None, timeout: float | None = None) -> RoutingDecision:
        choices = [str(item) for item in candidates if str(item)]
        if not choices:
            raise ValueError("at least one candidate is required")
        if client is not None and getattr(client, "available", True):
            try:
                return self._jev(client, state or {}, choices, timeout=timeout)
            except Exception as exc:
                reason = f"JEV {self.operation} failed: {exc}"[:300]
        else:
            if client is not None:
                note = getattr(client, "note_unavailable", None)
                if callable(note):
                    note(self.operation)
            reason = "JEV unavailable; deterministic safety fallback selected."
        return RoutingDecision(choices[0], 0.35, "Selected the first supplied candidate as a safety fallback.", choices, "local_fallback", True, reason, {choices[0]: 0.35})


class ModelRouter(_Router):
    operation = "model_route"


class ToolRouter(_Router):
    operation = "tool_route"


class TestSelector(_Router):
    operation = "test_select"


class MemoryDecision(_Router):
    operation = "memory_decision"
