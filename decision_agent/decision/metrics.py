"""Observation metrics derived from persisted decision evidence.

The Phase 1 metrics exist to evaluate whether model calls are useful, not to
count calls (docs/requirements.md: 观测指标). Everything here is computed from
the JSONL evidence the hooks already persist, so measuring never adds a model
call.

Definitions
-----------
``usable_call_ratio``
    Adopted JEV calls divided by attempted calls with complete adoption
    provenance. Legacy records without ``source``, ``fallback`` and
    ``fallback_reason`` are reported separately and excluded from this
    comparable ratio instead of being treated as failed JEV decisions.
``calls_invalid`` / ``calls_missing_provenance`` / ``calls_duplicate``
    Complete-provenance calls that were not adopted, legacy calls without
    complete provenance, and repeated calls with the same operation plus
    request digest inside one task.
``loop_count`` / ``escalations``
    Stop decisions that answered ``continue`` / ``escalate``.
``duration_s``
    Span between the first and the last evidence record of the task.
``tokens``
    Summed OpenRouter ``usage`` from the JEV call records.
``error_completion_rate``
    Share of tasks whose final Stop decision was ``stop`` while the last
    failing evidence has no later passing validation behind it.

A truncated or otherwise unreadable JSONL line is skipped and reported as
``unreadable_records`` instead of failing the whole aggregation; a hook
process that died mid-write must not blind the metrics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import EvidenceKind, EvidenceRecord
from .sufficiency import is_failed_evidence, is_positive_evidence, validation_correlation


class MetricsError(RuntimeError):
    """Raised when an evidence file cannot be read for aggregation."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _record_dict(record: EvidenceRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, EvidenceRecord):
        return record.to_dict()
    return dict(record)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class TaskMetrics:
    """Metrics for a single task (one evidence store)."""

    task_id: str
    session_id: str
    evidence_path: str
    records: int
    calls_attempted: int
    calls_ok: int
    calls_adopted: int
    calls_eligible: int
    calls_invalid: int
    calls_missing_provenance: int
    calls_unavailable: int
    calls_duplicate: int
    usable_call_ratio: float | None
    loop_count: int
    escalations: int
    completed: bool
    completed_with_error: bool | None
    duration_s: float | None
    unreadable_records: int = 0
    tokens: dict[str, float] = field(default_factory=dict)
    by_operation: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricsReport:
    """Project-level roll-up of the per-task metrics."""

    tasks: list[TaskMetrics] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": [task.to_dict() for task in self.tasks], "totals": dict(self.totals)}


class MetricsAggregator:
    """Derive the observation metrics from JEV call and decision records."""

    _fallback_sources = {"local", "local_fallback", "local_prescreen"}

    # -- aggregation -----------------------------------------------------

    def aggregate(
        self,
        records: Iterable[EvidenceRecord | Mapping[str, Any]],
        *,
        task_id: str = "",
        session_id: str = "",
        evidence_path: str = "",
        unreadable: int = 0,
    ) -> TaskMetrics:
        items = [_record_dict(record) for record in records]
        calls: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for item in items:
            metadata = _mapping(item.get("metadata"))
            if metadata.get("decision") != "jev_call":
                continue
            content = _mapping(item.get("content"))
            calls.append(
                (_mapping(content.get("jev")), _mapping(content.get("context")), metadata)
            )

        attempted = [entry for entry in calls if entry[0].get("called")]
        unavailable = len(calls) - len(attempted)
        eligible = [entry for entry in attempted if self._has_complete_provenance(entry[1])]
        adopted = [entry for entry in eligible if self._is_adopted(*entry[:2])]
        missing_provenance = len(attempted) - len(eligible)
        duplicate_keys: dict[tuple[str, str], int] = {}
        by_operation: dict[str, dict[str, int]] = {}
        for call, _context, _metadata in attempted:
            operation = str(call.get("operation") or "unknown")
            bucket = by_operation.setdefault(
                operation,
                {"calls": 0, "adopted": 0, "invalid": 0, "missing_provenance": 0},
            )
            bucket["calls"] += 1
            digest = str(call.get("request_digest") or "")
            if digest:
                key = (operation, digest)
                duplicate_keys[key] = duplicate_keys.get(key, 0) + 1
        for call, context, _metadata in attempted:
            operation = str(call.get("operation") or "unknown")
            bucket = by_operation.setdefault(
                operation,
                {"calls": 0, "adopted": 0, "invalid": 0, "missing_provenance": 0},
            )
            if not self._has_complete_provenance(context):
                bucket["missing_provenance"] += 1
            elif self._is_adopted(call, context):
                bucket["adopted"] += 1
            else:
                bucket["invalid"] += 1
        duplicates = sum(count - 1 for count in duplicate_keys.values())

        stop_statuses = self._stop_statuses(items)
        completed = bool(stop_statuses) and stop_statuses[-1] == "stop"

        return TaskMetrics(
            task_id=task_id,
            session_id=session_id,
            evidence_path=evidence_path,
            records=len(items),
            calls_attempted=len(attempted),
            calls_ok=sum(1 for call, _context, _metadata in attempted if call.get("ok")),
            calls_adopted=len(adopted),
            calls_eligible=len(eligible),
            calls_invalid=len(eligible) - len(adopted),
            calls_missing_provenance=missing_provenance,
            calls_unavailable=unavailable,
            calls_duplicate=duplicates,
            usable_call_ratio=round(len(adopted) / len(eligible), 4) if eligible else None,
            loop_count=sum(1 for status in stop_statuses if status == "continue"),
            escalations=sum(1 for status in stop_statuses if status == "escalate"),
            completed=completed,
            completed_with_error=self._has_unresolved_failure(items) if completed else None,
            duration_s=self._duration_seconds(items),
            unreadable_records=unreadable,
            tokens=self._tokens(attempted),
            by_operation=by_operation,
        )

    def aggregate_path(self, path: str | Path) -> MetricsReport:
        """Aggregate one evidence file, one store, or a whole evidence root."""

        root = Path(path)
        if root.is_file():
            return self.report([self._task_from_file(root)])
        if not root.exists():
            raise MetricsError(f"Evidence path does not exist: {root}")
        files = sorted(root.glob("**/evidence.jsonl"))
        # A project evidence root may contain a legacy shared file directly at
        # its root plus the current <session>/<turn> stores. That shared file
        # is not a task and must not distort task counts or ratios. Preserve a
        # direct file when it is the only store, so aggregate_path(store_dir)
        # remains useful for one standalone store.
        direct = root / "evidence.jsonl"
        if len(files) > 1 and direct in files:
            files.remove(direct)
        return self.report([self._task_from_file(item) for item in files])

    def report(self, tasks: Iterable[TaskMetrics]) -> MetricsReport:
        collected = list(tasks)
        attempted = sum(task.calls_attempted for task in collected)
        adopted = sum(task.calls_adopted for task in collected)
        eligible = sum(task.calls_eligible for task in collected)
        completed = [task for task in collected if task.completed]
        error_completed = [task for task in completed if task.completed_with_error]
        durations = [task.duration_s for task in collected if task.duration_s is not None]
        tokens: dict[str, float] = {}
        by_operation: dict[str, dict[str, int]] = {}
        for task in collected:
            for key, value in task.tokens.items():
                tokens[key] = tokens.get(key, 0) + value
            for operation, bucket in task.by_operation.items():
                merged = by_operation.setdefault(
                    operation,
                    {"calls": 0, "adopted": 0, "invalid": 0, "missing_provenance": 0},
                )
                for key, value in bucket.items():
                    merged[key] = merged.get(key, 0) + int(value)
        totals = {
            "tasks": len(collected),
            "calls_attempted": attempted,
            "calls_adopted": adopted,
            "calls_eligible": eligible,
            "calls_invalid": sum(task.calls_invalid for task in collected),
            "calls_missing_provenance": sum(task.calls_missing_provenance for task in collected),
            "calls_unavailable": sum(task.calls_unavailable for task in collected),
            "calls_duplicate": sum(task.calls_duplicate for task in collected),
            "unreadable_records": sum(task.unreadable_records for task in collected),
            "usable_call_ratio": round(adopted / eligible, 4) if eligible else None,
            "loop_count": sum(task.loop_count for task in collected),
            "escalations": sum(task.escalations for task in collected),
            "completed_tasks": len(completed),
            "completed_with_error": len(error_completed),
            "error_completion_rate": (
                round(len(error_completed) / len(completed), 4) if completed else None
            ),
            "mean_task_duration_s": (
                round(sum(durations) / len(durations), 3) if durations else None
            ),
            "tokens": tokens,
            "by_operation": by_operation,
        }
        return MetricsReport(tasks=collected, totals=totals)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _has_complete_provenance(context: Mapping[str, Any]) -> bool:
        adopted = context.get("adopted")
        if not isinstance(adopted, Mapping):
            return False
        source = adopted.get("source")
        fallback = adopted.get("fallback")
        fallback_reason = adopted.get("fallback_reason")
        return (
            isinstance(source, str)
            and bool(source.strip())
            and isinstance(fallback, bool)
            and isinstance(fallback_reason, str)
        )

    def _is_adopted(self, call: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        if not call.get("ok") or context.get("error"):
            return False
        if not self._has_complete_provenance(context):
            return False
        adopted = context["adopted"]
        source = adopted["source"]
        fallback = adopted["fallback"]
        if fallback:
            return False
        return source.lower() not in self._fallback_sources

    @staticmethod
    def _stop_statuses(items: list[dict[str, Any]]) -> list[str]:
        statuses: list[str] = []
        for item in items:
            content = _mapping(item.get("content"))
            if content.get("hook_event") != "Stop":
                continue
            status = str(_mapping(content.get("decision")).get("status") or "")
            if status:
                statuses.append(status)
        return statuses

    @staticmethod
    def _has_unresolved_failure(items: list[dict[str, Any]]) -> bool:
        records = [EvidenceRecord.from_dict(item) for item in items]
        failures: list[tuple[int, str]] = []
        passes: list[tuple[int, str]] = []
        for index, (item, record) in enumerate(zip(items, records)):
            content = _mapping(item.get("content"))
            is_failure = (
                str(item.get("severity") or "") == "error"
                or "failure_route" in content
                or is_failed_evidence(record)
            )
            correlation = validation_correlation(record)
            if is_failure:
                failures.append((index, correlation))
            if (
                record.kind_value in {EvidenceKind.TEST_RESULT.value, EvidenceKind.RUNTIME.value}
                and is_positive_evidence(record)
            ):
                passes.append((index, correlation))

        for failure_index, correlation in failures:
            if not correlation:
                return True
            if not any(
                pass_index > failure_index and pass_correlation == correlation
                for pass_index, pass_correlation in passes
            ):
                return True
        return False

    @staticmethod
    def _duration_seconds(items: list[dict[str, Any]]) -> float | None:
        times = [value for value in (_parse_time(item.get("created_at")) for item in items) if value]
        if len(times) < 2:
            return None
        return round((max(times) - min(times)).total_seconds(), 3)

    @staticmethod
    def _tokens(attempted: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for call, _context, _metadata in attempted:
            for key, value in _mapping(call.get("usage")).items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                totals[str(key)] = totals.get(str(key), 0) + float(value)
        return {
            key: (int(value) if float(value).is_integer() else round(value, 10))
            for key, value in totals.items()
        }

    def _task_from_file(self, path: Path) -> TaskMetrics:
        records: list[dict[str, Any]] = []
        unreadable = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        decoded = json.loads(line)
                    except json.JSONDecodeError:
                        unreadable += 1
                        continue
                    if not isinstance(decoded, Mapping):
                        # Valid JSON that is not an object is still unusable.
                        unreadable += 1
                        continue
                    records.append(dict(decoded))
        except OSError as exc:
            raise MetricsError(f"Could not read evidence from {path}: {exc}") from exc
        return self.aggregate(
            records,
            task_id=path.parent.name,
            session_id=path.parent.parent.name,
            evidence_path=str(path),
            unreadable=unreadable,
        )
