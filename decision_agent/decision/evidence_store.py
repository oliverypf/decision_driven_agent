"""A small JSONL-backed evidence store suitable for hook processes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..models import EvidenceKind, EvidenceRecord, RetentionPolicy


class EvidenceStoreError(RuntimeError):
    """Raised when persisted evidence cannot be read or written."""


class EvidenceStore:
    """Append-only evidence storage with enforced retention decisions.

    JSONL keeps writes cheap for PostToolUse-style hooks and makes the store
    inspectable without a database dependency. Mandatory evidence (user
    requests, requirements, acceptance criteria, diffs, test results, build
    failures, security risks, decisions and final acceptance) is always kept.
    Oversized logs and runtime records are compressed at write time. An
    explicit DROP is honored only for non-mandatory records, and the returned
    record still carries ``retention=drop`` plus ``metadata.dropped`` so a
    caller can tell the record was not persisted.
    """

    _always_keep = {
        EvidenceKind.REQUIREMENT.value,
        EvidenceKind.ACCEPTANCE_CRITERIA.value,
        EvidenceKind.GIT_DIFF.value,
        EvidenceKind.TEST_RESULT.value,
        EvidenceKind.SECURITY_RISK.value,
        EvidenceKind.BUILD_FAILURE.value,
        EvidenceKind.USER_REQUEST.value,
        EvidenceKind.DECISION.value,
        EvidenceKind.FINAL_ACCEPTANCE.value,
    }
    _compressible = {EvidenceKind.LOG.value, EvidenceKind.RUNTIME.value}
    _compress_threshold = 4000
    _compressed_limit = 2000

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / "evidence.jsonl"
        self.unreadable_records = 0

    def append(
        self,
        kind: EvidenceKind | str,
        content: Any,
        *,
        severity: str = "info",
        retention: RetentionPolicy | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        kind_value = kind.value if isinstance(kind, EvidenceKind) else str(kind)
        chosen_retention = self._decide_retention(kind_value, content, severity, retention)
        record_metadata = dict(metadata or {})
        stored_content = content
        if chosen_retention is RetentionPolicy.COMPRESS:
            record_metadata["compressed"] = True
            record_metadata["original_chars"] = self._content_size(content)
            stored_content = self._compress_content(content)
        elif chosen_retention is RetentionPolicy.DROP:
            record_metadata["dropped"] = True
        record = EvidenceRecord(
            kind=kind_value,
            content=stored_content,
            severity=severity,
            retention=chosen_retention,
            metadata=record_metadata,
        )
        if chosen_retention is RetentionPolicy.DROP:
            return record
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                # Windows hook stdin may contain unmatched surrogates. Escape
                # non-ASCII text on disk so evidence writing never breaks the
                # lifecycle hook or suppresses the decision call.
                handle.write(json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True))
                handle.write("\n")
        except OSError as exc:
            raise EvidenceStoreError(f"Could not append evidence to {self.path}: {exc}") from exc
        return record

    @staticmethod
    def _content_size(content: Any) -> int:
        if isinstance(content, str):
            return len(content)
        try:
            return len(json.dumps(content, ensure_ascii=True, default=str, sort_keys=True))
        except (TypeError, ValueError):
            return len(str(content))

    def _compress_content(self, content: Any) -> Any:
        if isinstance(content, str):
            encoded = content
        else:
            try:
                encoded = json.dumps(content, ensure_ascii=False, default=str, sort_keys=True)
            except (TypeError, ValueError):
                encoded = str(content)
        if len(encoded) <= self._compressed_limit:
            return encoded
        removed = len(encoded) - self._compressed_limit
        return encoded[: self._compressed_limit] + f"...[compressed {removed} chars]"

    def _decide_retention(
        self,
        kind_value: str,
        content: Any,
        severity: str,
        requested: RetentionPolicy | str | None,
    ) -> RetentionPolicy:
        mandatory = kind_value in self._always_keep or severity in {"error", "warning"}
        if requested is not None:
            requested_value = (
                requested.value if isinstance(requested, RetentionPolicy) else str(requested)
            )
        elif kind_value in self._compressible and self._content_size(content) > self._compress_threshold:
            requested_value = RetentionPolicy.COMPRESS.value
        else:
            requested_value = RetentionPolicy.KEEP.value
        if requested_value not in {item.value for item in RetentionPolicy}:
            raise EvidenceStoreError(f"Unknown retention policy: {requested_value}")
        if mandatory or kind_value in self._always_keep:
            return RetentionPolicy.KEEP
        return RetentionPolicy(requested_value)

    def read_all(self) -> list[EvidenceRecord]:
        self.unreadable_records = 0
        if not self.path.exists():
            return []
        records: list[EvidenceRecord] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        decoded = json.loads(line)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        decoded = None
                    if not isinstance(decoded, Mapping):
                        # A truncated hook write must not make the rest of the
                        # append-only evidence history inaccessible. The same
                        # applies to a valid JSON line that is not an object.
                        self.unreadable_records += 1
                        continue
                    try:
                        records.append(EvidenceRecord.from_dict(decoded))
                    except (TypeError, ValueError, AttributeError):
                        self.unreadable_records += 1
                        continue
        except OSError as exc:
            raise EvidenceStoreError(f"Could not read evidence from {self.path}: {exc}") from exc
        return records

    def iter_kind(self, kind: EvidenceKind | str) -> Iterable[EvidenceRecord]:
        kind_value = kind.value if isinstance(kind, EvidenceKind) else str(kind)
        return (record for record in self.read_all() if record.kind_value == kind_value)

    def clear(self) -> None:
        """Remove all persisted evidence for the current task."""

        try:
            if self.path.exists():
                self.path.unlink()
        except OSError as exc:
            raise EvidenceStoreError(f"Could not clear evidence at {self.path}: {exc}") from exc
