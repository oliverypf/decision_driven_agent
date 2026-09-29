"""Deterministic pre-screen for PostToolUse results.

PostToolUse fires for every tool call and has a 15 second budget, so the hook
must not spend a model call on every result. This module classifies the
result locally with a confidence estimate; the hook only submits ambiguous
results (weak failure signals, unclassified failure kinds) to JEV.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


TEST_COMMAND_RE = re.compile(
    r"^\s*(?:(?:python(?:\d(?:\.\d+)*)?|py)\s+-m\s+(?:pytest|unittest)\b|"
    r"pytest\b|unittest\b|npm\s+(?:run\s+)?test\b|yarn\s+test\b|"
    r"pnpm\s+(?:run\s+)?test\b|cargo\s+test\b|go\s+test\b|dotnet\s+test\b|"
    r"mvn\s+test\b|gradle\s+test\b|jest\b|vitest\b|mocha\b|tox\b|"
    r"nox\b|rspec\b|phpunit\b|ctest\b)",
    re.IGNORECASE,
)
BUILD_COMMAND_RE = re.compile(
    r"(?:\bbuild\b|\bcompile\b|\bnpm\s+run\s+build\b|\byarn\s+build\b|"
    r"\bpnpm\s+(?:run\s+)?build\b|\bcargo\s+build\b|\bdotnet\s+build\b|"
    r"\bmvn\s+(?:package|verify)\b|\bgradle\s+build\b)",
    re.IGNORECASE,
)
RUNTIME_COMMAND_RE = re.compile(
    r"(?:\bcurl\b|\binvoke-webrequest\b|\bhealth(?:check)?\b|\bsmoke\b)",
    re.IGNORECASE,
)

EVIDENCE_KINDS = ("test_result", "build_failure", "runtime", "other")

_STRONG_FAILURE_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("traceback", re.compile(r"traceback \(most recent call last\)", re.IGNORECASE)),
    ("assertion", re.compile(r"\bassertion(?:error| failed)\b", re.IGNORECASE)),
    ("assertion", re.compile(r"\bexpected .{0,80}? but got\b", re.IGNORECASE)),
    ("missing_dependency", re.compile(r"\bmodule\s?not\s?found\s?error\b|\bno module named\b", re.IGNORECASE)),
    (
        "command_not_found",
        re.compile(
            r"\bcommand not found\b|\bis not recognized as an internal or external command\b",
            re.IGNORECASE,
        ),
    ),
    ("missing_path", re.compile(r"\bno such file or directory\b", re.IGNORECASE)),
    ("permission", re.compile(r"\bpermission denied\b|\baccess is denied\b", re.IGNORECASE)),
    (
        "network",
        re.compile(
            r"\b(?:connection refused|could not resolve host|network is unreachable|address already in use)\b",
            re.IGNORECASE,
        ),
    ),
    ("npm_error", re.compile(r"\bnpm ERR!\b")),
    ("build_error", re.compile(r"\berror [A-Z]{2,6}\d{3,5}\b")),
    ("exception", re.compile(r"^\w*(?:Error|Exception):\s", re.MULTILINE)),
    ("failure_summary", re.compile(r"^(?:FAILED|ERROR) \S", re.MULTILINE)),
    ("failure_count", re.compile(r"\b\d+ (?:tests? )?failed\b|\b\d+ failing\b", re.IGNORECASE)),
)
_WEAK_FAILURE_RE = re.compile(r"\b(?:failed|failure|error|exception)\b", re.IGNORECASE)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def failure_text(response: Any, response_text: str) -> str:
    """Prefer structured error output over unrelated stdout."""

    if isinstance(response, Mapping):
        parts = [
            _stringify(response.get(key))
            for key in ("stderr", "error", "message")
            if response.get(key) is not None
        ]
        parts = [part for part in parts if part]
        if parts:
            return "\n".join(parts)
    return response_text


@dataclass(frozen=True)
class ToolResultPreScreen:
    kind: str = "other"
    failed: bool | None = None
    confidence: float = 0.0
    signals: list[str] = field(default_factory=list)
    needs_model: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "failed": self.failed,
            "confidence": self.confidence,
            "signals": list(self.signals),
            "needs_model": self.needs_model,
            "reason": self.reason,
        }


def prescreen_tool_result(
    *,
    tool_name: str = "",
    command: str = "",
    response_text: str = "",
    exit_code: int | None = None,
    response: Any = None,
) -> ToolResultPreScreen:
    """Classify one tool result locally and flag cases that need JEV."""

    signals: list[str] = []
    if is_read_only_command_chain(command):
        signals.append("command:read_only_inspection")
        failed = exit_code not in (None, 0)
        return ToolResultPreScreen(
            kind="other",
            failed=failed,
            confidence=0.95,
            signals=signals + ([f"exit_code:{exit_code}"] if failed else []),
            needs_model=False,
            reason="Read-only inspection output is logged but is not validation evidence.",
        )
    if TEST_COMMAND_RE.search(command):
        kind, kind_confidence = "test_result", 0.92
        signals.append("command:test_runner")
    elif BUILD_COMMAND_RE.search(command):
        kind, kind_confidence = "build_failure", 0.9
        signals.append("command:build")
    elif RUNTIME_COMMAND_RE.search(command):
        kind, kind_confidence = "runtime", 0.85
        signals.append("command:runtime")
    elif command.strip():
        kind, kind_confidence = "other", 0.3
        signals.append("command:unclassified")
    else:
        kind, kind_confidence = "other", 0.5

    is_execution = bool(command.strip())
    text = failure_text(response, response_text)
    explicit_error = isinstance(response, Mapping) and any(
        _stringify(response.get(key)).strip() for key in ("error", "stderr")
    )
    if explicit_error:
        signals.append("response:error_field")

    if exit_code not in (None, 0):
        failed, failure_confidence = True, 0.95
        signals.append(f"exit_code:{exit_code}")
    elif exit_code == 0:
        failed, failure_confidence = False, 0.95
        signals.append("exit_code:0")
    else:
        marker = next((label for label, pattern in _STRONG_FAILURE_MARKERS if pattern.search(text)), None)
        if marker:
            failed, failure_confidence = True, 0.8
            signals.append(f"marker:{marker}")
        elif explicit_error and text.strip():
            failed, failure_confidence = True, 0.7
        elif is_execution and _WEAK_FAILURE_RE.search(text):
            failed, failure_confidence = None, 0.45
            signals.append("marker:weak")
        elif is_execution:
            # An executed command without an exit code does not provide
            # positive success evidence merely because no failure marker was
            # found. Keep the result unknown so PostToolUse cannot record it
            # as a passed validation and mislead the Stop judge.
            failed, failure_confidence = None, 0.45
        else:
            failed, failure_confidence = False, 0.6

    needs_model = False
    reason = ""
    # An unclassified command with no error signal is ordinary execution
    # telemetry, not a useful model decision. Keep it local. JEV remains
    # involved for validation/runtime results and error interpretation.
    if failed is None and is_execution and (
        kind != "other" or "marker:weak" in signals or "response:error_field" in signals
    ):
        needs_model = True
        reason = "The failure signal is weak and the exit code is unknown."
    elif failed and kind == "other":
        needs_model = True
        reason = "The tool failed but the evidence kind is unclear."

    return ToolResultPreScreen(
        kind=kind,
        failed=failed,
        confidence=round(min(kind_confidence, failure_confidence), 2),
        signals=signals,
        needs_model=needs_model,
        reason=reason,
    )


# --- PreToolUse pre-screen ---------------------------------------------------

READ_ONLY_TOOL_NAMES = frozenset(
    {
        "read",
        "read_file",
        "cat",
        "type",
        "get-content",
        "get-childitem",
        "get-item",
        "test-path",
        "select-string",
        "measure-object",
        "list",
        "list_directory",
        "ls",
        "dir",
        "glob",
        "grep",
        "rg",
        "search",
        "find",
        "head",
        "tail",
        "codebase_search",
        "get_file_metadata",
    }
)
EDIT_TOOL_NAMES = frozenset(
    {"apply_patch", "edit", "write", "multiedit", "str_replace", "create_file"}
)

READ_ONLY_COMMAND_RE = re.compile(
    r"^\s*(?:rg|grep|cat|type|ls|dir|head|tail|wc|find|fd|"
    r"get-command|get-content|get-childitem|get-item|test-path|select-string|measure-object|"
    r"f(?:ormat)-(?:list|table|wide|custom)\b|"
    r"git\s+(?:status|diff|log|show|branch|remote|rev-parse|blame|ls-files)|"
    r"pip\s+(?:list|show)|where|which|echo|write-output)\b",
    re.IGNORECASE,
)

_DESTRUCTIVE_TOOL_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "remove_root",
        re.compile(
            r"\brm\s+(?:-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\s+(?:/|~|\$HOME|C:\\\\(?:\s|$))",
            re.IGNORECASE,
        ),
    ),
    (
        "disk_operation",
        # Format-* is a PowerShell output formatter. Keep the negative
        # lookahead local to format so a later real format command in the
        # same shell input is still detected.
        re.compile(r"(?:^|[;&|]\s*)(?:format(?!-)|diskpart|mkfs(?:\.\w+)?)\b", re.IGNORECASE),
    ),
    ("raw_disk_write", re.compile(r"\bdd\b[^|;&]*\bof=/dev/", re.IGNORECASE)),
    (
        "system_power",
        re.compile(r"(?:^|[;&|]\s*)(?:shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
    ),
    ("registry_wipe", re.compile(r"\breg\s+delete\s+HKLM", re.IGNORECASE)),
    (
        "system_tree_delete",
        re.compile(
            r"(?:rm|remove-item|del|rmdir)\b[^|;&]*(?:C:\\Windows|C:\\Program Files|%windir%|/etc|/usr|/var|/System)\b",
            re.IGNORECASE,
        ),
    ),
)

_HIGH_RISK_COMMAND_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "force_push",
        re.compile(r"\bgit\s+push\b[^|;&]*(?:--force\b(?!-with-lease)|-f\b)", re.IGNORECASE),
    ),
    ("hard_reset", re.compile(r"\bgit\s+(?:reset\s+--hard|clean\s+-[a-z]*f)", re.IGNORECASE)),
    (
        "remote_code_execution",
        re.compile(
            r"(?:\bcurl\b|\bwget\b|\biwr\b|\binvoke-webrequest\b)[^|;&]*\|\s*(?:ba?sh|sh|python\b|iex|invoke-expression)",
            re.IGNORECASE,
        ),
    ),
    (
        "dynamic_execution",
        re.compile(r"\b(?:iex|invoke-expression)\b|\bpython\s+-c\b|\bnode\s+-e\b", re.IGNORECASE),
    ),
    (
        "permission_change",
        re.compile(r"\b(?:chmod|chown|takeown|icacls|set-acl|set-executionpolicy)\b", re.IGNORECASE),
    ),
    ("registry_write", re.compile(r"\breg\s+(?:add|import)\b", re.IGNORECASE)),
    (
        "credential_access",
        re.compile(r"(?:\.ssh|id_rsa|\.aws/credentials|\.env\b)", re.IGNORECASE),
    ),
)


def extract_tool_command(tool_input: Any) -> str:
    """Extract the command-like field used by Codex tool payloads.

    Different tool adapters expose shell input as ``command``, ``cmd`` or
    ``script``. Keeping this extraction in the pre-screen module prevents the
    PostToolUse and PreToolUse hooks from classifying the same payload
    differently.
    """

    if isinstance(tool_input, Mapping):
        for key in ("command", "cmd", "script"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(tool_input, str):
        return tool_input
    return ""


# Backwards-compatible alias for callers that used the original private name.
_tool_use_command = extract_tool_command


def _has_shell_chain(command: str) -> bool:
    """Return whether a command contains a shell chain beyond a safe pipeline."""

    # A pipeline can still be classified locally when every stage is
    # read-only. Semicolons, background operators and conditional chains are
    # always left for JEV because later stages may change state.
    return bool(re.search(r"(?:&&|\|\||[;&\r\n])", command))


def _split_unquoted_shell_stages(command: str) -> list[str]:
    """Split simple shell operators without splitting quoted search patterns."""

    stages: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            current.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            current.append(char)
            escaped = True
            index += 1
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char in "|;&\r\n":
            if index + 1 < len(command) and command[index:index + 2] in {"&&", "||"}:
                index += 2
            else:
                index += 1
            stages.append("".join(current).strip())
            current = []
            continue
        current.append(char)
        index += 1
    if quote:
        return []
    stages.append("".join(current).strip())
    return stages


def _is_read_only_command(command: str) -> bool:
    """Return whether a command or every stage of its pipeline is read-only."""

    parts = _split_unquoted_shell_stages(command)
    if not parts or any(not part for part in parts):
        return False
    return all(READ_ONLY_COMMAND_RE.search(part) for part in parts)


def is_read_only_command_chain(command: str) -> bool:
    """Return whether every simple stage in a shell chain is read-only.

    Inspection output may quote historical test results. Never ask JEV to
    classify that output as a current validation run.
    """

    text = command.strip()
    if not text or any(marker in text for marker in ("$(", "`", ">", "<")):
        return False
    parts = _split_unquoted_shell_stages(text)
    return bool(parts) and all(part.strip() and _is_read_only_command(part.strip()) for part in parts)


@dataclass(frozen=True)
class ToolUsePreScreen:
    risk: str = "medium"
    recommendation: str = "proceed"
    appropriate: bool | None = None
    confidence: float = 0.0
    signals: list[str] = field(default_factory=list)
    needs_model: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "recommendation": self.recommendation,
            "appropriate": self.appropriate,
            "confidence": self.confidence,
            "signals": list(self.signals),
            "needs_model": self.needs_model,
            "reason": self.reason,
        }


def prescreen_tool_use(*, tool_name: str = "", tool_input: Any = None) -> ToolUsePreScreen:
    """Classify a planned tool call and flag the ones that need JEV judgment.

    Read-only tools, validation commands, dedicated file-edit tools and
    unambiguously destructive commands are decided locally. Everything else is
    submitted to JEV for the risk and tool-appropriateness decision.
    """

    name = (tool_name or "").strip().lower()
    command = extract_tool_command(tool_input)
    signals: list[str] = []

    if command:
        for label, pattern in _DESTRUCTIVE_TOOL_MARKERS:
            if pattern.search(command):
                signals.append(f"destructive:{label}")
                return ToolUsePreScreen(
                    risk="high",
                    recommendation="block",
                    appropriate=False,
                    confidence=0.95,
                    signals=signals,
                    needs_model=False,
                    reason=(
                        "The command matches a destructive pattern that the local safety "
                        f"pre-screen refuses ({label}); no model call is needed to block it."
                    ),
                )

    if command:
        for label, pattern in _HIGH_RISK_COMMAND_MARKERS:
            if pattern.search(command):
                signals.append(f"high_risk:{label}")
                return ToolUsePreScreen(
                    risk="high",
                    recommendation="confirm",
                    appropriate=None,
                    confidence=0.55,
                    signals=signals,
                    needs_model=True,
                    reason=(
                        f"The command matches the high-risk signal {label}; "
                        "JEV must judge the risk and the tool choice."
                    ),
                )

    if name in READ_ONLY_TOOL_NAMES:
        signals.append("tool:read_only")
        return ToolUsePreScreen(
            risk="low",
            recommendation="proceed",
            appropriate=True,
            confidence=0.85,
            signals=signals,
            needs_model=False,
            reason="Read-only tool; the local pre-screen is sufficiently confident.",
        )
    if name in EDIT_TOOL_NAMES:
        signals.append("tool:file_edit")
        return ToolUsePreScreen(
            risk="low",
            recommendation="proceed",
            appropriate=True,
            confidence=0.8,
            signals=signals,
            needs_model=False,
            reason=(
                "File edits are recorded by PostToolUse and validated by the Stop gate; "
                "no separate pre-tool model call is needed."
            ),
        )
    if command and not _has_shell_chain(command) and _is_read_only_command(command):
        signals.append("command:read_only")
        return ToolUsePreScreen(
            risk="low",
            recommendation="proceed",
            appropriate=True,
            confidence=0.85,
            signals=signals,
            needs_model=False,
            reason="Read-only command; the local pre-screen is sufficiently confident.",
        )
    if command and not _has_shell_chain(command) and "|" not in command and (
        TEST_COMMAND_RE.search(command) or BUILD_COMMAND_RE.search(command)
    ):
        signals.append("command:validation")
        return ToolUsePreScreen(
            risk="low",
            recommendation="proceed",
            appropriate=True,
            confidence=0.8,
            signals=signals,
            needs_model=False,
            reason="Validation command; the local pre-screen is sufficiently confident.",
        )


    signals.append("command:unclassified" if command else "tool:unclassified")
    return ToolUsePreScreen(
        risk="medium",
        recommendation="confirm",
        appropriate=None,
        confidence=0.3,
        signals=signals,
        needs_model=True,
        reason="The local pre-screen cannot judge this tool; JEV must decide.",
    )
