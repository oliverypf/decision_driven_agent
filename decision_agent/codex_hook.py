"""Codex lifecycle hook adapter for the Phase 1 decision layer.

The adapter deliberately keeps the Codex-facing protocol small: read one JSON
object from stdin and write one JSON object to stdout. Evidence is isolated by
Codex session and turn so a later request in the same conversation cannot
inherit an earlier task's validation records.
"""

from __future__ import annotations

import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from .controller import DecisionController
from .decision.failure_router import FailureInput
from .decision.prescreen import extract_tool_command, prescreen_tool_result
from .models import FAILURE_TYPES, EvidenceKind, FailureRoute


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _problem_id(requirement: str) -> str:
    digest = hashlib.sha256(requirement.strip().encode("utf-8", "replace")).hexdigest()[:16]
    return f"problem-{digest}"

# A prompt such as "要", "继续" or "ok" confirms the current goal without
# restating it. The raw reply is still recorded, but the effective requirement
# for the turn carries the previous informative requirement forward so the JEV
# Stop judge judges the actual task instead of a bare acknowledgement.
_LOW_INFORMATION_STEMS = {
    "要", "好", "是", "对", "可以", "行", "继续", "嗯",
    "ok", "okay", "k", "yes", "yeah", "yep", "sure", "go", "proceed", "continue",
}
_LOW_INFORMATION_PHRASES = {
    "do it", "go ahead", "keep going", "go on",
    # `_normalized_reply` strips trailing particles such as "了" and "吧",
    # so variants like "现在可以了吧" normalize to this shorter form.
    "现在可以",
    "现在可以了吗", "现在可以了吧", "现在可以了", "可以了吗", "可以了吧",
    "现在好了么", "现在好了吗", "现在修好了吗", "现在完成了吗",
    "stop可以了吗", "stop现在可以了吗", "hook现在可以了吗",
    "再定位一下，可以stop了吗", "再定位一下可以stop了吗", "再定位一下stop可以了吗",
    "再定位一下现在可以stop了吗", "再定位一下hook可以stop了吗",
    "怎样才能通过呢", "怎么才能通过呢", "如何才能通过呢",
    "怎样才算通过", "怎么才算通过", "如何才算通过",
}
_REPLY_TRIM = " \t\r\n。.!！?？~～,，、:：;；\"'“”‘’()（）"
_REPLY_FILLER_PREFIXES = ("嗯", "呃", "那", "就")
_REPLY_FILLER_SUFFIXES = ("吧", "呀", "啊", "啦", "哦", "嘛", "的", "了")
_CARRIED_MARKER_PREFIX = "[The user replied only"

# Per-hook JEV budgets. The Stop hook runs two serial decisions inside a 45s
# hook budget; PostToolUse and PreToolUse run at most one decision inside a
# 15s budget. Budgets stay below the hook timeout so a slow model response
# degrades to a marked local fallback instead of the hook being killed.
_HOOK_JEV_TIMEOUTS = {
    "UserPromptSubmit": 4.0,
    "PreToolUse": 8.0,
    "PostToolUse": 10.0,
    "Stop": 18.0,
}


def _safe_id(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    safe = _SAFE_ID_RE.sub("_", raw).strip("._")
    return safe[:128] or fallback


def _jsonable(value: Any, *, limit: int = 24_000) -> Any:
    """Bound persisted tool output without failing on non-JSON values."""

    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        encoded = str(value)
    if len(encoded) <= limit:
        return value
    return encoded[:limit] + "...[truncated]"


def _text(value: Any, *, limit: int = 24_000) -> str:
    bounded = _jsonable(value, limit=limit)
    if isinstance(bounded, str):
        return bounded
    try:
        return json.dumps(bounded, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(bounded)


def _extract_exit_code(response: Any) -> int | None:
    if isinstance(response, Mapping):
        for key in ("exit_code", "exitCode", "returncode", "return_code"):
            value = response.get(key)
            if isinstance(value, bool):
                continue
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                pass
    text = _text(response, limit=8_000)
    match = re.search(r"(?:exited|exit(?:ed)? with code|return code)\D+(-?\d+)", text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        parts: list[str] = []
        for key in ("stdout", "stderr", "output", "message", "error", "content"):
            if key in response and response[key] is not None:
                parts.append(_text(response[key]))
        if parts:
            return "\n".join(parts)
    return _text(response)


_DIFF_OUTPUT_RE = re.compile(r"(?m)^(?:diff --git\s|---\s+\S|\+\+\+\s+\S|@@\s)")


def _contains_diff_output(response_text: str) -> bool:
    """Return whether tool output contains an actual unified diff payload."""

    return bool(_DIFF_OUTPUT_RE.search(response_text))


def _project_root(payload: Mapping[str, Any]) -> Path:
    """Resolve the repository root without relying on a Git checkout."""

    cwd_value = payload.get("cwd")
    if cwd_value:
        cwd = Path(str(cwd_value)).resolve()
        for candidate in (cwd, *cwd.parents):
            if (candidate / "decision_agent").is_dir() or (candidate / ".codex" / "decision_agent").is_dir():
                return candidate
    module_path = Path(__file__).resolve()
    if module_path.parent.parent.name == ".codex":
        return module_path.parents[2]
    for candidate in (module_path.parent, *module_path.parents):
        if (candidate / "decision_agent").is_dir() and (candidate / "decision_agent" / "codex_hook.py").exists():
            return candidate
        if (candidate / ".codex" / "decision_agent").is_dir():
            return candidate
    return module_path.parents[1]


def _state_path(root: Path, session_id: str) -> Path:
    return root / ".decision" / "sessions" / f"{_safe_id(session_id, 'default')}.json"


def _default_state(root: Path, session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "task_id": "uninitialized",
        "requirement": "",
        "problem_id": "",
        "problem_statement": "",
        "prompt": "",
        "requirement_source": "prompt",
        "carried_from": "",
        "acceptance_criteria": [],
        "domain": "unknown",
        "evidence_dir": str(root / ".decision" / "evidence" / _safe_id(session_id, "default") / "uninitialized"),
        "stop_attempts": 0,
    }


def _load_state(root: Path, session_id: str) -> dict[str, Any]:
    path = _state_path(root, session_id)
    if not path.exists():
        return _default_state(root, session_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return _default_state(root, session_id)
    return dict(value) if isinstance(value, Mapping) else _default_state(root, session_id)


def _save_state(root: Path, state: Mapping[str, Any]) -> None:
    session_id = str(state.get("session_id") or "default")
    path = _state_path(root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # Hook payloads can contain unmatched UTF-16 surrogates on Windows.
    # Escape them on disk so state recovery cannot crash the Stop gate.
    temporary.write_text(json.dumps(dict(state), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _normalized_reply(text: Any) -> str:
    value = str(text or "").strip(_REPLY_TRIM).lower()
    changed = True
    while changed and value:
        changed = False
        for prefix in _REPLY_FILLER_PREFIXES:
            if value.startswith(prefix) and len(value) > len(prefix):
                value = value[len(prefix):]
                changed = True
        for suffix in _REPLY_FILLER_SUFFIXES:
            if value.endswith(suffix) and len(value) > len(suffix):
                value = value[: -len(suffix)]
                changed = True
    return value.strip(_REPLY_TRIM)


def _is_low_information(text: Any) -> bool:
    """True when a prompt depends on the current goal instead of restating it."""

    normalized = _normalized_reply(text)
    return (
        not normalized
        or normalized in _LOW_INFORMATION_STEMS
        or normalized in _LOW_INFORMATION_PHRASES
    )


def _requirement_from_records(path: Path) -> str:
    """Return the newest informative requirement recorded in one evidence file."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-200:]):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(record, Mapping) or str(record.get("kind")) not in {"requirement", "user_request"}:
            continue
        content = record.get("content")
        if isinstance(content, str) and content.strip() and not _is_low_information(content):
            return content.strip()
    return ""


def _goal_from_packet(path: Path) -> str:
    """Return the goal recorded in one decision-space packet, if any."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    goal = data.get("goal") if isinstance(data, Mapping) else None
    return goal.strip() if isinstance(goal, str) and goal.strip() else ""


def _recent_requirement(root: Path, session_id: str, previous: Mapping[str, Any]) -> tuple[str, str]:
    """Find the most recent informative requirement for this Codex session."""

    carried = str(previous.get("requirement") or "").strip()
    if carried and not _is_low_information(carried):
        return carried, "previous_turn"
    session_dir = root / ".decision" / "evidence" / _safe_id(session_id, "default")
    candidates: list[tuple[str, str, str]] = []
    for pattern, reader, source in (
        ("*/decision-space.json", _goal_from_packet, "session_packet"),
        ("*/evidence.jsonl", _requirement_from_records, "session_evidence"),
    ):
        try:
            paths = sorted(
                session_dir.glob(pattern),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            paths = []
        for path in paths[:5]:
            text = reader(path)
            if text and not _is_low_information(text):
                candidates.append((path.parent.name, text, source))
                break
    if not candidates:
        return "", ""
    # Codex turn ids are time-ordered, so the largest id is the newest context.
    turn_key, text, source = max(candidates, key=lambda item: item[0])
    return text, f"{source}:{turn_key}"


def _start_task(
    payload: Mapping[str, Any],
    root: Path,
    *,
    jev_timeout: float | None = None,
) -> tuple[DecisionController, dict[str, Any]]:
    session_id = str(payload.get("session_id") or "default")
    turn_id = str(payload.get("turn_id") or "prompt")
    session_key = _safe_id(session_id, "default")
    task_key = _safe_id(turn_id, "prompt")
    evidence_dir = root / ".decision" / "evidence" / session_key / task_key
    previous = _load_state(root, session_id)
    prompt = str(payload.get("prompt") or payload.get("requirement") or "").strip()
    requirement = prompt
    carried_from = ""
    if _is_low_information(prompt):
        carried, carried_from = _recent_requirement(root, session_id, previous)
        if carried:
            base = "\n".join(
                line for line in carried.splitlines() if not line.startswith(_CARRIED_MARKER_PREFIX)
            ).strip()
            requirement = (
                f"{(base or carried)[:4000]}\n"
                f"[The user replied only {prompt!r} on this turn and did not restate a goal.]"
            )
    acceptance_criteria = [str(item) for item in (payload.get("acceptance_criteria") or [])]
    if not acceptance_criteria and carried_from:
        acceptance_criteria = [str(item) for item in (previous.get("acceptance_criteria") or [])]
    domain = payload.get("domain") or payload.get("task_domain")
    if not domain and carried_from:
        previous_domain = str(previous.get("domain") or "").strip()
        if previous_domain and previous_domain.lower() != "unknown":
            domain = previous_domain
    state = {
        "session_id": session_id,
        "task_id": turn_id,
        "requirement": requirement,
        "problem_id": _problem_id(requirement),
        "problem_statement": requirement,
        "prompt": prompt,
        "requirement_source": "carried" if carried_from else "prompt",
        "carried_from": carried_from,
        "acceptance_criteria": acceptance_criteria,
        "domain": domain or "unknown",
        "evidence_dir": str(evidence_dir),
        "stop_attempts": 0,
        "model": payload.get("model"),
    }
    controller = DecisionController.from_directory(evidence_dir, jev_timeout=jev_timeout)
    controller.set_trace_context(session_id=session_id, turn_id=turn_id,
                                 problem_id=state["problem_id"],
                                 problem_statement=state["problem_statement"])
    provenance = {
        "session_id": session_id,
        "turn_id": turn_id,
        "source": "UserPromptSubmit",
        "requirement_source": state["requirement_source"],
        "carried_from": carried_from or None,
    }
    controller.record_evidence(
        EvidenceKind.USER_REQUEST,
        prompt,
        metadata=provenance,
    )
    controller.record_evidence(
        EvidenceKind.REQUIREMENT,
        requirement,
        metadata=provenance,
    )
    if state["acceptance_criteria"]:
        controller.record_evidence(
            EvidenceKind.ACCEPTANCE_CRITERIA,
            state["acceptance_criteria"],
            metadata={"session_id": session_id, "turn_id": turn_id, "source": "UserPromptSubmit"},
        )
    _save_state(root, state)
    return controller, state


def _current_task(
    payload: Mapping[str, Any],
    root: Path,
    *,
    jev_timeout: float | None = None,
) -> tuple[DecisionController, dict[str, Any]]:
    session_id = str(payload.get("session_id") or "default")
    state = _load_state(root, session_id)
    # Stop/PostToolUse can arrive without a preceding prompt hook (for
    # example after a hook process restart). Do not send an empty synthetic
    # requirement to JEV when this session already has a real requirement in
    # an earlier turn's evidence or decision-space packet.
    if not str(state.get("requirement") or "").strip():
        recovered, recovered_from = _recent_requirement(root, session_id, state)
        if recovered:
            state["requirement"] = recovered
            state["requirement_source"] = recovered_from
            state["carried_from"] = recovered_from
            if str(state.get("task_id") or "") == "uninitialized":
                session_dir = root / ".decision" / "evidence" / _safe_id(session_id, "default")
                candidates = sorted(
                    session_dir.glob("*/evidence.jsonl"),
                    key=lambda item: item.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    state["task_id"] = candidates[0].parent.name
                    state["evidence_dir"] = str(candidates[0].parent)
            # Persist recovery before any later JEV/evidence operation. If
            # Stop is interrupted or JEV is unavailable, the next hook must
            # still see the recovered requirement instead of recreating the
            # uninitialized state.
            _save_state(root, state)
    if payload.get("requirement") and not state.get("requirement"):
        state["requirement"] = str(payload["requirement"])
    if payload.get("acceptance_criteria") and not state.get("acceptance_criteria"):
        state["acceptance_criteria"] = [str(item) for item in payload["acceptance_criteria"]]
    if (payload.get("domain") or payload.get("task_domain")) and str(state.get("domain") or "unknown").lower() == "unknown":
        state["domain"] = payload.get("domain") or payload.get("task_domain")
    evidence_dir = Path(str(state.get("evidence_dir") or root / ".decision" / "evidence" / _safe_id(session_id, "default") / "uninitialized"))
    controller = DecisionController.from_directory(evidence_dir, jev_timeout=jev_timeout)
    controller.set_trace_context(
        session_id=session_id,
        turn_id=str(payload.get("turn_id") or state.get("task_id") or "uninitialized"),
        problem_id=str(state.get("problem_id") or _problem_id(str(state.get("requirement") or ""))),
        problem_statement=str(state.get("problem_statement") or state.get("requirement") or ""),
    )
    return controller, state


def _handle_user_prompt(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    from .direction import prompt_context
    timeout = _HOOK_JEV_TIMEOUTS["UserPromptSubmit"]
    controller, state = _start_task(payload, root, jev_timeout=timeout)
    return prompt_context(controller, state, timeout=timeout)


def _handle_post_tool_use(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    controller, state = _current_task(payload, root, jev_timeout=_HOOK_JEV_TIMEOUTS["PostToolUse"])
    tool_name = str(payload.get("tool_name") or "unknown")
    tool_input = payload.get("tool_input")
    tool_response = payload.get("tool_response")
    metadata = {
        "session_id": state.get("session_id"),
        "turn_id": payload.get("turn_id") or state.get("task_id"),
        "tool_name": tool_name,
        "tool_use_id": payload.get("tool_use_id"),
        "source": "PostToolUse",
    }
    controller.record_evidence(
        EvidenceKind.LOG,
        {"tool_name": tool_name, "tool_input": _jsonable(tool_input), "tool_response": _jsonable(tool_response)},
        metadata=metadata,
    )

    command = extract_tool_command(tool_input)
    response_text = _response_text(tool_response)
    exit_code = _extract_exit_code(tool_response)
    # Local pre-screen: classify from the command and structured error output,
    # not arbitrary stdout. Source files often contain words such as
    # test_failed or build_failure. Only low-confidence results are escalated
    # to JEV, so a routine tool call never spends a model invocation.
    prescreen = prescreen_tool_result(
        tool_name=tool_name,
        command=command,
        response_text=response_text,
        exit_code=exit_code,
        response=tool_response,
    )
    kind = prescreen.kind
    failed = prescreen.failed
    classification: dict[str, Any] = {
        "kind": kind,
        "failed": failed,
        "confidence": prescreen.confidence,
        "signals": list(prescreen.signals),
        "needs_model": prescreen.needs_model,
        "reason": prescreen.reason,
        "source": "local_prescreen",
        "fallback": False,
    }
    escalation: dict[str, Any] = {}
    if prescreen.needs_model:
        escalation = controller.classify_tool_evidence(
            tool_name=tool_name,
            command=command,
            response_text=response_text,
            exit_code=exit_code,
            prescreen=prescreen.to_dict(),
            use_model=True,
            timeout=_HOOK_JEV_TIMEOUTS["PostToolUse"],
        )
        kind = str(escalation.get("kind") or kind)
        escalated_failed = escalation.get("failed")
        failed = failed if escalated_failed is None else bool(escalated_failed)
        classification.update(
            {
                "kind": kind,
                "failed": failed,
                "confidence": escalation.get("failed_confidence", classification["confidence"]),
                "kind_confidence": escalation.get("kind_confidence"),
                "source": str(escalation.get("source") or classification["source"]),
                "fallback": bool(escalation.get("fallback")),
                "fallback_reason": str(escalation.get("fallback_reason") or ""),
            }
        )

    is_patch_command = bool(re.search(r"\bapply_patch\b", command, re.IGNORECASE))
    is_diff_command = bool(re.search(r"(?:\bdiff\s+--git\b|\bgit\s+diff\b)", command, re.IGNORECASE))
    edit_tools = {"apply_patch", "Edit", "Write", "node_repl", "mcp__node_repl__js"}
    is_dedicated_edit = tool_name in edit_tools or tool_name.lower() in {name.lower() for name in edit_tools}
    if (is_dedicated_edit or is_patch_command) and failed is not True:
        controller.record_evidence(
            EvidenceKind.GIT_DIFF,
            _jsonable(tool_input),
            metadata={**metadata, "evidence_source": "edit_tool_input"},
        )
    elif is_diff_command and failed is not True and _contains_diff_output(response_text):
        controller.record_evidence(
            EvidenceKind.GIT_DIFF,
            _jsonable(tool_response),
            metadata={**metadata, "evidence_source": "tool_response"},
        )

    if kind == "test_result":
        # ``failed is None`` means the local pre-screen could not resolve the
        # result without an exit code. Recording that as "passed" would hand
        # the Stop judge false validation evidence, so it stays explicitly
        # unknown.
        test_status = "failed" if failed else ("passed" if failed is False else "unknown")
        controller.record_evidence(
            EvidenceKind.TEST_RESULT,
            {"command": _jsonable(command), "output": _jsonable(tool_response), "exit_code": exit_code},
            severity="error" if failed else "info",
            metadata={
                **metadata,
                "status": test_status,
                "classified_by": classification["source"],
            },
        )
    elif kind == "build_failure" and failed is False:
        # A build that actually succeeded is validation evidence; only the
        # failed case is stored as BUILD_FAILURE below.
        controller.record_evidence(
            EvidenceKind.TEST_RESULT,
            {
                "command": _jsonable(command),
                "output": _jsonable(tool_response),
                "exit_code": exit_code,
                "validation": "build",
            },
            metadata={**metadata, "status": "passed", "classified_by": classification["source"]},
        )
    elif kind == "runtime" and failed is False:
        controller.record_evidence(
            EvidenceKind.RUNTIME,
            {"command": _jsonable(command), "output": _jsonable(tool_response), "exit_code": exit_code},
            metadata={**metadata, "status": "passed", "classified_by": classification["source"]},
        )

    if prescreen.needs_model or failed:
        controller.record_evidence(
            EvidenceKind.DECISION,
            {"tool_evidence": classification},
            metadata={**metadata, "decision": "tool_evidence_classification"},
        )

    if failed:
        route = None
        if (
            str(escalation.get("source")) == "JEV"
            and escalation.get("failure_type") in FAILURE_TYPES
        ):
            failure_type = str(escalation["failure_type"])
            route = FailureRoute(
                type=failure_type,
                confidence=float(escalation.get("failure_type_confidence") or 0.0),
                reason="JEV classified the failure from the tool-evidence submission.",
                signals=["jev:tool_evidence"],
                reasoning_required=bool(
                    escalation.get(
                        "failure_type_reasoning_required",
                        failure_type == "UNKNOWN",
                    )
                ),
                source="JEV",
                fallback=False,
                scores={
                    str(key): float(value)
                    for key, value in dict(escalation.get("failure_type_scores") or {}).items()
                },
            )
        if route is None:
            route = controller.route_failure(
                FailureInput(
                    message=response_text,
                    environment=response_text,
                    recent_diff=_text(tool_input),
                    exit_code=exit_code,
                    test_name=command if kind == "test_result" else "",
                ),
                timeout=_HOOK_JEV_TIMEOUTS["PostToolUse"],
            )
        controller.record_evidence(
            EvidenceKind.DECISION,
            {"failure_route": route.to_dict()},
            severity="error",
            metadata={**metadata, "decision": "failure_route", "command": command},
        )
        if kind == "build_failure":
            controller.record_evidence(
                EvidenceKind.BUILD_FAILURE,
                {"command": _jsonable(command), "output": _jsonable(tool_response), "route": route.to_dict()},
                severity="error",
                metadata=metadata,
            )
    return {}


def _handle_pre_tool_use(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Judge a planned tool call with JEV before Codex runs it.

    Read-only calls, validation commands, dedicated file-edit tools and
    unambiguously destructive commands are decided by the local pre-screen.
    Everything else is submitted to JEV. A destructive local verdict is the
    only path that denies the tool call; JEV verdicts are advisory.
    """

    controller, state = _current_task(payload, root, jev_timeout=_HOOK_JEV_TIMEOUTS["PreToolUse"])
    tool_name = str(payload.get("tool_name") or "unknown")
    tool_input = payload.get("tool_input")
    decision = controller.judge_tool_use(
        tool_name=tool_name,
        tool_input=tool_input,
        requirement=str(state.get("requirement") or ""),
        domain=str(state.get("domain") or "unknown"),
        timeout=_HOOK_JEV_TIMEOUTS["PreToolUse"],
    )
    metadata = {
        "session_id": state.get("session_id"),
        "turn_id": payload.get("turn_id") or state.get("task_id"),
        "tool_name": tool_name,
        "tool_use_id": payload.get("tool_use_id"),
        "source": "PreToolUse",
        "decision": "tool_risk",
    }
    controller.record_evidence(
        EvidenceKind.DECISION,
        {"hook_event": "PreToolUse", "tool_decision": decision.to_dict()},
        severity="warning" if decision.risk == "high" else "info",
        metadata=metadata,
    )
    if decision.risk == "high":
        controller.record_evidence(
            EvidenceKind.SECURITY_RISK,
            {"tool_name": tool_name, "tool_decision": decision.to_dict()},
            severity="warning",
            metadata=metadata,
        )

    summary = (
        f"[{decision.source}] risk={decision.risk} "
        f"recommendation={decision.recommendation}: {decision.reason}"
    )
    local_destructive_block = (
        decision.source == "local_prescreen"
        and decision.recommendation == "block"
        and any(signal.startswith("destructive:") for signal in decision.signals)
    )
    if local_destructive_block:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": summary[:1000],
            }
        }
    if decision.recommendation != "proceed" or decision.risk == "high":
        return {
            "systemMessage": f"PreToolUse decision: {summary}",
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": f"PreToolUse decision: {summary}",
            },
        }
    # PreToolUse has no supported non-blocking `continue` field. An empty
    # JSON object is a successful advisory result; Codex proceeds normally.
    return {}


def _handle_stop(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    controller, state = _current_task(payload, root, jev_timeout=_HOOK_JEV_TIMEOUTS["Stop"])
    requirement = str(state.get("requirement") or payload.get("requirement") or "").strip()
    requirement_source = str(state.get("requirement_source") or "prompt")
    raw_dir = str(state.get("evidence_dir") or "").strip()
    evidence_dir = Path(raw_dir) if raw_dir else None
    if _is_low_information(requirement) and evidence_dir and evidence_dir.is_dir():
        # The prompt was only an acknowledgement. Prefer the packet goal the
        # reasoning model wrote for this turn, then any informative requirement
        # recorded in this turn's evidence, and mark which text was submitted.
        packet_goal = _goal_from_packet(evidence_dir / "decision-space.json")
        if packet_goal and not _is_low_information(packet_goal):
            requirement, requirement_source = packet_goal, "turn_packet_goal"
        else:
            recorded = _requirement_from_records(evidence_dir / "evidence.jsonl")
            if recorded:
                requirement, requirement_source = recorded, "turn_evidence"
    criteria = [str(item) for item in (state.get("acceptance_criteria") or payload.get("acceptance_criteria") or [])]
    domain = state.get("domain") or payload.get("domain") or payload.get("task_domain") or "unknown"
    if not requirement:
        requirement = "No user requirement was recorded for this Codex session."
    iteration = int(state.get("stop_attempts") or 0)

    completion_text = (
        payload.get("last_assistant_message")
        or payload.get("assistant_response")
        or payload.get("final_response")
        or payload.get("completion_claim")
        or payload.get("response")
    )
    if completion_text:
        controller.record_evidence(
            EvidenceKind.FINAL_ACCEPTANCE,
            {"response": _jsonable(completion_text)},
            metadata={
                "session_id": state.get("session_id"),
                "turn_id": payload.get("turn_id") or state.get("task_id"),
                "source": "Stop",
                "role": "assistant_response",
                "domain": domain,
                "status": "unverified",
            },
        )
    decision = controller.judge_stop(
        requirement=requirement,
        acceptance_criteria=criteria,
        iteration=iteration,
        agent_requested_stop=True,
        domain=domain,
        # Always request the JEV decision. When the client is unavailable the
        # controller records the failed attempt and returns an explicitly marked
        # local_fallback instead of an unmarked local judgment.
        use_model=True,
        timeout=_HOOK_JEV_TIMEOUTS["Stop"],
    )
    trace_session_id = str(state.get("session_id") or payload.get("session_id") or "unknown")
    trace_turn_id = str(payload.get("turn_id") or state.get("task_id") or "unknown")
    trace_suffix = f" [session_id={trace_session_id} turn_id={trace_turn_id}]"
    controller.record_evidence(
        EvidenceKind.DECISION,
        {"hook_event": "Stop", "decision": decision.to_dict()},
        metadata={
            "session_id": state.get("session_id"),
            "turn_id": payload.get("turn_id") or state.get("task_id"),
            "domain": decision.domain,
            "requirement_source": requirement_source,
        },
    )

    status = decision.status.value if hasattr(decision.status, "value") else str(decision.status)
    # Once JEV has resolved a concrete domain, freeze it for the remainder of
    # this task. Re-running domain classification on the growing transcript
    # was the source of implementation/investigation/unknown drift.
    if decision.domain and str(decision.domain).lower() != "unknown":
        state["domain"] = decision.domain
    if status == "continue":
        state["stop_attempts"] = iteration + 1
        _save_state(root, state)
        return {"decision": "block", "reason": str(decision.reason) + trace_suffix}
    if status == "escalate":
        state["escalated"] = True
        _save_state(root, state)
        return {
            "continue": True,
            "systemMessage": f"Decision layer escalation: {decision.reason}{trace_suffix}",
        }

    controller.record_evidence(
        EvidenceKind.FINAL_ACCEPTANCE,
        {"approved": True, "requirement": requirement, "missing": decision.missing, "domain": decision.domain},
        metadata={
            "session_id": state.get("session_id"),
            "turn_id": payload.get("turn_id") or state.get("task_id"),
            "domain": decision.domain,
            "requirement_source": requirement_source,
        },
    )
    state["completed"] = True
    _save_state(root, state)
    return {"continue": True}


def handle_hook(payload: Mapping[str, Any], *, root_dir: str | Path | None = None) -> dict[str, Any]:
    """Handle one Codex hook payload and return the JSON response object."""

    root = Path(root_dir).resolve() if root_dir else _project_root(payload)
    event = str(payload.get("hook_event_name") or "")
    if event == "UserPromptSubmit":
        return _handle_user_prompt(payload, root)
    if event == "PostToolUse":
        return _handle_post_tool_use(payload, root)
    if event == "PreToolUse":
        return _handle_pre_tool_use(payload, root)
    if event == "Stop":
        return _handle_stop(payload, root)
    return {"continue": True}


def main(*, root_dir: str | Path | None = None) -> int:
    """CLI entry point used by the repository-local Codex hook command."""

    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    raw = sys.stdin.read()
    payload: Mapping[str, Any] = {}
    try:
        value = json.loads(raw or "{}")
        if not isinstance(value, Mapping):
            raise ValueError("Hook input must be a JSON object")
        payload = value
        result = handle_hook(payload, root_dir=root_dir)
    except Exception as exc:  # Hooks should not make Codex unusable when storage is unavailable.
        print(f"decision hook error: {exc}", file=sys.stderr)
        if payload.get("hook_event_name") == "Stop":
            # A completion gate must fail closed: if evidence storage or the
            # decision layer crashes, do not allow an unverified stop.
            result = {
                "decision": "block",
                "reason": "Decision layer unavailable; completion evidence could not be verified.",
                "systemMessage": f"Decision layer error: {exc}",
            }
        else:
            # `continue` is not a valid PreToolUse response. Returning an
            # empty object keeps the fail-open behavior without making Codex
            # mark the hook run as malformed.
            result = {} if payload.get("hook_event_name") == "PreToolUse" else {"continue": True}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
