import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import decision_agent.codex_hook as codex_hook
from decision_agent.codex_hook import handle_hook


class CodexHookTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def _prompt(self, root: str, *, session: str = "session-1", turn: str = "turn-1") -> dict:
        return handle_hook(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": session,
                "turn_id": turn,
                "cwd": root,
                "prompt": "Implement login",
            },
            root_dir=root,
        )

    def test_user_prompt_and_tool_results_are_persisted(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "tool_name": "apply_patch",
                    "tool_input": {"command": "diff --git a/src/auth.py b/src/auth.py"},
                    "tool_response": "patched",
                },
                root_dir=root,
            )
            handle_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "python -m unittest"},
                    "tool_response": {"exit_code": 0, "stdout": "2 tests passed"},
                },
                root_dir=root,
            )
            response = handle_hook(
                {"hook_event_name": "Stop", "session_id": "session-1", "turn_id": "turn-1"},
                root_dir=root,
            )

            self.assertEqual(response, {"continue": True})
            evidence_files = list(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            self.assertEqual(len(evidence_files), 1)
            records = [json.loads(line) for line in evidence_files[0].read_text(encoding="utf-8").splitlines()]
            kinds = {record["kind"] for record in records}
            self.assertTrue({"user_request", "requirement", "git_diff", "test_result", "final_acceptance"} <= kinds)

    def test_short_reply_carries_the_previous_requirement(self):
        with tempfile.TemporaryDirectory() as root:
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "cwd": root,
                    "prompt": "为登录接口添加失败重试并运行单元测试",
                },
                root_dir=root,
            )
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-2",
                    "cwd": root,
                    "prompt": "要",
                },
                root_dir=root,
            )
            state = json.loads(
                Path(root, ".decision", "sessions", "session-1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["requirement_source"], "carried")
            self.assertIn("为登录接口添加失败重试并运行单元测试", state["requirement"])
            self.assertIn("要", state["requirement"])
            records = [
                json.loads(line)
                for line in Path(root, ".decision", "evidence", "session-1", "turn-2", "evidence.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            by_kind = {record["kind"]: record for record in records}
            self.assertEqual(by_kind["user_request"]["content"], "要")
            self.assertIn("为登录接口添加失败重试并运行单元测试", by_kind["requirement"]["content"])
            self.assertEqual(by_kind["requirement"]["metadata"]["requirement_source"], "carried")
            self.assertEqual(by_kind["requirement"]["metadata"]["carried_from"], "previous_turn")
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-3",
                    "cwd": root,
                    "prompt": "继续",
                },
                root_dir=root,
            )
            state = json.loads(
                Path(root, ".decision", "sessions", "session-1.json").read_text(encoding="utf-8")
            )
            self.assertIn("为登录接口添加失败重试并运行单元测试", state["requirement"])
            self.assertEqual(state["requirement"].count("[The user replied only"), 1)

    def test_informative_prompt_is_recorded_verbatim(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            state = json.loads(
                Path(root, ".decision", "sessions", "session-1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["requirement"], "Implement login")
            self.assertEqual(state["requirement_source"], "prompt")

    def _write_packet(self, root: str, turn: str, goal: str) -> None:
        packet = Path(root, ".decision", "evidence", "session-1", turn, "decision-space.json")
        packet.parent.mkdir(parents=True, exist_ok=True)
        packet.write_text(
            json.dumps(
                {
                    "goal": goal,
                    "candidates": [{"id": "a", "action": "run tests"}],
                    "criteria": ["c"],
                    "constraints": ["k"],
                    "evidence": ["e"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_stop_uses_the_packet_goal_when_the_prompt_was_an_acknowledgement(self):
        with tempfile.TemporaryDirectory() as root:
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "cwd": root,
                    "prompt": "要",
                },
                root_dir=root,
            )
            self._write_packet(root, "turn-1", "Observe how the Stop hook handles a short acknowledgement")
            handle_hook(
                {"hook_event_name": "Stop", "session_id": "session-1", "turn_id": "turn-1"},
                root_dir=root,
            )
            records = [
                json.loads(line)
                for line in Path(root, ".decision", "evidence", "session-1", "turn-1", "evidence.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            stop_records = [
                record
                for record in records
                if record["kind"] == "decision" and record["content"].get("hook_event") == "Stop"
            ]
            self.assertTrue(stop_records)
            self.assertEqual(stop_records[-1]["metadata"]["requirement_source"], "turn_packet_goal")

    def test_short_reply_carries_the_packet_goal_when_no_requirement_survived(self):
        with tempfile.TemporaryDirectory() as root:
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "cwd": root,
                    "prompt": "要",
                },
                root_dir=root,
            )
            self._write_packet(root, "turn-1", "为登录接口添加失败重试并运行单元测试")
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-2",
                    "cwd": root,
                    "prompt": "继续",
                },
                root_dir=root,
            )
            state = json.loads(
                Path(root, ".decision", "sessions", "session-1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["requirement_source"], "carried")
            self.assertEqual(state["carried_from"], "session_packet:turn-1")
            self.assertIn("为登录接口添加失败重试并运行单元测试", state["requirement"])

    def test_failed_tool_is_classified_without_blocking_post_tool_use(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            response = handle_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "pytest"},
                    "tool_response": {"exit_code": 1, "stderr": "AssertionError: expected 1 but got 2"},
                },
                root_dir=root,
            )
            self.assertEqual(response, {})
            self.assertNotIn("hookSpecificOutput", response)
            evidence_files = list(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            self.assertEqual(len(evidence_files), 1)
            records = [json.loads(line) for line in evidence_files[0].read_text(encoding="utf-8").splitlines()]
            decision_records = [record for record in records if record.get("kind") == "decision"]
            self.assertTrue(decision_records, "expected a decision evidence record")
            payload = decision_records[-1].get("content") or decision_records[-1].get("payload") or {}
            self.assertEqual(payload.get("failure_route", {}).get("type"), "TEST_ERROR")

    def test_node_repl_edit_is_recorded_as_git_diff(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook({
                "hook_event_name": "PostToolUse", "session_id": "session-1", "turn_id": "turn-1",
                "tool_name": "mcp__node_repl__js", "tool_input": {"code": "write source file"},
                "tool_response": "edited",
            }, root_dir=root)
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [json.loads(line) for line in evidence_file.read_text(encoding="utf-8").splitlines()]
            diff_records = [record for record in records if record["kind"] == "git_diff"]
            self.assertEqual(len(diff_records), 1)
            self.assertEqual(diff_records[0]["metadata"]["evidence_source"], "edit_tool_input")

    def test_empty_diff_output_does_not_create_git_diff_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook({
                "hook_event_name": "PostToolUse", "session_id": "session-1", "turn_id": "turn-1",
                "tool_name": "Bash", "tool_input": {"command": "git diff"},
                "tool_response": "",
            }, root_dir=root)
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [json.loads(line) for line in evidence_file.read_text(encoding="utf-8").splitlines()]
            self.assertFalse([record for record in records if record["kind"] == "git_diff"])

    def test_real_diff_output_creates_git_diff_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook({
                "hook_event_name": "PostToolUse", "session_id": "session-1", "turn_id": "turn-1",
                "tool_name": "Bash", "tool_input": {"command": "git diff"},
                "tool_response": "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n+++ b/src/auth.py\n",
            }, root_dir=root)
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [json.loads(line) for line in evidence_file.read_text(encoding="utf-8").splitlines()]
            diff_records = [record for record in records if record["kind"] == "git_diff"]
            self.assertEqual(len(diff_records), 1)
            self.assertEqual(diff_records[0]["metadata"]["evidence_source"], "tool_response")

    def test_successful_validation_closes_prior_failed_tool(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook({
                "hook_event_name": "PostToolUse", "session_id": "session-1", "turn_id": "turn-1",
                "tool_name": "apply_patch", "tool_input": {"command": "diff --git a/app.py b/app.py"},
                "tool_response": "patched",
            }, root_dir=root)
            handle_hook({
                "hook_event_name": "PostToolUse", "session_id": "session-1", "turn_id": "turn-1",
                "tool_name": "Bash", "tool_input": {"command": "pytest"},
                "tool_response": {"exit_code": 1, "stderr": "AssertionError"},
            }, root_dir=root)
            handle_hook({
                "hook_event_name": "PostToolUse", "session_id": "session-1", "turn_id": "turn-1",
                "tool_name": "Bash", "tool_input": {"command": "pytest"},
                "tool_response": {"exit_code": 0, "stdout": "1 passed"},
            }, root_dir=root)
            response = handle_hook(
                {"hook_event_name": "Stop", "session_id": "session-1", "turn_id": "turn-1"},
                root_dir=root,
            )
            self.assertEqual(response, {"continue": True})

    def test_unresolved_tool_result_is_not_recorded_as_passed(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "python -m unittest"},
                    "tool_response": "error: something odd",
                },
                root_dir=root,
            )
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [
                json.loads(line)
                for line in evidence_file.read_text(encoding="utf-8").splitlines()
            ]
            test_records = [record for record in records if record["kind"] == "test_result"]
            self.assertEqual(len(test_records), 1)
            self.assertEqual(test_records[0]["metadata"]["status"], "unknown")
            self.assertNotEqual(test_records[0]["metadata"]["status"], "passed")

    def test_missing_exit_code_is_not_recorded_as_passed(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "pytest"},
                    "tool_response": "collected 2 items",
                },
                root_dir=root,
            )
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [
                json.loads(line)
                for line in evidence_file.read_text(encoding="utf-8").splitlines()
            ]
            test_records = [record for record in records if record["kind"] == "test_result"]
            self.assertEqual(len(test_records), 1)
            self.assertEqual(test_records[0]["metadata"]["status"], "unknown")
            self.assertNotIn(
                "passed",
                [record["metadata"].get("status") for record in test_records],
            )

    def test_post_tool_use_accepts_cmd_and_script_fields(self):
        for field in ("cmd", "script"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                self._prompt(root)
                handle_hook(
                    {
                        "hook_event_name": "PostToolUse",
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "tool_name": "Bash",
                        "tool_input": {field: "pytest"},
                        "tool_response": {"exit_code": 0, "stdout": "1 passed"},
                    },
                    root_dir=root,
                )
                evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
                records = [json.loads(line) for line in evidence_file.read_text(encoding="utf-8").splitlines()]
                test_records = [record for record in records if record["kind"] == "test_result"]
                self.assertEqual(len(test_records), 1)
                self.assertEqual(test_records[0]["content"]["command"], "pytest")
                self.assertEqual(test_records[0]["metadata"]["status"], "passed")

    def test_successful_build_is_recorded_as_validation_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "cargo build"},
                    "tool_response": {"exit_code": 0, "stdout": "Compiling ok"},
                },
                root_dir=root,
            )
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [
                json.loads(line)
                for line in evidence_file.read_text(encoding="utf-8").splitlines()
            ]
            validation = [
                record
                for record in records
                if record["kind"] == "test_result"
                and record["metadata"].get("status") == "passed"
            ]
            self.assertEqual(len(validation), 1)
            self.assertEqual(validation[0]["content"]["validation"], "build")
            self.assertFalse(
                [record for record in records if record["kind"] == "build_failure"]
            )

    def test_information_domain_uses_answer_evidence_without_diff(self):
        with tempfile.TemporaryDirectory() as root:
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-information",
                    "cwd": root,
                    "prompt": "Which hooks fired?",
                    "domain": "information",
                },
                root_dir=root,
            )
            response = handle_hook(
                {
                    "hook_event_name": "Stop",
                    "session_id": "session-1",
                    "turn_id": "turn-information",
                    "assistant_response": "UserPromptSubmit, PostToolUse, and Stop fired.",
                },
                root_dir=root,
            )
            self.assertEqual(response, {"continue": True})
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [json.loads(line) for line in evidence_file.read_text(encoding="utf-8").splitlines()]
            final_records = [record for record in records if record["kind"] == "final_acceptance"]
            self.assertTrue(final_records)
            self.assertEqual(final_records[0]["metadata"]["domain"], "information")

    def test_stop_reads_codex_last_assistant_message(self):
        with tempfile.TemporaryDirectory() as root:
            handle_hook(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "turn_id": "turn-information",
                    "cwd": root,
                    "prompt": "Which hooks fired?",
                    "domain": "information",
                },
                root_dir=root,
            )
            response = handle_hook(
                {
                    "hook_event_name": "Stop",
                    "session_id": "session-1",
                    "turn_id": "turn-information",
                    "last_assistant_message": "UserPromptSubmit, PostToolUse, and Stop fired.",
                },
                root_dir=root,
            )

            self.assertEqual(response, {"continue": True})
            evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            records = [json.loads(line) for line in evidence_file.read_text(encoding="utf-8").splitlines()]
            final_records = [record for record in records if record["kind"] == "final_acceptance"]
            self.assertTrue(final_records)
            self.assertEqual(final_records[0]["content"]["response"], "UserPromptSubmit, PostToolUse, and Stop fired.")
            self.assertEqual(final_records[0]["metadata"]["role"], "assistant_response")
            stop_decisions = [
                record
                for record in records
                if record["kind"] == "decision" and record.get("content", {}).get("hook_event") == "Stop"
            ]
            self.assertTrue(stop_decisions)
            self.assertEqual(stop_decisions[-1]["content"]["decision"]["status"], "stop")

    def test_stop_blocks_then_escalates_at_iteration_limit(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            responses = [
                handle_hook(
                    {"hook_event_name": "Stop", "session_id": "session-1", "turn_id": "turn-1"},
                    root_dir=root,
                )
                for _ in range(4)
            ]
            self.assertTrue(all(response["decision"] == "block" for response in responses[:3]))
            self.assertTrue(responses[3]["continue"])
            self.assertIn("escalation", responses[3]["systemMessage"].lower())

    def test_new_turn_uses_a_new_evidence_store(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root, turn="turn-1")
            self._prompt(root, turn="turn-2")
            evidence_files = list(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
            self.assertEqual(len(evidence_files), 2)

    def test_stop_without_jev_marks_the_local_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            handle_hook(
                {
                    "hook_event_name": "Stop",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                },
                root_dir=root,
            )
            records = [
                json.loads(line)
                for line in Path(
                    root, ".decision", "evidence", "session-1", "turn-1", "evidence.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            stop_decisions = [
                record
                for record in records
                if record["kind"] == "decision"
                and isinstance(record.get("content"), dict)
                and record["content"].get("hook_event") == "Stop"
            ]
            self.assertTrue(stop_decisions)
            decision = stop_decisions[-1]["content"]["decision"]
            self.assertEqual(decision["source"], "local_fallback")
            self.assertTrue(decision["fallback"])
            self.assertTrue(decision["fallback_reason"])
            audits = [
                record
                for record in records
                if record["metadata"].get("decision") == "jev_call"
            ]
            self.assertTrue(audits, "an unavailable Stop decision must still be audited")
            self.assertFalse(audits[-1]["content"]["jev"]["called"])

    def test_unverified_implementation_claim_cannot_stop_without_diff_or_validation(self):
        with tempfile.TemporaryDirectory() as root:
            self._prompt(root)
            response = handle_hook(
                {
                    "hook_event_name": "Stop",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "assistant_response": "I implemented login.",
                },
                root_dir=root,
            )

            self.assertEqual(response["decision"], "block")
            evidence_file = Path(root, ".decision", "evidence", "session-1", "turn-1", "evidence.jsonl")
            records = [json.loads(line) for line in evidence_file.read_text(encoding="utf-8").splitlines()]
            stop_decisions = [
                record["content"]["decision"]
                for record in records
                if record["kind"] == "decision" and record["content"].get("hook_event") == "Stop"
            ]
            self.assertEqual(stop_decisions[-1]["status"], "continue")
            self.assertIn("git_diff", stop_decisions[-1]["missing"])
            self.assertIn("validation", stop_decisions[-1]["missing"])


class _FakeJevClient:
    """Test double for the JEV-backed decisions (no network calls)."""

    def __init__(self, *, evidence_answer=None, failure_answer=None, risk_answer=None, available=True):
        self.available = available
        self.evidence_answer = evidence_answer
        self.failure_answer = failure_answer
        self.risk_answer = risk_answer
        self.evidence_requests = []
        self.failure_requests = []
        self.risk_requests = []
        self._calls = []

    def classify_tool_evidence(self, **kwargs):
        self.evidence_requests.append(kwargs)
        self._calls.append({"called": True, "ok": True, "operation": "tool_evidence"})
        return dict(self.evidence_answer or {})

    def judge_failure_type(self, **kwargs):
        self.failure_requests.append(kwargs)
        self._calls.append({"called": True, "ok": True, "operation": "failure_type"})
        return dict(self.failure_answer or {})

    def judge_tool_risk(self, **kwargs):
        self.risk_requests.append(kwargs)
        self._calls.append({"called": True, "ok": True, "operation": "tool_risk"})
        return dict(self.risk_answer or {})

    def note_unavailable(self, operation):
        self._calls.append({"called": False, "available": False, "operation": operation})

    @property
    def call_history(self):
        return [dict(call) for call in self._calls]

    @property
    def last_call(self):
        return dict(self._calls[-1]) if self._calls else None


class CodexHookDirectionTimeoutTests(unittest.TestCase):
    def test_user_prompt_direction_uses_the_four_second_budget(self):
        class DirectionClient:
            available = True

            def __init__(self, *args, **kwargs):
                self.timeout = kwargs.get("timeout")
                self.requests = []

            def decide(self, **kwargs):
                self.requests.append(kwargs)
                answers = {
                    f"q{index}": {"noul": 0.9 if index == 0 else 0.0}
                    for index in range(5)
                }
                return {"answers": answers, "model": "test-direction"}

            @property
            def call_history(self):
                return [
                    {"called": True, "ok": True, "operation": "direction_route"}
                    for _ in self.requests
                ]

            @property
            def last_call(self):
                return self.call_history[-1] if self.call_history else None

        with tempfile.TemporaryDirectory() as root:
            client = DirectionClient()
            with patch("decision_agent.controller.JevClient", return_value=client):
                response = handle_hook(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "cwd": root,
                        "prompt": "Implement login",
                    },
                    root_dir=root,
                )

            self.assertEqual(response["continue"], True)
            self.assertTrue(client.requests)
            self.assertEqual(client.requests[0]["timeout"], 4.0)


class CodexHookJevRoutingTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def _records(self, root):
        evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
        return [
            json.loads(line)
            for line in evidence_file.read_text(encoding="utf-8").splitlines()
        ]

    def _prompt(self, root):
        return handle_hook(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "cwd": root,
                "prompt": "Implement login",
            },
            root_dir=root,
        )

    def test_low_confidence_result_is_escalated_in_one_call(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient(
                evidence_answer={
                    "kind": "test_result",
                    "kind_confidence": 0.81,
                    "kind_certain": True,
                    "kind_scores": {"test_result": 0.81},
                    "failed": True,
                    "failed_confidence": 0.9,
                    "failure_type": "TEST_ERROR",
                    "failure_type_confidence": 0.88,
                    "failure_type_certain": True,
                    "failure_type_reasoning_required": False,
                    "failure_type_scores": {"TEST_ERROR": 0.88},
                }
            )
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                response = handle_hook(
                    {
                        "hook_event_name": "PostToolUse",
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "tool_name": "Bash",
                        "tool_input": {"command": "python scripts/check.py"},
                        "tool_response": {"exit_code": 1, "stderr": "2 checks failed"},
                    },
                    root_dir=root,
                )

            self.assertEqual(response, {})
            records = self._records(root)
            test_records = [record for record in records if record["kind"] == "test_result"]
            self.assertEqual(len(test_records), 1)
            self.assertEqual(test_records[0]["metadata"]["status"], "failed")
            self.assertEqual(test_records[0]["metadata"]["classified_by"], "JEV")
            decisions = [record for record in records if record["kind"] == "decision"]
            route = decisions[-1]["content"]["failure_route"]
            self.assertEqual(route["type"], "TEST_ERROR")
            self.assertEqual(route["source"], "JEV")
            self.assertFalse(route["fallback"])
            self.assertEqual(len(fake.evidence_requests), 1)
            self.assertEqual(fake.failure_requests, [])

    def test_tool_evidence_fallback_preserves_the_reason(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient(evidence_answer={})
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                handle_hook(
                    {
                        "hook_event_name": "PostToolUse",
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "tool_name": "exec_command",
                        "tool_input": {"command": "check-widget"},
                        "tool_response": {"message": "ambiguous result"},
                    },
                    root_dir=root,
                )

            classifications = [
                record["content"]["tool_evidence"]
                for record in self._records(root)
                if record["kind"] == "decision"
                and "tool_evidence" in record.get("content", {})
            ]
            self.assertTrue(classifications)
            self.assertEqual(classifications[-1]["source"], "local_fallback")
            self.assertTrue(classifications[-1]["fallback"])
            self.assertIn("Invalid JEV tool-evidence response", classifications[-1]["fallback_reason"])

    def test_invalid_jev_failure_type_is_never_adopted(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient(
                evidence_answer={
                    "kind": "test_result",
                    "kind_confidence": 0.81,
                    "failed": True,
                    "failed_confidence": 0.9,
                    "failure_type": "NOT_A_FAILURE_TYPE",
                    "failure_type_confidence": 0.88,
                    "failure_type_scores": {"TEST_ERROR": 0.88},
                }
            )
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                handle_hook(
                    {
                        "hook_event_name": "PostToolUse",
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "tool_name": "Bash",
                        "tool_input": {"command": "python scripts/check.py"},
                        "tool_response": {"exit_code": 1, "stderr": "checks failed"},
                    },
                    root_dir=root,
                )

            records = self._records(root)
            routes = [
                record["content"]["failure_route"]
                for record in records
                if record["kind"] == "decision" and "failure_route" in record["content"]
            ]
            self.assertTrue(routes)
            self.assertNotEqual(routes[-1]["source"], "JEV")
            self.assertTrue(routes[-1]["fallback"])

    def test_high_confidence_failure_uses_jev_failure_router(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient(
                failure_answer={
                    "failure_type": "CODE_ERROR",
                    "confidence": 0.9,
                    "reasoning_required": False,
                    "scores": {"CODE_ERROR": 0.9, "TEST_ERROR": 0.2},
                    "reason": "JEV failure-type scores: CODE_ERROR=0.90, TEST_ERROR=0.20",
                }
            )
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                handle_hook(
                    {
                        "hook_event_name": "PostToolUse",
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "tool_name": "Bash",
                        "tool_input": {"command": "pytest"},
                        "tool_response": {"exit_code": 1, "stderr": "AssertionError: expected 1 but got 2"},
                    },
                    root_dir=root,
                )

            decisions = [record for record in self._records(root) if record["kind"] == "decision"]
            route = decisions[-1]["content"]["failure_route"]
            # The local rule would answer TEST_ERROR; the JEV decision wins.
            self.assertEqual(route["type"], "CODE_ERROR")
            self.assertEqual(route["source"], "JEV")
            self.assertEqual(len(fake.failure_requests), 1)
            self.assertEqual(fake.evidence_requests, [])

    def test_non_command_output_word_error_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient()
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                handle_hook(
                    {
                        "hook_event_name": "PostToolUse",
                        "session_id": "session-1",
                        "turn_id": "turn-1",
                        "tool_name": "Read",
                        "tool_input": {"file_path": "docs/log.txt"},
                        "tool_response": "The document lists error codes used by the service.",
                    },
                    root_dir=root,
                )

            records = self._records(root)
            self.assertFalse(
                [record for record in records if "failure_route" in record.get("content", {})],
                "a non-command result that mentions error codes must not be routed as a failure",
            )
            self.assertEqual(fake.evidence_requests, [])
            self.assertEqual(fake.failure_requests, [])


class CodexHookPreToolUseTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def _records(self, root):
        evidence_file = next(Path(root, ".decision", "evidence").rglob("evidence.jsonl"))
        return [
            json.loads(line)
            for line in evidence_file.read_text(encoding="utf-8").splitlines()
        ]

    def _prompt(self, root):
        return handle_hook(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "cwd": root,
                "prompt": "Implement login",
            },
            root_dir=root,
        )

    def _tool_call(self, root, *, tool_name, tool_input):
        return handle_hook(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
            root_dir=root,
        )

    def test_read_only_command_skips_the_model(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient()
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                response = self._tool_call(
                    root,
                    tool_name="exec_command",
                    tool_input={"command": "rg -n hook README.md"},
                )

            self.assertEqual(response, {})
            self.assertEqual(fake.risk_requests, [])
            tool_decisions = [
                record for record in self._records(root)
                if isinstance(record.get("content"), dict)
                and record["content"].get("hook_event") == "PreToolUse"
            ]
            self.assertEqual(len(tool_decisions), 1)
            decision = tool_decisions[0]["content"]["tool_decision"]
            self.assertEqual(decision["source"], "local_prescreen")
            self.assertEqual(decision["risk"], "low")
            self.assertFalse(decision["fallback"])

    def test_destructive_command_is_denied_by_the_local_safety_screen(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient()
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                response = self._tool_call(
                    root,
                    tool_name="exec_command",
                    tool_input={"command": "rm -rf /"},
                )

            output = response["hookSpecificOutput"]
            self.assertEqual(output["hookEventName"], "PreToolUse")
            self.assertEqual(output["permissionDecision"], "deny")
            self.assertTrue(output["permissionDecisionReason"])
            self.assertEqual(fake.risk_requests, [])
            security_records = [
                record for record in self._records(root) if record["kind"] == "security_risk"
            ]
            self.assertEqual(len(security_records), 1)

    def test_unclassified_command_is_judged_by_jev(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient(
                risk_answer={
                    "risk": "medium",
                    "risk_confidence": 0.7,
                    "risk_certain": True,
                    "risk_scores": {"low": 0.2, "medium": 0.7, "high": 0.1},
                    "appropriate": True,
                    "appropriate_confidence": 0.8,
                    "recommendation": "proceed",
                    "confidence": 0.7,
                    "reasoning_required": False,
                    "reason": "JEV tool-risk scores: medium=0.70",
                }
            )
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                response = self._tool_call(
                    root,
                    tool_name="exec_command",
                    tool_input={"command": "git commit -m 'update'"},
                )

            self.assertEqual(response, {})
            self.assertEqual(len(fake.risk_requests), 1)
            self.assertEqual(fake.risk_requests[0]["timeout"], 8.0)
            records = self._records(root)
            tool_decisions = [
                record for record in records
                if isinstance(record.get("content"), dict)
                and record["content"].get("hook_event") == "PreToolUse"
            ]
            self.assertEqual(tool_decisions[0]["content"]["tool_decision"]["source"], "JEV")
            audits = [
                record
                for record in records
                if record["metadata"].get("decision") == "jev_call"
                and record["metadata"].get("operation") == "tool_risk"
            ]
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["content"]["context"]["adopted"]["risk"], "medium")

    def test_unavailable_jev_marks_the_local_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient(available=False)
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                response = self._tool_call(
                    root,
                    tool_name="exec_command",
                    tool_input={"command": "git commit -m 'update'"},
                )

            self.assertNotIn("continue", response)
            self.assertIn("PreToolUse decision", response.get("systemMessage", ""))
            tool_decisions = [
                record for record in self._records(root)
                if isinstance(record.get("content"), dict)
                and record["content"].get("hook_event") == "PreToolUse"
            ]
            decision = tool_decisions[0]["content"]["tool_decision"]
            self.assertEqual(decision["source"], "local_fallback")
            self.assertTrue(decision["fallback"])
            self.assertTrue(decision["fallback_reason"])

    def test_jev_block_remains_advisory(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeJevClient(
                risk_answer={
                    "risk": "medium",
                    "recommendation": "block",
                    "reason": "JEV recommends review before proceeding.",
                    "confidence": 0.8,
                    "appropriate": False,
                }
            )
            self._prompt(root)
            with patch("decision_agent.controller.JevClient", return_value=fake):
                response = self._tool_call(
                    root,
                    tool_name="exec_command",
                    tool_input={"command": "git commit -m 'update'"},
                )

            self.assertNotIn("continue", response)
            self.assertNotIn("permissionDecision", response.get("hookSpecificOutput", {}))

    def test_pretooluse_exception_returns_a_contract_safe_response(self):
        root = tempfile.mkdtemp()
        with patch.object(codex_hook, "_handle_pre_tool_use", side_effect=RuntimeError("storage down")):
            with patch("sys.stdin") as stdin, patch("builtins.print") as output:
                stdin.read.return_value = json.dumps({"hook_event_name": "PreToolUse"})
                codex_hook.main(root_dir=root)
                result = json.loads(output.call_args[0][0])
        self.assertEqual(result, {})

    def test_stop_handler_exception_blocks_completion(self):
        root = tempfile.mkdtemp()
        with patch.object(codex_hook, "_handle_stop", side_effect=RuntimeError("storage down")):
            with patch("sys.stdin") as stdin, patch("builtins.print") as output:
                stdin.read.return_value = json.dumps({"hook_event_name": "Stop"})
                codex_hook.main(root_dir=root)
                result = json.loads(output.call_args[0][0])
        self.assertEqual(result["decision"], "block")
        self.assertIn("could not be verified", result["reason"])


if __name__ == "__main__":
    unittest.main()
