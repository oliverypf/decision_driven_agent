import unittest

from decision_agent.decision.prescreen import prescreen_tool_result, prescreen_tool_use


class ToolUsePreScreenTests(unittest.TestCase):
    def test_read_only_command_is_decided_locally(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "rg -n hook README.md"},
        )
        self.assertEqual(result.risk, "low")
        self.assertEqual(result.recommendation, "proceed")
        self.assertFalse(result.needs_model)

    def test_powershell_format_list_is_not_a_disk_operation(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "Get-Command python | Format-List *"},
        )

        self.assertEqual(result.risk, "low")
        self.assertEqual(result.recommendation, "proceed")
        self.assertFalse(result.needs_model)

    def test_real_format_after_a_formatter_is_still_blocked(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "Get-Command python | Format-List *; format C:"},
        )

        self.assertEqual(result.risk, "high")
        self.assertEqual(result.recommendation, "block")
        self.assertIn("destructive:disk_operation", result.signals)

    def test_mutating_pipeline_is_not_read_only(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "git status | git commit -m update"},
        )

        self.assertTrue(result.needs_model)
        self.assertEqual(result.risk, "medium")

    def test_validation_command_is_decided_locally(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "python -m unittest discover -s tests"},
        )
        self.assertFalse(result.needs_model)
        self.assertEqual(result.recommendation, "proceed")

    def test_file_edit_tool_is_decided_locally(self):
        result = prescreen_tool_use(tool_name="apply_patch", tool_input={"patch": "*** Begin Patch"})
        self.assertFalse(result.needs_model)
        self.assertIn("tool:file_edit", result.signals)

    def test_root_delete_is_blocked_locally(self):
        result = prescreen_tool_use(tool_name="exec_command", tool_input={"command": "rm -rf /"})
        self.assertEqual(result.risk, "high")
        self.assertEqual(result.recommendation, "block")
        self.assertFalse(result.needs_model)

    def test_system_tree_delete_is_blocked_locally(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "Remove-Item -Recurse -Force C:\\Windows"},
        )
        self.assertEqual(result.recommendation, "block")

    def test_force_push_asks_the_model(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "git push --force origin main"},
        )
        self.assertTrue(result.needs_model)
        self.assertEqual(result.risk, "high")
        self.assertIn("high_risk:force_push", result.signals)

    def test_unclassified_command_asks_the_model(self):
        result = prescreen_tool_use(
            tool_name="exec_command",
            tool_input={"command": "git commit -m 'update'"},
        )
        self.assertTrue(result.needs_model)
        self.assertEqual(result.risk, "medium")


class PreScreenTests(unittest.TestCase):
    def test_exit_code_is_authoritative_for_failure(self):
        result = prescreen_tool_result(command="pytest", response_text="all good", exit_code=1)
        self.assertTrue(result.failed)
        self.assertEqual(result.kind, "test_result")
        self.assertFalse(result.needs_model)

    def test_successful_command_stays_local(self):
        result = prescreen_tool_result(command="npm test", response_text="134 passing", exit_code=0)
        self.assertFalse(result.failed)
        self.assertEqual(result.kind, "test_result")
        self.assertFalse(result.needs_model)

    def test_unclassified_failure_requests_jev(self):
        result = prescreen_tool_result(
            command="python scripts/check.py",
            response_text="2 checks failed",
            exit_code=1,
        )
        self.assertTrue(result.failed)
        self.assertEqual(result.kind, "other")
        self.assertTrue(result.needs_model)
        self.assertIn("evidence kind is unclear", result.reason)

    def test_weak_signal_without_exit_code_requests_jev(self):
        result = prescreen_tool_result(
            command="make verify",
            response_text="warning: error budget exceeded",
        )
        self.assertIsNone(result.failed)
        self.assertTrue(result.needs_model)

    def test_execution_without_exit_code_or_failure_signal_stays_unknown(self):
        result = prescreen_tool_result(
            command="pytest",
            response_text="collected 2 items",
        )
        self.assertIsNone(result.failed)
        self.assertTrue(result.needs_model)

    def test_strong_marker_without_exit_code_is_a_failure(self):
        result = prescreen_tool_result(
            command="python -m unittest",
            response_text="AssertionError: expected 1 but got 2",
        )
        self.assertTrue(result.failed)
        self.assertEqual(result.kind, "test_result")
        self.assertFalse(result.needs_model)

    def test_plain_text_without_command_is_not_a_failure(self):
        result = prescreen_tool_result(
            tool_name="Read",
            response_text="The document lists error codes used by the service.",
        )
        self.assertFalse(result.failed)
        self.assertFalse(result.needs_model)

    def test_build_and_runtime_commands_are_classified(self):
        build = prescreen_tool_result(command="cargo build", response_text="ok", exit_code=0)
        self.assertEqual(build.kind, "build_failure")
        runtime = prescreen_tool_result(
            command="curl http://localhost/healthcheck",
            response_text="ok",
            exit_code=0,
        )
        self.assertEqual(runtime.kind, "runtime")


if __name__ == "__main__":
    unittest.main()
