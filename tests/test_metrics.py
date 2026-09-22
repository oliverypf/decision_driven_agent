import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from decision_agent.controller import DecisionController
from decision_agent.decision.failure_router import FailureInput
from decision_agent.decision.metrics import MetricsAggregator, MetricsError


def _call_record(
    operation,
    *,
    ok=True,
    called=True,
    digest="",
    adopted=None,
    error="",
    usage=None,
    created_at="2026-09-21T10:00:00+00:00",
):
    jev = {"called": called, "ok": ok, "operation": operation}
    if called:
        jev["request_chars"] = 100
    if digest:
        jev["request_digest"] = digest
    if usage is not None:
        jev["usage"] = usage
    context = {}
    if adopted is not None:
        context["adopted"] = adopted
    if error:
        context["error"] = error
    return {
        "kind": "decision",
        "content": {"jev": jev, "context": context},
        "metadata": {"decision": "jev_call", "operation": operation},
        "created_at": created_at,
        "severity": "info",
    }


def _stop_record(status, created_at="2026-09-21T10:05:00+00:00"):
    return {
        "kind": "decision",
        "content": {"hook_event": "Stop", "decision": {"status": status}},
        "metadata": {"session_id": "s", "turn_id": "t"},
        "created_at": created_at,
        "severity": "info",
    }


def _test_result(status, created_at, command="python -m unittest"):
    return {
        "kind": "test_result",
        "content": {"command": command, "exit_code": 0 if status == "passed" else 1},
        "metadata": {"status": status},
        "created_at": created_at,
        "severity": "info" if status == "passed" else "error",
    }


def _failure_route(created_at, command="python -m unittest"):
    return {
        "kind": "decision",
        "content": {"failure_route": {"type": "TEST_ERROR"}},
        "metadata": {"decision": "failure_route", "command": command},
        "created_at": created_at,
        "severity": "error",
    }


class MetricsAggregatorTests(unittest.TestCase):
    def setUp(self):
        self.aggregator = MetricsAggregator()

    def test_usable_call_ratio_counts_only_adopted_calls(self):
        records = [
            _call_record(
                "stop",
                digest="a",
                adopted={
                    "status": "stop",
                    "source": "JEV",
                    "fallback": False,
                    "fallback_reason": "",
                },
            ),
            _call_record(
                "tool_evidence",
                ok=False,
                digest="b",
                error="JEV request failed",
                adopted={
                    "kind": "other",
                    "source": "local_fallback",
                    "fallback": True,
                    "fallback_reason": "JEV request failed",
                },
            ),
            _call_record("failure_type", called=False, ok=False),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(metrics.calls_attempted, 2)
        self.assertEqual(metrics.calls_adopted, 1)
        self.assertEqual(metrics.calls_invalid, 1)
        self.assertEqual(metrics.calls_unavailable, 1)
        self.assertEqual(metrics.usable_call_ratio, 0.5)

    def test_marked_local_fallback_is_never_counted_as_usable(self):
        records = [
            _call_record(
                "failure_type",
                digest="a",
                adopted={
                    "type": "UNKNOWN",
                    "source": "local_fallback",
                    "fallback": True,
                    "fallback_reason": "JEV unavailable",
                },
            ),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(metrics.calls_adopted, 0)
        self.assertEqual(metrics.usable_call_ratio, 0.0)

    def test_incomplete_provenance_is_not_counted_as_adopted(self):
        records = [
            _call_record("stop", digest="a", adopted={"status": "stop", "source": "JEV"}),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(metrics.calls_adopted, 0)
        self.assertEqual(metrics.calls_eligible, 0)
        self.assertEqual(metrics.calls_invalid, 0)
        self.assertEqual(metrics.calls_missing_provenance, 1)
        self.assertIsNone(metrics.usable_call_ratio)

    def test_legacy_provenance_is_excluded_from_comparable_ratio(self):
        records = [
            _call_record(
                "stop",
                digest="new",
                adopted={
                    "status": "stop",
                    "source": "JEV",
                    "fallback": False,
                    "fallback_reason": "",
                },
            ),
            _call_record("stop", digest="legacy"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(metrics.calls_attempted, 2)
        self.assertEqual(metrics.calls_eligible, 1)
        self.assertEqual(metrics.calls_adopted, 1)
        self.assertEqual(metrics.calls_missing_provenance, 1)
        self.assertEqual(metrics.usable_call_ratio, 1.0)

    def test_duplicate_decisions_are_counted_per_operation_and_digest(self):
        records = [
            _call_record("evidence_sufficiency", digest="same"),
            _call_record("evidence_sufficiency", digest="same"),
            _call_record("evidence_sufficiency", digest="other"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(metrics.calls_attempted, 3)
        self.assertEqual(metrics.calls_duplicate, 1)

    def test_loop_count_and_completion_come_from_stop_decisions(self):
        records = [
            _stop_record("continue"),
            _stop_record("continue", "2026-09-21T10:06:00+00:00"),
            _stop_record("stop", "2026-09-21T10:07:00+00:00"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(metrics.loop_count, 2)
        self.assertEqual(metrics.escalations, 0)
        self.assertTrue(metrics.completed)
        self.assertFalse(metrics.completed_with_error)

    def test_completed_task_with_unresolved_failure_is_an_error_completion(self):
        records = [
            _test_result("passed", "2026-09-21T10:01:00+00:00"),
            _failure_route("2026-09-21T10:02:00+00:00"),
            _stop_record("stop"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertTrue(metrics.completed)
        self.assertTrue(metrics.completed_with_error)

    def test_failure_followed_by_a_passing_check_is_not_an_error_completion(self):
        records = [
            _failure_route("2026-09-21T10:01:00+00:00"),
            _test_result(
                "passed",
                "2026-09-21T10:02:00+00:00",
                command="python -m unittest",
            ),
            _stop_record("stop"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertTrue(metrics.completed)
        self.assertFalse(metrics.completed_with_error)

    def test_unrelated_passing_check_does_not_close_a_failure(self):
        records = [
            _failure_route(
                "2026-09-21T10:01:00+00:00",
                command="python -m unittest tests.test_login",
            ),
            _test_result(
                "passed",
                "2026-09-21T10:02:00+00:00",
                command="python -m unittest tests.test_docs",
            ),
            _stop_record("stop"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertTrue(metrics.completed)
        self.assertTrue(metrics.completed_with_error)

    def test_tokens_are_summed_from_call_usage(self):
        records = [
            _call_record(
                "stop",
                digest="a",
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ),
            _call_record("failure_type", digest="b", usage={"prompt_tokens": 50, "total_tokens": 50}),
            _call_record("tool_risk", digest="c"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(
            metrics.tokens,
            {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170},
        )

    def test_duration_spans_the_first_and_last_record(self):
        records = [
            _call_record("stop", digest="a", created_at="2026-09-21T10:00:00+00:00"),
            _stop_record("stop", created_at="2026-09-21T10:01:30+00:00"),
        ]

        metrics = self.aggregator.aggregate(records)

        self.assertEqual(metrics.duration_s, 90.0)


class MetricsReportTests(unittest.TestCase):
    def test_aggregate_path_scans_a_project_evidence_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store_dir = Path(temp_dir, "session-1", "turn-1")
            store_dir.mkdir(parents=True)
            records = [
                _call_record(
                    "stop",
                    digest="a",
                    adopted={
                        "status": "stop",
                        "source": "JEV",
                        "fallback": False,
                        "fallback_reason": "",
                    },
                ),
                _call_record(
                    "stop",
                    digest="a",
                    adopted={
                        "status": "stop",
                        "source": "JEV",
                        "fallback": False,
                        "fallback_reason": "",
                    },
                ),
                _stop_record("stop"),
            ]
            (store_dir / "evidence.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            report = MetricsAggregator().aggregate_path(temp_dir)

            self.assertEqual(len(report.tasks), 1)
            self.assertEqual(report.tasks[0].task_id, "turn-1")
            self.assertEqual(report.tasks[0].session_id, "session-1")
            self.assertEqual(report.totals["tasks"], 1)
            self.assertEqual(report.totals["calls_attempted"], 2)
            self.assertEqual(report.totals["calls_adopted"], 2)
            self.assertEqual(report.totals["calls_duplicate"], 1)
            self.assertEqual(report.totals["completed_tasks"], 1)
            self.assertEqual(report.totals["error_completion_rate"], 0.0)
            self.assertEqual(report.totals["by_operation"]["stop"]["adopted"], 2)

    def test_missing_evidence_path_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir, "absent")

            with self.assertRaises(MetricsError):
                MetricsAggregator().aggregate_path(missing)

    def test_legacy_root_level_store_is_excluded_when_nested_stores_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "evidence.jsonl").write_text(
                json.dumps(_call_record("stop", digest="root")) + "\n",
                encoding="utf-8",
            )
            nested = Path(root, "session-2", "turn-2")
            nested.mkdir(parents=True)
            (nested / "evidence.jsonl").write_text(
                json.dumps(_call_record("stop", digest="nested")) + "\n",
                encoding="utf-8",
            )

            report = MetricsAggregator().aggregate_path(root)

            self.assertEqual(len(report.tasks), 1)
            self.assertEqual(report.totals["calls_attempted"], 1)

    def test_truncated_records_are_counted_instead_of_aborting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store_dir = Path(temp_dir, "session", "turn")
            store_dir.mkdir(parents=True)
            (store_dir / "evidence.jsonl").write_text(
                json.dumps(_call_record("stop", digest="a")) + "\n" + '{"content": "truncat' + "\n",
                encoding="utf-8",
            )

            report = MetricsAggregator().aggregate_path(temp_dir)

            self.assertEqual(report.totals["calls_attempted"], 1)
            self.assertEqual(report.totals["unreadable_records"], 1)
            self.assertEqual(report.tasks[0].unreadable_records, 1)


class MetricsControllerTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_controller_metrics_reports_unavailable_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            controller.route_failure(FailureInput(message="boom"))

            metrics = controller.metrics()

            self.assertEqual(metrics.calls_attempted, 0)
            self.assertEqual(metrics.calls_unavailable, 1)
            self.assertIsNone(metrics.usable_call_ratio)

    def test_controller_metrics_reports_unreadable_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            controller.record_evidence("user_request", "check metrics")
            with Path(temp_dir, "evidence.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{broken\n")

            metrics = controller.metrics()

            self.assertEqual(metrics.unreadable_records, 1)


class MetricsCliTests(unittest.TestCase):
    def test_metrics_cli_outputs_a_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            store_dir = Path(temp_dir, "session", "turn")
            store_dir.mkdir(parents=True)
            (store_dir / "evidence.jsonl").write_text(
                json.dumps(
                    _call_record(
                        "stop",
                        digest="a",
                        adopted={
                            "source": "JEV",
                            "fallback": False,
                            "fallback_reason": "",
                        },
                    )
                ) + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, "-m", "decision_agent", "metrics", temp_dir],
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=repo_root,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["totals"]["calls_attempted"], 1)
            self.assertEqual(payload["totals"]["calls_adopted"], 1)
            self.assertEqual(payload["tasks"][0]["task_id"], "turn")


if __name__ == "__main__":
    unittest.main()
