import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from decision_agent.controller import DecisionController
from decision_agent.decision.evidence_store import EvidenceStore, EvidenceStoreError
from decision_agent.decision.failure_router import FailureInput, FailureRouter
from decision_agent.decision.sufficiency import EvidenceSufficiencyJudge
from decision_agent.decision.stop_judge import StopJudge
from decision_agent.models import (
    FAILURE_TYPES,
    EvidenceKind,
    FailureType,
    RetentionPolicy,
    StopStatus,
)
from decision_agent.jev_client import JevClient, JevDecisionError


class EvidenceStoreTests(unittest.TestCase):
    def test_evidence_round_trips_as_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            record = controller.record_evidence(
                EvidenceKind.USER_REQUEST,
                "Implement login",
                metadata={"source": "user"},
            )
            restored = controller.evidence()

            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].id, record.id)
            self.assertEqual(restored[0].metadata["source"], "user")
            self.assertTrue(Path(temp_dir, "evidence.jsonl").exists())

    def test_oversized_log_is_compressed_on_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            record = controller.record_evidence(EvidenceKind.LOG, "x" * 9000)

            self.assertEqual(record.retention_value, "compress")
            self.assertTrue(record.metadata["compressed"])
            self.assertEqual(record.metadata["original_chars"], 9000)
            restored = controller.evidence()
            self.assertEqual(len(restored), 1)
            self.assertLess(len(restored[0].content), 9000)
            self.assertIn("[compressed", restored[0].content)

    def test_explicit_drop_is_honored_for_non_mandatory_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EvidenceStore(temp_dir)
            record = store.append(
                EvidenceKind.LOG,
                "transient output",
                retention=RetentionPolicy.DROP,
            )

            self.assertEqual(record.retention_value, "drop")
            self.assertTrue(record.metadata["dropped"])
            self.assertEqual(store.read_all(), [])

    def test_mandatory_and_warning_evidence_cannot_be_dropped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EvidenceStore(temp_dir)
            mandatory = store.append(
                EvidenceKind.USER_REQUEST,
                "keep me",
                retention=RetentionPolicy.DROP,
            )
            warning = store.append(
                EvidenceKind.LOG,
                "warning output",
                severity="warning",
                retention=RetentionPolicy.DROP,
            )

            self.assertEqual(mandatory.retention_value, "keep")
            self.assertEqual(warning.retention_value, "keep")
            self.assertEqual(len(store.read_all()), 2)

    def test_explicit_compress_is_honored_for_small_payloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EvidenceStore(temp_dir)
            record = store.append(
                EvidenceKind.LOG,
                "small",
                retention=RetentionPolicy.COMPRESS,
            )

            self.assertEqual(record.retention_value, "compress")
            self.assertTrue(record.metadata["compressed"])
            self.assertEqual(record.content, "small")

    def test_unknown_retention_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EvidenceStore(temp_dir)

            with self.assertRaises(EvidenceStoreError):
                store.append(EvidenceKind.LOG, "value", retention="archive")

    def test_corrupt_jsonl_line_does_not_hide_valid_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EvidenceStore(temp_dir)
            store.append(EvidenceKind.USER_REQUEST, "keep me")
            with Path(temp_dir, "evidence.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{truncated\n")
            restored = store.read_all()
            self.assertEqual(len(restored), 1)
            self.assertEqual(store.unreadable_records, 1)


class SufficiencyTests(unittest.TestCase):
    class _JsonResponse:
        headers = {}

        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_complete_evidence_is_sufficient(self):
        judge = EvidenceSufficiencyJudge()
        result = judge.evaluate(
            "Implement login",
            ["valid login returns a token"],
            [
                {
                    "kind": "git_diff",
                    "content": "src/auth.py adds login token handling",
                },
                {
                    "kind": "test_result",
                    "content": "valid login returns a token: passed",
                    "metadata": {"status": "passed", "criterion_ids": [0]},
                },
            ],
        )

        self.assertTrue(result.implemented)
        self.assertTrue(result.evidence_sufficient)
        self.assertEqual(result.missing, [])

    def test_mixed_test_summary_is_not_treated_as_passed(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [
                {"kind": "git_diff", "content": "src/auth.py"},
                {"kind": "test_result", "content": "1 failed, 1 passed"},
            ],
        )

        self.assertFalse(result.evidence_sufficient)
        self.assertIn("validation", result.missing)
        self.assertIn("unresolved_failure", result.missing)

    def test_informational_task_does_not_require_implementation_evidence(self):
        controller = DecisionController.from_directory(tempfile.mkdtemp())
        decision = controller.judge_stop(requirement="Which hooks fired?", iteration=0)
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertIn("git_diff", decision.missing)

    def test_information_domain_can_stop_without_diff_or_tests(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Which hooks fired?",
            [],
            [{
                "kind": "final_acceptance",
                "content": {"response": "UserPromptSubmit, PostToolUse, and Stop fired."},
            }],
            domain="information",
        )
        self.assertTrue(result.evidence_sufficient)
        self.assertNotIn("git_diff", result.missing)
        self.assertNotIn("validation", result.missing)

    def test_empty_goal_payload_does_not_hide_a_later_answer(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Which hooks fired?",
            [],
            [
                {"kind": "final_acceptance", "content": {"response": ""}},
                {"kind": "final_acceptance", "content": {"response": "The three hooks fired."}},
            ],
            domain="information",
        )

        self.assertTrue(result.evidence_sufficient)

    def test_false_or_zero_payload_is_not_goal_evidence(self):
        for payload in ({"answer": False}, {"answer": 0}, {"answer": []}, {"answer": {}}):
            with self.subTest(payload=payload):
                result = EvidenceSufficiencyJudge().evaluate(
                    "Answer the question",
                    [],
                    [{"kind": "final_acceptance", "content": payload}],
                    domain="information",
                )
                self.assertFalse(result.evidence_sufficient)
                self.assertIn("goal_completion", result.missing)

    def test_non_boolean_structured_test_status_is_not_validation(self):
        for passed in (0, 1):
            with self.subTest(passed=passed):
                result = EvidenceSufficiencyJudge().evaluate(
                    "Implement login",
                    [],
                    [
                        {"kind": "git_diff", "content": "src/auth.py"},
                        {"kind": "test_result", "content": {"passed": passed}},
                    ],
                )
                self.assertFalse(result.evidence_sufficient)
                self.assertIn("validation", result.missing)

    def test_false_external_completion_is_not_action_evidence(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Send the deployment notification",
            [],
            [{"kind": "runtime", "content": {"completed": False}}],
            domain="external_action",
        )

        self.assertFalse(result.evidence_sufficient)
        self.assertIn("external_action_result", result.missing)

    def test_unrelated_success_does_not_close_a_build_failure(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [
                {"kind": "git_diff", "content": "src/auth.py"},
                {
                    "kind": "build_failure",
                    "content": {"command": "cargo build", "output": "compile failed"},
                    "metadata": {"status": "failed"},
                },
                {
                    "kind": "test_result",
                    "content": {"command": "npm run lint", "output": "passed"},
                    "metadata": {"status": "passed"},
                },
            ],
        )

        self.assertFalse(result.evidence_sufficient)
        self.assertIn("unresolved_failure", result.missing)

    def test_same_validation_id_can_close_an_earlier_failure(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [
                {"kind": "git_diff", "content": "src/auth.py"},
                {
                    "kind": "test_result",
                    "content": {"command": "first command", "output": "failed"},
                    "metadata": {"status": "failed", "validation_id": "login-suite"},
                },
                {
                    "kind": "test_result",
                    "content": {"command": "retry command", "output": "passed"},
                    "metadata": {"status": "passed", "validation_id": "login-suite"},
                },
            ],
        )

        self.assertTrue(result.evidence_sufficient)

    def test_non_boolean_validation_required_result_is_marked_fallback(self):
        client = Mock()
        client.decide_validation_required.return_value = "false"
        client.evaluate_sufficiency.return_value = {
            "noul": 0.9,
            "implemented": True,
            "evidence_sufficient": True,
            "missing": [],
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Update the requirements documentation",
            [],
            [{"kind": "git_diff", "content": "docs/requirements.md"}],
            use_model=True,
            model_client=client,
            domain="documentation",
        )

        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertIn("must be boolean", result.fallback_reason)
        self.assertNotIn("validation", result.missing)

    def test_external_action_uses_result_confirmation_without_diff_or_tests(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Send the deployment notification",
            [],
            [{
                "kind": "final_acceptance",
                "content": {"confirmation": "The provider accepted the notification."},
            }],
            domain="external_action",
        )

        self.assertTrue(result.implemented)
        self.assertTrue(result.evidence_sufficient)
        self.assertNotIn("git_diff", result.missing)
        self.assertNotIn("validation", result.missing)

    def test_external_action_assistant_claim_is_not_confirmation_evidence(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Send the deployment notification",
            [],
            [{
                "kind": "final_acceptance",
                "content": {"confirmation": "I sent it."},
                "metadata": {"role": "assistant_response", "source": "Stop"},
            }],
            domain="external_action",
        )

        self.assertFalse(result.evidence_sufficient)
        self.assertIn("external_action_result", result.missing)

    def test_question_about_tests_is_informational(self):
        controller = DecisionController.from_directory(tempfile.mkdtemp())
        decision = controller.judge_stop(requirement="Were the 23 tests using JEV?", iteration=0)
        self.assertEqual(decision.status, StopStatus.CONTINUE)

    def test_implementation_status_question_is_informational(self):
        controller = DecisionController.from_directory(tempfile.mkdtemp())
        decision = controller.judge_stop(requirement="现在实现到哪种程度了", iteration=0)
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertIn("git_diff", decision.missing)

    def test_missing_validation_prevents_stop(self):
        controller = DecisionController.from_directory(tempfile.mkdtemp())
        controller.record_evidence(EvidenceKind.GIT_DIFF, "changed src/auth.py")
        decision = controller.judge_stop(
            requirement="Implement login",
            acceptance_criteria=["valid login returns a token"],
            iteration=0,
        )

        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertIn("validation", decision.missing)

    def test_jev_can_waive_validation_for_documentation_only_change(self):
        client = Mock()
        client.decide_validation_required.return_value = False
        client.evaluate_sufficiency.return_value = {"noul": 0.9, "implemented": True, "evidence_sufficient": True}
        judge = EvidenceSufficiencyJudge()
        result = judge.evaluate(
            "Update the requirements documentation",
            [],
            [{"kind": "git_diff", "content": "docs/requirements.md"}],
            use_model=True,
            model_client=client,
        )
        self.assertTrue(result.evidence_sufficient)
        self.assertNotIn("validation", result.missing)
        client.decide_validation_required.assert_called_once()
        client.evaluate_sufficiency.assert_called_once()

    def test_jev_overrides_locally_sufficient_evidence(self):
        client = Mock()
        client.decide_validation_required.return_value = True
        client.evaluate_sufficiency.return_value = {
            "noul": 0.2,
            "implemented": False,
            "evidence_sufficient": False,
            "missing": ["boundary case"],
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [
                {"kind": "git_diff", "content": "src/auth.py"},
                {"kind": "test_result", "content": {"passed": True}},
            ],
            use_model=True,
            model_client=client,
        )
        self.assertFalse(result.implemented)
        self.assertFalse(result.evidence_sufficient)
        self.assertEqual(result.missing, ["boundary case"])
        client.evaluate_sufficiency.assert_called_once()

    def test_validation_is_required_when_jev_is_unavailable(self):
        client = Mock()
        client.decide_validation_required.side_effect = JevDecisionError("offline")
        client.evaluate_sufficiency.return_value = {"noul": 0.1, "missing": []}
        result = EvidenceSufficiencyJudge().evaluate(
            "Update the requirements documentation",
            [],
            [{"kind": "git_diff", "content": "docs/requirements.md"}],
            use_model=True,
            model_client=client,
        )
        self.assertIn("validation", result.missing)

    def test_validation_decision_failure_is_not_hidden_by_later_jev_success(self):
        client = Mock()
        client.decide_validation_required.side_effect = JevDecisionError("offline")
        client.evaluate_sufficiency.return_value = {
            "noul": 0.95,
            "implemented": True,
            "evidence_sufficient": True,
            "missing": [],
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
            model_client=client,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertIn("validation decision failed", result.fallback_reason)
        self.assertTrue(result.evidence_sufficient)

    def test_missing_jev_sufficiency_fields_uses_fallback(self):
        client = Mock()
        client.decide_validation_required.return_value = True
        client.evaluate_sufficiency.return_value = {"noul": 0.9, "missing": []}
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login", [], [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True, model_client=client,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertIn("implemented result was invalid", result.fallback_reason)

    def test_boolean_jev_sufficiency_probability_uses_fallback(self):
        client = Mock()
        client.decide_validation_required.return_value = True
        client.evaluate_sufficiency.return_value = {
            "noul": True,
            "implemented": True,
            "evidence_sufficient": True,
            "missing": [],
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
            model_client=client,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertIn("probability was invalid", result.fallback_reason)

    def test_non_finite_jev_sufficiency_probability_uses_fallback(self):
        for probability in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(probability=probability):
                client = Mock()
                client.decide_validation_required.return_value = True
                client.evaluate_sufficiency.return_value = {
                    "noul": probability,
                    "implemented": True,
                    "evidence_sufficient": True,
                    "missing": [],
                }
                result = EvidenceSufficiencyJudge().evaluate(
                    "Implement login",
                    [],
                    [{"kind": "git_diff", "content": "src/auth.py"}],
                    use_model=True,
                    model_client=client,
                )
                self.assertEqual(result.source, "local_fallback")
                self.assertTrue(result.fallback)

    def test_non_list_jev_missing_field_uses_fallback(self):
        client = Mock()
        client.decide_validation_required.return_value = True
        client.evaluate_sufficiency.return_value = {
            "noul": 0.9,
            "implemented": True,
            "evidence_sufficient": True,
            "missing": "none",
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
            model_client=client,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertIn("missing list was invalid", result.fallback_reason)

    def test_jev_missing_list_is_the_final_result(self):
        client = Mock()
        client.decide_validation_required.return_value = True
        client.evaluate_sufficiency.return_value = {
            "noul": 0.95,
            "implemented": True,
            "evidence_sufficient": True,
            "missing": [],
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            ["valid login returns a token"],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
            model_client=client,
        )
        # Locally the criterion is uncovered and validation is missing, but JEV
        # returned a valid sufficient answer, so its missing list is final.
        self.assertEqual(result.source, "JEV")
        self.assertEqual(result.missing, [])
        self.assertTrue(result.evidence_sufficient)

    def test_jev_answer_without_a_missing_key_keeps_local_detail(self):
        client = Mock()
        client.decide_validation_required.return_value = True
        client.evaluate_sufficiency.return_value = {
            "noul": 0.2,
            "implemented": False,
            "evidence_sufficient": False,
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            ["valid login returns a token"],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
            model_client=client,
        )
        self.assertEqual(result.source, "JEV")
        self.assertFalse(result.evidence_sufficient)
        self.assertIn("validation", result.missing)

    def test_jev_sufficiency_rejects_contradictory_completion(self):
        contradictory_answers = (
            {
                "noul": 0.9,
                "implemented": False,
                "evidence_sufficient": True,
                "missing": [],
            },
            {
                "noul": 0.9,
                "implemented": True,
                "evidence_sufficient": True,
                "missing": ["validation"],
            },
        )
        for answer in contradictory_answers:
            with self.subTest(answer=answer):
                client = Mock()
                client.decide_validation_required.return_value = True
                client.evaluate_sufficiency.return_value = answer
                result = EvidenceSufficiencyJudge().evaluate(
                    "Implement login",
                    [],
                    [{"kind": "git_diff", "content": "src/auth.py"}],
                    use_model=True,
                    model_client=client,
                )

                self.assertEqual(result.source, "local_fallback")
                self.assertTrue(result.fallback)
                self.assertIn("contradicts", result.fallback_reason)

    def test_sufficiency_without_a_client_is_marked_fallback(self):
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertTrue(result.fallback_reason)

    def test_stop_without_a_client_is_marked_fallback(self):
        result = StopJudge().evaluate(
            requirement="Implement login",
            acceptance_criteria=[],
            evidence=[{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertTrue(result.fallback_reason)

    def test_failure_router_without_a_client_is_marked_fallback(self):
        result = FailureRouter().classify(
            FailureInput(message="command not found: pytest"),
            use_model=True,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertTrue(result.fallback_reason)

    def test_iteration_limit_escalates(self):
        controller = DecisionController.from_directory(tempfile.mkdtemp())
        decision = controller.judge_stop(
            requirement="Implement login",
            acceptance_criteria=["valid login returns a token"],
            iteration=3,
        )

        self.assertEqual(decision.status, StopStatus.ESCALATE)

    def test_jev_can_decline_escalation_at_iteration_limit(self):
        client = Mock()
        client.judge_stop.return_value = {
            "status": "continue",
            "confidence": 0.4,
            "missing": ["validation"],
            "reason": "JEV selected continue.",
            "goal_completed": False,
            "goal_confidence": 0.4,
            "domain": "implementation",
            "domain_confidence": 0.8,
        }
        decision = StopJudge(max_iterations=3).evaluate(
            requirement="Implement login",
            acceptance_criteria=[],
            evidence=[],
            iteration=3,
            use_model=True,
            model_client=client,
        )
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertEqual(decision.source, "JEV")
        self.assertFalse(decision.fallback)

    def test_jev_is_used_only_as_secondary_judge(self):
        response = Mock()
        response.read.return_value = json.dumps({"answers": {"evidence_sufficiency": {"noul": 0.42, "implemented": False, "evidence_sufficient": False, "missing": ["boundary case"]}}}).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client = JevClient(api_key="test-key", opener=Mock(return_value=response))
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            ["valid login returns a token"],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
            model_client=client,
        )
        self.assertFalse(result.evidence_sufficient)
        self.assertIn("boundary case", result.missing)
        self.assertEqual(result.confidence, 0.42)

    def test_jev_decides_stop_from_the_complete_evidence(self):
        client = Mock()
        client.judge_stop.return_value = {
            "status": "stop",
            "confidence": 0.91,
            "missing": [],
            "reason": "JEV goal completion=0.91; domain=information; domain stop=0.9; selected stop.",
            "goal_completed": True,
            "goal_confidence": 0.91,
            "domain": "information",
            "domain_confidence": 0.9,
        }
        result = StopJudge().evaluate(
            requirement="现在看看 JEV 有没有被调过",
            acceptance_criteria=[],
            evidence=[{"kind": "decision", "content": {"jev": {"called": True}}}],
            use_model=True,
            model_client=client,
            domain="information",
        )
        self.assertEqual(result.status, StopStatus.STOP)
        client.judge_stop.assert_called_once()
        self.assertEqual(client.judge_stop.call_args.kwargs["evidence"][0]["kind"], "decision")
        self.assertEqual(client.judge_stop.call_args.kwargs["domain"], "information")
        self.assertTrue(result.goal_completed)
        self.assertEqual(result.domain, "information")

    def test_invalid_jev_stop_shape_uses_marked_fallback(self):
        client = Mock()
        client.judge_stop.return_value = {
            "status": "stop",
            "confidence": float("nan"),
            "missing": "validation",
            "goal_completed": "yes",
            "domain": 123,
        }
        result = StopJudge().evaluate(
            requirement="Implement login",
            acceptance_criteria=[],
            evidence=[],
            use_model=True,
            model_client=client,
        )

        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertIn("invalid", result.fallback_reason.lower())

    def test_real_client_stop_uses_goal_then_domain_decisions(self):
        class Response:
            headers = {}

            def __init__(self, body):
                self.body = body

            def read(self):
                return json.dumps(self.body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        domain_answers = {
            f"domain_{name}": {"noul": 0.01}
            for name in (
                "implementation",
                "documentation",
                "investigation",
                "configuration",
                "external_action",
                "unknown",
            )
        }
        domain_answers["domain_information"] = {"noul": 0.9}
        responses = [
            Response({"answers": {"goal_completed": {"noul": 0.92}, **domain_answers}}),
            Response({
                "answers": {
                    "domain_stop_allowed": {"noul": 0.88},
                    "domain_evidence_missing": {"noul": 0.1},
                    "escalate_required": {"noul": 0.01},
                }
            }),
        ]
        opener = Mock(side_effect=responses)
        client = JevClient(api_key="test-key", opener=opener)
        result = client.judge_stop(
            requirement="Which hooks fired?",
            acceptance_criteria=[],
            evidence=[{"kind": "final_acceptance", "content": {"response": "The hooks fired."}}],
            iteration=0,
            max_iterations=3,
            agent_requested_stop=True,
        )
        self.assertEqual(result["status"], "stop")
        self.assertEqual(result["domain"], "information")
        self.assertTrue(result["goal_completed"])
        self.assertEqual(opener.call_count, 2)
        self.assertEqual([call["operation"] for call in client.call_history], ["goal_completion", "domain_stop"])

    def test_domain_hint_does_not_override_confident_jev_domain(self):
        domain_answers = {
            f"domain_{name}": {"noul": 0.05}
            for name in (
                "implementation",
                "documentation",
                "investigation",
                "configuration",
                "external_action",
                "unknown",
            )
        }
        domain_answers["domain_information"] = {"noul": 0.9}
        client = JevClient(
            api_key="test-key",
            opener=lambda request, timeout=None: self._JsonResponse(
                {"answers": {"goal_completed": {"noul": 0.9}, **domain_answers}}
            ),
        )
        result = client.judge_goal_completion(
            requirement="Which hooks fired?",
            acceptance_criteria=[],
            evidence=[],
            domain="implementation",
            iteration=0,
            agent_requested_stop=True,
        )
        self.assertEqual(result["domain"], "information")
        self.assertEqual(result["domain_confidence"], 0.9)

    def test_escalate_score_controls_the_iteration_limit(self):
        def result_for(score):
            client = JevClient(
                api_key="test-key",
                opener=lambda request, timeout=None, score=score: self._JsonResponse(
                    {
                        "answers": {
                            "domain_stop_allowed": {"noul": 0.2},
                            "domain_evidence_missing": {"noul": 0.1},
                            "escalate_required": {"noul": score},
                        }
                    }
                ),
            )
            return client.judge_domain_stop(
                requirement="Implement login",
                acceptance_criteria=[],
                evidence=[],
                domain="implementation",
                goal_completed=0.4,
                iteration=3,
                max_iterations=3,
                agent_requested_stop=True,
            )

        self.assertEqual(result_for(0.2)["status"], "continue")
        escalated = result_for(0.8)
        self.assertEqual(escalated["status"], "escalate")
        self.assertIn("iteration_limit", escalated["missing"])


class JevEvidenceCompactionTests(unittest.TestCase):
    def test_completion_claim_survives_compaction(self):
        claim = "\u4e2d\u6587" * 100 + "CONCLUSION-MARKER"
        compact = JevClient._compact_evidence(
            [
                {"kind": "user_request", "id": "u", "content": "question"},
                {"kind": "final_acceptance", "id": "f", "content": {"response": claim}},
                {"kind": "log", "id": "l", "content": "x" * 5000},
            ]
        )
        final = next(item for item in compact if item["kind"] == "final_acceptance")
        encoded = json.dumps(final["content"], ensure_ascii=True)
        # The old 600-character cap hid everything past the first CJK sentence.
        self.assertGreater(len(encoded), 600)
        self.assertIn("CONCLUSION-MARKER", encoded)
        log = next(item for item in compact if item["kind"] == "log")
        self.assertLessEqual(len(json.dumps(log["content"], ensure_ascii=True)), 700)

    def test_validation_correlation_metadata_survives_compaction(self):
        compact = JevClient._compact_evidence(
            [
                {
                    "kind": "test_result",
                    "id": "failed-check",
                    "content": {"command": "python -m unittest tests.test_login"},
                    "metadata": {"status": "failed", "validation_id": "login-suite"},
                }
            ]
        )

        self.assertEqual(compact[0]["metadata"]["validation_id"], "login-suite")


class FailureRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = FailureRouter()

    def test_routes_environment_failure(self):
        result = self.router.classify(FailureInput(message="ModuleNotFoundError: No module named requests"))
        self.assertEqual(result.type, FailureType.ENVIRONMENT_ERROR)

    def test_routes_flaky_failure(self):
        result = self.router.classify(FailureInput(message="flaky test passed on retry"))
        self.assertEqual(result.type, FailureType.FLAKY)

    def test_routes_code_failure(self):
        result = self.router.classify(
            FailureInput(stack_trace="Traceback ... TypeError: cannot read properties of undefined")
        )
        self.assertEqual(result.type, FailureType.CODE_ERROR)

    def test_unknown_requests_strong_model(self):
        result = self.router.classify(FailureInput(message="something failed"))
        self.assertEqual(result.type, FailureType.UNKNOWN)
        self.assertTrue(result.reasoning_required)


class FailureRouterJevTests(unittest.TestCase):
    class _FakeClient:
        def __init__(self, *, answer=None, error=None, available=True):
            self.available = available
            self.answer = answer
            self.error = error
            self.requests = []
            self.unavailable_notes = []

        def judge_failure_type(self, **kwargs):
            self.requests.append(kwargs)
            if self.error is not None:
                raise self.error
            return dict(self.answer)

        def note_unavailable(self, operation):
            self.unavailable_notes.append(operation)

    def test_jev_answer_wins_over_local_rules(self):
        client = self._FakeClient(
            answer={
                "failure_type": "CODE_ERROR",
                "confidence": 0.83,
                "reasoning_required": False,
                "scores": {"CODE_ERROR": 0.83, "FLAKY": 0.11},
                "reason": "JEV failure-type scores: CODE_ERROR=0.83, FLAKY=0.11",
            }
        )
        route = FailureRouter().classify(
            FailureInput(message="flaky test passed on retry"),
            use_model=True,
            model_client=client,
        )
        # The local rule would answer FLAKY; the JEV decision is authoritative.
        self.assertEqual(route.type, FailureType.CODE_ERROR)
        self.assertEqual(route.source, "JEV")
        self.assertFalse(route.fallback)
        self.assertEqual(route.scores["CODE_ERROR"], 0.83)
        self.assertEqual(client.requests[0]["message"], "flaky test passed on retry")

    def test_jev_error_falls_back_to_marked_local_route(self):
        client = self._FakeClient(error=RuntimeError("gateway down"))
        route = FailureRouter().classify(
            FailureInput(message="flaky test passed on retry"),
            use_model=True,
            model_client=client,
        )
        self.assertEqual(route.type, FailureType.FLAKY)
        self.assertEqual(route.source, "local_fallback")
        self.assertTrue(route.fallback)
        self.assertIn("gateway down", route.fallback_reason)

    def test_unavailable_jev_is_noted_and_marked(self):
        client = self._FakeClient(available=False)
        route = FailureRouter().classify(
            FailureInput(message="ModuleNotFoundError: No module named requests"),
            use_model=True,
            model_client=client,
        )
        self.assertEqual(route.type, FailureType.ENVIRONMENT_ERROR)
        self.assertTrue(route.fallback)
        self.assertEqual(route.source, "local_fallback")
        self.assertEqual(client.unavailable_notes, ["failure_type"])

    def test_explicit_local_call_is_not_marked_as_fallback(self):
        route = FailureRouter().classify(FailureInput(message="flaky test passed on retry"))
        self.assertEqual(route.type, FailureType.FLAKY)
        self.assertEqual(route.source, "local")
        self.assertFalse(route.fallback)


class JevFailureClassificationTests(unittest.TestCase):
    class _Response:
        headers = {}

        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def test_client_ranks_failure_types(self):
        answers = {f"failure_type_{name}": {"noul": 0.02} for name in FAILURE_TYPES}
        answers["failure_type_TEST_ERROR"] = {"noul": 0.91}
        opener = Mock(side_effect=[self._Response({"answers": answers})])
        client = JevClient(api_key="test-key", opener=opener)
        result = client.judge_failure_type(message="AssertionError: expected 1 but got 2")
        self.assertEqual(result["failure_type"], "TEST_ERROR")
        self.assertFalse(result["reasoning_required"])
        self.assertEqual([call["operation"] for call in client.call_history], ["failure_type"])

    def test_client_returns_unknown_for_ambiguous_scores(self):
        answers = {f"failure_type_{name}": {"noul": 0.5} for name in FAILURE_TYPES}
        opener = Mock(side_effect=[self._Response({"answers": answers})])
        client = JevClient(api_key="test-key", opener=opener)
        result = client.judge_failure_type(message="something failed")
        self.assertEqual(result["failure_type"], "UNKNOWN")
        self.assertTrue(result["reasoning_required"])

    def test_client_rejects_invalid_scores(self):
        answers = {f"failure_type_{name}": {"noul": 0.02} for name in FAILURE_TYPES}
        answers["failure_type_TEST_ERROR"] = {"noul": 1.5}
        opener = Mock(side_effect=[self._Response({"answers": answers})])
        client = JevClient(api_key="test-key", opener=opener)
        with self.assertRaises(JevDecisionError):
            client.judge_failure_type(message="AssertionError")

    def test_client_resolves_ambiguous_tool_result_in_one_call(self):
        answers = {
            "evidence_test_result": {"noul": 0.8},
            "evidence_build_failure": {"noul": 0.1},
            "evidence_runtime": {"noul": 0.1},
            "evidence_other": {"noul": 0.2},
            "tool_failed": {"noul": 0.9},
        }
        for name in FAILURE_TYPES:
            answers[f"failure_type_{name}"] = {"noul": 0.05}
        answers["failure_type_TEST_ERROR"] = {"noul": 0.88}
        opener = Mock(side_effect=[self._Response({"answers": answers})])
        client = JevClient(api_key="test-key", opener=opener)
        result = client.classify_tool_evidence(
            tool_name="Bash",
            command="python scripts/check.py",
            response_summary="2 checks failed",
            exit_code=1,
            prescreen={"kind": "other", "failed": True, "needs_model": True},
        )
        self.assertEqual(result["kind"], "test_result")
        self.assertTrue(result["failed"])
        self.assertEqual(result["failure_type"], "TEST_ERROR")
        self.assertFalse(result["failure_type_reasoning_required"])
        self.assertEqual(opener.call_count, 1)


class FailureRouteAuditTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_unavailable_jev_fallback_records_call_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            route = controller.route_failure(FailureInput(message="something went wrong"))

            self.assertTrue(route.fallback)
            self.assertEqual(route.source, "local_fallback")
            audits = [
                record.to_dict()
                for record in controller.evidence()
                if record.metadata.get("decision") == "jev_call"
            ]
            self.assertEqual(len(audits), 1)
            call = audits[0]["content"]["jev"]
            self.assertFalse(call["called"])
            self.assertEqual(call["operation"], "failure_type")
            context = audits[0]["content"]["context"]
            self.assertIn("input_summary", context)
            self.assertEqual(context["adopted"]["source"], "local_fallback")
            self.assertTrue(context["adopted"]["fallback"])

    def test_jev_audit_records_one_entry_per_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            controller.route_failure(FailureInput(message="something went wrong"))
            controller.route_failure(FailureInput(message="something went wrong"))
            audits = [
                record
                for record in controller.evidence()
                if record.metadata.get("decision") == "jev_call"
            ]
            self.assertEqual(len(audits), 2)

    def test_tool_evidence_audit_records_full_provenance(self):
        valid_answer = {
            "kind": "test_result",
            "kind_confidence": 0.8,
            "kind_certain": True,
            "failed": False,
            "failed_confidence": 0.9,
            "failure_type_confidence": 0.1,
        }
        cases = (
            (valid_answer, "JEV", False, ""),
            ({"kind": "invalid"}, "local_fallback", True, "Invalid JEV tool-evidence response"),
        )
        for answer, source, fallback, reason_part in cases:
            with self.subTest(source=source):
                client = Mock()
                client.available = True
                client.call_history = [
                    {"called": True, "ok": True, "operation": "tool_evidence"}
                ]
                client.classify_tool_evidence.return_value = answer
                controller = DecisionController(
                    EvidenceStore(tempfile.mkdtemp()), jev_client=client
                )
                result = controller.classify_tool_evidence(
                    tool_name="Bash",
                    command="python -m unittest",
                    response_text="checks completed",
                    exit_code=0,
                    prescreen={"kind": "other", "failed": False},
                    use_model=True,
                )

                audits = [
                    record.to_dict()
                    for record in controller.evidence()
                    if record.metadata.get("operation") == "tool_evidence"
                ]
                adopted = audits[-1]["content"]["context"]["adopted"]
                self.assertEqual(adopted, result)
                self.assertEqual(adopted["source"], source)
                self.assertEqual(adopted["fallback"], fallback)
                self.assertIsInstance(adopted["fallback_reason"], str)
                if reason_part:
                    self.assertIn(reason_part, adopted["fallback_reason"])

    def test_tool_risk_audit_records_full_fallback_provenance(self):
        client = Mock()
        client.available = True
        client.call_history = [
            {"called": True, "ok": False, "operation": "tool_risk"}
        ]
        client.judge_tool_risk.side_effect = RuntimeError("gateway down")
        controller = DecisionController(
            EvidenceStore(tempfile.mkdtemp()), jev_client=client
        )
        result = controller.judge_tool_use(
            tool_name="exec_command",
            tool_input={"command": "git commit -m update"},
        )

        audits = [
            record.to_dict()
            for record in controller.evidence()
            if record.metadata.get("operation") == "tool_risk"
        ]
        adopted = audits[-1]["content"]["context"]["adopted"]
        self.assertEqual(adopted, result.to_dict())
        self.assertEqual(adopted["source"], "local_fallback")
        self.assertTrue(adopted["fallback"])
        self.assertIn("gateway down", adopted["fallback_reason"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_stop_cli_outputs_json(self):
        import subprocess
        import sys

        payload = {
            "requirement": "Implement login",
            "acceptance_criteria": ["valid login returns a token"],
            "evidence": [
                {"kind": "git_diff", "content": "src/auth.py"},
                {
                    "kind": "test_result",
                    "content": "valid login returns a token passed",
                    "metadata": {"status": "passed", "criterion_ids": [0]},
                },
            ],
        }
        completed = subprocess.run(
            [sys.executable, "-m", "decision_agent", "stop", "-"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(json.loads(completed.stdout)["status"], "stop")

    def test_tool_risk_cli_outputs_json(self):
        import subprocess
        import sys

        payload = {
            "tool_name": "exec_command",
            "tool_input": {"command": "rg -n hook README.md"},
            "requirement": "check the hooks",
        }
        completed = subprocess.run(
            [sys.executable, "-m", "decision_agent", "tool-risk", "-"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["risk"], "low")
        self.assertEqual(result["source"], "local_prescreen")
        self.assertEqual(result["recommendation"], "proceed")


class StopFallbackTransparencyTests(unittest.TestCase):
    def test_jev_failure_is_marked_as_a_local_fallback(self):
        class BrokenClient:
            available = True

            def judge_stop(self, **kwargs):
                raise RuntimeError("network down")

        decision = StopJudge().evaluate(
            requirement="Implement login",
            acceptance_criteria=[],
            evidence=[],
            use_model=True,
            model_client=BrokenClient(),
        )
        self.assertEqual(decision.source, "local_fallback")
        self.assertTrue(decision.fallback)
        self.assertIn("network down", decision.fallback_reason)
        self.assertIn("fallback_reason", decision.to_dict())

    def test_invalid_tool_risk_response_is_marked_fallback(self):
        client = Mock()
        client.available = True
        client.call_history = []
        client.judge_tool_risk.return_value = {
            "risk": "invalid", "recommendation": "proceed", "reason": "bad",
            "confidence": 0.5, "appropriate": True,
        }
        controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
        result = controller.judge_tool_use(
            tool_name="custom_tool", tool_input={"value": "x"}, use_model=True
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)

    def test_non_string_jev_sufficiency_reason_uses_fallback(self):
        client = Mock()
        client.evaluate_sufficiency.return_value = {
            "noul": 0.8,
            "implemented": True,
            "evidence_sufficient": True,
            "missing": [],
            "reason": 123,
        }
        result = EvidenceSufficiencyJudge().evaluate(
            "Implement login",
            [],
            [{"kind": "git_diff", "content": "src/auth.py"}],
            use_model=True,
            model_client=client,
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)
        self.assertIn("reason was invalid", result.fallback_reason)

    def test_non_finite_tool_risk_confidence_is_marked_fallback(self):
        client = Mock()
        client.available = True
        client.call_history = []
        client.judge_tool_risk.return_value = {
            "risk": "low", "recommendation": "proceed", "reason": "bad",
            "confidence": float("nan"), "appropriate": True,
        }
        controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
        result = controller.judge_tool_use(
            tool_name="custom_tool", tool_input={"value": "x"}, use_model=True
        )
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)

    def test_invalid_tool_risk_scores_are_marked_fallback(self):
        valid = {
            "risk": "low",
            "risk_confidence": 0.7,
            "risk_certain": True,
            "risk_scores": {"low": 0.7, "medium": 0.2, "high": 0.1},
            "appropriate": True,
            "appropriate_confidence": 0.8,
            "recommendation": "proceed",
            "confidence": 0.7,
            "reasoning_required": False,
            "reason": "JEV judged the call.",
        }
        invalid_values = (
            {"risk_confidence": float("nan")},
            {"appropriate_confidence": float("inf")},
            {"risk_scores": {"low": 0.7, "medium": 0.2, "extreme": 0.1}},
            {"risk_scores": {"low": float("nan"), "medium": 0.2, "high": 0.1}},
        )
        for changes in invalid_values:
            with self.subTest(changes=changes):
                client = Mock()
                client.available = True
                client.call_history = []
                client.judge_tool_risk.return_value = {**valid, **changes}
                controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
                result = controller.judge_tool_use(
                    tool_name="custom_tool", tool_input={"value": "x"}, use_model=True
                )
                self.assertEqual(result.source, "local_fallback")
                self.assertTrue(result.fallback)

    def test_invalid_failure_confidence_is_marked_fallback(self):
        client = Mock()
        client.available = True
        client.call_history = []
        client.judge_failure_type.return_value = {
            "failure_type": "CODE_ERROR", "confidence": 2.0,
            "reasoning_required": False, "scores": {}, "reason": "bad",
        }
        controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
        result = controller.route_failure(FailureInput(message="traceback"), use_model=True)
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)

    def test_non_string_failure_reason_is_marked_fallback(self):
        for reason in (None, 123, []):
            with self.subTest(reason=reason):
                client = Mock()
                client.available = True
                client.call_history = []
                client.judge_failure_type.return_value = {
                    "failure_type": "CODE_ERROR", "confidence": 0.9,
                    "reasoning_required": False, "scores": {}, "reason": reason,
                }
                controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
                result = controller.route_failure(FailureInput(message="traceback"), use_model=True)
                self.assertEqual(result.source, "local_fallback")
                self.assertTrue(result.fallback)
                self.assertIn("invalid reason", result.fallback_reason)

    def test_non_finite_failure_confidence_is_marked_fallback(self):
        client = Mock()
        client.available = True
        client.call_history = []
        client.judge_failure_type.return_value = {
            "failure_type": "CODE_ERROR", "confidence": float("inf"),
            "reasoning_required": False, "scores": {}, "reason": "bad",
        }
        controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
        result = controller.route_failure(FailureInput(message="traceback"), use_model=True)
        self.assertEqual(result.source, "local_fallback")
        self.assertTrue(result.fallback)

    def test_invalid_tool_evidence_boolean_is_marked_fallback(self):
        client = Mock()
        client.available = True
        client.call_history = []
        client.classify_tool_evidence.return_value = {
            "kind": "test_result", "kind_confidence": 0.8,
            "failed": "false", "failed_confidence": 0.1,
            "failure_type_confidence": 0.1,
        }
        controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
        result = controller.classify_tool_evidence(
            tool_name="custom", command="custom", response_text="ok", exit_code=0,
            prescreen={"kind": "log"}, use_model=True,
        )
        self.assertEqual(result["source"], "local_fallback")
        self.assertTrue(result["fallback"])

    def test_jev_decision_is_marked_as_jev(self):
        class StubClient:
            available = True

            def judge_stop(self, **kwargs):
                return {
                    "status": "continue",
                    "confidence": 0.5,
                    "missing": ["validation"],
                    "domain": "implementation",
                    "domain_confidence": 0.8,
                    "goal_completed": False,
                    "goal_confidence": 0.4,
                    "reason": "JEV selected continue.",
                }

        decision = StopJudge().evaluate(
            requirement="Implement login",
            acceptance_criteria=[],
            evidence=[],
            use_model=True,
            model_client=StubClient(),
        )
        self.assertEqual(decision.source, "JEV")
        self.assertFalse(decision.fallback)
        self.assertEqual(decision.fallback_reason, "")

    def test_local_only_decision_is_not_marked_as_fallback(self):
        decision = StopJudge().evaluate(
            requirement="Implement login",
            acceptance_criteria=[],
            evidence=[],
            use_model=False,
        )
        self.assertEqual(decision.source, "local")
        self.assertFalse(decision.fallback)

    def test_jev_stop_is_not_overridden_when_agent_did_not_request_stop(self):
        client = Mock()
        client.judge_stop.return_value = {
            "status": "stop",
            "confidence": 0.9,
            "missing": [],
            "reason": "JEV allowed stop.",
            "goal_completed": True,
            "goal_confidence": 0.9,
            "domain": "information",
            "domain_confidence": 0.9,
        }
        decision = StopJudge().evaluate(
            requirement="Answer the question",
            acceptance_criteria=[],
            evidence=[],
            agent_requested_stop=False,
            use_model=True,
            model_client=client,
        )
        self.assertEqual(decision.status, StopStatus.STOP)
        self.assertEqual(decision.source, "JEV")
        self.assertFalse(decision.fallback)

    def test_jev_stop_requires_its_own_task_domain(self):
        client = Mock()
        client.judge_stop.return_value = {
            "status": "continue",
            "confidence": 0.7,
            "missing": ["validation"],
            "reason": "JEV selected continue.",
            "goal_completed": False,
            "goal_confidence": 0.5,
        }
        decision = StopJudge().evaluate(
            requirement="Implement login",
            acceptance_criteria=[],
            evidence=[],
            domain="implementation",
            use_model=True,
            model_client=client,
        )
        self.assertEqual(decision.source, "local_fallback")
        self.assertTrue(decision.fallback)
        self.assertIn("invalid domain", decision.fallback_reason)

    def test_jev_stop_requires_goal_completion_output(self):
        client = Mock()
        client.judge_stop.return_value = {
            "status": "stop",
            "confidence": 0.9,
            "missing": [],
            "reason": "JEV allowed stop.",
            "domain": "information",
            "domain_confidence": 0.9,
        }
        decision = StopJudge().evaluate(
            requirement="Answer the question",
            acceptance_criteria=[],
            evidence=[],
            use_model=True,
            model_client=client,
        )
        self.assertEqual(decision.source, "local_fallback")
        self.assertTrue(decision.fallback)
        self.assertIn("goal_completed", decision.fallback_reason)

    def test_jev_stop_rejects_contradictory_completion(self):
        base = {
            "status": "stop",
            "confidence": 0.9,
            "missing": [],
            "reason": "JEV allowed stop.",
            "goal_completed": True,
            "goal_confidence": 0.9,
            "domain": "information",
            "domain_confidence": 0.9,
        }
        contradictory_answers = (
            {**base, "goal_completed": False},
            {**base, "missing": ["validation"]},
        )
        for answer in contradictory_answers:
            with self.subTest(answer=answer):
                client = Mock()
                client.judge_stop.return_value = answer
                decision = StopJudge().evaluate(
                    requirement="Answer the question",
                    acceptance_criteria=[],
                    evidence=[],
                    use_model=True,
                    model_client=client,
                )

                self.assertEqual(decision.source, "local_fallback")
                self.assertTrue(decision.fallback)
                self.assertIn("contradicts", decision.fallback_reason)


class StopAuditContextTests(unittest.TestCase):
    class _AuditClient:
        available = True

        def __init__(self):
            self._calls = []

        def judge_stop(self, **kwargs):
            self._calls.append({"called": True, "ok": True, "operation": "stop"})
            return {
                "status": "continue",
                "confidence": 0.5,
                "missing": ["validation"],
                "domain": "implementation",
                "domain_confidence": 0.8,
                "goal_completed": False,
                "goal_confidence": 0.4,
                "reason": "JEV selected continue.",
            }

        def decide_validation_required(self, **kwargs):
            self._calls.append({"called": True, "ok": True, "operation": "validation_required"})
            return True

        def evaluate_sufficiency(self, **kwargs):
            self._calls.append({"called": True, "ok": True, "operation": "evidence_sufficiency"})
            return {"noul": 0.9}

        @property
        def call_history(self):
            return [dict(call) for call in self._calls]

        @property
        def last_call(self):
            return dict(self._calls[-1]) if self._calls else None

    def test_stop_audit_records_the_adopted_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            controller.jev_client = self._AuditClient()
            controller.judge_stop(requirement="Implement login")

            audits = [
                record.to_dict()
                for record in controller.evidence()
                if record.metadata.get("decision") == "jev_call"
            ]
            self.assertEqual(len(audits), 1)
            context = audits[0]["content"]["context"]
            self.assertEqual(context["adopted"]["source"], "JEV")
            self.assertEqual(context["iteration"], 0)
            self.assertIn("requirement", context)

    def test_sufficiency_audit_records_the_adopted_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            controller.jev_client = self._AuditClient()
            controller.judge_sufficiency(requirement="Implement login")

            audits = [
                record.to_dict()
                for record in controller.evidence()
                if record.metadata.get("decision") == "jev_call"
            ]
            self.assertEqual(len(audits), 2)
            for audit in audits:
                adopted = audit["content"]["context"]["adopted"]
                self.assertIn("source", adopted)
                self.assertIn("fallback", adopted)
                self.assertIn("fallback_reason", adopted)

    def test_multi_call_stop_audits_are_per_call(self):
        class MultiCallClient(self._AuditClient):
            def judge_stop(self, **kwargs):
                self._calls.extend([
                    {"called": True, "ok": True, "operation": "goal_completion"},
                    {"called": True, "ok": True, "operation": "domain_stop"},
                ])
                return {
                    "status": "continue",
                    "confidence": 0.5,
                    "missing": ["validation"],
                    "domain": "implementation",
                    "domain_confidence": 0.8,
                    "goal_completed": False,
                    "goal_confidence": 0.4,
                    "reason": "JEV selected continue.",
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            controller = DecisionController.from_directory(temp_dir)
            controller.jev_client = MultiCallClient()
            controller.judge_stop(requirement="Implement login")

            audits = [
                record.to_dict()
                for record in controller.evidence()
                if record.metadata.get("decision") == "jev_call"
            ]
            self.assertEqual(len(audits), 2)
            self.assertEqual(
                {audit["metadata"]["operation"] for audit in audits},
                {"goal_completion", "domain_stop"},
            )
            for audit in audits:
                adopted = audit["content"]["context"]["adopted"]
                self.assertEqual(adopted["source"], "JEV")
                self.assertFalse(adopted["fallback"])
                self.assertEqual(adopted["fallback_reason"], "")
                self.assertEqual(
                    adopted["call_operation"],
                    audit["metadata"]["operation"],
                )


class JevTimeoutBudgetTests(unittest.TestCase):
    class _Response:
        headers = {}

        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def test_decide_honors_an_explicit_timeout_budget(self):
        seen = {}

        def opener(request, timeout=None):
            seen["timeout"] = timeout
            return self._Response({"answers": {"probe": {"noul": 0.9}}})

        client = JevClient(api_key="test-key", opener=opener, timeout=30.0)
        client.decide(
            state={},
            questions={"probe": {"type": "noul", "instructions": "score it"}},
            operation="probe",
            timeout=2.5,
        )
        self.assertEqual(seen["timeout"], 2.5)
        self.assertEqual(client.call_history[-1]["timeout_s"], 2.5)

    def test_judge_tool_risk_ranks_the_risk_scores(self):
        answers = {
            "risk_low": {"noul": 0.05},
            "risk_medium": {"noul": 0.2},
            "risk_high": {"noul": 0.85},
            "tool_appropriate": {"noul": 0.8},
        }
        client = JevClient(api_key="test-key", opener=lambda request, timeout=None: self._Response({"answers": answers}))
        result = client.judge_tool_risk(
            tool_name="exec_command",
            command="git push --force origin main",
            requirement="Ship the fix",
        )
        self.assertEqual(result["risk"], "high")
        self.assertTrue(result["risk_certain"])
        self.assertEqual(result["recommendation"], "confirm")
        self.assertTrue(result["appropriate"])
        self.assertEqual(client.call_history[-1]["operation"], "tool_risk")


class JevCallRecordTests(unittest.TestCase):
    class _Response:
        headers = {}

        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def test_call_record_captures_usage_and_request_digest(self):
        body = {
            "answers": {"probe": {"noul": 0.9}},
            "usage": {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
        }
        client = JevClient(api_key="test-key", opener=lambda request, timeout=None: self._Response(body))
        client.decide(state={"requirement": "x"}, questions={"probe": {"type": "noul"}}, operation="probe")

        call = client.call_history[-1]
        self.assertEqual(
            call["usage"],
            {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
        )
        self.assertEqual(len(call["request_digest"]), 16)

    def test_identical_requests_share_a_digest_and_changed_requests_do_not(self):
        body = {"answers": {"probe": {"noul": 0.5}}}
        client = JevClient(api_key="test-key", opener=lambda request, timeout=None: self._Response(body))
        client.decide(state={"requirement": "same"}, questions={"probe": {"type": "noul"}}, operation="probe")
        client.decide(state={"requirement": "same"}, questions={"probe": {"type": "noul"}}, operation="probe")
        client.decide(state={"requirement": "other"}, questions={"probe": {"type": "noul"}}, operation="probe")

        digests = [call["request_digest"] for call in client.call_history]
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[0], digests[2])

    def test_missing_usage_is_recorded_as_an_empty_mapping(self):
        client = JevClient(
            api_key="test-key",
            opener=lambda request, timeout=None: self._Response({"answers": {"probe": {"noul": 0.5}}}),
        )
        client.decide(state={}, questions={"probe": {"type": "noul"}}, operation="probe")

        self.assertEqual(client.call_history[-1]["usage"], {})

    def test_usage_is_normalized_from_the_decisions_endpoint_shape(self):
        body = {
            "answers": {"probe": {"noul": 0.5}},
            "usage": {"input_tokens": 282, "output_tokens": 20, "cost": 1.1844e-05},
        }
        client = JevClient(api_key="test-key", opener=lambda request, timeout=None: self._Response(body))
        client.decide(state={}, questions={"probe": {"type": "noul"}}, operation="probe")

        self.assertEqual(
            client.call_history[-1]["usage"],
            {
                "prompt_tokens": 282,
                "completion_tokens": 20,
                "total_tokens": 302,
                "cost": 1.1844e-05,
            },
        )


class JevQuestionShapeTests(unittest.TestCase):
    """Question types must stay inside the discriminator set the API accepts."""

    _SUPPORTED = {"noul", "choice", "score"}

    class _Response:
        headers = {}

        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def test_validation_required_sends_a_supported_question_type(self):
        captured = {}

        def opener(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return self._Response({"answers": {"validation_required": {"noul": 0.9}}})

        client = JevClient(api_key="test-key", opener=opener)

        self.assertTrue(
            client.decide_validation_required(requirement="Implement login", evidence=[])
        )
        self.assertEqual(captured["body"]["questions"]["validation_required"]["type"], "noul")

    def test_validation_required_thresholds_the_noul_score(self):
        client = JevClient(
            api_key="test-key",
            opener=lambda request, timeout=None: self._Response(
                {"answers": {"validation_required": {"noul": 0.2}}}
            ),
        )

        self.assertFalse(
            client.decide_validation_required(requirement="Update the docs", evidence=[])
        )

    def test_every_client_question_uses_a_supported_type(self):
        source = Path(__file__).resolve().parents[1] / "decision_agent" / "jev_client.py"
        types = set(re.findall(r'"type":\s*"([a-z_]+)"', source.read_text(encoding="utf-8")))

        self.assertTrue(types)
        self.assertLessEqual(types, self._SUPPORTED)


if __name__ == "__main__":
    unittest.main()
