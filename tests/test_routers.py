import tempfile
import unittest

from decision_agent.controller import DecisionController
from decision_agent.decision.evidence_store import EvidenceStore
from decision_agent.decision.routers import MemoryDecision, ModelRouter, TestSelector, ToolRouter


class FakeJevClient:
    """Minimal JEV double that mimics the real client's call audit."""

    available = True

    def __init__(self, scores, *, available=True):
        self.scores = scores
        self.available = available
        self._calls = []

    def decide(self, **kwargs):
        self._calls.append(
            {
                "called": True,
                "ok": True,
                "operation": kwargs.get("operation"),
                "question_keys": sorted(kwargs.get("questions") or {}),
            }
        )
        return {
            "answers": {
                f"candidate_{index}": {"noul": score}
                for index, score in enumerate(self.scores)
            }
        }

    def note_unavailable(self, operation):
        self._calls.append(
            {"called": False, "available": False, "operation": operation}
        )

    @property
    def call_history(self):
        return [dict(call) for call in self._calls]

    @property
    def last_call(self):
        return dict(self._calls[-1]) if self._calls else None


class RouterTests(unittest.TestCase):
    def test_model_router_uses_jev_choice(self):
        result = ModelRouter().decide(candidates=["small", "large"], client=FakeJevClient([0.9, 0.2]))
        self.assertEqual(result.choice, "small")
        self.assertEqual(result.source, "JEV")
        self.assertFalse(result.fallback)

    def test_tool_router_uses_jev_choice(self):
        result = ToolRouter().decide(candidates=["rg", "pytest"], client=FakeJevClient([0.7, 0.1]))
        self.assertEqual(result.choice, "rg")
        self.assertEqual(result.scores["rg"], 0.7)

    def test_test_selector_uses_jev_choice(self):
        result = TestSelector().decide(candidates=["tests/test_routers.py", "tests/test_phase1.py"], client=FakeJevClient([0.8, 0.5]))
        self.assertEqual(result.choice, "tests/test_routers.py")

    def test_memory_decision_uses_jev_choice(self):
        result = MemoryDecision().decide(candidates=["keep", "drop"], client=FakeJevClient([0.2, 0.8]))
        self.assertEqual(result.choice, "drop")

    def test_invalid_jev_response_is_explicit_fallback(self):
        class Client:
            available = True

            def decide(self, **kwargs):
                return {"answers": {"candidate_0": {"noul": "bad"}, "candidate_1": {"noul": 0.2}}}

        result = ModelRouter().decide(candidates=["small", "large"], client=Client())
        self.assertTrue(result.fallback)
        self.assertEqual(result.source, "local_fallback")
        self.assertIn("invalid JEV routing response", result.fallback_reason)

    def test_boolean_score_is_rejected_as_invalid(self):
        class Client:
            available = True

            def decide(self, **kwargs):
                return {"answers": {"candidate_0": {"noul": True}, "candidate_1": {"noul": 0.2}}}

        result = ModelRouter().decide(candidates=["small", "large"], client=Client())
        self.assertTrue(result.fallback)
        self.assertEqual(result.source, "local_fallback")

    def test_controller_records_each_router_decision(self):
        client = FakeJevClient([0.9, 0.2], available=False)
        controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
        results = [
            controller.route_model(["small", "large"]),
            controller.route_tool(["rg", "pytest"]),
            controller.select_tests(["tests/test_routers.py", "tests/test_phase1.py"]),
            controller.decide_memory(["keep", "drop"]),
        ]
        self.assertTrue(all(result.source == "local_fallback" for result in results))
        self.assertTrue(all(result.fallback for result in results))
        records = controller.evidence()
        operations = [
            record.metadata["decision"]
            for record in records
            if record.metadata.get("decision") != "jev_call"
        ]
        self.assertEqual(operations, ["model_route", "tool_route", "test_select", "memory_decision"])
        audits = [
            record
            for record in records
            if record.metadata.get("decision") == "jev_call"
        ]
        self.assertEqual(len(audits), 4)
        self.assertTrue(all(record.content["jev"]["called"] is False for record in audits))

    def test_router_jev_call_is_audited(self):
        client = FakeJevClient([0.9, 0.2])
        controller = DecisionController(EvidenceStore(tempfile.mkdtemp()), jev_client=client)
        decision = controller.route_model(["small", "large"], state={"requirement": "pick a model"})
        self.assertEqual(decision.source, "JEV")
        audits = [
            record
            for record in controller.evidence()
            if record.metadata.get("decision") == "jev_call"
        ]
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0].metadata["operation"], "model_route")
        self.assertTrue(audits[0].content["jev"]["called"])
        self.assertEqual(audits[0].content["context"]["adopted"]["choice"], "small")


if __name__ == "__main__":
    unittest.main()
