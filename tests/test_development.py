import unittest

from decision_agent.decision.development import DevelopmentPlanner


class FakeClient:
    available = True

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def decide(self, **kwargs):
        self.calls.append(kwargs)
        return {"answers": self.answers}


class DevelopmentPlannerTests(unittest.TestCase):
    def setUp(self):
        self.problem = {"problem_id": "problem-1", "problem_statement": "Fix the JEV authentication failure"}
        self.evidence = [
            {"id": "e1", "kind": "error", "content": "401"},
            {"id": "e2", "kind": "config", "content": "endpoint"},
        ]

    def test_chooses_evidence_bound_file_and_passes_problem(self):
        client = FakeClient({"candidate_0": {"noul": 0.2}, "candidate_1": {"noul": 0.9}})
        planner = DevelopmentPlanner(client)
        result = planner.choose_files(
            problem=self.problem,
            evidence=self.evidence,
            candidates=[
                {"id": "config", "action": "edit config", "evidence_ids": ["e2"], "path": "jev.config.json"},
                {"id": "header", "action": "edit header", "evidence_ids": ["e1"], "path": "jev_client.py"},
            ],
        )
        self.assertEqual(result["selected_id"], "header")
        self.assertEqual(result["source"], "JEV")
        self.assertEqual(client.calls[0]["state"]["problem"], self.problem)

    def test_rejects_candidate_without_evidence(self):
        planner = DevelopmentPlanner(FakeClient({}))
        with self.assertRaises(ValueError):
            planner.choose_direction(problem=self.problem, evidence=self.evidence, candidates=[{"id": "x", "action": "guess", "evidence_ids": ["missing"]}])

    def test_fallback_is_explicit(self):
        class Broken(FakeClient):
            def decide(self, **kwargs):
                raise RuntimeError("offline")
        result = DevelopmentPlanner(Broken({})).choose_next_action(
            problem=self.problem, evidence=self.evidence,
            candidates=[{"id": "inspect", "action": "inspect", "evidence_ids": ["e1"]}],
        )
        self.assertTrue(result["fallback"])
        self.assertEqual(result["source"], "local_fallback")

    def test_verifies_scope_and_problem_bound_validation(self):
        result = DevelopmentPlanner.verify_execution(
            problem=self.problem,
            selected_files=["jev_client.py"],
            changed_files=["jev_client.py"],
            validation_evidence=[{"id": "v1", "problem_id": "problem-1", "content": "tests passed"}],
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["validation_evidence_ids"], ["v1"])

    def test_rejects_change_outside_selected_scope(self):
        result = DevelopmentPlanner.verify_execution(
            problem=self.problem, selected_files=["jev_client.py"], changed_files=["controller.py"], validation_evidence=[]
        )
        self.assertEqual(result["status"], "scope_violation")


if __name__ == "__main__":
    unittest.main()
