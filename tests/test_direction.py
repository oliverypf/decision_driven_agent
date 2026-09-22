import contextlib
import io
import json
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from decision_agent.codex_hook import handle_hook
from decision_agent.direction import route


class Fake:
    available = True
    model = "fake"

    def __init__(self, winners):
        self.winners = iter(winners)
        self.calls = 0

    def decide(self, *, state, questions):
        self.calls += 1
        winner = next(self.winners)
        return {"answers": {key: {"noul": 0.9 if key == winner else 0.01} for key in questions}}


class RoutingTests(unittest.TestCase):
    def test_open(self):
        client = Fake(["q1"])
        self.assertEqual(route("investigate", client=client)["direction"], "BUILD_CANDIDATES")
        self.assertEqual(client.calls, 1)

    def test_guard(self):
        result = route("fix", client=Fake(["q0"]))
        self.assertEqual(result["reason"], "closure_guard_failed")
        self.assertEqual(result["source"], "local_fallback")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["calls"][0]["adopted"]["source"], "local_fallback")
        self.assertTrue(result["calls"][0]["adopted"]["fallback"])

    def test_closed_selection(self):
        space = {"candidates": [{"id": "a", "action": "run unit tests"}],
                 "criteria": ["coverage"], "constraints": ["read only"], "evidence": ["unit tests cover changed module"]}
        self.assertEqual(route("verify", space, client=Fake(["q0", "q0"]))["candidate_id"], "a")

    def test_none(self):
        space = {"candidates": [{"id": "a", "action": "test"}], "criteria": ["x"], "constraints": ["x"], "evidence": ["x"]}
        result = route("verify", space, client=Fake(["q0", "q1"]))
        self.assertEqual(result["direction"], "COLLECT_EVIDENCE")
        self.assertEqual(result["source"], "JEV")
        self.assertFalse(result["fallback"])
        self.assertEqual(result["fallback_reason"], "")

    def test_low_confidence_valid_response_is_adopted_from_jev(self):
        class LowConfidence:
            available = True
            model = "fake"

            def decide(self, *, state, questions):
                return {
                    "answers": {
                        key: {"noul": 0.2 if key == "q1" else 0.1}
                        for key in questions
                    }
                }

        result = route("fix", client=LowConfidence())
        self.assertEqual(result["direction"], "BUILD_CANDIDATES")
        self.assertEqual(result["source"], "JEV")
        self.assertFalse(result["fallback"])
        self.assertEqual(result["fallback_reason"], "")

    def test_oversized_state_is_marked_as_fallback(self):
        result = route(
            "verify",
            {"evidence": ["x" * 25_000]},
            client=Fake([]),
        )
        self.assertEqual(result["direction"], "ESCALATE")
        self.assertEqual(result["reason"], "state_too_large")
        self.assertEqual(result["source"], "local_fallback")
        self.assertTrue(result["fallback"])

    def test_prompt_context_and_audit(self):
        with TemporaryDirectory() as root, patch('decision_agent.controller.JevClient', return_value=Fake(['q1'])):
            result = handle_hook({'hook_event_name': 'UserPromptSubmit', 'session_id': 'test',
                                  'turn_id': 'one', 'prompt': 'investigate'}, root_dir=root)
            self.assertTrue(result['continue'])
            self.assertIn('BUILD_CANDIDATES', result['hookSpecificOutput']['additionalContext'])
            self.assertIn(str(Path(root) / '.codex' / 'hooks' / 'decision_route.py'),
                          result['hookSpecificOutput']['additionalContext'])
            records = list(Path(root).rglob('evidence.jsonl'))
            self.assertEqual(len(records), 1)
            self.assertIn('direction_route', records[0].read_text())

    def test_packet_goal_is_recorded_as_requirement_evidence(self):
        from decision_agent.direction import main as direction_main

        with TemporaryDirectory() as root:
            packet_dir = Path(root, 'evidence', 'sess', 'turn-2')
            packet_dir.mkdir(parents=True)
            packet = packet_dir / 'decision-space.json'
            packet.write_text(
                json.dumps({
                    'goal': '为登录接口添加失败重试并运行单元测试',
                    'candidates': [{'id': 'a', 'action': 'run unit tests'}],
                    'criteria': ['coverage'],
                    'constraints': ['read only'],
                    'evidence': ['unit tests cover changed module'],
                }),
                encoding='utf-8',
            )
            with patch('decision_agent.direction.JevClient', return_value=Fake(['q0', 'q0'])), \
                    patch('sys.argv', ['decision_route.py', str(packet)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                direction_main()
            records = [
                json.loads(line)
                for line in (packet_dir / 'evidence.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual(records[-1]['kind'], 'requirement')
            self.assertEqual(records[-1]['content'], '为登录接口添加失败重试并运行单元测试')
            self.assertEqual(records[-1]['metadata']['source'], 'decision_route')
            self.assertEqual(records[-1]['metadata']['turn_id'], 'turn-2')

    def test_direction_route_audits_each_jev_call_with_provenance(self):
        from decision_agent.direction import main as direction_main

        class CallClient:
            available = True
            model = "fake"

            def __init__(self):
                self.call_history = []

            def decide(self, *, state, questions, operation, timeout=None):
                self.call_history.append({
                    "called": True,
                    "ok": True,
                    "operation": operation,
                    "request_digest": operation,
                })
                return {
                    "answers": {
                        key: {"noul": 0.9 if key == "q0" else 0.1}
                        for key in questions
                    }
                }

        with TemporaryDirectory() as root:
            packet_dir = Path(root, 'evidence', 'sess', 'turn-audit')
            packet_dir.mkdir(parents=True)
            packet = packet_dir / 'decision-space.json'
            packet.write_text(
                json.dumps({
                    'goal': 'select a safe candidate and verify it',
                    'candidates': [{'id': 'a', 'action': 'run unit tests'}],
                    'criteria': ['tests pass'],
                    'constraints': ['read only'],
                    'evidence': ['existing tests cover the change'],
                }),
                encoding='utf-8',
            )
            with patch('decision_agent.direction.JevClient', return_value=CallClient()), \
                    patch('sys.argv', ['decision_route.py', str(packet)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                direction_main()

            records = [
                json.loads(line)
                for line in (packet_dir / 'evidence.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            audits = [
                record for record in records
                if record['metadata'].get('decision') == 'jev_call'
            ]
            self.assertEqual(
                [record['metadata']['operation'] for record in audits],
                ['direction_route', 'direction_select'],
            )
            for audit in audits:
                adopted = audit['content']['context']['adopted']
                self.assertEqual(adopted['source'], 'JEV')
                self.assertFalse(adopted['fallback'])
                self.assertEqual(adopted['fallback_reason'], '')

    def test_round_limit_is_decided_by_jev(self):
        from decision_agent.direction import main as direction_main

        with TemporaryDirectory() as root:
            packet_dir = Path(root, 'evidence', 'sess', 'turn-3')
            packet_dir.mkdir(parents=True)
            packet = packet_dir / 'decision-space.json'
            packet.write_text(
                json.dumps({
                    'goal': 'select a safe candidate and verify it',
                    'candidates': [{'id': 'a', 'action': 'run unit tests'}],
                    'criteria': ['tests pass'],
                    'constraints': ['read only'],
                    'evidence': ['existing tests cover the change'],
                }),
                encoding='utf-8',
            )
            ledger = packet_dir / 'direction-rounds.jsonl'
            ledger.write_text('{}\n{}\n{}\n', encoding='utf-8')
            with patch('decision_agent.direction.JevClient', return_value=Fake(['q4'])), \
                    patch('sys.argv', ['decision_route.py', str(packet)]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                direction_main()

            result = json.loads(output.getvalue())
            self.assertEqual(result['direction'], 'ESCALATE')
            self.assertEqual(result['source'], 'JEV')
            self.assertFalse(result['fallback'])
            self.assertEqual(len(result['calls']), 1)
            self.assertEqual(len(ledger.read_text(encoding='utf-8').splitlines()), 4)
