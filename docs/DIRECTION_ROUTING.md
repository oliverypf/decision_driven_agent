# Direction routing

## Source ownership and App integration

All maintained implementation is in this project's decision_agent package.
hmCodex/.codex/hooks/decision_agent_hook.py and decision_route.py are thin
adapters which import this source over the shared filesystem, not the old
embedded package. The lifecycle adapter explicitly passes hmCodex as root_dir,
so task state and evidence remain in the consuming project's .decision folder.
The old embedded files are retained but are not loaded by these adapters.
Do not update that copy; future fixes belong here. No global user hook was
installed, avoiding duplicate hooks and effects on unrelated projects.

The shared source path must remain accessible to the Codex App's Python
process. A direct adapter smoke test does not establish App trust or prove
that a real App task invoked it; verify a real session separately.

UserPromptSubmit now requests a direction, not a completion verdict.
JEV scores five fixed choices: SELECT_CANDIDATE, BUILD_CANDIDATES,
COLLECT_EVIDENCE, CLARIFY_GOAL and ESCALATE. Scores are independent
suitability scores, not a normalized probability distribution.
A score below 0.6 or a winner margin below 0.1 falls back to ESCALATE.

The hook returns advisory additionalContext to the coding model. It does not
replace the host scheduler or automatically execute a candidate. The model
must construct alternatives for open problems, collect evidence when needed,
and submit a packet using `.codex/hooks/decision_route.py PACKET_PATH`.

Packet schema:

```json
{
  "goal": "Choose the next verification",
  "candidates": [{"id": "unit", "action": "Run the existing unit suite"}],
  "criteria": ["Cover the changed behavior"],
  "constraints": ["No production writes"],
  "evidence": ["Inspection shows the suite covers the changed module"]
}
```

The structural closure guard requires 1-8 unique actionable candidates and
nonempty criteria, constraints and evidence. Semantic adequacy still depends
on the model and actual evidence; structure alone does not prove closure.
JEV then scores only those candidates plus NONE. No generated command is
executed automatically. Selection must still obey user permissions and tests.

Each packet directory permits three refinement calls; after that the result
is ESCALATE. Network failures and malformed scores also escalate without
blocking the user's task. Each request has a three-second network timeout.
The Stop completion check is JEV-driven in two stages: JEV first scores whether the user goal is complete and identifies the task domain, then scores whether stopping is allowed under that domain's completion contract. Information and investigation tasks do not inherit implementation-only requirements such as a git diff or tests. Local code only validates the response and enforces the hard iteration-limit safety guard. If JEV is unavailable, times out or returns an invalid answer, the decision is marked with `source="local_fallback"`, `fallback=true` and a `fallback_reason` before the hook uses a domain-aware conservative local fallback, so the fallback is never presented as a JEV conclusion.

The PreToolUse risk and tool-appropriateness judgment follows the same pattern: read-only calls, validation commands, dedicated file-edit tools and unambiguously destructive commands are decided by the local pre-screen (the destructive verdict is the only path that denies the call), and everything else is submitted to JEV. Each hook passes an explicit per-call timeout budget so a slow model call degrades to the marked fallback instead of the hook being killed.

Audit records use content.direction_route with direction, scores and calls.
The packet CLI appends to evidence.jsonl and direction-rounds.jsonl beside
the packet. Diagnostic session IDs must not be mistaken for real App tasks.
