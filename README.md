# Decision-Driven Agent

Phase 2 of the Decision-Driven Codex Agent design. Phase 1 is complete and the current work extends the decision layer with second-stage routing capabilities. The package is dependency-free and exposes JSON-compatible components for hook integration:

- `EvidenceStore`: append-only JSONL evidence persistence with enforced retention. Mandatory evidence
  (requirements, acceptance criteria, diffs, test results, build failures, security risks, user requests,
  decisions and final acceptance) is always kept, `error`/`warning` records are never dropped, oversized
  logs and runtime records are compressed at write time, and an explicit `DROP` is honored only for
  non-mandatory records (the returned record reports `retention=drop` and `metadata.dropped`).
- `EvidenceSufficiencyJudge`: checks implementation, acceptance coverage, validation evidence, and unresolved failures.
- `StopJudge`: asks JEV whether the goal is complete, applies a JEV-selected task-domain gate, and returns `stop`, `continue`, or `escalate`.
- `FailureRouter`: submits failure evidence to JEV and adopts the returned failure type; local rules run only as a marked safety fallback when JEV is unavailable or its answer is invalid.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Hook-friendly CLI

The CLI accepts a JSON object from a file or stdin and prints a JSON decision:

```powershell
Get-Content .\stop-input.json | python -m decision_agent stop -
python -m decision_agent sufficiency .\sufficiency-input.json
python -m decision_agent failure-route .\failure-input.json
python -m decision_agent tool-risk .\tool-risk-input.json
```

The stop and sufficiency payloads accept `requirement`, `acceptance_criteria`, `evidence`, and an optional `domain`. Supported domains are `implementation`, `documentation`, `information`, `investigation`, `configuration`, `external_action`, and `unknown`. Evidence records may include `metadata.criterion_ids` to associate validation with a specific acceptance criterion. The failure route payload maps to `FailureInput` fields: `message`, `stack_trace`, `environment`, `recent_diff`, `exit_code`, and `test_name`.

Failure routing is JEV-driven: the route output carries `source` (`JEV` or `local_fallback`), `fallback`, `fallback_reason` and the per-type JEV scores, and every JEV call is recorded in the evidence store with its call status, submission digest, adopted output and error. A `local_fallback` result is never presented as a JEV conclusion.

Tool-risk payloads accept `tool_name`, `tool_input`, an optional `requirement` and an optional `domain`. The result carries the risk level, the tool-appropriateness verdict, the recommendation (`proceed`, `confirm` or `block`) and the same `source`/`fallback`/`fallback_reason` markers as the other JEV-backed decisions.

## Codex project hooks

The repository includes a project-level `.codex/hooks.json` and a Windows-compatible hook entrypoint. It connects the decision layer to:

- `UserPromptSubmit`: starts a task evidence store and records the requirement.
- `PreToolUse`: asks JEV to judge a planned tool call (risk level and tool appropriateness) before it runs. Read-only calls, validation commands, dedicated file-edit tools and unambiguously destructive commands are decided by a local pre-screen; only the deterministic destructive verdict denies the call, and JEV verdicts stay advisory. Every judgment is recorded in the evidence store, and a high-risk verdict is additionally stored as security-risk evidence.
- `PostToolUse`: records tool input/output, diffs, tests, runtime checks and build failures. A local pre-screen classifies high-confidence results without a model call; weak or unclassified failures are escalated to JEV in a single call that resolves the evidence kind, the failure status and the failure type.
- `Stop`: asks JEV to judge goal completion first, then applies the domain-specific completion contract; it blocks missing goal/domain evidence and escalates after the iteration limit. The decision carries `source`, `fallback` and `fallback_reason`, so a local safety fallback is never presented as a JEV conclusion.

The evidence path is `.decision/evidence/<session>/<turn>/evidence.jsonl`; session state is stored under `.decision/sessions/`. A new user turn gets a new evidence store even when Codex keeps the same conversation session.

A turn whose prompt is only an acknowledgement (`要`, `继续`, `ok`, `do it`, ...) does not restate the goal.
The hook still records the raw reply as `user_request` evidence, but the effective requirement for the turn
carries the most recent informative requirement forward - the previous turn's requirement, the newest
session `decision-space.json` goal, or the newest informative requirement record - and marks it with
`metadata.requirement_source` and `metadata.carried_from`. The Stop hook applies the same fallback and
prefers the packet goal the reasoning model wrote for the turn, so JEV judges the real task instead of a
bare acknowledgement. Running `decision_route.py` also writes the packet goal back as `requirement`
evidence with `metadata.source="decision_route"`.

To activate the project hook in Codex, open this project and review/trust the unmanaged hook when prompted, or use `/hooks` in Codex. For a one-off CLI run outside the trusted project flow, Codex supports `--dangerously-bypass-hook-trust`; only use that when the hook source has been reviewed.

When `JEV_API_KEY` is configured, the Stop hook makes two JEV decisions: goal completion/domain classification, followed by a domain-specific stop gate. Information and investigation tasks can complete with an answer or supported finding without a git diff or tests; implementation and configuration tasks still require the evidence appropriate to those domains. Local code only validates the response shape and enforces hard safety constraints such as the iteration limit. If JEV is unavailable, times out or returns an invalid answer, the hook records `source="local_fallback"` with the failure reason and falls back to a domain-aware conservative decision rather than presenting the local result as a JEV conclusion. Each hook passes an explicit per-call JEV timeout budget (Stop 18s per decision, PostToolUse 10s, PreToolUse 8s) so a slow model call degrades to the marked fallback instead of the hook being killed.

JEV connection defaults can be changed in a project-root `jev.config.json` (copy `jev.config.example.json` to start). Supported fields are `base_url`, `model`, `api_key` and `timeout`. Keep the API key in `JEV_API_KEY` when possible; `jev.config.json` is ignored by Git. Explicit `JevClient(...)` arguments take precedence over the file, then the file takes precedence over built-in defaults. Set `DECISION_AGENT_CONFIG` when the config file lives elsewhere. Invalid or missing configuration falls back to the built-in endpoint, model and timeout.

## Efficiency routers

## Evidence-bound development loop

开发任务可以按以下顺序调用 `development-plan`，每次只让 JEV 从有限候选中选择一个结果：

1. `next_action`：选择下一步动作；
2. `files`：选择允许修改的文件；
3. `direction`：选择修改方向；
4. `verification`：选择验证方案。

每个候选项必须包含 `id`、`action` 和已有证据的 `evidence_ids`。请求中的 `problem` 必须包含完整的
`problem_id` 与 `problem_statement`，JEV 会在每个阶段收到同一问题上下文和相关证据。JEV 不负责直接改文件或执行命令；
Codex/脚本执行选中的候选后，调用 `development-verify` 检查实际改动是否越出选定文件范围，并确认验证证据绑定同一问题。
JEV 不可用时会显式返回 `source=local_fallback` 和 `fallback_reason`，不会伪装成 JEV 结论。

示例：

```powershell
python -m decision_agent development-plan .\development-plan.json
python -m decision_agent development-verify .\development-verify.json
```

`development-plan.json` 的核心结构为：

```json
{
  "stage": "files",
  "problem": {"problem_id": "problem-123", "problem_statement": "修复认证失败"},
  "candidates": [
    {"id": "client", "action": "修复请求认证头", "path": "decision_agent/jev_client.py", "evidence_ids": ["e1"]}
  ],
  "evidence": [{"id": "e1", "kind": "error", "content": "401 Missing Authentication header"}]
}
```

`development-verify.json` 需要提供 `problem`、`selected_files`、`changed_files` 和 `validation_evidence`。

The repository includes first-pass routing decisions for reducing wasted model and tool calls:

- `ModelRouter`: choose between model candidates.
- `ToolRouter`: choose between tool candidates.
- `TestSelector`: choose which tests or validation commands to run.
- `MemoryDecision`: choose whether context or evidence candidates should be kept and reused.

Each router returns a structured `RoutingDecision` with the chosen candidate, confidence, reason, candidate scores and explicit `source`/`fallback` markers. JEV remains the decision authority when available; a local decision is only selected as an explicitly marked `local_fallback` when JEV is unavailable or its response is invalid. `DecisionController` exposes them as `route_model`, `route_tool`, `select_tests` and `decide_memory`.

## Observation metrics

`python -m decision_agent metrics [PATH]` aggregates the observation metrics from persisted evidence
(`.decision/evidence` by default; a single `evidence.jsonl`, one store directory, or a whole evidence root
all work). A legacy shared `evidence.jsonl` directly under a project evidence root is excluded when
task-scoped stores are present. Calls without complete adoption provenance are reported as
`calls_missing_provenance` and excluded from the comparable `usable_call_ratio`; pass `--store-dir`
to decision CLI commands when durable audit persistence is required, otherwise they use an isolated
temporary store. The report carries per-task entries plus project totals:

- `usable_call_ratio`: attempted JEV calls whose answer was adopted, divided by all attempted calls.
- `calls_invalid` / `calls_duplicate`: attempted calls that were not adopted, and repeated calls with the
  same operation plus request digest inside one task.
- `loop_count` / `escalations`: Stop decisions that answered `continue` / `escalate`.
- `mean_task_duration_s`: span between the first and the last evidence record.
- `tokens`: `prompt_tokens`, `completion_tokens`, `total_tokens` and `cost`, summed from the JEV call
  records that capture the OpenRouter usage block.
- `error_completion_rate`: completed tasks whose last failing evidence has no later passing validation.

Measurement never adds a model call: everything is derived from the existing `metadata.decision="jev_call"`
audit records and decision records. Truncated JSONL lines are skipped and counted under
`unreadable_records` instead of failing the whole report.

