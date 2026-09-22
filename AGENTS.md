# Decision-Driven Agent - 仓库工作约定

本仓库实现《Decision-Driven Codex Agent》（需求原文见 `docs/requirements.md`）。第一阶段基线已完成，当前已进入第二阶段。
在这里工作的 agent 必须遵守以下约定。

## 目标口径

- 目标是**提高模型调用效用、避免无效调用**，让方向判断更准确。
- 调用次数、重复分析、耗时的下降是结果指标，不是目标，不得为了减少调用而牺牲判断质量。

## 决策分层（不可越层）

1. LLM Expand：开放式问题、方案生成、复杂推理、根因分析、代码生成。
2. JEV Decision Model Reduce：分类、路由、候选选择、排序、验证、Gate、Continue/Stop、升级、失败分类。
3. Program Execute：测试、构建、工具调用、数据采集、状态验证。

所有决策项统一交给 JEV。本地代码只允许：采集和持久化证据、构造并校验 JEV 输入、校验 JEV 输出格式、
在 JEV 不可用或输出非法时执行**明确标记**的安全兜底。本地规则不得静默替代 JEV 决策；兜底结果必须
随 `source` / `fallback` / `fallback_reason` 一起记录，不得伪装成 JEV 结论。

## Hook 契约（`.codex/hooks.json`，入口 `decision_agent.codex_hook`）

- `UserPromptSubmit`：启动本轮证据库、记录需求，并调用 JEV 做方向路由，输出 advisory。
- `PreToolUse`：JEV 做工具风险与工具适当性判断；只读、测试/构建、专用编辑工具与明确的破坏性命令
  由本地预筛处理（标记 `local_prescreen`），其中只有确定性的破坏性判定会拒绝调用，JEV 结论保持建议性。
- `PostToolUse`：采集工具输入输出、diff、测试、运行时状态与构建失败；先本地预筛，低置信度结果升级给
  JEV 一次调用完成证据分类 / 失败判定 / 失败类型判定。
- `Stop`：JEV 先判断目标完成度并识别任务域，再按任务域完成契约判断是否允许停止；超过迭代上限交由 JEV
  判断是否升级。

## 每轮工作流

1. 收到 UserPromptSubmit advisory 后，把决策空间写入
   `.decision/evidence/<session_id>/<turn_id>/decision-space.json`，键为
   `goal`（字符串，必须完整复述真正任务，短回复时不得只写"要"）、
   `candidates`（`[{id, action}]`，1-8 个）、`criteria`、`constraints`、`evidence`（均为非空字符串数组）。
2. 在仓库根目录运行 `python .codex/hooks/decision_route.py "<packet 路径>"`（advisory 里给的是同一个脚本的绝对路径）。
3. 严格按返回方向执行：`SELECT_CANDIDATE` 只执行返回的 `candidate_id` 并验证结果；`BUILD_CANDIDATES`
   构造有限可执行候选集；`COLLECT_EVIDENCE` 先取证不要猜；`CLARIFY_GOAL` 只问必要澄清；
   `ESCALATE` 用推理模型分析而不是猜一个 JEV 选项。
4. 最多精炼三轮；每轮的路线与得分记录在 `direction-rounds.jsonl`。`decision_route.py` 会把 packet 的
   `goal` 回写为 requirement 证据，供 Stop 与本轮之后的继承使用。

## 证据与溯源

- 证据按 `.decision/evidence/<session_id>/<turn_id>/evidence.jsonl` 持久化；会话状态在
  `.decision/sessions/<session_id>.json`。
- 用户只回复"要 / 继续 / ok"这类确认时，hook 会把原始回复记为 `user_request`，并把最近一次可读需求
  （上一轮需求、最近的 packet `goal`、最近的需求记录）继承为本轮需求，用 `requirement_source` 与
  `carried_from` 标记来源；Stop 同样会优先使用本轮 packet 的 `goal`。不要把这套继承当作需要修复的缺陷。
- 每次 JEV 调用都会写入 `metadata.decision="jev_call"` 审计记录（调用状态、请求摘要、采纳结果、错误）。
- 观测指标用 `python -m decision_agent metrics [PATH]` 复算，不额外产生模型调用。

## 开发约定

- 只依赖 Python 标准库；不修改 Codex 核心。
- 测试：`python -m unittest discover -s tests`（PowerShell 下 stderr 的 NativeCommandError 属正常现象，
  以 `Ran N / OK` 为准）。
- 保持改动聚焦，不要顺手修无关缺陷；不要提交未经要求的 git commit。
