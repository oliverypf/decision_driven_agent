# Decision-Driven Codex Agent 需求记录

来源：用户于 2026-09-19 提供的《Decision-Driven Codex Agent 改造设计方案》文本。

## 项目目标

将现有基于 LLM 的 Agent 执行模式改造成 Decision-Driven Agent：提高模型调用效用、避免无效调用，让方向判断更准确，并把分类、路由、验证、停止等判断从强模型中剥离（调用次数、重复分析与耗时的下降是结果指标，而非目标）。

本项目的所有决策项统一使用 JEV（当前决策模型）实现。JEV 是分类、路由、候选选择、排序、验证、Gate、Continue/Stop、升级和失败分类的实际决策者；本地代码不得以确定性规则静默替代这些决策。

总体分层：

1. LLM Expand：开放式问题、方案生成、复杂推理、根因分析、代码生成。
2. JEV Decision Model Reduce：由 JEV 负责分类、路由、选择、排序、验证、Gate、Continue/Stop、升级和失败分类。
3. Program Execute：测试、构建、工具调用、数据采集、状态验证。

JEV 决策约束：

- 每个需要判断的决策项都必须形成结构化问题、候选项、约束和证据，并提交给 JEV。
- 本地代码只负责采集和持久化证据、构造和校验 JEV 输入、校验 JEV 输出格式，以及在 JEV 不可用或输出非法时执行明确的安全兜底。
- 安全兜底不得伪装成 JEV 结论；必须记录 JEV 调用状态、输入摘要、输出、错误和最终采用的兜底结果。
- 测试中的 mock JEV 只能验证调用协议和决策编排，不能视为真实 JEV 服务调用。

## 目标架构

统一 Decision Layer，规划以下模块：

```text
decision/
├── router
├── planner
├── evidence
├── verifier
├── safety
├── memory
└── evolution
```

统一决策接口需要表达：当前状态、判断问题、候选结果、已有证据，以及最终选择、置信度、是否需要强模型和缺失信息。

## 阶段范围

当前状态（2026-09-22）：第一阶段基线已经完成，第二阶段已启动。第二阶段在既有 Decision Layer、Evidence Store 和 hook 契约基础上继续演进；当前仓库已包含 Test Selector、Model Router、Tool Router 和 Memory Decision 的初步实现。

### 第一阶段基线

第一阶段的目标是验证调用效用是否提高、无效调用是否减少，必须实现：

1. **Stop Judge**：先调用 JEV 判断用户目标是否完成并识别任务域，再由 JEV 按任务域判断是否允许停止；避免把 `git_diff`、`validation` 等实现型证据要求施加到信息查询等其他任务上，并在超过循环上限时判断是否升级到强模型。
2. **Evidence Store**：保存需求、验收标准、Git diff、测试结果、运行时状态、日志、决策历史等证据。测试失败、安全风险、构建失败、用户要求和最终验收证据必须保留。
3. **Evidence Sufficiency Judge**：将需求、验收标准、测试结果、运行时证据、日志和 diff 提交给 JEV，由 JEV 输出 `implemented`、`evidence_sufficient`、`missing` 等结果。
4. **Failure Router**：将失败证据提交给 JEV，由 JEV 分类为 `CODE_ERROR`、`TEST_ERROR`、`ENVIRONMENT_ERROR`、`FLAKY`、`MISSING_EVIDENCE` 或 `UNKNOWN`；不得用本地分类结果替代 JEV 决策。

Test Selector、Model Router、Tool Router 和 Memory Decision 属于第二阶段的演进范围；Agent Router、Self Evolution、Safety Profile 和 Agent Trust Model 仍属于后续阶段。第二阶段的完整验收范围以新增需求和后续设计记录为准。

## 集成约束

不修改 Codex 核心，采用 `AGENTS.md`、hooks、Decision Controller 和 Evidence Store 集成。目标 hook 契约为：

- `UserPromptSubmit`：调用 JEV 进行问题分类和方向路由。
- `PreToolUse`：调用 JEV 进行风险和工具判断。
- `PostToolUse`：采集测试结果、日志和执行状态。
- `Stop`：调用 JEV 进行最终验收和 Continue/Stop 判断。

Stop 的任务域可以由 hook 输入提供，也可以由 JEV 从需求和证据中识别。当前域包括 `implementation`、`documentation`、`information`、`investigation`、`configuration`、`external_action` 和 `unknown`。JEV 必须先返回目标完成度，再针对解析出的域返回 `domain_stop_allowed`；本地代码只校验格式并执行迭代上限等安全约束。

本仓库提供独立 Python 包、Controller 和 JSON in/out CLI，供上述 hooks 调用；第一阶段能力保持兼容，第二阶段能力在此基础上扩展。

## 观测指标

当前通过 JEV 审计记录和 `python -m decision_agent metrics [PATH]` 记录、复算（用于评估调用效用，而非以调用数量为目标）：有效调用占比、无效或重复判定次数、循环次数、任务完成时间、token 消耗和错误完成率。

## 验收标准

- 每个决策项都有可追溯的 JEV 调用记录，包括输入证据摘要、候选项、模型、输出和调用状态。
- 用户目标未完成或当前任务域所需证据不足时，JEV 驱动的 Stop Judge 返回 `continue`。
- 目标完成且满足当前任务域的完成契约时，JEV 驱动的 Stop Judge 返回 `stop`；信息查询和调查任务不因缺少 `git_diff` 或测试而自动阻塞。
- 连续不足且达到迭代上限时，由 JEV 判断并返回 `escalate`，不得无限继续。
- Evidence Store 能持久化并恢复 JSONL 证据。
- 常见环境错误、测试错误、代码错误、 flaky 和缺失证据可由 JEV 驱动的 Failure Router 稳定分类。
- 未匹配的失败由 JEV 返回 `UNKNOWN` 并标记 `reasoning_required=true`。
