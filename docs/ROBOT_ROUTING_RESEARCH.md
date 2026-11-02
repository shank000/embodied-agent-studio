# 机器人路由研究：任务、观测与成本契约

本轮把复合任务评价、路由调用证据和费用归因接到现有实验记录中，为后续 harness、路由与策略研究准备数据。尚未启动强化学习训练，也不以工程回归结果代替模型任务成功率。基础接口与 IDE 方案见 [实验与训练接入](EXPERIMENTS_AND_TRAINING.md)。

## 1. 先运行可验收的复合任务

选择 MuJoCo，再选择下表场景与任务。建议先使用 Panda；双/三 Panda 可在共享分拣场景研究分工。两项任务的推荐预算均为 **180 秒仿真时间、100 次动作**；模型轮数、请求超时和墙钟预算仍需单独配置。推荐预算不是完成任务的保证。

| 任务 ID | 场景 | 顺序与物理验收 |
| --- | --- | --- |
| `order_fulfillment` | `sorting`、`cooperative` | 红块入绿盒并释放、稳定 0.3 秒 → 蓝块入蓝盒并释放、稳定 0.3 秒 → 所有末端撤离物体至少 8 cm。已交付物必须继续满足放置条件；提前交付蓝块、黄色非目标方块水平移动超过 3 cm，判本次任务失败。 |
| `drawer_cycle` | `drawer` | 抽屉拉开超过 12 cm、保持 0.3 秒 → 关至 2 cm 以内且速度低于 3 cm/s、保持 0.3 秒 → 脱离手部接触、所有末端距把手至少 8 cm，保持 0.3 秒。 |

长度和速度阈值随场景缩放；任务阶段全部满足后，还需最终整体条件连续保持 0.3 秒。评价依据仿真位置、速度、接触与关节状态，模型声明完成不能替代验收。

任务图版本为 `order-fulfillment-v1`、`drawer-cycle-v1`。阶段事件记录 `id`、`type`、`stage_id`、`time`；`time` 为仿真时间，类型有 `achieved`、`revoked`、`violation`。首次达成附带 `first_achievement`，违规附带 `reason`。维持型阶段失效会撤销；抽屉的 `opened` 是保留的历史前置条件，不要求关闭时仍打开。

每轮保留动作前后特权评价与新增阶段事件，和策略可见观察分开。稀疏奖励 `native-success-transition-v1` 只在有效原生判据首次转为成功时取 1，其余有效转移为 0，评价缺失为 `null`。`robot-decision-measurements-v1` 另外记录：

- `task_success`、`elapsed_wall_s`：成功奖励与真实墙钟耗时。
- `stage_potential_delta`、`first_achievements`：当前阶段完成比例的变化、该轮首次达成次数；阶段撤销时前者可以为负。
- `constraint_violation`：任务规则导致的失败；未经验证的评价仍为未知。

这些分量尚未加权为统一密集奖励。修改奖励权重应注册独立实验协议，不能把不同后端的进度值直接比较。物理成功或明确任务图违规可终止；外部预算截断另记为 `truncated`。MuJoCo 评价代码指纹包含 `server/task_graph.py`，修改任务图后旧协议需重新注册。

## 2. 路由 API 接入与逐轮关联

继续使用现有模型配置中的地址、密钥、模型名称及协议。OpenAI Responses、OpenAI 兼容 Chat、Anthropic、自定义 Agent，以及 HTTP 策略/工具请求，发送当前已记录调用的关联头；不改变官方请求体协议：

| HTTP 头 | 含义 |
| --- | --- |
| `X-EmbodiWorks-schema-version` | `robot-routing-trace-v1` |
| `X-EmbodiWorks-trace-id` | 当前实验 ID |
| `X-EmbodiWorks-decision-id` | `<run_id>:round:<轮次>` |
| `X-EmbodiWorks-request-id` | 本次真实调用的 `call_id` |
| `X-EmbodiWorks-parent-call-id` | 同一实验中已记录的嵌套父调用；无父调用时不发送 |

HTTP 头名不区分大小写。一次决策可以产生多次调用，重试分别留档；实验、轮次、调用不可混作一个计数。平台生成的 ID 不被返回值覆盖，关联头不携带密钥或评价真值。

网关可以在正常响应中附加以下**可选片段**，也支持放在响应的 `metadata` 对象内。仍需返回所选接口要求的动作/内容及真实 `usage`：

```json
{
  "routing": {
    "resolved_provider": "实际服务提供者",
    "resolved_model": "实际执行模型",
    "router_policy_version": "router-v1",
    "selected_capability": "vision_action",
    "reason_code": "visual_manipulation",
    "selection_probability": null,
    "billing_source": "服务端计价来源",
    "usage_completeness": "服务端用量完整性说明"
  },
  "billing": {"scope": "self"}
}
```

字符串字段缺失时保留未知；概率只接受服务明确返回的有限数值 `[0, 1]`，不从置信度文字推算。`requested_model` 保存请求模型，`model` 保存返回/解析后的模型。网关没有返回路由元数据时显示不可用，不假装已经完成智能路由。额外片段不会自动生成网关内部工具的逐次调用树；如服务只返回总用量，归因粒度只能到该服务。

## 3. 费用证据与避免重复计算

`billing.scope` 说明**当前已记录调用**的费用范围，不是支付凭证：

| 值 | 汇总行为 |
| --- | --- |
| `self`（默认） | 当前调用自身费用；已记录子调用另计。 |
| `inclusive` | 当前金额包含同一 trace 中的后代调用；子调用明细保留，汇总不再相加。 |
| `included` | 当前费用已含在祖先账单内；只有能关联到明确的 `inclusive` 祖先时才排除。 |

同一 `(trace_id, call_id)` 合计一次。缺少含子项祖先、关联循环或未知 scope 标为不可完整估算；含子项父调用缺少价格时也不会把整笔费用当零。仅返回 `billing` 元数据不提供真实扣款金额。

金额证据分开显示：API Token 单价计算为 `price_estimate`；Codex 登录的 API 等价金额为 `subscription_reference`，不计入现金成本；人工登记的真实非模型账单为 `actual_invoice`。本地策略没有 API 费不意味着 GPU、存储或人工免费。

通过 `POST /api/runs/{run_id}/expenses` 登记 compute/storage/tool/network/human/other，沿用数量、单位、美元单价、来源和说明，并可补充：

- `evidence_type`：`price_estimate` 或 `actual_invoice`；后者必须提供 `source_invoice_id`。
- `source_invoice_id`：来源方的账单编号；与 `source` 一起确定账单身份。
- `allocation_key`、`allocation_rule`：分摊标识与规则；分摊必须有账单编号和规则。

同一来源、账单和分摊标识合计一次；历史重复中实际账单优先于估算。不同分摊标识可以登记，但**尚未验证分摊之和不超过原账单总额**。同一账单不能同时登记整额和分摊项：新写入会拒绝；历史或父子实验中发现混合范围时，保存全部记录、报告 `invoice_conflicts`，整张冲突账单暂不计入已知合计。核对后先作废重复范围，再登记正确项目。不要把已记录 API 费用再次作为非模型开销登记。

`GET /api/runs/{run_id}/analysis` 的 `cost_attribution` 提供阶段/模型来源、逐次 `routing`、恢复与重试关联、两类金额证据及 `coverage_details`。覆盖信息列出未计价/缺用量的调用、被父费用包含的调用、冲突账单和未计量资源类别。`known_total_usd` 是已记录的已知金额，不是完整实际账单；未知费用不可当零。

## 4. 训练导出与研究有效性

结束后调用 `GET /api/runs/{run_id}/training`，当前返回 `embodied-transitions-v2`；同时下载 `/bundle` 获取图片等资产。批量研究、对比、修复和 HarnessPAI 汇总应选择实际执行动作的子实验导出。

v2 保留观察、动作、后继观察、稀疏奖励、终止/截断、阶段时间、公开调用内容及独立的特权评价，新增或明确输出 `decision_id`、`stage_events`、`reward_components`、`call_cost`、`routing_decisions`。费用范围为该轮已记录模型/工具调用；资源费用仍在实验账本中。

| 标记 | 当前筛选条件 |
| --- | --- |
| `training_eligible` | 前后观察、动作和奖励齐全；无反馈错误；不是 error/cancelled/interrupted；不在首次终止之后。 |
| `routing_training_eligible` | 上述条件满足，至少一条调用有 `selected_capability`，且这些路由记录都有 `router_policy_version`。 |
| `propensity_evaluation_eligible` | 路由训练条件满足，所有这些记录有真实返回且在 `(0, 1]` 内的选择概率。 |

这些标记是数据结构筛选，不证明策略质量、因果可识别性或离线评估无偏。概率缺失时仍保存决策证据，但不冒充满足重要性采样条件。特权评价仅用于奖励、评价或显式允许特权信息的 critic，不混入同条件视觉策略输入。

LIBERO 不能仅靠数字 seed 不同宣称留出集独立：实际初始化使用 `seed % state_count`。研究实验首次 reset 后、任何模型执行前，平台读取官方状态池身份和内容哈希，核验全部 development/validation/test 映射：

- 身份包含任务 ID、官方状态文件 SHA-256、状态数、索引和各状态内容 SHA-256；同索引别名及不同索引但相同内容均可发现。
- 同一划分或不同划分重复初态都会报告冲突；任务身份参与比较，不把不同任务的索引相同当成同一初态。
- `native_split_validation` 保存 `passed`、`conflict` 或 `unverified` 及证据；无法验证或池在运行中改变时停止研究执行。其他后端为 `not_applicable`，不强加 LIBERO 字段。
- 旧检查点仍可读取，但缺少初态来源时标记 `unverified_legacy`，不能据此通过研究验证或晋升。历史记录不补造验证证据。

## 5. 已验证范围与下一步

本轮工程回归：**Python 547 项，475 通过、72 跳过；UI 87 项通过**。跳过项不等于通过，其原生 SDK/环境条件仍需满足。HTTP 路由经过本地 stub 请求、元数据与记录链路验收，不能据此声称真实路由 API 已联调成功，或 LLM 已达到某一任务成功率。

当前没有 RL 训练、自动 GPU 计费或任意内部工具的自动追踪。下一步应先接一个真实路由服务，冻结任务、预算、奖励版本与独立初态划分，重复测量成功率、墙钟时间、已知费用和费用覆盖率，再决定训练目标。IDE 仍按现有文档采用 VS Code 接入方案。
