# Harness 与模型配置验收

[使用指南](USAGE.md#harness-与模型分别配置) · [安装指南](INSTALL.md)

## 2026-11-02 实际界面验收基线

验收环境为 Windows、MuJoCo 3.13.0、Franka Panda、桌面场景、布局种子 42。操作从浏览器界面完成，包括配置、保存、关闭后重开、切换、启动、停止和失败恢复。本节覆盖工作台视觉闭环、Codex CLI 与自定义 HTTP harness。

真实模型通过本机 Codex 登录调用 `gpt-6-astra`。其他 API 组合连接本机协议测试服务，使用合成动作和用量；测试服务通过真实 SDK 或 CLI 接收三路相机与状态，再由工作台执行 IK 和 MuJoCo 物理动作。协议测试不代表官方模型能力、账号授权或真实费用验证。

### 12 次界面试验

| 场景 | 次数 | 实际结果 |
| --- | ---: | --- |
| Codex 本机登录，1 秒抬高手臂 | 1 | 调用与动作执行完成；未满足持续保持判据，保留为未通过环境验收。 |
| 工作台 + Responses、Chat Completions、Claude Messages | 3 | 三种协议均保存、切换并执行动作；各请求包含预期模型名、三路图片和正确鉴权。 |
| Codex CLI + 独立 Responses 地址 | 1 | CLI 使用配置的模型地址完成结构化输出，再执行仿真动作。 |
| HTTP harness，由服务管理模型 | 1 | 独立 harness 地址与鉴权生效，模型标识传递正确。 |
| HTTP harness + 独立模型 API | 2 | 默认不发送模型密钥；显式授权后才转交。更换 harness 地址会取消授权。 |
| 无效 JSON 动作 | 1 | 保存原始解析错误，任务没有执行无效动作；界面说明自动诊断处于关闭状态。 |
| 故障暂停后更换 harness 与模型，再重试 | 1 | 保留原实验和第 1 轮，使用新配置恢复执行；失败、诊断及重试均保留，混合 harness 单独统计。诊断同样使用本机故障注入服务。 |
| 延迟响应期间停止 | 1 | 运行中配置锁定；停止后丢弃迟到输出，仿真时间与末端位置没有继续变化。 |
| Codex 本机登录，3 秒抬高手臂 | 1 | 完成末端沿世界 Z 轴抬高 2 cm，环境评价器确认成功；墙钟 24.32 秒，模型调用 20.90 秒。 |

这些检查没有改变环境的误差容限或持续保持要求。单轮结束、正常进程退出和任务成功是不同结果；测试没有把合成协议请求计作模型 benchmark 成绩。

### 已修复的公开行为

- 故障恢复时保存脱敏配置变更；运行状态使用实际模型与 harness，历史调用保持原样。分析根据主执行调用识别混合 harness，不将故障诊断模型当作主 harness。
- 无轮次时显示“共 0 轮”，不再显示“1–0”。
- 已结束调用显示“当前无进行中的推理请求”；从未调用时仍显示“暂无推理请求”。

该基线通过 116 项前端回归、42 项配置／鉴权／恢复／分析回归，以及另外执行的 18 项 Agent 回归；这些检查组可能覆盖相同模块，不应相加成独立案例总数。生产构建和差异检查通过。

## Claude Code 与 dsh 验收

新增接入使用官方 Claude Code **2.1.316** 和 DeepSeek Harness **0.2.1-rc.5**，Node.js 至少 **22.19.0**。测试分为两层：普通回归使用受控子进程检查取消、错误和日志边界；原生协议测试启动真正的官方 CLI，模型地址固定为本机 HTTP 服务，响应、图片和用量均为合成测试数据。

| 验收路径 | 检查内容 |
| --- | --- |
| Claude Code → 本机 Messages SSE | 模型 ID、独立 API Key、三张图片、effort 和输出上限传递正确；仅开放 CLI 的 `StructuredOutput` 机制，返回动作通过工作台 Schema。 |
| Claude Code → 不存在的模型 | 保留官方 CLI 返回的模型／访问错误；流式请求失败后 CLI 自身可转为非流式重试，但不改变模型。 |
| dsh → 本机 Responses、Chat Completions、Messages | 三种官方模型适配器均接收图片、模型、鉴权与推理配置；未向模型开放本机工具，各成功轮次只发出一次模型请求。 |
| dsh → 服务失败 | 保留真实错误；关闭隐式 SDK 重试，失败不会自动换模型或接口。 |
| 两个适配器的受控子进程回归 | 配置隔离、密钥不进入命令行／输出、无效 JSON 拒绝、超时／取消终止子进程、不支持组合提前拒绝、公开文本和实际返回用量归档。 |

Claude Code 的汇总用量与 `total_cost_usd`、`modelUsage` 保存为 CLI 报告值。CLI 可能内部调用多次，汇总输入不足以确定每个请求的长上下文价格档位，因此这些字段不构成逐请求官方单价精确复算或真实账单验证。

### 真实 CLI 协议检查

完成 `npm ci` 后，Windows PowerShell：

```powershell
$env:EMBODIWORKS_CLAUDE_NATIVE_TESTS = '1'
$env:RUN_DSH_RUNTIME_TESTS = '1'
node tools/python.mjs -m unittest tests.test_claude_code_native tests.test_dsh_harness.DshOfficialRuntimeTests -v
Remove-Item Env:EMBODIWORKS_CLAUDE_NATIVE_TESTS, Env:RUN_DSH_RUNTIME_TESTS
```

Linux / macOS：

```sh
EMBODIWORKS_CLAUDE_NATIVE_TESTS=1 RUN_DSH_RUNTIME_TESTS=1 node tools/python.mjs -m unittest tests.test_claude_code_native tests.test_dsh_harness.DshOfficialRuntimeTests -v
```

这组测试不需要 API Key、订阅登录或 GPU，也不访问收费模型；测试服务仅监听 `127.0.0.1`。跨平台 CI 单独启用这些检查，真实供应商账号、视觉任务语义和机器人任务成功率仍以各自实验记录为准。

## 可复用检查方法

```sh
npm run test:ui
npm run test:launcher
npm run build
node tools/python.mjs -m unittest tests.test_agent_harness tests.test_harness_accounting tests.test_harness_analysis tests.test_claude_code_harness tests.test_dsh_harness tests.test_recovery tests.test_recovery_provider_operations tests.test_analysis_simulators -v
```

每个新增 harness 至少检查：合法模型协议、保存后重开、不支持组合的错误、两份密钥的目标与脱敏、实际执行身份、取消与超时、失败恢复、公开输出与用量记录。CLI 测试应区分真实 CLI + 本机协议服务、受控子进程夹具与真实模型账号，分别报告证据。

## 验证范围

本页不声明所有模型、任务或仿真后端已完整通过。远程供应商 API 的真实账号调用、完整任务成功率、长期连续运行、RoboDojo／BEHAVIOR 等独立原生 SDK 验收需要各自的运行环境、凭据与实验记录。具体环境状态见[多后端指南](simulators.md)和[环境完整性记录](ENVIRONMENT_INTEGRITY.md)。

原始测试记录保留在操作者本机，发布版本只包含这份验收摘要和回归代码；不发布认证文件、模型密钥、原始推理转录或个人目录。
