# HarnessPAI 程序执行、演化与验证

入口：**RSI 实验室 → HarnessPAI 程序验证**。

接入 [Darwin-Agent/HarnessPAI](https://github.com/Darwin-Agent/HarnessPAI) 的官方执行器、`evolution.loop.evolve` 调度、候选复制、图节点经验库与程序保存工具。在本平台的真实 MuJoCo / Franka Panda 上运行程序，通过失败反馈演化程序，再用分离的保留种子验收。当前工作台场景不会被重置；验证使用独立场景、串行运行，不与其他控制任务并发。**不训练、不微调任何模型。**

## 已实现与适用范围

### 官方方法中的训练边界

核对日期：2026-10-27。官方论文 v2（2026-10-21 修订）的主流程冻结动作模型，只演化程序与技能记忆；另有使用成功轨迹监督微调 π0.5 和 DreamZero 的后训练实验。强化学习是文中讨论的扩展方向，该后训练实验并未使用 RL。论文也未把微调后的模型再次送回 harness 反复共同演化。依据：[§4.5 Model Feedback](https://arxiv.org/html/2609.29166v2#S4.SS5)、[§5.6 Post-Training](https://arxiv.org/html/2609.29166v2#S5.SS6)。

本平台当前执行“冻结权重的 harness 适配”，不启动上述微调。锁定上游提交的 `harnesspai/feedback/collect.py` 仍是抛出 `NotImplementedError` 的采集接口占位；它不能证明训练流水线已接入，也不否定论文中报告的训练实验。[锁定版本源代码](https://github.com/Darwin-Agent/HarnessPAI/blob/3af7d561d531ce3e3a3884a3f801c41b8b98e2fe/harnesspai/feedback/collect.py)。

| 能力 | 实际行为 |
| --- | --- |
| 自动任务入口 | 选择场景、任务及模型/工具后，按任务与接口检索程序。已有程序先复测，通过则跳过编码；无匹配或复测失败时进入官方演化流程。 |
| 冻结候选 | 每次保存源程序、SHA-256、任务、种子、预算与上游版本，调用官方 `prepare_candidate_workspace` 创建候选副本。 |
| 固定程序执行 | 官方 `run_rollout` 启动本平台的原生 episode 适配器；每个种子单独启动 MuJoCo。 |
| 跨实验演化 | 官方调度完成改写、入口检查、失败种子探针、全批次验收、自我分析和下一轮改写；每次执行期间代码保持固定。 |
| 模型接入 | 编码、失败分析和经验整理复用平台 Codex / OpenAI / 兼容 API / Anthropic / 自定义接口；URL、模型与密钥可配置，密钥仅在内存保留。 |
| 动作与工具接入 | 动作模型独立配置 VLA/WAM 的 `embodied-policy-v1` 服务；视觉/决策工具使用具名 `embodied-tool-v1` 服务。模型可替换，但接口、动作空间和 Panda 适配必须一致，每次仍会重新验收。 |
| 经验保存 | 全部演化种子与保留种子通过后，调用官方图节点经验整理工具。先在本次副本写入，检查通过后保存不可变版本并更新共享库指针；经验保存失败会单独报告，不覆盖旧库。 |
| 持续反馈 | 程序可反复读取三路相机、机器人状态，执行绝对末端动作，读取真实执行反馈；IK 和底层控制沿用平台。 |
| 独立验收 | MuJoCo 原生任务判据读取物理状态、阈值和持续时间。程序返回值、正常退出均不等同于成功。 |
| 失败分析 | 每个种子保留异常、标准输出、原生判据、逐步观察和动作、耗时、状态快照及可选视频。无结果或超时不能通过。 |
| 候选验证 | 至少两个不同种子全部通过，且代码哈希保持不变，才能保存版本。自动模式完成开发集及最终验证后自动归档；相同程序复用原版本，保留新的实验记录。高级模式可手动保存。 |
| 实验管理 | 汇总进入“实验结果与分析”，可重命名、移入回收站、查看逐种子证据、下载完整实验；已验证程序可重新载入编辑。 |
| 停止 | 正常动作在下一反馈检查点停止并完成日志/视频；不响应的代码在 5 秒后终止进程树，保留已产生文件。强制终止时当前视频可能未完成，不能认定成功。 |

当前验证范围为 **MuJoCo / Panda 的七类平台场景**；程序能否完成某个任务取决于所写策略。附带 `reach_up.py` 是本平台的“末端抬高 2 cm”接口示例，不是上游任务解法，也不证明抓取、抽屉等复杂任务已被解出。改变布局种子会改变场景初态，但这个末端示例不依赖物体布局，其两种子通过仅验证执行与验收链路。

本机尚未部署论文的 π0.5、SAM3、Molmo2/GraspNet 服务。当前是 **官方演化调度在平台 MuJoCo 任务上的适配验证**，不是原论文 LIBERO＋π0.5 结果复现。当前诊断读取原生反馈和异常，不使用原版 Gemini 全视频诊断；任务图使用平台任务节点，不声称完成演示视频图抽取和 VLA 能力评估。编码和诊断调用已接入平台 token、耗时、模型公开输出与费用记录。用户自编程序自行发起的外部请求尚未计量，费用面板不能代表这部分总成本。

## 安装

先按 [安装指南](INSTALL.md)准备主平台 Python 3.13、MuJoCo 模型资产、Node 和 FFmpeg。HarnessPAI 要求 Python `>=3.10,<3.13`，因此另建 Python 3.11 环境，**不要装入主平台 `.venv`**。

已验证上游提交：`3af7d561d531ce3e3a3884a3f801c41b8b98e2fe`。从项目根目录执行：

```sh
git clone --filter=blob:none --sparse https://github.com/Darwin-Agent/HarnessPAI.git vendor/harnesspai
git -C vendor/harnesspai checkout 3af7d561d531ce3e3a3884a3f801c41b8b98e2fe
git -C vendor/harnesspai sparse-checkout set harnesspai configs environments scripts
```

Windows（已经安装 Python 3.11 / Python Launcher）：

```powershell
py -3.11 -m venv .venv-harnesspai
.venv-harnesspai/Scripts/python.exe -m pip install -e vendor/harnesspai
.venv-harnesspai/Scripts/python.exe -X utf8 -m harnesspai --help
```

Linux / macOS：

```sh
python3.11 -m venv .venv-harnesspai
.venv-harnesspai/bin/python -m pip install -e vendor/harnesspai
.venv-harnesspai/bin/python -X utf8 -m harnesspai --help
```

仅安装上游轻量执行核心及其声明依赖；无需初始化大型仿真子模块或下载模型权重。此适配在 Windows 真机执行通过；Linux/macOS 的 HarnessPAI 端到端运行尚待对应机器验证，主平台各系统的既有验证范围保持不变。安装完成后点击卡片内“刷新运行环境”。版本不符、Python 失效或导入错误会显示真实原因，不能显示为可运行。

## 使用

1. 打开 RSI 实验室，选择场景与任务，保留默认“自动任务”。默认任务为抬起红色方块；快速检查链路可选择末端抬高。
2. 配置编码模型。已有程序复测通过时不会调用编码模型；新任务或失败修复才需要它。
3. 按需展开“动作模型与视觉／决策工具”，配置模型名、推理地址与密钥。默认直接使用 MuJoCo IK，不代表已运行 π0.5。服务需遵守[程序接口](HARNESSPAI_PROGRAM_API.md)。
4. 设定开发种子和最终验证种子、预算及视频选项。确认执行本机 Python，点击“开始任务”。无需手写代码或预先注册程序。
5. 查看初始复测、演化批次和最终验证。通过后自动归档，后续同任务会检索复用；失败、停止与归档错误均保留记录。

高级模式保留固定程序验证和从指定程序演化。开发种子（例如 `0,1,2`）与最终验证种子（例如 `42,43`）不得重叠。官方演化轮数预算至少为 5，批次全部通过时提前结束。历史程序的复测不代表重新获得未见种子成绩；研究评估需预先确定未参与开发的种子与版本。

代码修改通过结构化文件替换提交，只允许改候选 `program.py`；平台负责落盘和执行检查，编码模型无需直接写磁盘或调用命令。未返回有效代码修改时停止演化并展示模型说明，避免重复运行原程序。只读诊断不能写文件。经验整理只允许本次经验副本中的 Markdown/JSON。此范围检查用于防止模型误改工作台；执行 Python 仍按下述本机可信代码约定。

编码和诊断会收到最近演化 episode 的初始/最终相机画面、公开机器人状态、动作反馈，以及[完整程序接口](HARNESSPAI_PROGRAM_API.md)。最终保留种子的结果不回传编码模型。自动任务模式会为新任务检索或生成程序；高级固定模式原样执行用户提供的代码。页面区分同一轮内的小批次检查和完整批次，最终验证种子数量单独统计。

程序接口为同步函数：

```python
def run(ctx):
    observation = ctx.observe()  # 原生三路图像和机器人状态，不含评价真值
    pose = dict(observation['tcp'])
    pose['position_m'] = [*pose['position_m']]
    pose['position_m'][2] += 0.02
    pose.update(status='move', gripper_open=1.0, duration_s=1.0,
                explanation='抬高末端')
    feedback = ctx.move_tcp(pose)  # 绝对位姿，米，四元数 wxyz
    ctx.note(feedback['motion']['status'])
```

`ctx.seed` 和 `ctx.task` 提供本轮布局种子与任务标识。`ctx.move_tcp` 受平台动作验证、步数和仿真时长预算约束。`ctx.budget()` 返回已用动作、剩余动作与实际仿真/墙钟预算；动作后的物理稳定阶段也消耗仿真时间，程序应为松手、撤离和验收保持时间预留预算。每个种子的墙钟超时包含环境启动与程序执行；不会把失败动作隐式重试成成功。程序可依据反馈自行重试，此时每次动作单独记录。

Windows 进度文件短暂被读取占用时，写入会在一秒内重试原子替换，期间保留完整旧快照。持续占用或权限问题仍停止并报告真实原因，不跳过记录。执行器出错时页面显示简明原因，可点击“查看完整执行日志”读取完整上下文。

**本机 Python 拥有本机用户权限**：候选副本与独立进程用于防止意外覆盖和串扰，不是文件系统/网络安全沙箱。只运行可信代码，不把密钥写入程序或 `ctx.note`；源代码与日志会进入实验下载包。公开部署应另行使用受限容器和认证，当前入口面向本机研究工作台。

## 记录与复现证据

每条验证记录保存在 `runs/<run_id>/`：

- `program.py`、`job.json`：提交程序、哈希、完整验证配置。
- `candidate/workbench_candidate/program.py`：官方工具创建的运行副本。
- 自动演化中，`program.py` 保留初始程序，`candidate/workbench_candidate/program.py` 是最终冻结程序；`job.json` 同时保存初始和最终哈希。
- `baseline/`、`evolution/runs/round_*/`：初始对照、各代代码差异、探针、全批次执行、失败分析；`evolution/transcript.md` 是官方完整演化记录。
- `model-request.json`、`model-response-*.json` 及标准 `events.jsonl`：模型公开输出与调用遥测，不保存 API Key。
- `memory/skills/`：本次经验副本；通过检查的版本写入 `methods/harnesspai-memory/versions/<run_id>/`，下一次运行读取 `latest.json` 指向的版本。
- `progress.json`、`summary.json`：逐种子结果和最终归档。
- `episodes/seed-*/`：官方执行器 stdout/stderr、原生 `result.json`、`diagnostics.jsonl`。
- `episodes/seed-*/records/<id>/`：标准 `RunLog` 事件、动作前后图像、物理状态、视频。
- `methods/harnesspai/<run_id>/`：通过验证后保存的程序和适用范围清单。

顶层是多种子汇总，不作为单条强化学习轨迹导出；包内每个 episode 的逐步日志使用既有观察/动作/原生稀疏奖励格式。连接统一训练导出与模型服务是后续阶段，不能把这阶段描述为完成强化学习训练。

检查命令：

```powershell
$env:HARNESSPAI_INTEGRATION = '1'
.venv/Scripts/python.exe -X utf8 -m unittest tests.test_harnesspai tests.test_harnesspai_evolution tests.test_harnesspai_automation tests.test_harnesspai_files -v
npm run test:ui
npm run build
```

原生检查覆盖：真实末端动作和视频、两个种子的成功验证、未打开抽屉不能自报成功、异常保留、停止后不启动后续种子。普通测试还覆盖语法/任务/种子检查、代码修改后禁止晋升和界面状态。

## 与原论文复现的区别

平台复用上游真实执行与演化代码，但没有复现论文成绩。该固定提交的 `code_library` 仅提供模板和管理工具，没有发布可直接运行的演化任务解法；`feedback/collect.py` 的采集器仍为未实现占位。平台使用已有记录与原生验收能力补齐反馈。原版主要实验冻结动作模型，模型微调是单独实验，因此当前阶段完全不需要训练。

要复现原版主实验，还需要 Linux 模型服务（π0.5-LIBERO、SAM3、Molmo2，其他任务按需 GraspNet / WAM）、原版仿真和数据、编码与诊断服务及按论文约定执行的任务/种子集。8GB 显存本机用于控制台与轻量调度；原版组合不能因“不训练”就免去推理显存。具体部署沿用[官方环境与服务配置](https://github.com/Darwin-Agent/HarnessPAI#environments)。在这些服务未验证可用前，不显示为原论文评测已就绪。

上游 `pyproject.toml` 声明 MIT；本次检查未发现仓库根目录 LICENSE 文件。上游源代码保留在忽略版本管理的 `vendor/harnesspai`，本仓库仅提供适配层、示例和安装说明。后续分发上游模型、数据或资产前应分别确认其授权。
