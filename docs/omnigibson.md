# BEHAVIOR / OmniGibson

工作台通过独立进程接入原生 OmniGibson 场景、R1Pro 移动双臂、整机控制动作、相机和 BEHAVIOR 的 BDDL 评价。它使用 Isaac Sim / PhysX / RTX，不使用已有的 Newton 或 MuJoCo 环境替代运行。

当前集成的范围是交互研究、平台实验记录及独立官方评估器桥接。**本次未安装或运行原生 SDK，不能将接口测试视作官方任务已经通过。** 环境卡分别显示准备检查、接口能力和原生验收状态；没有资源或资产时会保留当前环境，并给出缺少的准备项。

## 固定版本与运行条件

| 组件 | 本适配的固定版本 |
| --- | --- |
| BEHAVIOR-1K 源码 | `v3.9.4`，提交 `fe66cc74538f7eba3c57255da0ee88e8ce113c87` |
| Python | 独立 Python 3.11；可使用官方 Conda 环境 |
| OmniGibson 包元数据 | 该源码标签仍声明 `3.9.3`，所以同时检查 Git 提交与导入路径 |
| Isaac Sim | 5.1.0 / 5.1.0.0 |
| BEHAVIOR 资产 | `behavior-1k-assets`，版本 3.9.0 |
| 机器人资产 | `omnigibson-robot-assets`，版本 3.8.2 |

不要直接跟随 `main` 安装到主工作台 `.venv`。上游 `main` 的 Python / Isaac Sim 依赖可能不同；该固定版本也不自动等于特定官方挑战的获准提交版本。

官方安装文档支持 Windows 和 Ubuntu。工作台按 Isaac Sim 5.1 的最低配置预检 **32 GB 系统内存、16 GB NVIDIA 显存**；显卡还必须满足 RTX / RT 核心和驱动要求。内存预检使用操作系统可见容量，给 32 GB 系统预留硬件保留差额。GPU 驱动、首帧渲染和实际任务仍需要原生验证。

本次开发主机为 Windows 11、约 16 GB 内存、8 GB 显存，不满足上述准备条件。资产压缩包约 31.5 GB，还需为解压、Isaac Sim、缓存和实验录像准备空间；不要仅按压缩包大小分配磁盘。平台不会自动下载大体积依赖或自动接受条款。

来源：[上游安装文档](https://behavior.stanford.edu/getting_started/installation.html)、[固定版本安装脚本](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/setup.ps1)、[Isaac Sim 5.1 要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)、[官方资产](https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/main)。

## 安装与本机注册

先在有足够资源的主机上使用官方安装器，阅读并自行接受所需许可。以下示例路径需替换为自己的磁盘目录，不要在现有工作台依赖环境中安装。

Windows PowerShell：

```powershell
git clone -b v3.9.4 https://github.com/StanfordVL/BEHAVIOR-1K.git E:\simulation\BEHAVIOR-1K
Set-Location E:\simulation\BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Dataset
```

Linux：

```sh
git clone -b v3.9.4 https://github.com/StanfordVL/BEHAVIOR-1K.git /data/simulation/BEHAVIOR-1K
cd /data/simulation/BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --dataset
```

如需将资产放在其他盘，按上游说明设置 `OMNIGIBSON_DATA_PATH` 后再安装。该目录下应包含 `behavior-1k-assets`、`omnigibson-robot-assets`，以及安装器准备的任务实例。配置工具使用同一个数据根目录。

固定 SDK 的预采样加载器从 `2026-challenge-task-instances` 读取任务；旧版 `behavior-1k-assets/scenes` 中的同名模板不能替代它。若安装器尚未准备该目录，在**上游独立环境**中使用官方下载入口：

```sh
python -m omnigibson.utils.asset_utils --download_2026_challenge_task_instances
```

此命令会下载数据并导入原生 SDK，须在满足运行条件、已完成上游许可流程的主机上执行；工作台目录检查不会代为运行它。来源：[固定版本资产工具](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/OmniGibson/omnigibson/utils/asset_utils.py)。

回到**工作台仓库根目录**，用工作台 Python 注册官方运行环境，以下以 Windows 为例：

```powershell
node tools/python.mjs -m tools.omnigibson_setup `
  --python C:\path\to\miniconda3\envs\behavior\python.exe `
  --source-root E:\simulation\BEHAVIOR-1K `
  --data-path E:\simulation\behavior-data
```

Linux 的参数相同，将解释器改为 `/path/to/conda/envs/behavior/bin/python` 等实际路径。带空格的路径请加引号。

注册命令检查解释器版本、包元数据、SDK 来源路径、Git 提交和 GPU 容量，不导入或启动 OmniGibson。成功检查的路径记录在被 Git 忽略的 `.venv-omnigibson/runtime.json`。硬件或资产仍不足时会显示具体未就绪项，退出码为 2；这不是原生任务验收结果。

只读检查当前配置：

```sh
node tools/python.mjs -m tools.omnigibson_setup --check
```

`--check` 不改写配置。修复包版本或路径后，重新提供三个路径和 `--replace` 更新准备证据；原配置保存为 `runtime.previous.json`。工具不会覆盖未知格式配置或已有备份。运行时正式加载前还会重新核对依赖与 GPU，不只信任历史标记。

在网页 **仿真环境 → BEHAVIOR / OmniGibson** 中查看准备项。全部满足后可点击加载。首次着色器编译可能较慢；只运行一个重型仿真后端，加载和实验过程中不要修改 SDK 或资产。

## 任务、动作与独立评价

- 任务目录从已安装的官方预采样模板发现可选的场景／任务组合，保留官方示例说明。目录数量表示找到的定义或模板，不表示全部任务原生验收通过。
- 布局种子在此后端选择预采样实例索引；超出实例范围明确报错，不取模复用初态。任意文字任务、任意房间或任意机器人不保证兼容。
- 默认机器人是官方 R1Pro。普通 Agent 使用后端声明的整机动作向量、维度、范围和控制组；自定义 VLA/WAM 也需要输出相同的原生动作合同。末端 TCP 或其他机器人动作不能直接传入。
- 使用真实原生控制器和物理抓取；不会在执行失败时改用直接设置状态／位置的符号动作原语。若后续增加 assisted 或 symbolic 模式，应单独记录和对比。
- 完全成功读取官方 BDDL 目标逻辑，部分完成度 Q 单独保存。模型说“完成”只触发验证；未满足目标不能记为成功。官方分数、平台成本和墙钟时间分别记录。
- 模型只能读取声明的相机与机器人观察。用于验收的目标谓词、物体真值和成功证据不自动输入视觉策略。
- 当前物理状态恢复及配对研究以后端能力声明为准。只记录配置／状态摘要不代表完整检查点；涉及物体生成、切割和粒子的任务尤其不能宣称逐比特恢复。

| 平台功能 | 当前接入范围 |
| --- | --- |
| 普通 Agent、自定义 VLA/WAM、单次组合决策 | 使用整机原生动作协议；模型地址、密钥与模型名称沿用现有配置 |
| 默认 SmolVLA | SO-101 映射不适用于 R1Pro，须配置匹配本体的自定义策略 |
| 观察、录像与训练导出 | 动态 RGB/深度、本体状态、动作、原生奖励及终止/截断证据；没有原生运行时不产生物理成绩 |
| 配对 RSI 实验、完整物理回放 | 尚未验收原生状态恢复，因此禁用需要同一物理初态恢复的流程 |
| HarnessPAI 程序验证 | 原有 MuJoCo/Panda 适配范围不变，本次未将其原语移植到 R1Pro |
| BEHAVIOR 官方评测 | 独立原生评估器与外部 WebSocket 策略；不将部分任务结果当作整个挑战分数 |

训练导出的 `reward` 继续使用已有的成功跃迁稀疏奖励合同。原生 Q 与 `native_reward` 保留在独立评价证据中，其中 `native_reward` 是最后一个 SDK 步的返回值，不是一个多步动作块的累计奖励；训练器不能将其直接当作整个决策轮的奖励和。当前没有隐含的 Q 奖励塑形、权重训练或物理回放。

来源：[Environment API](https://behavior.stanford.edu/omnigibson/environments.html)、[BDDL](https://behavior.stanford.edu/behavior_components/behavior_tasks.html)、[官方评价协议](https://behavior.stanford.edu/challenge/evaluation.html)、[状态保存限制](https://behavior.stanford.edu/tutorials/save_load.html)。

## 验收与证据边界

无 SDK 的接口回归：

```sh
node tools/python.mjs -m unittest tests.test_omnigibson_runtime tests.test_omnigibson_backend tests.test_environment_registry -v
node --test tests/test_simulators_ui.mjs
```

原生验收必须在已准备的运行环境执行，且记录源码版本、资产版本、GPU、场景任务及实例索引。至少覆盖：

1. 原生首次加载、首帧、实际相机和本体状态；相机图片能保存并回读。
2. 原生动作产生可观测物理状态变化，非法维度／越界动作不会污染状态。
3. 完整任务正例与未完成负例，Q 与 BDDL 完全成功分别记录。
4. 失败、取消、预算耗尽和进程错误，正确区分终止与截断。
5. 一轮真实 Worker → Agent/策略 → 动作 → 评价 → 日志／视频 → 训练导出。
6. 反复加载与退出后释放进程和显存；画面失败不影响历史记录访问。

本次资源不足，上述原生条目仍为待验收。流体、布料、热状态及物体转换任务应在刚体任务链路通过后逐类验收；不能用接口模拟测试替代物理结果。

## 远程部署与官方基准

Windows 并非必须作为仿真主机。可在有 RTX GPU 的 Linux 或 Windows 工作站上部署同一工作台与 OmniGibson，浏览器继续在本机访问。Linux 服务器可通过 SSH 端口转发访问后端；当前没有“只填写任意远程地址即可获得仿真”的通用 RPC 服务。

交互研究之外，平台还提供独立的 **BEHAVIOR 官方评估器** 入口：调用固定 `v3.9.4` 的 `python -m omnigibson.eval.eval`，通过上游 WebSocket 协议连接用户的策略服务，使用上游默认 R1Pro、观察封装与评测器。普通 HTTP 大模型接口不能直接填作该 WebSocket 服务。启动前先释放交互中的重型仿真，使用同一台机器逐实例串行运行，`num_envs=1`。

固定版本的官方 [`r1pro.yaml`](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/OmniGibson/omnigibson/eval/r1pro.yaml) 使用 **`grasping_mode: assisted`**，不归一化动作，默认 RGB 为 1080×1080。评测桥保留该配置，并在运行计划和实验后端元数据中记录抓取模式。工作台交互环境使用 `physical` 抓取，两者不能混合统计为同一物理控制条件的成绩。外部导入 JSON 未提供抓取配置时，模式保持未知。

该入口需要额外安装固定版本上游的 Eval／JoyLo 依赖及 `2026-challenge-task-instances`。启动前检查 Python、SDK、固定源码、GPU、任务元数据和评测依赖；未就绪时仍可预览计划或导入结果，但不会启动假仿真。`public_test` 使用索引 0–19，对应固定版本原生实例 ID 301–320；`train` 则直接使用训练实例 ID。未开放隐藏测试实例。

以下 `behavior-eval.json` 示例可用于工作台主 Python 的同一 CLI：

```json
{
  "task_name": "turning_on_radio",
  "instance_indices": [0],
  "mode": "public_test",
  "policy_host": "127.0.0.1",
  "policy_port": 8001,
  "policy_scheme": "ws",
  "policy_name": "my-harness",
  "num_rollouts": 1,
  "max_steps": null,
  "record_video": true,
  "max_wall_time_s": 3600
}
```

```sh
node tools/python.mjs -m tools.behavior_benchmark plan --spec behavior-eval.json
node tools/python.mjs -m tools.behavior_benchmark run --spec behavior-eval.json
node tools/python.mjs -m tools.behavior_benchmark import --result turning_on_radio_301_0.json --policy-name my-harness --mode public_test
```

`max_steps: null` 保留上游按演示长度计算的默认预算；设置具体值会显式传递 `--max-steps`。CLI 运行时 Ctrl+C 会停止本次拥有的进程树并保留已生成结果。Windows 使用独立 Job Object，Linux 使用私有进程组；不按进程名称批量结束应用。结果、原生视频、实际命令、诊断输出与总墙钟保存在本次 `runs/<run_id>/benchmark/`，同时进入实验结果管理。取消、超时或进程错误产生的有效部分结果也会保留；进程退出码为零但缺少结果不算完成。

结果导入接受单份或列表形式的上游逐回合 JSON，保留 `task / instance_id / rollout_id / success / q_score / time / agent_distance` 等原始字段。Q 使用上游报告值；工作台给出的平均 Q 和成功率只覆盖本次实际收到的回合，不将缺失回合算为零，也不伪造整个挑战总分。外部策略没有上报的 Token、推理耗时和费用保持未知；导入结果的执行来源未经工作台独立验证。

固定 `v3.9.4` 与官网挑战说明的 `v3.9.3` 存在版本差异，本入口不宣称官方榜单合规或认证成绩。正式提交还需与上游确认版本和提交规则。自定义成本奖励与训练实验结果不替代官方评测。

评测桥契约测试（真实轻量子进程，未加载 SDK）：

```sh
node tools/python.mjs -m unittest tests.test_behavior_benchmark -v
```

来源：[固定评测 CLI](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/OmniGibson/omnigibson/eval/eval.py)、[上游结果生成](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/OmniGibson/omnigibson/eval/evaluator.py)、[固定实例与控制约定](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/OmniGibson/omnigibson/eval/utils/eval_utils.py)。

## 许可

BEHAVIOR 源码的 MIT 许可与场景／物体数据许可独立。官方 Data Bundle 条款限制非商业学术研究及 OmniGibson 内使用，禁止再分发密钥与资产。本仓库提供适配代码、配置工具和文档，不包含数据包或解密密钥。Isaac Sim 也有单独条款。

来源：[代码许可](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/LICENSE)、[官方安装器中的数据条款](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.4/setup.sh)。
