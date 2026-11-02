# RoboDojo 交互仿真与官方基准

RoboDojo 已同时接入“仿真环境”和“官方基准”。交互适配器复用固定版本的官方 `EvalEnv`、布局、双 ARX X5、相机、CuRobo IK、控制器和任务判据，供工作台 Agent、自定义 VLA/WAM 与组合策略驱动；官方基准页保留独立客户端、协议和 `_result.json` 归档。

当前 Windows 电脑可以检查运行条件、选择任务、下载计划和导入已有结果。**完整 Linux GPU / Isaac Sim 5.1 环境尚未提供；本次完成平台适配及接口回归，未完成 RoboDojo 原生画面、动作与任务成功率验收。** 源码或依赖预检通过不会显示成原生实验通过。

RoboDojo 使用自己的 **Isaac Sim 5.1 + 官方定制 IsaacLab / CuRobo** 运行栈，独立管理仿真生命周期。工作台已有的 MuJoCo、Isaac Lab 3 / Newton 等环境不替代它。当前没有自动 SSH 部署；后续 GPU 服务器可运行同一工作台，或执行导出的官方命令后把结果导回本机。

## 当前可用范围

| 功能 | 行为与条件 |
| --- | --- |
| 交互仿真 | “仿真环境 → RoboDojo → 加载环境”；运行条件未满足时展示缺项并禁用加载，API 加载失败保留当前环境 |
| 可视化与双臂 | 原生 head / left_wrist / right_wrist；左、右 ARX X5，关节 / 夹爪与绝对末端动作；原生 Linux GPU 验收待完成 |
| 实验记录 | 复用逐轮输入输出、用量与费用、观察、动作、评价与可选视频；缺失的调用用量仍为未知 |
| 官方目录 | 固定版本的 54 个可运行任务配置、5 个能力维度、双 ARX X5 本体 |
| 评测计划 | 选择任务、种子、外部策略和回合数，下载 JSON 或可移植 Linux 脚本；不会启动 SDK 或模型 |
| 原生执行入口 | Linux 上完成独立依赖和资产配置、静态检查通过后，启动官方评测客户端，保留日志和终局结果 |
| 结果导入 | 无需仿真运行环境；校验并保存官方 `_result.json`，进入现有实验结果与分析 |
| 视频 | 可选复制与该结果一致的本机目录中的官方回合视频；默认不复制 |
| 策略 | 官方 XPolicyLab WebSocket 服务；工作台普通 HTTP / Codex 模型配置不会自动转换成该协议 |

交互适配器当前只提供三路原生相机预设；不支持自由相机、完整物理检查点恢复或物理轨迹重播。因此配对初态比较与冻结协议批量研究暂不启用，单次 Agent、策略与组合实验可用。记录可用于视频回看和轨迹分析，不能据此声称物理重放或官方榜单分数。各环境本次检查结果见[完整性验收矩阵](ENVIRONMENT_INTEGRITY.md)。

## 交互模式的控制与评估

- 场景与任务 ID 使用同一个官方配置，共 54 个。平台种子映射为 `eval_seed = seed % 3`、`layout_index = seed // 3`；超出任务布局池会报错，不循环复用布局冒充新初态。日志记录布局内容摘要。
- 图像直接来自官方三相机。Agent 观察包含机器人自身状态及原生任务指令；终局判据与特权奖励保留在环境评价和实验记录中，不混入策略观察。
- 左臂为 `arm1`、右臂为 `arm2`。TCP 使用单环境世界坐标、米和 `wxyz` 四元数；未选机械臂保持其关节目标。末端动作使用官方 IK 与关节控制；无解会报告失败，不吞掉动作错误。
- 自定义 VLA/WAM 沿用 `embodied-policy-v1` HTTP 接口，关节名须与所选机器人观察一致；每帧为六个弧度关节和一个 0–1 夹爪开度。周期须为观察中 `limits.control_timestep_s` 的正整数倍，固定版本通常为 0.04 秒；例如配置 0.04 或 0.08，而非默认 0.05。默认 SmolVLA 仍只适配 SO-101。
- 任务成功取自官方 `is_episode_end` 与奖励状态；模型声明不能替代验收。任务终止但失败时结束实验，保留失败原因；运行时长和动作数不等同于成功。
- 交互模式由工作台提供动作，不启动 XPolicyLab 策略服务。按官方协议产出评分时，使用下文独立“官方基准”流程。

## 固定版本与源码准备

接入参考版本如下，避免更新子模块后仍沿用旧实验解释：

| 组件 | 固定提交 |
| --- | --- |
| RoboDojo | `f57e1633ffed59761bc8e568cfe7530c4b33381c` |
| XPolicyLab | `5d0be1499634d3176258c8f2e31f5fe816497b0a` |
| RoboDojo IsaacLab fork | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| RoboDojo CuRobo fork | `d17b54ce32cba095c0b000c4c58777075d11de0e` |

在项目根目录预览准备操作，再获取固定源码和 XPolicyLab：

```bash
python tools/setup-robodojo.py --dry-run
python tools/setup-robodojo.py
```

默认放在 `vendor/robodojo`，可用 `--source-root` 指定其他目录。工具保留用户源码修改和已有资产；它不会安装 SDK、下载仿真资产、接受 NVIDIA 许可、修改系统驱动或安装训练模型。固定 gitlink 的子模块使用 `--checkout`，不会追踪 `--remote` 的最新分支。

源码准备会为未修改的官方 Shell 脚本恢复 LF 换行，解决 Windows 检出后在 Linux / WSL 中出现 `pipefail\r` 的问题。存在实际内容修改时会停止并保留文件。

在独立 Linux GPU 机器上，按[官方安装说明](https://robodojo-benchmark.com/doc/usage/install-and-download/)配置 Python 3.11、Torch 2.7.0 / CUDA 12.8、Isaac Sim 5.1.0，以及上表两个官方 fork。官方建议 Ubuntu 22.04、32 GB 内存、16 GB 显存和匹配 NVIDIA 驱动；以固定版本文档及实际兼容检查为准。官方安装脚本包含系统安装、环境初始化和许可相关操作，应先阅读再按该机器条件执行。

完成独立环境安装后，在**实际运行仿真的 Linux 机器**记录已有 Python 的绝对路径：

```bash
python tools/setup-robodojo.py \
  --source-root /opt/robodojo \
  --python /opt/conda/envs/RoboDojo/bin/python
```

路径替换为该机器实际位置。配置写入本项目 `.venv-robodojo/runtime.json`，示例：

```json
{
  "kind": "native",
  "source_root": "/opt/robodojo",
  "python": "/opt/conda/envs/RoboDojo/bin/python"
}
```

该文件不包含 API Key，也不代表 SDK 已通过原生运行验证。页面会检查本机 Linux 平台、固定源码 / 子模块、资产目录和 SDK 版本；静态检查通过后还需运行真实回合。Windows 上填写 Linux 路径不会变成远程执行器；当前 RoboDojo 入口也没有复用其他后端的 WSL 桥。

交互子进程还需 `numpy`、`Pillow` 与 `pydantic>=2,<3`（平台动作协议）。保留官方 SDK 的固定依赖，勿把工作台主环境的全部依赖强行安装到 Isaac 环境。未来服务器上运行同一份工作台并配置上述独立解释器，服务绑定 `127.0.0.1`；本机可通过 `ssh -L 8000:127.0.0.1:8000 user@host` 访问。该方式由服务器承担仿真，页面传输画面和控制；无需将无认证控制端口直接暴露到公网。

## 资产、空间与许可

官方资产来源是 [Hugging Face 的 RoboDojo 数据集](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)。截至本次只读清点，revision `269457a3b42d8d0ae69898a601f2c1487c2f826e` 的 `Assets/` 有 15,365 个文件，下载负载约 **41.27 GB（38.44 GiB）**，不含 SDK、缓存、模型和实验视频。主要为材质约 26.07 GB、物体约 12.62 GB、房间约 1.69 GB。布局、机器人、传感器与轨迹等也应保留官方引用关系。

评测无需下载全部训练数据。官方列出的演示、LeRobot、HDF5、深度和真实机器人数据是额外数据资产；其中部分规模可达 TB，不能与必需仿真资产混为一次默认安装。工作台源码准备工具不下载这些内容。

RoboDojo 代码为 MIT，XPolicyLab 为 Apache-2.0，Hugging Face 数据卡标注 Apache-2.0。SDK、第三方资产和策略权重仍分别遵循各自许可；主项目代码许可不能覆盖它们。此次资产树清点没有发现 `Assets/` 内单独的 LICENSE / NOTICE 文件，使用和再分发前需核对官方说明与具体资产来源。

## 任务与评测语义

原生客户端按官方流程保留视频；工作台作业默认只复制结果 JSON，避免重复占用视频空间。API 计划参数 `copy_videos: true` 可同时归档视频，也可在运行结束后通过“导入官方结果”指定结果目录并勾选复制视频。

官方当前包含 42 个基础计分任务，展开泛化随机变体后共 54 个可运行配置。五个能力维度为泛化、记忆、精密操作、长程任务和开放任务。完整协议跨种子 `0、1、2`：泛化任务使用标准 25 回合及随机变体 25 回合，其他任务每种子 50 回合；完整基础任务 × 种子覆盖为 126 格。

页面的 `native` 使用任务规定回合数；整数回合数适合接线与小规模检查，官方客户端会取请求值与该任务原生数量的较小值。`smoke` 检查官方客户端、结果产物与链路能否完成，**不能证明策略已完成机器人任务**。任务成功由官方环境终局与奖励 / 任务判据提供。单任务或部分回合不会生成“官方完整总分”；本地成绩也没有经过官方云端反作弊认证。

官方入口可用于独立环境检查和手动评测；示例在 RoboDojo 源码根目录运行：

```bash
bash scripts/robodojo.sh dimensions
bash scripts/robodojo.sh tasks
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
bash scripts/robodojo.sh client --task stack_bowls \
  --env-cfg arx_x5 --policy-name demo_policy \
  --policy-host 127.0.0.1 --policy-port 9999 \
  --ckpt external --action-type ee --seed 0 --eval-num 1 --dry-run
```

`doctor` 的跳过选项只检查相应源码 / 配置，不能当作完整依赖验收。去掉 `--dry-run` 前应完成 SDK、资产和策略服务准备。`demo_policy` 是接线示例，不保证任务成功。官方另有 `server`、`eval`、`smoke`、`benchmark` 和 `summarize` 入口；完整协议以[官方快速评测说明](https://robodojo-benchmark.com/doc/usage/quick-evaluation/)及本次固定版本源码为准。

工作台“生成计划”输出官方 `client` 命令的参数数组及源码目录，生成与下载不执行其中内容。运行按钮使用配置的独立 Python 启动同一版本 `src/eval_client/main.py`，单环境顺序执行选定任务和种子；每次调用有独立运行 ID、墙钟超时和日志。策略服务由用户在外部启动，工作台不会隐式启动或付费调用一个普通模型。

在“官方基准 → RoboDojo”选择任务并配置策略后，点击“生成计划”，再点击“生成 Linux 运行脚本”。脚本在已准备好的官方源码根目录运行（或用 `ROBODOJO_ROOT` 指定），支持 `--dry-run` 检查命令，不含本机 Windows 绝对路径。实际运行前需激活官方依赖环境并启动 XPolicyLab 策略服务。脚本核对固定提交、覆盖继承的回合数、为每个任务生成独立运行 ID，并执行整份计划的墙钟上限；产物保留在官方 `eval_result/RoboDojo` 下，可在工作台导入。

修改任务或参数会作废之前的计划和下载链接，须重新生成。界面可设置整次评测墙钟上限和视频复制选项；原生作业显示当前任务、种子和已归档数量。停止、超时或异常退出后仍归档已经生成的有效终局，父作业保留实际停止／错误状态。已归档数量不代表所有回合完成，也不代表机器人任务成功。

## 策略接入边界

RoboDojo 的 [XPolicyLab 固定版本](https://github.com/XPolicyLab/XPolicyLab/tree/5d0be1499634d3176258c8f2e31f5fe816497b0a)使用 **WebSocket + msgpack / NumPy 二进制协议**，不是 OpenAI JSON HTTP。接入模型需要在 XPolicyLab 创建对应策略目录与 `deploy.py`，实现官方的 `reset → update_obs → get_action`；客户端机器即便连接远程策略服务，也需要该目录的部署适配文件。

观察包含官方真实头部、左腕、右腕相机，以及双臂 / 夹爪状态。动作支持 `joint` 或 `ee` 绝对目标；末端姿态采用 `[x,y,z,qw,qx,qy,qz]`，夹爪取值与左右臂字段遵循对应官方配置。官方 CLI 的 `action_type` 是记录标签，必须与策略实际输出保持一致；环境按返回动作字段执行，它不会替模型转换或强制切换动作格式。不要直接将工作台其他环境的 TCP JSON、增量动作、四元数顺序或假三相机套用到这个接口。

优先复用官方 `WsModelClient`、消息 ID / 试次 ID 与图像 codec。协议区分推理请求步与物理控制步；网络重试或服务重启不能被解释为新增一次机器人成功回合。官方图像编码包含兼容版本标记，应由官方解码器读取。

**官方基准流程**不自动把 Codex 登录、普通 API 模型、VLA HTTP 服务或组合策略转换为 XPolicyLab。用户可以在外部策略适配器中接入这些模型，保持官方观察与动作合同；其用量和费用需要显式上报，外部策略未提供的 token、推理时长和费用均为**未知**。**交互仿真流程**可以使用工作台已有模型和 HTTP 策略接口，但不因此生成官方协议成绩。

## 结果归档与分析

每个任务 / 策略 / 本体 / 种子有独立官方 `_result.json`。主要字段单位如下：

| 字段 | 含义 |
| --- | --- |
| `eval_time` | 有效终局回合数；**不是秒数** |
| `success_rate` | 任务成功比例，范围 0–1 |
| `score` | 过程得分均值 × 100，范围 0–100 |
| `details[...].success` | 单回合官方原生成功布尔值 |
| `details[...].score` | 单回合过程分数，范围 0–1 |
| `details[...].layout_id` | 该回合布局编号 |

在页面选择 `_result.json` 或粘贴内容，核对任务 / 种子 / 策略后显式点击导入。读取文件本身不会导入。归档保留原始结果；分析按有效终局 `details` 重算汇总，源汇总不一致时保留原值和差异警告。缺失排除布局、异常尝试与全部分配分母时，不把缺失数据计作零，也不据此猜测额外成功率。

若需保存视频，填写工作台服务器可读取的本机结果目录，并勾选复制视频。该目录必须含与本次导入相同的 `_result.json`。当前每个视频上限 2 GiB、一次导入总量上限 8 GiB；页面播放的是已归档的原生视频。默认不复制视频。单任务 JSON 上限 10 MiB。

记录会进入“实验结果与分析”，可查看终局明细、源汇总差异、覆盖率、原始日志与下载归档。原生运行的任务日志和已生成结果目录在错误 / 停止时保留；未形成终局 `_result.json` 的回合不伪造结果。RoboDojo 原生结果不含模型 token、API 价格或推理耗时，导入记录不能声称零费用、零 token 或零推理时长。工作台单独记录其负责执行作业的墙钟时间。

## Linux GPU 首次验收

上线前按以下顺序取证，并保留固定源码、依赖、模型、种子和完整日志：

1. 完整 `doctor`、平台依赖检查通过。选择“仿真环境 → RoboDojo”，验收三路真实图像及双臂实际状态，不用接口测试图片替代。
2. 分别执行左 / 右臂小幅安全关节、夹爪及 TCP 动作，确认另一臂保持目标；检查无解、过期场景动作及无效周期被拒绝。测试暂停、继续、停止、任务切换与重置资源回收。
3. 使用已有 Agent 或自定义策略跑一回合，核对官方成功与失败、动作 / 仿真 / 墙钟时间、模型调用和视频。验证相同种子布局摘要一致、不同种子按官方池加载、越界不复用。
4. 在“官方基准”单任务 / 单种子 / 单回合启动外部策略，验证真实动作推进和官方终局；分别保留成功与失败，不能只验进程退出码。对照 `_result.json`、视频和平台解析，核验 SR、过程分数、有效回合分母及排除信息。
5. 运行中停止与超时均回收本次进程组，保留已完成日志、视频和结果；重试不复用过往结果冒充本次产物。切回 MuJoCo 验证旧 GPU 资源释放。
6. 先验证多回合与五个任务维度，再覆盖全部 54 配置及官方种子 / 回合协议。只有完整覆盖后才考虑官方协议的汇总与提交；社区榜单认证遵循官方流程。

这些是待具备 GPU 运行环境后的验收项，不是当前已完成实验的声明。

安装完成后可先运行以下真实 SDK 冒烟测试（默认不启用，以免无准备时隐式启动 Isaac）：

```bash
EMBODIWORKS_ROBODOJO_NATIVE_TEST=1 /opt/conda/envs/RoboDojo/bin/python \
  -m unittest tests.test_robodojo_backend.RoboDojoNativeAcceptance -v
```

该测试只覆盖默认任务加载、三相机、一次保持动作、重置和关闭；它不证明机器人解决了任务，完整动作和成功 / 失败验收仍按上述清单执行。
