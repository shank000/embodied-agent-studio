# 仿真环境与任务套件

EmbodiWorks 在同一个工作台选择后端，复用模型接入、实验记录、视频与独立任务验收。每次只加载一套环境；正在执行任务时须先停止，切换会重置当前仿真。

左侧 **仿真环境 → 选择后端 → 加载环境**。目录按适配器及本机独立运行环境显示可用状态，安装后点击 **刷新**。首次 SDK 初始化、编译或资产下载可能较慢，加载结束前请等待结果。候选后端加载失败会保留原有可用环境并报告真实原因，不会静默换引擎继续实验。

## 当前范围

本项目维护范围为 **MuJoCo、Genesis、ALFWorld、AI2-THOR、LIBERO、ManiSkill、Isaac 和 RoboCasa365**，另提供 RoboDojo 官方基准入口。Genesis 是已有后端；其他未适配的仿真栈不列入接入计划。

| 后端 / 任务套件 | 当前实现 | 本次验收与安装说明 |
| --- | --- | --- |
| RoboCasa365 | 官方 65 项原子技能与 300 项复合厨房任务目录；PandaOmron 移动机械臂、原生控制与三相机 | 本机 Windows 的导航、抽屉、柜体抓放与烹饪组合任务样本，以及真实 Worker/Agent/视频回放链路通过；独立 Python 3.11，详见[RoboCasa365 指南](robocasa365.md)。未逐一运行全部 365 任务。 |
| MuJoCo | Panda、SO-101、RM65 + GaiaHand、双/三 Panda；七类场景、三相机、关节/末端控制及任务评价 | 现有完整主后端。Windows 本机、Linux 渲染和三平台 CI 的各自边界见[安装文档](INSTALL.md)。 |
| Genesis World | Panda 桌面抬起、放置、末端到达；Genesis 物理、IK、PD 和三相机 | 本机原生 Windows CPU 物理及主 Worker 集成通过；其他系统待实机验收。独立 `.venv-genesis`，安装见下文。 |
| ManiSkill / SAPIEN | 官方 `PickCube-v1` / `PushCube-v1`、Panda；SAPIEN CPU PhysX、Vulkan 三相机及官方成功条件 | 本机 Windows 六项原生回归与一项真实主 Worker / Agent 集成通过；覆盖视频解码、回放及官方判据拒绝假完成。使用独立 Python 3.11，详见[ManiSkill 指南](maniskill.md)。 |
| LIBERO | 官方五个套件、130 个任务目录；Panda、robosuite 控制器与官方 BDDL 评价 | 本机六项模块回归（五项原生、一项目录）与一项真实主 Worker / Agent 集成通过；任务和初态资产已核对，不将目录数量当作策略求解数量。详见[LIBERO 指南](libero.md)。 |
| ALFWorld | 六类官方家庭任务；TextWorld/PDDL 文字观察与逐轮命令 | 本机 Ubuntu WSL 运行，六类任务正例及完整 Agent/记录闭环通过。预览是文字排版，**没有相机**。详见[家庭环境指南](household-environments.md)。 |
| AI2-THOR | 官方 Unity 房间、第一人称 RGB、导航和物体交互；按原生物体元数据验收 | 本机 Ubuntu WSLg 的真实 SDK 与主 Worker 桥接通过；没有虚构侧视/腕部相机。逐房间和任务的验收边界见[家庭环境指南](household-environments.md)。 |
| Isaac Lab / Newton | 独立 Isaac Lab 3.0、Newton MJWarp、官方 IK/驱动、Panda 桌面与三相机 | 本机 Windows RTX 5060 Ti 的八项原生回归、两项接口契约与一项真实主 Worker / Agent 集成通过；覆盖加载后立即观察、官方初态、实际末端到达、物理验收、视频和回放。详见[Isaac 指南](isaac.md)。 |
| 完整 Isaac Sim | 与 Newton 分开的 Isaac Lab / PhysX / RTX 入口 | 本次未安装或实机验收完整 Sim；需要符合官方资源要求、手动完成许可流程。不会用 Newton 的验证替代完整 Sim 的证明。详见[Isaac 指南](isaac.md)。 |

现有 MuJoCo 机器人与场景不会自动转换到其他引擎。加载后以当前后端返回的模型、任务和能力为准；不适用的控制会禁用或明确拒绝。ALFWorld / AI2-THOR 使用原生离散动作，机械臂 TCP、关节动作块及连续 VLA 接口不适用于它们。

环境切换完成后，任务目录、机器人控制和自动生成的任务草稿随当前会话更新，用户编辑过的指令保留。切换前发出的迟到响应不会覆盖新环境。子进程退出、原生环境重建失败或检查点重放失败时，工作台显示真实原因并锁定操作；在“仿真环境”选择原后端并重新加载可建立新会话。

实验启动确认后立即停止，也会保存一条已停止记录并解除控制占用。尚未调用模型的实验显示“无模型调用”，有缺失信息的历史记录仍保持未知，不据此估算为零花费。

LIBERO 是基于 robosuite / MuJoCo 的任务套件；ManiSkill 使用 SAPIEN 的物理与渲染；Isaac Lab 是可选择物理运行栈的框架。这些层次分别保存在实验身份中，不把任务套件都当作独立物理引擎。

RoboCasa365 基于 robosuite / MuJoCo，环境身份单独记录其任务判据和版本。RoboDojo 则在“官方基准”页管理，由独立的官方 Isaac Sim 客户端运行，使用 XPolicyLab WebSocket 策略；本机未配置完整 Linux GPU 运行时也可导出计划、导入终局结果和视频。详见[RoboDojo 指南](robodojo.md)。

## Genesis 独立环境

主工作台继续使用 Python 3.13 `.venv`；Genesis 的 PyTorch、编译器与渲染依赖安装在 `.venv-genesis`。

Windows PowerShell，在仓库根目录执行：

```powershell
py -3.13 -m venv .venv-genesis
.venv-genesis\Scripts\python.exe -m pip install --upgrade pip
.venv-genesis\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv-genesis\Scripts\python.exe -m pip install --no-compile -r requirements-genesis.txt
```

Linux CPU 对应命令为：

```sh
python3.13 -m venv .venv-genesis
.venv-genesis/bin/python -m pip install --upgrade pip
.venv-genesis/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv-genesis/bin/python -m pip install --no-compile -r requirements-genesis.txt
```

macOS Apple Silicon 使用对应 `bin/python` 路径，PyTorch 命令改为 `python -m pip install torch==2.10.0`，不使用 Linux CPU wheel 索引。以上是安装路径说明，不代表本项目已完成 Genesis 的 Linux / macOS 实机验收。[Genesis 官方安装与平台支持](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)。

安装后正常 `npm start` 启动主工作台，在环境卡刷新并加载 Genesis。本机首次 `Scene.build` 编译约 **241 秒**，缓存后初始化约 **33 秒**；这是单台 Windows 机器的测量，不是加载时间保证。

当前适配使用 CPU 物理和原生离屏相机，相机仍需要可用的 OpenGL 驱动。Genesis SDK 使用 MuJoCo 解析官方 MJCF 资产，但本适配的动力学、IK、PD 和图像来自 Genesis。Windows 中文资源路径会通过未改动官方资产的临时 ASCII 副本加载；不会修改 SDK 源码。

![本机 Genesis Panda 原生仿真](images/genesis.jpg)

## 安装和验证原则

- 各 SDK 放入自己的独立环境；主 `.venv` 不混装可选仿真依赖。家庭环境的 Windows WSL 解释器映射保存在本机 `runtime.json`，不提交个人机器路径或凭据。
- 每次只启动一个重型 SDK。16 GB 内存机器尤其需要关闭其他仿真进程；安装完成、显卡可见和图像实际可用分别验证。
- 后端特定原生测试必须在对应 SDK 环境运行，主环境会明确跳过可选测试。主环境的普通契约回归不能替代原生验收。
- 集成测试只用本地固定响应替代模型推理；物理/文字状态转换、观察、验收、录像与回放使用真实后端。测试使用临时运行目录，不调用收费模型，不改写个人实验。

Genesis 的原生与集成回归示例：

```powershell
.venv-genesis\Scripts\python.exe -m unittest tests.test_genesis_backend -v
$env:EMBODI_TEST_GENESIS='1'
.venv\Scripts\python.exe -m unittest tests.test_genesis_integration -v
Remove-Item Env:\EMBODI_TEST_GENESIS
```

其他后端的完整依赖、资产和回归命令分别见 [ManiSkill](maniskill.md)、[LIBERO](libero.md)、[ALFWorld / AI2-THOR](household-environments.md)、[Isaac](isaac.md)。远程 GPU 仿真服务未随本次部署，不能仅填写地址就获得远程后端。

## 实验可比性与维护

记录保存后端、SDK 版本、任务、评价器和动作约定。模型的“完成”只是验收申请；实际成功由官方任务条件或独立环境状态判据决定。特权对象坐标、PDDL / BDDL 证据和评价结果不会注入策略观察。

连续物理后端用仿真秒，ALFWorld / AI2-THOR 用 `environment_step`；墙钟和推理耗时始终单独计量。分析按环境与评价契约分组，不能把文字动作步与物理秒直接汇总。

方法版本追溯后端代码与依赖；严格评测协议冻结指定环境和评价器。升级 SDK 或控制实现后，需要新方法和新协议；历史记录、视频和协议保留原样。同一个种子只保证各后端自己的生成/初态选择规则，不能据此宣称跨引擎场景或动力学相同。

不同 SDK 的检查点和轨迹不能跨后端恢复。ALFWorld / AI2-THOR 用官方动作重执行并验证最终观察，其他后端以各自保存的原生状态为准，不统一宣称求解器暖启动缓存或逐比特复现。

源码、机器人、场景和数据资产遵循各自上游许可；本仓库的 MIT 许可不覆盖下载的第三方资产，也不代表可以重新分发训练数据或权重。具体来源见各后端文档和上游许可。
