# LIBERO 官方操作任务

LIBERO 是建立在 robosuite / MuJoCo 上的机器人操作基准。本适配器使用官方任务资产、BDDL 目标、Panda 与 robosuite 控制器，不将它标为另一套物理引擎。实验同时保存 LIBERO、robosuite、MuJoCo 版本和可读取的官方源码提交号。

## 安装

原生 Windows 验收使用 Python **3.11**、LIBERO **0.1.0**（官方提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`）、robosuite **1.4.0**、MuJoCo **3.2.7**、NumPy **1.26.4** 和 CPU PyTorch **2.5.1**。主工作台的 Python 3.13 环境保持独立。需要 Git 和可用的 OpenGL 显卡驱动，不需要另启工作台服务。

在仓库根目录运行 PowerShell：

```powershell
py -3.11 -m venv .venv-libero
.venv-libero\Scripts\python.exe -m pip install --upgrade pip
.venv-libero\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv-libero\Scripts\python.exe -m pip install --no-compile -r requirements-libero.txt
.venv-libero\Scripts\python.exe -m pip check
```

依赖清单从固定的官方 Git 提交进行 editable 安装。该官方版本使用 namespace 包，清单已将 `editable_mode=compat` 附在对应的 requirement 上，避免现代 setuptools 生成不包含 LIBERO 的空 editable finder。不要用 PyPI 上名称相同的其他维护分支直接替代该固定版本。

官方源码自带约 **422 MB 任务资产和 12 MB 初态**，本适配不下载大规模演示数据。Git / pip 还需要下载 robosuite、PyTorch 及其依赖，实际磁盘占用高于资产大小。在工作台的“仿真环境”卡刷新并加载 **LIBERO**，再选择官方任务套件和具体任务。

Linux 可在独立 Python 3.11 环境中使用同一清单，但本轮完整适配验收在原生 Windows 上完成。Linux 的 EGL / OSMesa、macOS 和 WSL 完整流程尚未在本项目验收；上游支持与当前项目实测范围应分别判断。[LIBERO 官方安装](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html)、[robosuite 官方安装与 Windows 条件](https://robosuite.ai/docs/installation.html)。

## 任务、控制与评价

目录收录官方五个套件，共 **130 个任务**：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 各 10 项，以及 LIBERO-90 和 LIBERO-10。任务编号如 `libero_goal:0` 使用官方原始顺序；该任务是打开柜子的中间抽屉。初始化选择官方 `.pruned_init` 中的 `seed % 初态数`，相同任务、版本与种子选取相同初态。相同数字种子不代表跨引擎初态等价。

三路观察分别来自原生 `agentview`、`frontview`、`robot0_eye_in_hand` 相机，对应工作台的 front / side / wrist，提供同步图像和真实相机外参。策略观察包含机器人状态，不包含物体特权坐标、检查点或 BDDL 判据结果。

Agent 的绝对世界末端位姿交给官方 `OSC_POSE` 控制器，由它计算关节力矩；该路径没有可伪装成 IK 输出的关节目标。直接策略与手动关节目标经位置反馈映射到官方 `JOINT_POSITION` 控制器的归一化关节增量。原生控制频率 **20 Hz**，策略动作间隔为 **0.05–0.5 秒**，单块不超过 50 帧。夹爪使用官方二值速度命令逼近请求开度，动力学与工作台原有 MuJoCo Panda 模型不保证相同。

完成判据始终调用官方 `_check_success()`，并记录各项官方 BDDL 谓词从当前仿真状态得到的结果。模型的“done”只发起验收；抽屉未打开时，完成声明会被拒绝。该适配保留 LIBERO 的原始瞬时谓词标准，不另加或替换为自定义“抬升完成”等目标。[官方任务说明和许可](https://github.com/Lifelong-Robot-Learning/LIBERO)。

支持轨迹、可选视频、同任务状态回放、检查点、自由相机和暂停/继续。检查点限定相同后端身份、版本、场景和任务；跨引擎回放会明确拒绝。演示抓取、随机投放、物理参数开关及中文快捷命令不在当前 LIBERO 能力集合中，按界面实际可用项操作。

## Windows 的原生 SDK 配置

启动逻辑只在隔离进程中执行，不修改 SDK Python 源码、控制器公式、任务 XML 或 BDDL：

- **中文资产路径与路径长度**：MuJoCo 3.2.7 的 Windows 原生 mesh 加载对本项目中文目录报告文件不存在。适配器将未修改的官方 LIBERO 和 robosuite 包复制到短 ASCII 缓存 `%TEMP%\ew-libero\<版本标识>`，优先从该缓存导入。短前缀也避免官方长任务文件名超过 Windows 原生路径长度。初次需额外约 650 MB 缓存；后续复用。`TEMP` 必须是可写的英文路径。源码提交号会从原始 Git checkout 读取并保留在缓存元数据中。
- **MuJoCo DLL**：按 robosuite 官方 Windows 安装要求，若 `robosuite\utils\mujoco.dll` 缺失，复制同一个独立环境内 `mujoco\mujoco.dll` 到该 SDK 的 `utils` 目录。使用 ASCII 缓存时，目标位于缓存内；不从系统其他位置查找 DLL，也不覆盖已有文件。
- **OpenGL 和日志宏**：robosuite 1.4.0 在包初始化时提前导入渲染绑定，会在 Windows 强制选择不受该平台支持的 EGL。适配器先加载官方 `macros.py`，设置公开配置 `MUJOCO_GPU_RENDERING=False`、`FILE_LOGGING_LEVEL=None`，再由 SDK 使用 GLFW/OpenGL。这里关闭的是 Linux EGL 的强制选项，相机仍需要图形驱动；没有用绘图占位图替代原生渲染。SDK 日志由隔离进程标准错误收集。

缓存缺失会从原安装重新准备。升级依赖后应重新验收，运行中的实验不会自动升级引擎。若源 checkout 自身被删除，需重装固定官方来源，不能把缓存当作独立安装包。

## 验收与许可

2026-10-24，本台原生 Windows 环境通过 **6 项模块回归（1 项目录、5 项原生 SDK）和 1 项工作台集成验收**。集成验收实际保存并用 FFmpeg 解码了轨迹视频，确认错误完成声明被官方抽屉判据拒绝，且日志、回放与跨引擎隔离正常。Linux / macOS / WSL 不计入此次 LIBERO 实机验收。

在仓库根目录运行：

```powershell
$env:EMBODI_TEST_LIBERO = '1'
.venv-libero\Scripts\python.exe -m unittest tests.test_libero_backend -v
.venv\Scripts\python.exe -m unittest tests.test_libero_integration -v
Remove-Item Env:EMBODI_TEST_LIBERO
```

原生回归覆盖实际三相机、末端控制、关节策略、官方初态复现、检查点与回放，以及使用真实抽屉关节构造的正/负 BDDL 评价状态。正状态是在验证评价器，并不表示模型学会了打开抽屉。130 项任务的 BDDL 和官方初态均已核对存在，未将逐项资产检查表述为 130 项任务策略求解成功。

集成测试仅用本地固定响应替代模型推理，实际运行主 Worker、隔离 SDK、原生控制器、录像与回放；它先移动末端，再故意声明抽屉任务完成，预期环境拒绝此误报。检测到 FFmpeg 时会保存并解码视频。普通主环境测试会明确跳过这些可选原生验收，不把跳过计作通过。

LIBERO 主代码为 **MIT**，官方演示数据标为 **CC BY 4.0**。本仓库的官方任务名元数据保留其 MIT 许可和来源。外部物体、纹理和派生资产应遵循各自原始来源的许可；主代码许可不自动覆盖所有资产。本项目通过安装固定上游源码使用资产，不将其或演示数据复制发布为本项目的新资产包。[固定上游源码与 LICENSE](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)。
