# RoboCasa365

RoboCasa365 是基于 robosuite / MuJoCo 的官方厨房任务集。该后端直接加载官方厨房、PandaOmron 移动机械臂、控制器和任务成功判据；不依赖 Isaac，不需要下载训练数据或策略模型权重。

## 固定版本

| 项目 | 版本 |
| --- | --- |
| RoboCasa | 1.0.1，源码 `2ce1aecdd76e29b52d32fdb160436c5a3e230b07` |
| robosuite | 1.5.2，官方 master 源码 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| Python | 独立 `.venv-robocasa365`，推荐 Python 3.11 |
| MuJoCo / NumPy | 上游要求 3.3.1 / 2.2.5 |
| Numba / SciPy | 0.61.2 / 1.15.3 |

使用官方 master 的 robosuite，而不是 PyPI 的同名 1.5.2 包：后者的强制 mink 依赖要求 NumPy<2，与 RoboCasa 固定的 NumPy 2.2.5 冲突。主工作台和其他仿真环境的依赖保持独立。

## 安装与资产

使用仓库中的 `tools/setup-robocasa365.py` 安装器。它将官方源固定到上述提交，建立独立环境，并提供按组安装资产的入口；安装器的 `--help` 列出当前参数。

```shell
python tools/setup-robocasa365.py --stage all --asset-set complete --python /path/to/python3.11
```

`source`、`runtime`、`assets` 可分阶段执行。`kitchen` 资产组用于导航等无任务物品场景；默认抽屉和常用操作任务应选择 `complete`，包含官方默认注册表所需的 Objaverse 和 LightWheel。

Windows 的 MuJoCo 原生资产读取需要英文路径。后端将未改动的官方 Python 包复制到专属缓存，并从那里加载。默认缓存为虚拟环境所在盘的 `EmbodiWorksCache/robocasa365/<源码提交前12位>`；Linux/macOS 默认 `~/.cache/embodiworks/robocasa365/<提交前12位>`。设置 `EMBODI_ROBOCASA_CACHE` 可改变提交目录的父目录。保留缓存中的官方 attribution / license 文件。

资产合并到缓存的 `robocasa/models/assets/`，同时保留源码内的 scenes、arenas 和配置文件。官方上游 Box 下载链接在本次准备时返回 404，因此采用作者和 NVIDIA 的官方 Hugging Face 资源库：

- [RoboCasa 作者资源库](https://huggingface.co/datasets/robocasa/robocasa-assets)：`textures.zip`、`fixtures.zip`；按需安装 `objaverse.zip`、`aigen_objs.zip`、`generative_textures.zip`。
- [NVIDIA LightWheel MJCF 资源库](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF)：`fixtures_lightwheel/*.zip`、`objects_lightwheel/*.zip`。

基础厨房配置包含 88 个官方压缩包，共 2,091,467,204 字节压缩数据。`complete` 加入 Objaverse，共 89 包、4,255,351,925 字节压缩数据；本机已实际下载这些资产，合计解压 10,242,072,049 字节。它不含 AI 生成物体和生成纹理。`NavigateKitchen` 不需要任务物品；抽屉、抓取和许多组合任务需要物品资产。后端保持官方物品采样配置，缺失时报告原始错误和资产目录，不移除物品以绕过任务要求。作者 Objaverse 压缩包为 2,163,884,721 字节；AI 生成物体为 5,773,223,039 字节；生成纹理为 1,184,675,536 字节，后两项不属于默认物品注册表，未下载。下载和解压需要额外空间。

上述文件是仿真资产，不是演示数据集。运行仿真不需要 RoboCasa 人类演示、MimicGen 数据或 VLA 权重。

## 任务与动作

界面目录包含官方 65 项原子任务和 300 项组合任务。目录来源是固定源码中的任务类；其中 48 项组合任务没有 `dataset_registry` 的演示 horizon，目录保留空值。目录数量不表示每个任务已经逐一运行、策略能够完成任务，或本机已安装所有资产。

默认机器人为 `panda_omron`，默认场景为 `robocasa_atomic`，默认任务为 `OpenDrawer`。环境使用官方 target 厨房布局/风格配对、对应物品划分和默认物品注册表。随机种子与实际 layout/style 进入实验记录。

官方 reset 会先稳定物理场景约 0.5 秒，工作台保留这个真实仿真时间，并在状态记录 `initial_time_s`；实验时长按开始后的差值计算。除 `env.rng`，官方 Counter 还用旧 `np.random` 生成碰撞几何颜色，因此隔离进程同步设置 NumPy/Python 随机种子，完整模型 XML 指纹与物理初态可复现。检查点保留物理状态、原生任务/控制器状态、随机状态，以及烹饪颜色、炉火/水流指示等动态模型字段。

target 的十组布局/风格配对直接来自上游 `utils/env_utils.py:create_env(split="target")`。环境 horizon 采用官方任务数据注册表的值；未提供时保留 Kitchen 原生默认 1000 步。交互实验采用 `ignore_done=True`，停止由工作台的轮数、仿真时长和验收设置决定；这不是官方排行榜评测协议，界面实验成功率不可直接声称官方榜单分数。

Agent 绝对末端动作使用世界坐标、WXYZ 四元数，通过官方 OSC 的 `absolute/world` 配置执行，底盘速度保持为零。模型只看到原生三相机、机械臂与底盘状态、当前任务语言；成功判据来自环境特权状态，不发给模型。

当前视觉 Agent 的动作接口控制机械臂并保持底盘。导航任务及需要移动底盘的长程任务应使用完整原生策略接口或对应移动控制，不应把固定底盘的 TCP Agent 当作导航策略。

自定义策略通过 `policy_chunk` 发送 12 维归一化原生命令，`joint_names` 字段承载以下动作名称，**不是关节角或关节位置**：

```text
arm_dx, arm_dy, arm_dz, arm_drx, arm_dry, arm_drz,
gripper_command, base_vx_command, base_vy_command,
base_yaw_command, torso_command, base_mode
```

每维范围 `[-1,1]`。机械臂恢复官方 `delta/base` OSC；底盘为官方速度控制器，躯干为官方增量位置控制器。`base_mode > 0` 时使用官方移动模式，使机械臂目标随底盘参考系移动；`base_mode <= 0` 时使用机械臂模式。上游控制器执行具体缩放。策略间隔为 0.05–0.5 秒，且须为 20Hz 周期的整数倍。物理关节位置、底盘世界位姿、躯干高度作为独立测量状态返回，不与归一化控制混用。

动作完成和任务成功分别记录。`NavigateKitchen` 的官方判据检查底盘距目标不超过 0.20m、朝向余弦不小于 0.98。抽屉任务按官方开度判据验收；模型输出 `done` 只提出验收请求。

## 验收

静态契约测试：

```shell
python -m unittest tests.test_robocasa_backend.RoboCasaContracts -v
```

完成安装后，独占原生 SDK 资源运行测试（PowerShell）：

```powershell
$env:EMBODI_TEST_ROBOCASA365 = '1'
.venv-robocasa365/Scripts/python.exe -m unittest discover -s tests -p test_robocasa_backend.py -v
```

原生测试覆盖构造后立即观察、三路相机、2cm 世界坐标末端动作、底盘及躯干动作、检查点/回放、跨后端拒绝、种子复现和官方终局判据。实际运行结果记录于 `artifacts/validation/robocasa365/`；未运行的测试不视为通过。

本机 Windows / RTX 5060 Ti 8GB / 16GB RAM 的实测结果：

| 验收 | 结果 |
| --- | --- |
| 契约与原生回归 | 11/11 通过，58.473 秒；3 项离线契约、8 项原生测试 |
| 已采样任务 | NavigateKitchen、OpenDrawer、PickPlaceCounterToCabinet、TurnOffSimmeredSauceHeat |
| 真实动作 | 2cm 绝对末端动作、底盘速度、躯干动作、native 结束后夹爪保持 |
| 成功判据校准 | 抽屉关闭→原生完整开度→检查点恢复，官方判据依次 false→true→false；这是判据校准，不是策略成功率 |
| 组合任务回放 | 真实烹饪颜色、任务时间及物理状态恢复通过 |
| 工作台完整链路 | 1/1 通过，40.057 秒；真实 Worker/环境、两轮替代推理、假 done 拒绝、日志、视频完整解码、回放和跨引擎拒绝 |
| 自由视图回归 | 按官方相机外参初始化自由视角，三个操作场景复跑通过，84.112 秒；避免出生朝向变化后初始镜头落在墙外 |
| 长接触历史 | 1000 点、超过50KB的 NumPy 接触历史完整恢复；离线3/3通过，0.203秒 |

原生测试的已缓存启动约16–19秒。独立环境119个包通过 `uv pip check`。记录包括 `backend-final-tests.log`、`workbench-integration.log`、`viewer-final-tests.log`、`contracts-final-tests.log`、`assets-manifest.json` 和原生 JPEG。工作台链路只替换模型推理返回，不替换物理引擎、机器人、控制器、图像或成功判据。365 项目录未逐一完成策略评测。

Windows 上部分上游学习工具会安装自己的顶层 `tests` 包，因此独立环境使用上述 `unittest discover` 命令，避免模块名冲突。机器人不需要可选 GR1/mink、mimicgen 或 robosuite_models；上游对此产生的可选依赖提示不表示 PandaOmron 缺失。

## 来源与许可

[官方安装文档](https://robocasa.ai/docs/build/html/introduction/installation.html)、[固定 RoboCasa 源码](https://github.com/robocasa/robocasa/tree/2ce1aecdd76e29b52d32fdb160436c5a3e230b07)、[固定 robosuite 源码](https://github.com/ARISE-Initiative/robosuite/tree/5ce6643f3092639d08f7b0f90ed1c6a84f50552c)。RoboCasa 代码采用 MIT；官方发布资产和数据为 CC BY 4.0，需保留来源与署名。
