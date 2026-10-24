# ManiSkill / SAPIEN 官方操作任务

该适配器实际加载 ManiSkill **3.0.1** 的官方任务，使用 SAPIEN **3.0.3** 的 CPU PhysX 与 Vulkan 相机。它没有把现有 MuJoCo 场景改名为 ManiSkill，也没有将“目录中存在任务”当作模型求解成功。

## 独立安装

本次原生 Windows 验收使用 Python **3.11**、CPU PyTorch **2.10.0**。不要安装到主工作台的 Python 3.13 `.venv`。

在项目根目录运行 PowerShell：

```powershell
py -3.11 -m venv .venv-maniskill
.venv-maniskill\Scripts\python.exe -m pip install --upgrade pip
.venv-maniskill\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv-maniskill\Scripts\python.exe -m pip install --no-compile -r requirements-maniskill.txt
.venv-maniskill\Scripts\python.exe -m pip check
```

需要可用的 Vulkan 图形驱动。CPU 物理不等于可以省略原生图像渲染条件；本适配明确选择 `sim_backend=physx_cpu`、`render_backend=gpu`，GPU 用于 Vulkan 图像，未开启批量 GPU PhysX。

官方任务和 Panda 资产来自已安装的 SDK，本次没有下载训练轨迹或策略权重。安装后在工作台的“仿真环境”卡刷新并加载 **ManiSkill / SAPIEN**。Linux 可在独立 Python 3.11 环境中使用同一清单，但本轮完整适配验收在原生 Windows 上完成；不要直接复用 Windows 虚拟环境。[官方安装与平台表](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)。

## 当前任务和控制契约

| 场景 / 任务 ID | 官方任务 | 独立成功条件 |
| --- | --- | --- |
| `pick_cube` | `PickCube-v1` | 官方 `evaluate()` 确认方块距目标不超过 2.5 cm、机械臂静止，再连续保持 0.3 秒。 |
| `push_cube` | `PushCube-v1` | 官方 `evaluate()` 确认方块进入目标区并保持在桌面，再连续保持 0.3 秒。 |

机器人为官方 Panda。模型观察是原生 front / side / wrist 三路同步图像、相机标定和机器人状态，不包含任务特权对象坐标或官方评价字典。评价器单独读取 `env.evaluate()`，记录官方条件、有限状态检查和持续时间；模型声明完成不能替代这些证据。

绝对末端动作遵循世界坐标、米和 WXYZ 四元数，交给 **PyTorch Kinematics** 的受限伪逆 IK，再通过官方 `pd_joint_pos` 控制器、PhysX 驱动及末端反馈执行。IK 使用未改动的官方 Panda URDF；这不是声称 SAPIEN SDK 提供了本平台当前的 IK 求解器。单轮平移限制 12 cm、旋转 35°、最大关节变化 0.65 rad，不可达目标会报告原因。

策略块使用七个关节弧度和一个工作台夹爪开度值，当前最多 50 帧，动作间隔从控制周期到 0.5 秒，执行时限制关节速度。策略输出的关节名称与当前观察提供的顺序必须一致。默认 SmolVLA 的原有 SO-101 适配不自动变为 Panda 策略；Panda 应选择对应的自定义策略服务。

支持暂停/继续、单步、关节控制、末端动作、自由相机、检查点、轨迹与视频。恢复检查点需匹配引擎、SDK/运行依赖版本和状态格式；跨引擎回放会拒绝。其他机器人、复杂 MuJoCo 场景、演示脚本、随机投放和物理参数开关不在当前适配范围，以界面返回的能力集合为准。

## Windows 中文路径

SAPIEN 的原生资源读取在本机中文工程目录下出现过“文件不存在”。适配器在初始化阶段将官方 `mani_skill` 包和 SAPIEN Vulkan shader 复制到 `%TEMP%\embodiworks-maniskill-*` 的独立 ASCII 临时目录，使用 SAPIEN 提供的 shader 搜索路径入口后加载；关闭时清理自己的目录。官方 URDF、网格、shader、控制器及 SDK Python 源码均不修改。

`TEMP` 必须指向可写的英文路径，并留出官方 SDK 副本所需空间。图形驱动或 Vulkan 本身缺失时仍会返回真实错误，ASCII 路径处理不会替代渲染器。

## 验收范围

运行真实 SDK 回归：

```powershell
.venv-maniskill\Scripts\python.exe -m unittest tests.test_maniskill_backend -v
$env:EMBODI_TEST_MANISKILL='1'
.venv\Scripts\python.exe -m unittest tests.test_robot_backends_integration -v
Remove-Item Env:\EMBODI_TEST_MANISKILL
```

本机六项原生检查通过：关节驱动与重力、三路不同的同步真实相机、IK 移动和过期动作拒绝、官方成功条件及持续时间、检查点和跨后端拒绝、PushCube 场景与种子/策略限制。成功判据正例使用特权测试准备设置目标，用于验证评价器，并不表示某个模型已学会抓取或推动。

真实主 Worker / Agent 集成一项通过（41.624 秒）：固定响应先将末端抬高 2 cm，再声明 PickCube 完成，官方判据必须拒绝这个误报。测试实际保存轨迹与视频、使用 FFmpeg 完整解码、恢复记录状态并拒绝跨 MuJoCo 回放；仅模型推理由本地响应替代，没有调用外部收费模型。

主环境测试明确跳过未安装在主解释器中的 SDK 项目；跳过不计为通过。本轮证据对应单环境 CPU PhysX，不代表 Linux GPU 并行物理、所有 ManiSkill 任务或所有相机后端均已验收。

任务代码和 Panda 资源来自 [ManiSkill 官方仓库](https://github.com/mani-skill/ManiSkill)。代码与可下载资产具有不同许可范围，尤其数据/资产的非商业条款不能由本仓库的源码许可替代。
