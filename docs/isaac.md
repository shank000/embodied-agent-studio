# Isaac Lab / Newton 与完整 Isaac Sim

平台提供两个明确分开的入口：

| 入口 | 原生运行栈 | 资源与验收 |
| --- | --- | --- |
| Isaac Lab / Newton | Isaac Lab 3.0、Newton 1.6.1、官方默认 MJWarp 求解器、Warp 相机渲染 | 独立 `.venv-isaac`；本台 Windows RTX 5060 Ti 已通过 8 项原生回归、2 项接口契约与 1 项真实主 Worker / Agent 集成。 |
| Isaac Sim / PhysX | Isaac Lab 3.0、Isaac Sim 6.2、PhysX、RTX 相机 | 独立 `.venv-isaacsim`；本次不安装完整 Sim，不会自动接受 NVIDIA 许可。需要符合官方要求的工作站后实机验收。 |

Isaac Lab 3.0 正式支持不启动 Kit 的 Newton 路线。它使用官方 USD 导入、动力学、PD 驱动、DifferentialIKController 和相机组件；Newton 的默认 MJWarp 求解器是 MuJoCo-Warp，这一事实保存在实验元数据 `compute_backend=newton_mjwarp` 中。已有 MuJoCo 工作台的物理世界、对象与求解代码不会替代该入口。

完整 Sim 入口必须实际启动 `isaacsim_physx` 与 RTX；没有安装时返回安装错误，不会改用 Newton。当前 16 GB RAM / 8 GB VRAM 电脑低于完整 Isaac Sim 的官方最低配置，不能用 kitless 验证代替完整 Sim 的运行证明。

## 本地 Newton 环境

Python 3.12，在仓库根目录运行：

```powershell
py -3.12 -m venv .venv-isaac
.venv-isaac\Scripts\python.exe -m pip install --upgrade pip
.venv-isaac\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
.venv-isaac\Scripts\python.exe -m pip install -r requirements-isaac.txt
.venv-isaac\Scripts\python.exe -m pip check
```

Linux 可用同一依赖文件，创建虚拟环境后将 `Scripts\python.exe` 换为 `bin/python`。这不是 Linux 已完成本项目实机验收的声明。需要 CUDA 13.0 兼容驱动与可用 NVIDIA GPU；不要把这些依赖安装到平台主 `.venv`。

加载时官方 Franka USD 可能首次下载外部引用；第一次 Warp 物理与渲染内核会编译，随后复用 SDK 缓存。需要访问 NVIDIA 官方资产服务器，网络受限时应先为启动进程配置相应网络环境，再进行加载。

本机最终配置的缓存启动（含首次物理步、四路相机预热、恢复初态后立即观察检查）为 **41.516 秒**，10 项回归共 **46.357 秒**。这些是本机测量值。首次编译与下载需要更长时间；不要把缓存启动时间当成其他机器的保证。

Windows 中文目录下，NVRTC 对 Warp 的头文件路径存在读取问题。适配器只为该情况建立受控英文临时目录，复制官方 Python/头文件并通过目录符号链接复用二进制，退出后清理。需要 Windows 允许创建目录符号链接；没有权限时会明确提示启用开发者模式或使用英文安装路径，不会修改已安装的 SDK。

原生回归：

```powershell
.venv-isaac\Scripts\python.exe -m unittest tests.test_isaac_backend -v
$env:EMBODI_TEST_ISAAC='1'
.venv\Scripts\python.exe -m unittest tests.test_robot_backends_integration -v
Remove-Item Env:\EMBODI_TEST_ISAAC
```

主 Worker 集成在加载后未执行 settle 时立即采集三相机并严格检查 JSON 有限性，再使用本地固定响应实际执行两个回合的末端移动与完成申请。物理、三路图像、独立成功判据、中央实验记录、视频保存及完整解码、回放和跨引擎拒绝均运行真实 SDK；本机该项通过，用时 **52.249 秒**，没有调用外部收费模型。

## 当前工作台契约

- 官方 Franka Panda、`isaac_tabletop` 桌面与红方块；任务为抬起红块和末端向上移动 2 cm。
- 初次加载明确写入官方默认关节位置与速度，再保存用于重置的状态；保持暂停和仿真时间 0。加载后立即观察经过原生回归验证，不依赖用户先执行 settle。腕相机直接使用 TCP 挂载四元数，保持完整光轴与滚转；相机标定出现非有限数值时报告具体相机与字段，不输出不可序列化的观察。
- 所有公开动作采用世界坐标、米、WXYZ 四元数；Isaac Lab 3.0 的 XYZW 在适配器中明确转换。机械臂末端动作交给官方 DifferentialIKController 和原生 PD 驱动，每轮平移限 12 cm、转动限 35°。
- Newton 采用官方 reach/lift 任务使用的 `implicitfast` 积分与两个物理子步，并复用官方 Franka reach IK 的 `MujocoRigidBodyCfg(gravcomp=1.0)` 机器人重力补偿。保留官方高 PD Panda 的增益，世界与方块仍受重力影响；不能直接将裸配置默认的 Euler 积分与这些增益组合使用。积分器、子步、机器人重力补偿与时间步长记录到后端元数据。
- Agent 观察是三路同步原生图像与机器人本体状态；对象真值和任务评价不进入 Agent 观察。独立评价读取模拟器中的对象/末端坐标，并要求条件持续满足 0.3 秒。
- 使用同一实验记录、视频采样、关节动作块、检查点与状态回放入口。检查点核对引擎、SDK 版本和状态格式；状态恢复不声明包含求解器暖启动缓存的逐比特复现。
- 自由视角支持旋转、平移、缩放。未接入的场景、其他机器人与不支持的控制显式报错。

## 完整 Isaac Sim 环境

在符合资源要求的机器上单独创建 Python 3.12 的 `.venv-isaacsim`，按照 NVIDIA 官方安装文档检查驱动、磁盘和显存，再安装 `requirements-isaacsim.txt`。该文件包含大体积完整 Sim/扩展缓存，不能混装到 `.venv-isaac`。

首次启动会按 NVIDIA 流程要求操作者阅读并接受许可。本项目不会设置自动同意 EULA 的环境变量。模型资产服从各自 NVIDIA 资产条款；仓库不重新分发下载的完整资产包。

官方依据：[Isaac Lab 3.0 发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)、[Isaac Sim 资源与驱动要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)、[Isaac Sim Python 环境安装](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)。
