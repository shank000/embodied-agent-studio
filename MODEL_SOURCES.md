# 模型来源与本地场景设置

模型仓库：[Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

固定版本：`822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7`

| 模型 | 源文件 | 来源与许可 |
| --- | --- | --- |
| Franka Panda | `franka_emika_panda/panda.xml` | [Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7/franka_emika_panda)，基于 Franka 的公开 URDF 描述，Apache-2.0 |
| SO-101 | `robotstudio_so101/so101.xml` | [Menagerie SO101](https://github.com/google-deepmind/mujoco_menagerie/tree/822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7/robotstudio_so101)，源自 The Robot Studio SO-ARM100/SO-101 模型，Apache-2.0 |

原始模型文件、网格及 LICENSE 保存在 `vendor/mujoco_menagerie/`，未在磁盘上修改。网页模型缩略图来自同一模型目录。

## RM65 + GaiaHand

- RM65：[RealMan 官方 ROS 2 仓库](https://github.com/RealManRobot/ros2_rm_robot)，固定提交 `c941b565e4f9174afa36561f143ef5fbbb744750`。本地 `vendor/realman/rm_description/urdf/rm_65.urdf` 与原始 STL；仓库的 package.xml 许可字段未明确（TODO），不要将其视作 Apache/MIT 授权。
- GaiaHand：[Stella-robot 官方仓库](https://github.com/Stella-robot/GaiaHand)，固定提交 `eb61f952ce0086a3fea4ecb6559970c5ea1f09a5`，MIT。本地 `vendor/gaia_hand/Software/urdf/right_gaiahand`。该公开右手模型有 **15 个可动关节**，并非 Gaia Hand 20。
- `tools/download_robot_models.py` 下载源 URDF、引用网格与来源清单，不改写官方模型。运行时将手根通过固定关节连接到 RM65 Link6，偏移 0.012 m，MuJoCo 原生编译 URDF；保留几何、关节范围与惯性参数。
- 添加仿真位置伺服：机械臂 kp=1000、kv=60，使用 URDF 力矩限制；手部 kp=6、kv=0.12、力矩 ±1 Nm。公开 Gaia URDF 的 effort/velocity 为零，手部伺服值是本工作台仿真设定，未经真机标定。
- TCP 位于手根局部 z=0.16 m，添加腕部相机。可逐个调节 15 个手指关节，也可使用整手张合预设或 Agent 输出的 15 维手指目标。
- `assets/rm65_gaia.png` 为本地 MuJoCo 原生渲染缩略图。

载入时在内存中添加以下工作台设置：

- 添加桌面、地面、三颗自由方块和目标区域，以及用于页面展示的灯光、背景和离屏分辨率。
- 仿真统一为 2 ms 时间步和 `implicitfast` 积分器，默认重力为 `0 0 -9.81` m/s²。保留原有惯性、关节、执行器、碰撞和约束参数。
- Panda 额外添加 TCP 观测 site（手部局部坐标 `0 0 0.1034`），不改变动力学；SO-101 使用模型内的 `gripperframe`。
- 为两种模型增加前方和侧方固定相机；Panda 增加腕部相机，SO-101 保留原生 `wrist_cam`。这些相机没有质量，不改变机器人动力学。三路观察使用相同的 `MjData` 时刻。
- 原模型 keyframe 不直接使用，因为新加入自由物体后 qpos 维度发生变化。工作台用显式初始关节姿态初始化模型。
- 初始姿态与展示台尺寸分别按两种机械臂设置；**没有缩放机械臂本体**。SO-101 周围工作台和物体采用较小尺寸。
- 模型路径通过 MuJoCo 虚拟文件系统解析，以支持当前 Windows 中文项目路径。

这些模型用于机械臂交互和物理仿真，未经过实体设备参数标定。Panda 使用双指腱传动控制（原始控制范围 0–255），SO-101 夹爪为旋转关节；界面分别显示两者实际物理量。
