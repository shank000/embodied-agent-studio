# 自定义 VLA / WAM 策略接口

在“策略模型”中选择自定义 VLA 或 WAM，填写模型名称、完整 HTTP POST 推理地址和可选 API Key。本机服务可用 HTTP，远程服务使用 HTTPS。工作台不会把语言模型回答或预设轨迹当作 VLA/WAM 输出。

## 请求

```json
{
  "protocol": "embodied-policy-v1",
  "model": "your-model",
  "device": "cpu",
  "task": "将红色方块放入绿色托盘。",
  "seed": 42,
  "reset": true,
  "horizon": 10,
  "dt_s": 0.05,
  "observation": {
    "robot": "so101",
    "generation": 1,
    "time": 1.0,
    "actuators": [],
    "cameras": [],
    "hand_joints": []
  }
}
```

示例的空数组用于省略长数据。实际请求包含：

- `cameras`：同一物理时刻的 `front`、`side`、`wrist`，每项包含 JPEG `image_base64`、时间、世界坐标位置、旋转矩阵和视场角。
- `actuators`：有序的 `name`、`unit`、`position`、`target`、`min`、`max`。关节位置用弧度；Panda 最后一项执行器的 **target/min/max 用 0–255 控制量，position 用单指米数**，不能直接将该 position 作为动作。其余执行器目标和实际位置均用弧度。
- 机械臂本体状态还包含 TCP 位姿、关节速度、关节限制与动作范围。Gaia 的 `hand_joints` 是 15 项名称、实际角度和范围。
- 不提供物体真值位置或场景评价器输出。策略通过图像和本体状态预测。
- `reset=true` 表示一个新试验，请清空动作队列和时序缓存。后续块为 false。每轮种子为场景种子加轮次减一。
- 若填写 API Key，请求头为 `Authorization: Bearer <key>`。密钥不写入日志。

## 关节动作块响应

```json
{
  "protocol": "embodied-policy-v1",
  "action_type": "actuator_targets",
  "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
  "actions": [[0.0, -0.5, 0.8, 0.5, 0.0, 1.0]],
  "dt_s": 0.05,
  "model": "your-model",
  "summary": "策略动作说明",
  "model_output": {"optional": "公开原始预测"}
}
```

上面是协议格式示例，不是预训练策略。**关节名称必须从当前请求的 actuators 原样取出且顺序完全一致**，不得硬编码另一机器人名称。每帧维度必须为当前执行器数：Panda 8、SO-101 6、RM65 + GaiaHand 21。只接受有限数值，1–50 帧，帧间隔 0.02–0.5 秒。

目标按执行器范围和 2 rad/s 的目标变化速率限幅；Panda 夹爪为 255 控制量/s。限幅项数及实际下发目标写入日志。MuJoCo 通过执行器跟踪目标，不改写实际关节状态。长动作块仍受试验的仿真时间上限约束。

## 末端动作响应

也可返回 `action_type: "tcp_pose"` 和 `pose`，其余遵循同一个动作契约：

```json
{
  "protocol": "embodied-policy-v1",
  "action_type": "tcp_pose",
  "pose": {
    "status": "move",
    "frame": "world",
    "position_m": [0.4, 0.0, 0.4],
    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    "gripper_open": 1.0,
    "duration_s": 1.0,
    "hand_joint_positions_rad": null,
    "explanation": "下一步绝对末端目标"
  }
}
```

此位姿仅演示字段，是否可达由当前机器人 IK 检查。Gaia 可提供按观察顺序排列的 15 个手指角度；null 使用整手张合预设。`status: done` 请求环境验收；未通过时重新采集观察并继续，在原 `task` 字符串中附加公开的验收未通过提示，v1 请求字段保持不变。成功由 `task-evaluation-v2` 读取 MuJoCo 状态与持续时间判定，特权坐标与判据实测值不会发送给策略。

## 默认 SmolVLA

本地独立服务 `server.policy_service` 使用官方 LeRobot 0.6.1、`lerobot/smolvla_base` 和原始归一化处理器，直接预测动作块。默认适配 SO-101 的 6 维动作，前五个关节按当前 MJCF 范围映射至 [-100,100]，夹爪映射至 [0,100]，输出做反向映射。三张图依次映射至 checkpoint 中的三个图像特征。

这是一套明确记录在每次输出中的仿真标定。基础权重未经本工作台场景微调，图像视角与原始训练数据也不同；实际任务成功率由实验获得。可更换匹配此映射的微调权重；其他本体、标定或相机布局通过自定义 VLA 接口接入。

WAM 目前提供相同输入/动作协议的自定义接口，未内置 WAM 权重或世界预测可视化。`model_output` 可保存服务公开返回的世界预测摘要。
