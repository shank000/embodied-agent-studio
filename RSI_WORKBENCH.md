# RSI 实验室使用与接口

本轮提供可组合的执行与研究基础：方法配方登记、真实 episode 记录、同初态版本对比、验证晋升和回滚。当前子代由用户登记；尚未自动生成修改、训练权重或证明递归改进。

## 方法与模型组合

打开页面的“RSI 实验室”。主 Agent 的提供者、模型 ID、接口和参数均可配置，不绑定某个 GPT 型号。内置 Codex 本机登录、OpenAI Responses、OpenAI 兼容、Claude Messages 和自定义 Agent 服务。

- `agent`：视觉 Agent 输出一个绝对 TCP 动作。
- `policy`：SmolVLA 或自定义 VLA/WAM 输出 TCP 或关节动作块。
- `hybrid`：主 Agent 按初始、子目标过期、执行受阻等事件规划；本地技能或配置的策略产生候选；独立决策器选择候选；统一执行者完成动作。

决策器支持 TypeSafe Jev 和 `embodied-decision-v1` 自定义服务。开源 Jev 类模型可以自行部署为该服务，不会自动下载或冒充官方权重。策略组合支持：

- `single`：一个决策器选择动作。
- `cascade`：首个决策器选择，后续决策器逐个复核；拒绝或低置信会请求重规划。
- `weighted_vote`：各决策器在同一观察和候选上并行投票；声明权重严格超过总权重一半才执行；平票或不足多数时重规划。票份额不是经过校准的任务成功概率。服务出错不会静默丢票。

模型只产出提案或选择；它们不各自控制 MuJoCo。执行时仍检查动作结构、IK、关节限位和场景版本。物理在推理期间暂停，界面分别统计观察、规划、候选生成、决策、策略推理、执行和加载耗时。

## 不可变方法版本

保存时将当前具体配置冻结为 `methods/method-*/manifest.json`，包括名称、父代、进化代次、变更说明、角色配置、代码和提示词材料以及运行环境足迹。密钥不进入配方或磁盘材料。

从列表选择“继承”创建子代，编辑参数并写明变更。父代记录不覆盖。实际生效的技能参数是 `translation_step_m`、`duration_s`、`max_subgoal_steps`、`candidate_source`；控制参数是 `decision_min_confidence`。其他键会被拒绝，避免登记无效修改。

开始前会核对当前代码、提示和环境是否与登记版本一致。历史源代码材料用于追溯，不会动态执行；代码升级后需要在当前环境登记新版本。远端模型实际 revision、策略权重、独立策略运行时和机器人资源字节尚未全部锁定，清单会列明这些未核实项，因此不能据此宣称跨机器完全复现。

Agent／策略可复用同提供者、同地址的现有内存密钥。多个决策器的密钥必须逐索引配置，例如 `{"0":"first-key","1":"second-key"}`。密钥只在后端内存，页面发送后清空，服务重启后重新填写；不会把一个接口的密钥自动发给其他决策接口。

## 冻结协议与版本比较

协议固定机器人、场景、任务、三组互不重叠种子、随机运行顺序、仿真时限、墙钟时限、步数与晋升门槛。默认开发种子 42，验证种子 43/44/45，测试种子 46/47/48。协议按内容编号，修改需另存新协议。

选择两个或更多方法版本后，在同一 split 运行。每个种子先稳定物理并保存完整积分状态；各方法恢复同一初态，按登记的随机顺序依次运行。所有 seed × method 试验在开始时预分配，错误、停止、超时和未完成试验都保留在成功率分母；另外展示故障数。

晋升只接受完整结束的验证集实验。候选须是基线的直接子代，每对必须共享同一物理初态，配对数、成功率差与错误数满足登记门槛。测试集不能用于晋升。晋升记录标为 `exploratory`，配对 bootstrap 区间仅作描述，不是独立确认性证据。回滚会追加历史，不删除原版本或试验。

这些规则验证的是既定任务和预算中的观察改进；自动改进器、封存任务服务、跨链 RSI 统计、策略微调与课程生成属于后续阶段。

## 完整任务记录与回放

新运行保存到 `runs/<run_id>/`：

- `events.jsonl`：公开模型输出、候选、决策、实际反馈、各阶段耗时和用量。
- `observations/<step>/{front,side,wrist}.jpg` 与 `observations/manifest.jsonl`：真实同步相机序列、元数据和图像摘要。
- `checkpoints/initial.json`：完整 MuJoCo 初态。
- `trajectory.jsonl`：逐物理步积分状态、qpos、qvel、ctrl、执行器力、接触数和独立评分。
- `summary.json`：运行配置、结果和附件清单。
- `trajectory.mp4`：运行前勾选“保存轨迹视频”后生成的主视角 H.264 视频（默认关闭）。

“查看任务记录”可翻阅三路图像，并将选定物理帧加载到原生视图。状态回放保持暂停，不再调用模型，不把设置已记录状态称为重新执行动力学成功。历史未保存轨迹的日志不会生成虚构回放。完整 ZIP 可下载图像、状态和事件；逐步状态会随任务长度增加磁盘用量。

视频使用本机 FFmpeg（需 libx264），按仿真时间 20 FPS 记录当前主视角；暂停推理不增加视频时长。运行停止或出错时也会封装已录内容；没有启用的视频不事后补造。编码错误会写入 `artifacts.video.error` 并在页面显示，不替换任务本身的结果。所有启动接口接受 `record_video: false`，在配对实验开启时为每个版本／种子的子试验各录一份，开关不改变方法配方或评价协议。

## 自定义 Agent 接口

`provider=custom` 的 `base_url` 是完整 HTTP POST 地址。本机可使用 HTTP 且不填 Key，远端使用 HTTPS 和专用 Bearer Key。服务可以封装其他模型或已有 Agent 框架，但应将本次调用结果归约到请求指定的结构，不直接接管工作台执行接口。

请求包含 `protocol: "embodied-agent-v1"`、`model`、`task`、`prompt`、`system_contract`、三张真实 `cameras`、`response_schema` 和 `response_schema_name`。响应：

```json
{
  "protocol": "embodied-agent-v1",
  "output": {
    "status": "move", "frame": "world", "position_m": [0.3, 0.0, 0.2],
    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "gripper_open": 1.0,
    "duration_s": 0.7, "hand_joint_positions_rad": null, "explanation": "下一步末端目标"
  },
  "model": "your-agent-version",
  "usage": {"input_tokens": 100, "output_tokens": 50}
}
```

上面展示直接 Agent 的字段示例，坐标是否可达仍由当前机器人 IK 检查。`output` 必须严格符合本次 schema：直接 Agent 为 TCP 动作，主规划 Agent 为子目标对象；不能用自由文本或空动作代替。

## 自定义决策接口

POST 到 `decision.url`，请求 `protocol: "embodied-decision-v1"`，包含 `episode_id`、`observation_id`、`scene_generation`、任务／子目标、状态与带 ID 的动作候选。响应须回传本次 episode 和 observation 标识，返回已存在的 `candidate_id`，或者显式请求重规划。陌生候选、迟到选择和无效数值不会执行。多决策链的每一层均记录原始结果与汇总。

视觉开关为 false 时，决策器接收带时间戳的主 Agent 视觉摘要和当前本体状态，不能认为它读取了当前三路图像。为 true 时发送真实图像；TypeSafe 图像通路按其官方实验性协议启用，自定义服务也必须实际支持输入图像。

现有 VLA/WAM 动作协议继续使用 [POLICY_PROTOCOL.md](POLICY_PROTOCOL.md)。只预测未来世界状态的模型应在服务端组合规划器后返回可执行动作，不能直接当作动作策略。

## 本地接口

- `GET/POST /api/research/methods`：列出／登记版本。
- `POST /api/research/methods/{id}/credentials`：只更新该方法的内存凭据。
- `GET /api/research/defaults`：当前无密钥的角色配置。
- `POST /api/research/start`、`stop`：组合任务。
- `GET/POST /api/research/protocols`：冻结协议。
- `GET/POST /api/research/experiments`、`POST .../stop`：版本配对实验。
- `POST /api/research/promote`、`rollback`：验证晋升与回退。
- `GET /api/runs/{id}/episode`、`artifact/{path}`、`bundle`：任务附件与完整下载。
- `POST /api/research/replay`：将已结束记录的 `frame_index` 加载到原生视图。

所有启动、配置、停止、回放接口继续受本机来源和自动控制互斥约束。普通回归检查使用真实 MuJoCo 加本地 HTTP 边界夹具；它们验证机制，不评价预训练模型任务成功率。
