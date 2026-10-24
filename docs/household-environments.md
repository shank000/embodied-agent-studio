# 家庭任务环境：ALFWorld 与 AI2-THOR

这两套环境使用各自官方 SDK 的状态转换、观察和任务证据。ALFWorld 提供文字任务；AI2-THOR 提供 Unity 第一人称视觉与离散交互。它们按环境动作逐轮推进，不使用机械臂 TCP、IK、关节控制或连续 VLA 动作。MuJoCo、Genesis 等机械臂后端的控制方式保持独立。

## 本机安装

Windows 版本通过**本机已有的 WSL Ubuntu**运行这两个 SDK，工作台网页与主服务继续运行在 Windows。它不是远程服务器，也没有把 Linux 构建声明为 Windows 原生构建。

前提：已安装 WSL Ubuntu、WSLg 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。安装工具不会安装操作系统、修改驱动或替换系统 Python。TextWorld 的官方系统支持为 Linux/macOS；AI2-THOR 官方发布的 Unity 构建在本次 Windows 机器上通过 WSLg 实测成功。[TextWorld 安装说明](https://github.com/microsoft/TextWorld#installation)、[AI2-THOR 安装说明](https://ai2thor.allenai.org/ithor/documentation/)。

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe tools\setup-household.py --engine all --wsl Ubuntu
```

也可分别使用 `--engine alfworld` 或 `--engine ai2thor`。发行版名称不同则调整 `--wsl`；安装位置如下：

| 内容 | 位置 |
| --- | --- |
| ALFWorld 独立 Python 3.12 | WSL `~/.local/share/embodiworks/alfworld/bin/python` |
| AI2-THOR 独立 Python 3.12 | WSL `~/.local/share/embodiworks/ai2thor/bin/python` |
| Windows 端解释器桥接配置 | 项目 `.venv-alfworld/runtime.json`、`.venv-ai2thor/runtime.json` |
| ALFWorld 官方回合 | 项目 `vendor/alfworld_data/json_2.1.1/valid_unseen/` |
| AI2-THOR 官方 Unity 构建 | WSL 用户 `~/.ai2thor/` |

安装后会实际启动 SDK、读取观察和画面、关闭环境；通过后才写入 WSL 的可用标记。不要手工把未经验证的 `validated` 或 `assets_ready` 改成 `true`。可选 `--skip-verify` 仅安装，WSL 环境仍显示未就绪；以后去掉该选项重新运行验证。路径配置保存在本机忽略目录中，不提交到 Git，也不包含 API Key。

Linux/macOS 可在项目根目录运行以下命令，安装工具使用项目 `.venv-alfworld` 和 `.venv-ai2thor`，无需 WSL 配置：

```bash
python tools/setup-household.py --engine all
```

已完成的整条工作台实测环境是 Windows + Ubuntu WSL2 + WSLg + Python 3.12。其他系统仍需在相应硬件上运行下方回归；macOS 能否使用特定 AI2-THOR Unity 构建取决于官方平台支持。

## 资产与下载范围

ALFWorld 使用官方 [0.4.2 文字环境归档](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip)，约 36.5 MB。安装工具仅解压 `valid_unseen` 中的 134 个 `game.tw-pddl` 文件，约 12 MB；游戏文件本身包含官方 PDDL、语法和任务定义。它不下载训练集、视觉检测模型、预训练策略或演示视频。已有官方归档时可以避免重复下载：

```powershell
.\.venv\Scripts\python.exe tools\setup-household.py --engine alfworld --wsl Ubuntu --archive "D:\downloads\json_2.1.3_tw-pddl.zip"
```

工具仅接收指定路径结构下的游戏文件，保留已经存在的文件。若自己设置了 `ALFWORLD_DATA`，适配器将使用该目录；该目录同样需要提供 `json_2.1.1/valid_unseen/.../game.tw-pddl`。

AI2-THOR 首次真实启动由官方 SDK 下载 Unity 构建。本次 AI2-THOR 5.0.0 的 Linux64 构建下载约 **769 MB**；这是房间与运行时资产，不是训练权重。下载完成后后续启动复用本机缓存。WSLg 的显示服务必须可用，不能将缺少渲染依赖时的空白图像当作验证通过。

## 环境交互与任务

在“仿真环境”中加载相应后端后，使用“环境交互”面板：阅读任务与原生观察，点击可执行命令，或输入文字动作。Agent 接管也使用同一原生动作接口。自动任务运行时手动交互会锁定；原生动作失败会保留 SDK 原因。

ALFWorld 使用官方 TextWorld/PDDL 与 `AlfredDemangler`，任务类型如下：

| 任务 ID | 任务 |
| --- | --- |
| `pick_and_place_simple` | 取物并放置 |
| `look_at_obj_in_light` | 在灯光下检查物品 |
| `pick_clean_then_place_in_recep` | 清洁后放置 |
| `pick_heat_then_place_in_recep` | 加热后放置 |
| `pick_cool_then_place_in_recep` | 冷却后放置 |
| `pick_two_obj_and_place` | 放置两件物品 |

布局种子用于在对应任务族的官方回合中作确定性选择，不会凭空生成一个新房间。动作示例包括 `go to desk 1`、`open drawer 1`、`take mug 1 from desk 1`；以当前 SDK 返回的可执行命令为准。**ALFWorld 场景预览是当前原生文字观察的排版，不是相机，也不是 3D 仿真画面**；模型观察中的 `cameras` 是空列表。

AI2-THOR 提供 `FloorPlan1`（厨房）、`FloorPlan201`（客厅）、`FloorPlan301`（卧室）、`FloorPlan401`（浴室）房间入口，任务为：

| 任务 ID | 原生目标判据 |
| --- | --- |
| `thor_open_object` | 指定 `objectId` 的 `isOpen` 为真 |
| `thor_pickup_object` | `inventoryObjects` 包含指定 `objectId` |
| `thor_toggle_object` | 指定 `objectId` 的 `isToggled` 为真 |

种子从房间中符合条件且初始未完成的原生物体选择任务目标；当前房间没有相应物体时，环境会报告真实原因，不虚构替代目标。模型能看到实际第一人称 RGB、可见物体状态/可用操作和动作反馈。物体以稳定的不透明 `object_...` 句柄表示；SDK 的原生 ID 含有世界坐标，原生 ID、精确 `distance` 和物体位置均不发送给模型。目标文本、物品栏、动作列表及错误反馈也使用相同句柄。

导航直接使用 `MoveAhead`、`RotateRight` 等名称，物体交互使用面板中的 JSON 字符串，例如（句柄以当前观察为准）：

```json
{"action":"OpenObject","objectId":"object_0123456789abcdef01234567"}
```

物体句柄必须来自当前环境；适配器在内部反解为原生 ID 后再调用官方 Controller，未知句柄或直接提交原生 ID 都会被拒绝。AI2-THOR 不伪造侧视和腕部相机。调试、瞬移、`forceAction`、直接改物体位置等参数不开放给策略动作接口；控制对象不可见或原生交互失败时，记录去除原生坐标标识的错误供下一轮调整。

## 成功判定、日志与回放

- ALFWorld 由官方 `GameState.won` 判定，来源标记为 `alfworld_native_task`。
- AI2-THOR 由原生目标物体状态或物品栏判定，来源标记为 `ai2thor_native_task`。这些是本平台明确限定的原生状态任务，不冒充 ALFRED 完整基准成功率。
- 模型输出“完成”只触发环境验收，不能改变判定结果。PDDL 事实、专家解法、全场景元数据和评价器成功值不注入模型观察。
- `state`、模型观察和轨迹检查点声明 `time_unit="environment_step"`，环境时间以动作步为单位，`timestep=1`。实验指标用 `environment_steps` 记录步数，`sim_time_s` 为 `null`；墙钟总耗时和模型推理耗时仍分别以秒记录，不能把环境步当作连续动力学秒数分析。评价器同时提供任务名称、指令和明确判据，供界面展示；原生成功证据仍只用于独立验收。
- 检查点恢复和轨迹回放通过官方 SDK 重置回合并重新执行保存的动作，再比对最终原生观察。它是**原生动作重放**；观察不一致时明确拒绝精确恢复声明，不把不同状态伪装成原检查点。它不承诺 Unity 物理逐位一致。
- AI2-THOR 不透明动作句柄使用 `ai2thor-integration-v2` 状态格式；旧版直接使用原生 ID 的动作轨迹不兼容，不能跨格式恢复。

## 实际验收与复测

本次验收使用 ALFWorld 0.4.2 / TextWorld 1.7.0 和 AI2-THOR 5.0.0：

- ALFWorld：六个任务族、种子 42 的官方解法均触发真实 `won=true`；检查文字观察、空相机列表、过期动作拒绝、原生状态转换、回放与种子重复选择。
- AI2-THOR：`FloorPlan1` 的真实 Unity 相机、旋转动作引起图像变化、普通 `OpenObject` 改变目标原生状态并触发成功、导航动作检查点重放。正例测试的导航定位使用 SDK 交互位姿作测试准备，不代表已经验证某个模型能自主完成导航。
- UI：手动交互锁定、单请求执行、原生失败原因、旧环境事件与迟到响应隔离、SDK 文本安全显示。

未把接口存在等同于所有房间、所有模型组合都完成端到端评测。AI2-THOR 其余房间及拾取/开关任务可加载测试，实际研究使用前应按对应任务补充正负例验收。

Windows 下可复测原生 SDK，主环境不需要安装这些可选包：

```powershell
wsl -d Ubuntu --cd /mnt/d/workspace/机械臂具身仿真研究工作台 --exec /usr/bin/env EMBODI_TEST_ALFWORLD=1 /home/administrator/.local/share/embodiworks/alfworld/bin/python -m unittest tests.test_household_backends -v
wsl -d Ubuntu --cd /mnt/d/workspace/机械臂具身仿真研究工作台 --exec /usr/bin/env EMBODI_TEST_THOR=1 /home/administrator/.local/share/embodiworks/ai2thor/bin/python -m unittest tests.test_household_backends -v
```

替换示例中的工作区和 WSL 用户路径；不要在同一台内存受限机器同时启动多个重型 SDK。前端回归使用 `node --test tests/test_environment_ui.mjs`。测试结果区分普通契约检查、原生环境验收与可选跳过，成功接入 SDK 不表示模型策略质量已得到证明。

主 Python 环境还可验证完整 Worker → 独立进程 → SDK → Agent 闭环。推理由固定本地测试动作替代，不调用收费模型；日志、检查点和视频放在临时目录，测试结束后清理：

```powershell
$env:EMBODI_TEST_HOUSEHOLD='1'
.\.venv\Scripts\python.exe -m unittest tests.test_household_integration -v
Remove-Item Env:\EMBODI_TEST_HOUSEHOLD
```

ALFWorld 集成回归检查两次错误“完成”声明均被环境拒绝、中间实际执行原生命令、文字观察与视频归档，以及切换引擎后旧动作和旧轨迹被拒绝。AI2-THOR 桥接测试使用独立开关 `EMBODI_TEST_HOUSEHOLD_THOR=1`，会启动真实 Unity；请先关闭其他重型仿真进程。
