# 家庭任务环境：ALFWorld 与 AI2-THOR

家庭任务有三个独立入口，使用各自官方 SDK 的状态转换、观察和任务证据：

| 选择项 | 引擎 ID | 观察与任务 |
| --- | --- | --- |
| ALFWorld · 文字 | `alfworld` | TextWorld/PDDL 文字观察与高层文字命令，没有相机 |
| ALFWorld · 视觉 | `alfworld_thor` | 同一官方 ALFRED 回合的 Unity 三维房间、第一人称 RGB、低层导航与物体交互 |
| AI2-THOR | `ai2thor` | AI2-THOR 5.0 房间与本平台指定的开门、拾取、开关任务 |

三个入口都按环境动作逐轮推进，不使用机械臂 TCP、IK、关节控制或连续 VLA 动作。要查看 **ALFWorld 的三维仿真画面**，请选择 `ALFWorld · 视觉`；单独的 AI2-THOR 5.0 房间不替代 ALFWorld 官方回合。

## 本机安装

Windows 版本通过**本机已有的 WSL Ubuntu**运行这些独立 SDK，工作台网页与主服务继续运行在 Windows。它不是远程服务器，也没有把 Linux 构建声明为 Windows 原生构建。

前提：已安装 WSL Ubuntu 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。视觉环境在 WSL 中使用独立 Xvfb 显示，不依赖 Windows 桌面或远程桌面连接；请先在 Ubuntu 安装 `sudo apt-get install xvfb xauth`。独立 Python 的 PyVirtualDisplay 随对应 requirements 安装。安装工具不会安装操作系统、修改驱动或替换系统 Python。[TextWorld 安装说明](https://github.com/microsoft/TextWorld#installation)、[AI2-THOR 安装说明](https://ai2thor.allenai.org/ithor/documentation/)。

WSL 的 `DISPLAY=:0` 即使保留 socket，也可能因桌面会话断开而拒绝 X11 连接。适配器为每个 Unity 子进程启动独立、带认证且不监听 TCP 的后台显示，并在 Unity 关闭后释放；原生相机画面和物理环境仍来自官方 Unity 构建。常规 Linux 桌面继续使用已有显示，Linux 无显示时也需要 Xvfb。

在项目根目录安装 ALFWorld 三维视觉环境：

```powershell
.\.venv\Scripts\python.exe tools\setup-household.py --engine alfworld_thor --wsl Ubuntu
```

文字模式使用 `--engine alfworld`，AI2-THOR 5.0 使用 `--engine ai2thor`；`--engine all` 依次准备三者。发行版名称不同则调整 `--wsl`。ALFWorld 三维视觉固定为 **Python 3.11、ALFWorld 0.4.2、AI2-THOR 2.1.0**，不能与 AI2-THOR 5.0 环境混装。

| 内容 | 位置 |
| --- | --- |
| ALFWorld 文字独立 Python 3.12 | WSL `~/.local/share/embodiworks/alfworld/bin/python` |
| ALFWorld 视觉独立 Python 3.11 | WSL `~/.local/share/embodiworks/alfworld_thor/bin/python` |
| AI2-THOR 独立 Python 3.12 | WSL `~/.local/share/embodiworks/ai2thor/bin/python` |
| Windows 端解释器桥接配置 | 项目 `.venv-alfworld/runtime.json`、`.venv-alfworld_thor/runtime.json`、`.venv-ai2thor/runtime.json` |
| ALFWorld 官方回合 | 项目 `vendor/alfworld_data/json_2.1.1/valid_unseen/` |
| ALFWorld 视觉专用 Unity 2.1 构建 | WSL `~/.cache/embodiworks/alfworld_thor/` |
| AI2-THOR 5.0 官方 Unity 构建 | WSL `~/.ai2thor/` |

安装后会实际启动 SDK、读取观察和画面、关闭环境；通过后才写入 WSL 的可用标记。重装前先清除旧验证状态，依赖或原生验证失败时保持未就绪，保留真实错误。不要手工把未经验证的 `validated` 或 `assets_ready` 改成 `true`。可选 `--skip-verify` 仍会安装依赖和准备资产，但不启动原生环境；以后去掉该选项重新运行验证。路径配置保存在本机忽略目录中，不提交到 Git，也不包含 API Key。

Linux/macOS 可在项目根目录运行以下命令，安装工具使用对应的项目 `.venv-alfworld`、`.venv-alfworld_thor` 或 `.venv-ai2thor`，无需 WSL 配置：

```bash
python tools/setup-household.py --engine alfworld_thor
```

本机使用 Windows + Ubuntu WSL2 + WSLg。文字模式与 AI2-THOR 5.0 的工作台链路使用 Python 3.12，ALFWorld 视觉模式使用独立 Python 3.11；三者均已完成原生回归与工作台集成验证，具体范围见下文。其他系统仍需在相应硬件上运行回归；macOS 能否使用特定 AI2-THOR Unity 构建取决于官方平台支持。

## 资产与下载范围

ALFWorld 文字模式使用官方 [0.4.2 文字环境归档](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip)，约 36.5 MB。安装工具仅解压 `valid_unseen` 中的 134 个 `game.tw-pddl` 文件，约 12 MB；游戏文件本身包含官方 PDDL、语法和任务定义。已有归档时可以避免重复下载：

```powershell
.\.venv\Scripts\python.exe tools\setup-household.py --engine alfworld --wsl Ubuntu --archive "D:\downloads\json_2.1.3_tw-pddl.zip"
```

视觉模式还需要官方 [json_2.1.1_json.zip](https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip) 的 `traj_data.json`，用于恢复房间、物体状态、初始视角与 ALFRED 任务。安装器把它放在对应 `game.tw-pddl` 的同一回合目录。已有两份归档时可执行：

```powershell
.\.venv\Scripts\python.exe tools\setup-household.py --engine alfworld_thor --wsl Ubuntu --archive "D:\downloads\json_2.1.3_tw-pddl.zip" --visual-archive "D:\downloads\json_2.1.1_json.zip"
```

工具只提取指定 `valid_unseen` 路径下的任务元数据，保留已存在文件；不下载训练图像、视觉检测器、策略权重或演示视频。若自己设置了 `ALFWORLD_DATA`，适配器将使用该目录，视觉模式需要每个选中回合同时具备 `game.tw-pddl` 与 `traj_data.json`；安装器默认准备的位置仍是项目 `vendor/alfworld_data`。缺少视觉元数据时明确报错，不用文字图或无关房间代替。

ALFWorld 视觉的 Unity 2.1 构建在安装阶段通过官方 SDK 指定的 HTTPS 地址下载，校验官方 SHA256 后放入独立缓存。运行任务只打开已准备的可执行文件，避免旧 SDK 的默认缓存清理逻辑删除 AI2-THOR 5.0 构建。现有不完整构建目录会保留并提示检查，不会被覆盖。

AI2-THOR 首次真实启动由官方 SDK 下载 Unity 构建。本次 AI2-THOR 5.0.0 的 Linux64 构建下载约 **769 MB**；这是房间与运行时资产，不是训练权重。下载完成后后续启动复用本机缓存。WSLg 的显示服务必须可用，不能将缺少渲染依赖时的空白图像当作验证通过。

## 环境交互与任务

在“仿真环境”中加载相应后端后，使用“环境交互”面板：阅读任务与原生观察，点击可执行命令，或输入文字动作。Agent 接管也使用同一原生动作接口。自动任务运行时手动交互会锁定；原生动作失败会保留 SDK 原因。

ALFWorld 两种模式使用相同的六个任务族：

| 任务 ID | 任务 |
| --- | --- |
| `pick_and_place_simple` | 取物并放置 |
| `look_at_obj_in_light` | 在灯光下检查物品 |
| `pick_clean_then_place_in_recep` | 清洁后放置 |
| `pick_heat_then_place_in_recep` | 加热后放置 |
| `pick_cool_then_place_in_recep` | 冷却后放置 |
| `pick_two_obj_and_place` | 放置两件物品 |

相同任务族、种子和资产目录在两种 ALFWorld 模式中选择同一官方回合；种子不会凭空生成一个新房间。动作层次和状态表示不同，文字命令及检查点不能直接转成视觉动作或跨模式恢复。

文字模式使用官方 TextWorld/PDDL 与 `AlfredDemangler`。动作示例包括 `go to desk 1`、`open drawer 1`、`take mug 1 from desk 1`，以当前 SDK 返回的可执行命令为准。**仅文字模式的预览是原生文字排版**，模型观察中的 `cameras` 为空。

三维视觉模式由官方 `alfworld.env.thor_env.ThorEnv` 恢复对应 ALFRED 场景与任务，提供一张 **640×480 第一人称 RGB**、可见物体状态、任务指令与动作反馈。模型使用 `MoveAhead`、`RotateRight` 等低层动作以及下方 JSON 物体交互；清洁、加热、冷却等任务由官方环境维护状态。策略不获得 Oracle 导航、专家计划、PDDL 参数或全场景物体坐标。

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

ALFWorld 三维视觉和 AI2-THOR 均要求物体句柄来自当前环境；适配器在内部反解为原生 ID 后调用官方环境，未知句柄或直接提交原生 ID 都会被拒绝。两者均只有原生第一人称相机。调试、瞬移、`forceAction`、直接改物体位置等参数不开放给策略动作接口；控制对象不可见或原生交互失败时，记录去除原生坐标标识的错误供下一轮调整。ALFWorld 2.1 的 `PutObject` 参数由适配器按当前手持物品和目标容器转换，不需要模型自行构造旧版 SDK 参数。

## 成功判定、日志与回放

- ALFWorld 文字模式由官方 `GameState.won` 判定，来源标记为 `alfworld_native_task`。
- ALFWorld 三维视觉由官方 `ThorEnv.get_goal_satisfied()` 判定，来源为 `alfworld_thor_native_task`；`get_goal_conditions_met()` 提供满足条件数与总数，用于结果分析。
- AI2-THOR 由原生目标物体状态或物品栏判定，来源标记为 `ai2thor_native_task`。这些是本平台明确限定的原生状态任务，不冒充 ALFRED 完整基准成功率。
- 模型输出“完成”只触发环境验收，不能改变判定结果。PDDL 事实、专家解法、全场景元数据和评价器成功值不注入模型观察。
- `state`、模型观察和轨迹检查点声明 `time_unit="environment_step"`，环境时间以动作步为单位，`timestep=1`。实验指标用 `environment_steps` 记录步数，`sim_time_s` 为 `null`；墙钟总耗时和模型推理耗时仍分别以秒记录，不能把环境步当作连续动力学秒数分析。评价器同时提供任务名称、指令和明确判据，供界面展示；原生成功证据仍只用于独立验收。
- 检查点恢复和轨迹回放通过官方 SDK 重置回合并重新执行保存的动作，再比对最终原生观察。它是**原生动作重放**；输入校验在重置前完成。如果重置或重放后失败、观察不一致，当前环境会标记为未就绪，请在“仿真环境”中重新加载；不会继续操作部分恢复的回合。它不承诺 Unity 物理逐位一致。
- AI2-THOR 不透明动作句柄使用 `ai2thor-integration-v2` 状态格式；旧版直接使用原生 ID 的动作轨迹不兼容，不能跨格式恢复。
- ALFWorld 三维视觉使用 `alfworld-thor-lowlevel-v1`，同时记录 ALFWorld 与原生 AI2-THOR 版本；其检查点不与文字模式或 AI2-THOR 5.0 共用。

## 实际验收与复测

本机验收分别使用 ALFWorld 0.4.2 / TextWorld 1.7.0、ALFWorld 0.4.2 / AI2-THOR 2.1.0，以及独立 AI2-THOR 5.0.0：

ALFWorld 文字模式加载后直接展示“环境交互”中的官方任务、文字观察与可执行命令。文字预览图仍用于录像和回放，工作区不再重复显示同一份文字图片。用户保存的布局顺序保留。家庭环境的布局种子均接受 `0–4294967295`，从对应官方回合或目标目录中选择任务。

- ALFWorld 文字：四项原生与十项契约检查通过；六个任务族、种子 42 的官方解法均触发真实 `won=true`。覆盖文字观察、空相机列表、过期动作拒绝、原生状态转换、回放与种子重复选择；独立 THOR 测试在文字运行环境中按设计跳过，不计作通过。
- ALFWorld 三维视觉：安装器完成真实启动验证，得到一张 640×480 原生相机画面；五项原生与十五项契约检查全部通过。覆盖六个任务族、种子 42 的场景加载及初始未完成判定、旋转导致画面变化、检查点重放与旧动作拒绝。取物放置正例既验证原始官方参考动作，也去除 `forceAction` 并通过公开低层接口完成；这验证接口与原生目标，不代表模型自主求解六类任务。
- ALFWorld 视觉工作台链路：一项真实 Worker / Agent 集成通过，覆盖原生 RGB、公开 `RotateRight` 动作、环境拒绝假完成、轮数终止、日志与轨迹、可解码视频、动作回放以及切回 MuJoCo。只用本地固定响应替代模型推理，未调用收费模型。
- AI2-THOR：七项原生回归通过，覆盖四个房间与三类任务的全部 12 种目录组合、原生 RGB 和初始未完成判定，以及旋转画面变化和导航动作重放。`FloorPlan1` 的公开 `OpenObject`、`PickupObject`、`ToggleObjectOn` 动作分别改变目标状态并触发成功。正例的导航定位使用 SDK 交互位姿作测试准备，不代表模型能自主导航；实际目标交互不使用 `forceAction` 或直接改物体状态。
- 文字与 AI2-THOR 工作台链路：两项 ALFWorld 文字 Worker / Agent 回归和一项 AI2-THOR 5.0 Worker 回归通过，覆盖原生观察、动作与独立验收、记录与回放，以及恢复异常时不继续发布部分状态。
- UI：手动交互锁定、单请求执行、原生失败原因、旧环境事件与迟到响应隔离、SDK 文本安全显示。

已验证的安装契约另覆盖元数据配对、文件保留、路径边界及失败后的状态失效。上述覆盖范围区分目录样本加载、固定动作正例与工作台集成；没有把 12 种房间/任务组合的加载都计为成功求解。

Windows 下可复测原生 SDK，主环境不需要安装这些可选包：

```powershell
wsl -d Ubuntu --cd /mnt/d/workspace/机械臂具身仿真研究工作台 --exec /usr/bin/env EMBODI_TEST_ALFWORLD=1 /home/administrator/.local/share/embodiworks/alfworld/bin/python -m unittest discover -s tests -p test_household_backends.py -v
wsl -d Ubuntu --cd /mnt/d/workspace/机械臂具身仿真研究工作台 --exec /usr/bin/env EMBODI_TEST_ALFWORLD_THOR=1 /home/administrator/.local/share/embodiworks/alfworld_thor/bin/python -m unittest discover -s tests -p test_alfworld_thor.py -v
wsl -d Ubuntu --cd /mnt/d/workspace/机械臂具身仿真研究工作台 --exec /usr/bin/env EMBODI_TEST_THOR=1 /home/administrator/.local/share/embodiworks/ai2thor/bin/python -m unittest discover -s tests -p test_household_backends.py -v
```

替换示例中的工作区和 WSL 用户路径；不要在同一台内存受限机器同时启动多个重型 SDK。前端回归使用 `node --test tests/test_environment_ui.mjs`。测试结果区分普通契约检查、原生环境验收与可选跳过，成功接入 SDK 不表示模型策略质量已得到证明。

主 Python 环境还可验证完整 Worker → 独立进程 → SDK → Agent 闭环。推理由固定本地测试动作替代，不调用收费模型；日志、检查点和视频放在临时目录，测试结束后清理：

```powershell
$env:EMBODI_TEST_HOUSEHOLD='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_household_integration.py -v
Remove-Item Env:\EMBODI_TEST_HOUSEHOLD
```

ALFWorld 文字集成回归检查错误“完成”声明被环境拒绝、中间实际执行原生命令、文字观察与视频归档，以及切换引擎后旧动作和旧轨迹被拒绝。AI2-THOR 桥接测试使用独立开关 `EMBODI_TEST_HOUSEHOLD_THOR=1`。

ALFWorld 三维视觉的 Worker / Agent、录像与回放验收单独启用：

```powershell
$env:EMBODI_TEST_ALFWORLD_THOR_WORKBENCH='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_household_integration.py -k ALFWorldVisualIntegrationTests -v
Remove-Item Env:\EMBODI_TEST_ALFWORLD_THOR_WORKBENCH
```

视觉原生回归包含六个任务族样本加载、导航导致真实 RGB 变化、官方目标正负例、低层公开动作与回放；正例使用固定参考动作，验证接口与任务设置，不代表模型自主求解能力。视觉集成回归用本地固定响应执行动作，检查假完成拒绝和真实视频归档。两类测试都会启动 Unity，应与其他重型 SDK 串行运行。
