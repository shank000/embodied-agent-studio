# 安装与平台说明

[返回首页](../README.md) · [使用指南](USAGE.md)

EmbodiWorks 的界面、API 与仿真共用一份源码。启动器会按系统寻找 `.venv/Scripts/python.exe` 或 `.venv/bin/python`，不依赖终端已激活哪个 Python，也不会静默切换到系统环境。

## 验证范围

| 环境 | 已验证 | 尚未验证 |
| --- | --- | --- |
| Windows | 工作台、真实模型、图像渲染、实验记录与本地接口回归 | 各供应商与各型号的真实账号兼容性需用户验证 |
| Ubuntu 26.04 / WSL2 x86_64 | Python 3.13.15、MuJoCo 3.13.0；Panda/SO-101 动力学、960×640 原生画面、同步三路相机、前端状态与启动器测试 | EGL、CUDA 策略与视频完整流程 |
| Ubuntu GitHub runner | 安装、构建、前端与启动器回归、真实模型运动；OSMesa 原生图像、暂停、换视角、单步、切换模型与恢复运行 | 完整实验与外部模型账号流程 |
| macOS arm64 GitHub runner | 安装、构建、前端与启动器回归、CLI 路径、Panda/SO-101 真实模型运动 | Mac 真机 CGL 渲染、策略与完整工作台流程 |

WSL2 本轮使用 WSLg 的 GLFW，OpenGL 实际为 llvmpipe 软件渲染；这项验证不能当作 NVIDIA GPU 性能测试。CI 分为三平台基础检查和独立的 Linux OSMesa 原生渲染检查。

本轮 Windows 标准 runner 报告 WGL 驱动不可用，macOS 标准 runner 在 CGL 选择像素格式时报告 `invalid pixel format`（[原始检查记录](https://github.com/shank000/embodied-agent-studio/actions/runs/39941573370)）。因此 CI 不把它们当作图像验收环境，也不通过改测试或静默切换后端掩盖错误；Windows 图像已在本机验证，Mac 原生图像仍需有可用 OpenGL 的真机确认。

## 1. 准备工具与源码

- Python **3.13**，Node.js **22.12 或以上受支持版本**，Git。
- 能创建 OpenGL 上下文的图形驱动。Linux 无桌面环境见下文。
- 可选视频录制：安装带 `libx264` 的 [FFmpeg](https://ffmpeg.org/download.html)，将 `ffmpeg` 与 `ffprobe` 放入 PATH。

```sh
git clone https://github.com/shank000/embodied-agent-studio.git
cd embodied-agent-studio
```

在不同系统分别创建环境；不要复用 Windows 的 `.venv`、`.venv-policy` 或 `node_modules`。

**Windows PowerShell：**

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Linux / macOS：**

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

若 Python 3.13 不是默认 Python，请使用实际安装路径。macOS 的 Apple Silicon 环境应使用原生 arm64 Python 与 Node，避免混用不同架构的依赖。

## 2. 下载机器人资源

以下 Git 命令适用于三个系统。Panda 与 SO-101 使用固定的 MuJoCo Menagerie 提交：

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
git -C vendor/mujoco_menagerie fetch --depth 1 origin 822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7
git -C vendor/mujoco_menagerie checkout 822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7
git -C vendor/mujoco_menagerie sparse-checkout set franka_emika_panda robotstudio_so101
```

RM65 + GaiaHand 从厂商仓库单独下载：

```sh
# 先完成下面的 npm ci，再运行；启动器会选中本机 .venv。
node tools/python.mjs tools/download_robot_models.py
```

模型不打包进源码。RM65 上游许可尚未明确，下载和使用须核对厂商条款；公开 GaiaHand 为 15 关节版本。详见[模型来源](../MODEL_SOURCES.md)。已有 `vendor` 时检查当前提交后再更新，勿直接覆盖个人修改。

## 3. 构建与运行

```sh
npm ci
node tools/python.mjs tools/download_robot_models.py
npm run build
npm start
```

打开 **http://127.0.0.1:8000**。一个后端进程同时提供网页、仿真与 API；在终端按 **Ctrl+C** 退出。Windows 也可双击根目录的 `启动MuJoCo工作台.cmd`。

前端开发另开终端执行 `npm run dev`，访问 http://127.0.0.1:5173。修改 Python 后需要重启服务；修改前端后重新构建供 8000 端口使用。重启会停止运行中的实验，并清除仅在内存中的 API Key；实验日志和视频保留。

## Linux 无头环境

普通 Linux 桌面或 WSLg 可先使用默认 GLFW。没有 DISPLAY 的服务器需要可用的离屏 OpenGL 后端，例如：

```sh
# 安装并配置了 NVIDIA/Mesa EGL 驱动后：
MUJOCO_GL=egl npm start

# 安装了 Mesa OSMesa 库后，以软件渲染运行：
MUJOCO_GL=osmesa npm start
```

这两个变量只选择后端，不会自动安装驱动。在 Debian/Ubuntu，软件渲染所需系统库通常为 `libosmesa6`；EGL 依赖实际显卡驱动与容器 GPU 透传。不要通过静默更换后端掩盖图形初始化错误，先检查终端中的原始报错。依据：[MuJoCo Python 与渲染说明](https://mujoco.readthedocs.io/en/stable/python.html#rendering)。

macOS 的本项目使用 `mujoco.Renderer` 离屏画面，经 CGL 创建上下文，不使用 `viewer.launch_passive`；该 viewer 对 `mjpython` 的要求不能直接套用到本工作台。MuJoCo 的 [CGL 实现](https://github.com/google-deepmind/mujoco/blob/3.13.0/python/mujoco/cgl/__init__.py)请求硬件加速与多重采样像素格式，无法提供该格式的虚拟环境会初始化失败。Mac 完整运行仍待实机确认。

## 模型接入

网页模型卡片支持 Codex 登录、OpenAI、Claude、兼容 API 与自定义 Agent。项目内固定版本的 Codex CLI 随 `npm ci` 安装；程序按当前 OS 和 CPU 架构选择原生文件。也可安装本机原生 CLI，或通过 `CODEX_CLI_PATH` 指定原生可执行文件。

使用 Codex 时先在该系统执行 `npx --no-install codex login`。WSL 请从 Ubuntu 终端启动，并在 Linux 内分别安装 Node 依赖和登录 Codex；不要使用继承自 Windows PATH 的 npm 包装脚本。工作台不复制 Windows 登录令牌。其余服务在页面填写地址、模型与 Key；密钥仅存内存。

## 可选：SmolVLA 策略环境

视觉 Agent 和外部 VLA/WAM 接口不需要本地 Torch。默认 SmolVLA 使用独立 `.venv-policy`，避免与 MuJoCo 环境混装。

Windows 可运行 `安装策略环境.cmd`（需 [uv](https://docs.astral.sh/uv/)），NVIDIA CUDA 环境可运行 `安装策略环境.cmd cuda`。Linux CPU 环境：

```sh
python3.13 -m venv .venv-policy
.venv-policy/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv-policy/bin/python -m pip install -r requirements-policy.txt
```

macOS 使用 PyPI 的本机原生轮子：

```sh
python3.13 -m venv .venv-policy
.venv-policy/bin/python -m pip install -r requirements-policy.txt
```

Linux CUDA 请先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装与固定 Torch / torchvision 版本相符的包，再安装其余依赖；不要使用 Windows 专用的 `+cpu` / `+cu128` 安装命令套用到 Mac。

界面目前只提供 `cpu` 和 `cuda`；macOS 选 CPU，尚未实现 MPS。首次运行会下载 `lerobot/smolvla_base`；基础权重未在本项目任务上微调。策略环境与模型实际运行仍需分别验证，安装成功不代表任务成功。

## 检查

```sh
npm run test:launcher
npm run test:ui
npm test
npm run build
```

完整 Python 回归需要所有机器人资源，视频测试还需要 FFmpeg / ffprobe。普通测试不调用付费模型；`tests/live_agent_smoke.py` 只有显式运行才消耗模型额度。[跨平台 CI](../.github/workflows/platform-smoke.yml)运行三平台基础检查和 Linux 渲染 smoke，不代表所有平台的全部功能已验收。
