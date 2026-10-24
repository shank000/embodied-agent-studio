"""Supported simulator identities and capabilities; importing never loads optional engines."""
from pathlib import Path, PurePosixPath
import importlib
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = {'mujoco': ('mujoco_privileged_state', 'task-evaluation-v2'),
              'genesis': ('genesis_privileged_state', 'genesis-task-v1'),
              'maniskill': ('maniskill_official_evaluate', 'maniskill-task-v1'),
              'libero': ('libero_bddl_state', 'libero-bddl-v1'),
              'alfworld': ('alfworld_native_task', 'alfworld-task-v1'),
              'ai2thor': ('ai2thor_native_task', 'ai2thor-task-v1'),
              'isaac': ('isaaclab_newton_state', 'isaaclab-task-v1'),
              'isaacsim': ('isaacsim_privileged_state', 'isaaclab-task-v1'),
              'robocasa365': ('robocasa_official_state', 'robocasa365-task-v1')}
RUNTIMES = {
    'robocasa365': ('robocasa', 'RoboCasa365', 'robocasa_backend', 'RoboCasaSimulation', 'https://robocasa.ai/'),
    'genesis': ('genesis_world', 'Genesis World', 'genesis_backend', 'GenesisSimulation', 'https://genesis-world.readthedocs.io/'),
    'maniskill': ('mani_skill', 'ManiSkill / SAPIEN', 'maniskill_backend', 'ManiSkillSimulation', 'https://maniskill.readthedocs.io/'),
    'libero': ('libero', 'LIBERO', 'libero_backend', 'LiberoSimulation', 'https://github.com/Lifelong-Robot-Learning/LIBERO'),
    'alfworld': ('alfworld', 'ALFWorld', 'alfworld_backend', 'ALFWorldSimulation', 'https://github.com/alfworld/alfworld'),
    'ai2thor': ('ai2thor', 'AI2-THOR', 'thor_backend', 'ThorSimulation', 'https://ai2thor.allenai.org/'),
    'isaac': ('isaaclab', 'Isaac Lab / Newton', 'isaac_backend', 'IsaacSimulation', 'https://isaac-sim.github.io/IsaacLab/'),
    'isaacsim': ('isaacsim', 'Isaac Sim', 'isaac_backend', 'IsaacSimSimulation', 'https://docs.isaacsim.omniverse.nvidia.com/'),
}
COMMON = ['load', 'scene', 'reset', 'pause', 'step', 'control', 'home', 'stop',
          'instruction', 'observe', 'move_tcp', 'policy_chunk', 'checkpoint', 'restore',
          'replay_frame', 'settle', 'view', 'camera', 'speed', 'run_limits', 'verify_task',
          'render_suspend', 'select_robot']


def runtime_python(engine_id):
    if engine_id not in RUNTIMES:
        raise ValueError('该仿真后端尚未实现本地适配。')
    folder = ROOT / f'.venv-{engine_id}'
    return folder / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')


def runtime_info(engine_id):
    python = runtime_python(engine_id)
    marker = ROOT / f'.venv-{engine_id}' / 'runtime.json'
    if marker.is_file():
        value = json.loads(marker.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict) or any(key in value and type(value[key]) is not bool for key in ('validated', 'assets_ready')):
            raise ValueError('运行环境配置须为对象，validated 与 assets_ready 须为布尔值。')
        if value.get('kind') != 'wsl' or sys.platform != 'win32':
            raise ValueError('运行环境配置仅支持 Windows 上的固定 WSL 解释器。')
        distro, executable = value.get('distribution'), value.get('python')
        if (not isinstance(distro, str) or not distro or len(distro) > 100 or distro.startswith('-')
                or any(c in distro for c in '\r\n\x00') or not isinstance(executable, str)
                or not executable.startswith('/') or any(c in executable for c in '\r\n\x00')
                or PurePosixPath(executable).name not in ('python', 'python3', 'python3.11', 'python3.12')):
            raise ValueError('WSL 运行环境须提供发行版名称与绝对 Python 路径。')
        return {**value, 'installed': value.get('validated') is True and value.get('assets_ready') is True}
    folder = python.parent.parent
    package = RUNTIMES[engine_id][0]
    pattern = f'Lib/site-packages/{package}-*.dist-info' if sys.platform == 'win32' else f'lib/python*/site-packages/{package}-*.dist-info'
    distributions = sorted(folder.glob(pattern))
    return {'kind': 'native', 'python': str(python), 'installed': python.is_file() and bool(distributions),
            'version': distributions[-1].name.removeprefix(package+'-').removesuffix('.dist-info') if distributions else None}


def runtime_command(engine_id, generation_base=0, python_path=None, runtime_script=None):
    script = runtime_script or ROOT/'server'/'backends'/'runtime.py'
    if python_path is not None:
        python = Path(python_path)
        if not python.is_file(): raise ValueError(f'未找到独立 Python 环境：{python}。请按 docs/simulators.md 安装后再加载。')
        return [str(python), '-u', str(script), '--engine', engine_id,
                '--generation-base', str(generation_base)]
    value = runtime_info(engine_id)
    if not value['installed']:
        raise ValueError(f'未找到 {RUNTIMES[engine_id][1]} 独立运行环境。请按 docs/simulators.md 安装并完成资源准备。')
    arguments = ['--engine', engine_id, '--generation-base', str(generation_base)]
    if value['kind'] == 'wsl':
        root = ROOT.resolve()
        if len(root.drive) != 2 or root.drive[1] != ':': raise ValueError('WSL 运行需要本机盘符下的工作区。')
        linux_root = '/mnt/' + root.drive[0].lower() + root.as_posix()[2:]
        return ['wsl.exe', '--distribution', value['distribution'], '--cd', linux_root, '--exec',
                value['python'], '-u', '-m', 'server.backends.runtime', *arguments]
    return [value['python'], '-u', str(script), *arguments]


def backend_identity(engine_id='mujoco', version=None):
    if engine_id not in EVALUATORS:
        raise ValueError('未知或未经适配的仿真后端。')
    if version is None and engine_id == 'mujoco':
        import mujoco
        version = mujoco.__version__
    source, evaluator = EVALUATORS[engine_id]
    discrete = engine_id in ('alfworld', 'ai2thor')
    return {'engine_id': engine_id, 'engine_version': version,
            'evaluator_source': source, 'evaluator_version': evaluator,
            'state_schema': 'ai2thor-integration-v2' if engine_id == 'ai2thor' else f'{engine_id}-integration-v1',
            'control_frame': None if discrete else 'world_z_up_m_wxyz',
            'action_mode': ('text' if engine_id == 'alfworld' else 'discrete') if discrete else 'tcp_pose',
            'capabilities': (['load','scene','reset','pause','stop','instruction','environment_action','observe','verify_task',
                              'run_limits','checkpoint','restore','replay_frame','render_suspend'] if discrete else
                             COMMON + (['demo', 'flags', 'gravity', 'drop'] if engine_id == 'mujoco' else []))}


def supported_catalog(engine_id='mujoco'):
    from .simulation import MODELS
    from .scenes import SCENES, TASKS
    if engine_id not in EVALUATORS: raise ValueError('未知仿真后端。')
    if engine_id == 'mujoco': return MODELS, SCENES, TASKS
    if engine_id == 'libero':
        from .backends.libero_catalog import catalog, MODELS as libero_models
        scenes, tasks = catalog()
        return libero_models, scenes, tasks
    if engine_id == 'robocasa365':
        from .backends.robocasa_catalog import MODELS, SCENES, TASKS
        return MODELS, SCENES, TASKS
    if engine_id != 'genesis':
        module = importlib.import_module('.backends.'+RUNTIMES[engine_id][2], __package__)
        return module.MODELS, module.SCENES, module.TASKS
    scenes = {'tabletop': SCENES['tabletop']}
    return {'panda': MODELS['panda']}, scenes, {key: TASKS[key] for key in scenes['tabletop']['tasks']}


def validate_selection(engine_id, robot, scene_id, task_id):
    models, scenes, _ = supported_catalog(engine_id)
    if robot not in models: raise ValueError(f'{engine_id} 当前不支持机械臂 {robot}。')
    if scene_id not in scenes: raise ValueError(f'{engine_id} 当前不支持场景 {scene_id}。')
    if task_id not in scenes[scene_id]['tasks']: raise ValueError('当前场景不支持该任务。')


def catalog(active='mujoco', session_id=None):
    import mujoco
    values = [
        {'id': 'mujoco', 'name': 'MuJoCo', 'version': mujoco.__version__, 'implemented': True,
         'installed': True, 'status': 'available', 'description': '现有完整机械臂、复杂场景、多臂和任务评价。',
         'requirements': '工作台主 Python 环境', 'docs': 'https://mujoco.readthedocs.io/'},
    ]
    descriptions = {
        'robocasa365': ('官方 65 项原子与 300 项复合厨房任务；移动 PandaOmron、三相机和原生成功判据。', '独立 Python 3.11 .venv-robocasa365；需准备所选场景与物品资产，详见 docs/robocasa365.md。'),
        'genesis': ('原生 Genesis 物理、Panda、三相机与 IK；桌面三项任务。', '独立 .venv-genesis；首次加载编译物理内核。'),
        'maniskill': ('官方 PickCube / PushCube；SAPIEN/PhysX、原生图像和官方成功判据。', '独立 .venv-maniskill；Windows CPU 物理 + Vulkan，Linux 可用 GPU。'),
        'libero': ('官方五套任务、Panda、robosuite 控制器与 BDDL 成功谓词。', '独立 .venv-libero，准备官方任务、初态与场景资产，无需演示训练集。'),
        'alfworld': ('官方 ALFWorld / TextWorld 家务任务；文字观察与可执行命令。', 'Linux/macOS；本机 Windows 通过已配置的 Ubuntu WSL 运行。文字画面不冒充三维相机。'),
        'ai2thor': ('官方 Unity 房间、视觉导航与物体交互，依据 Unity 元数据验收。', '本机通过 Ubuntu WSLg 运行官方 Linux Unity；首次加载需下载官方构建。'),
        'isaac': ('Isaac Lab / Newton 机器人场景、原生控制与相机；独立运行无需完整 Isaac Sim。', '独立 Python 3.12 .venv-isaac；CUDA 与显卡驱动须匹配所选物理和相机。'),
        'isaacsim': ('Isaac Lab 的 Isaac Sim 物理与渲染路径；与 Newton 单独记录。', '独立 .venv-isaacsim；完整 Isaac Sim 需满足官方硬件条件并手动完成 EULA。'),
    }
    for engine_id, (_, name, module, _, docs) in RUNTIMES.items():
        error = None
        try: info = runtime_info(engine_id)
        except (ValueError, OSError) as exc: info = {'installed': False, 'version': None}; error = str(exc)
        implemented = importlib.util.find_spec(f'{__package__}.backends.{module}') is not None
        installed = bool(info['installed'])
        description, requirement = descriptions[engine_id]
        values.append({'id': engine_id, 'name': name, 'version': info.get('version'),
                       'implemented': implemented, 'installed': installed,
                       'status': 'available' if implemented and installed else 'not_installed',
                       'description': description, 'requirements': requirement, 'docs': docs,
                       'runtime_kind': info.get('kind'), 'error': error})
    for item in values:
        if item['implemented']:
            models, scenes, _ = supported_catalog(item['id'])
            item.update(models=list(models), scenes=list(scenes), capabilities=backend_identity(item['id'])['capabilities'])
        else:
            item.update(models=[], scenes=[], capabilities=[])
    return {'active': active, 'session_id': session_id, 'backends': values}
