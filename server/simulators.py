"""Supported simulator identities and capabilities; importing never loads optional engines."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = {'mujoco': ('mujoco_privileged_state', 'task-evaluation-v2'),
              'genesis': ('genesis_privileged_state', 'genesis-task-v1')}
COMMON = ['load', 'scene', 'reset', 'pause', 'step', 'control', 'home', 'stop',
          'instruction', 'observe', 'move_tcp', 'policy_chunk', 'checkpoint', 'restore',
          'replay_frame', 'settle', 'view', 'camera', 'speed', 'run_limits', 'verify_task',
          'render_suspend', 'select_robot']


def runtime_python(engine_id):
    if engine_id != 'genesis':
        raise ValueError('该仿真后端尚未实现本地适配。')
    folder = ROOT / '.venv-genesis'
    return folder / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')


def backend_identity(engine_id='mujoco', version=None):
    if engine_id not in EVALUATORS:
        raise ValueError('未知或未经适配的仿真后端。')
    if version is None and engine_id == 'mujoco':
        import mujoco
        version = mujoco.__version__
    source, evaluator = EVALUATORS[engine_id]
    return {'engine_id': engine_id, 'engine_version': version,
            'evaluator_source': source, 'evaluator_version': evaluator,
            'state_schema': f'{engine_id}-integration-v1',
            'control_frame': 'world_z_up_m_wxyz',
            'capabilities': COMMON + (['demo', 'flags', 'gravity', 'drop'] if engine_id == 'mujoco' else [])}


def supported_catalog(engine_id='mujoco'):
    from .simulation import MODELS
    from .scenes import SCENES, TASKS
    if engine_id not in EVALUATORS: raise ValueError('未知仿真后端。')
    if engine_id == 'mujoco': return MODELS, SCENES, TASKS
    scenes = {'tabletop': SCENES['tabletop']}
    return {'panda': MODELS['panda']}, scenes, {key: TASKS[key] for key in scenes['tabletop']['tasks']}


def validate_selection(engine_id, robot, scene_id, task_id):
    models, scenes, _ = supported_catalog(engine_id)
    if robot not in models: raise ValueError(f'{engine_id} 当前不支持机械臂 {robot}。')
    if scene_id not in scenes: raise ValueError(f'{engine_id} 当前不支持场景 {scene_id}。')
    if task_id not in scenes[scene_id]['tasks']: raise ValueError('当前场景不支持该任务。')


def catalog(active='mujoco', session_id=None):
    import mujoco
    python = runtime_python('genesis')
    folder = python.parent.parent
    pattern = 'Lib/site-packages/genesis_world-*.dist-info' if sys.platform == 'win32' else 'lib/python*/site-packages/genesis_world-*.dist-info'
    distributions = sorted(folder.glob(pattern))
    installed = python.is_file() and bool(distributions)
    genesis_version = distributions[-1].name.removeprefix('genesis_world-').removesuffix('.dist-info') if distributions else None
    values = [
        {'id': 'mujoco', 'name': 'MuJoCo', 'version': mujoco.__version__, 'implemented': True,
         'installed': True, 'status': 'available', 'description': '现有完整机械臂、复杂场景、多臂和任务评价。',
         'requirements': '工作台主 Python 环境', 'docs': 'https://mujoco.readthedocs.io/'},
        {'id': 'genesis', 'name': 'Genesis World', 'version': genesis_version, 'implemented': True,
         'installed': installed, 'status': 'available' if installed else 'not_installed',
         'description': '独立进程中的原生 Genesis 物理、Panda、三相机与 IK。当前支持桌面三项任务。',
         'requirements': '独立 .venv-genesis；首次加载需编译物理内核，请耐心等待。详见 docs/simulators.md。',
         'docs': 'https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html'},
        {'id': 'maniskill', 'name': 'ManiSkill / SAPIEN', 'description': '标准操作任务、控制器和原生成功判据；待实现适配器。',
         'requirements': 'Windows CPU 物理 + Vulkan 渲染；Linux 可用 GPU 物理。', 'docs': 'https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html'},
        {'id': 'isaac', 'name': 'Isaac Sim / Isaac Lab', 'description': '高保真传感器与训练生态；Lab 3.0 另有 Newton 工作流。待实现适配器。',
         'requirements': '完整 Isaac Sim 官方最低 32 GB RAM / 16 GB VRAM；建议后续独立 GPU 服务。', 'docs': 'https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html'},
        {'id': 'omnigibson', 'name': 'OmniGibson / BEHAVIOR', 'description': '家庭长程任务和语义评价；待实现适配器。',
         'requirements': '基于 Isaac Sim；独立环境、资产许可与至少 32 GB RAM。', 'docs': 'https://behavior.stanford.edu/getting_started/installation'},
    ]
    for item in values:
        if item['id'] in EVALUATORS:
            models, scenes, _ = supported_catalog(item['id'])
            item.update(models=list(models), scenes=list(scenes), capabilities=backend_identity(item['id'])['capabilities'])
        else:
            item.update(version=None, implemented=False, installed=False, status='planned', models=[], scenes=[], capabilities=[])
    return {'active': active, 'session_id': session_id, 'backends': values}
