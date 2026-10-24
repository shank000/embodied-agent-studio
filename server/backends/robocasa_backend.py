"""Official RoboCasa365 kitchens and PandaOmron, in an isolated MuJoCo runtime.

The native mobile controller has normalized commands, not twelve joint angles.
World TCP actions use the same official OSC with its documented absolute/world
settings. Native policy chunks restore the official delta/base controller mode.
"""
from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image

from ..actions import PoseAction
from ..simulators import backend_identity
from .robocasa_catalog import MODELS, SCENES, TASKS, SOURCE_COMMIT, SOURCE_URL

CAMERAS = {'front': 'robot0_agentview_left', 'side': 'robot0_agentview_right', 'wrist': 'robot0_eye_in_hand'}
ACTION_NAMES = ['arm_dx', 'arm_dy', 'arm_dz', 'arm_drx', 'arm_dry', 'arm_drz',
                'gripper_command', 'base_vx_command', 'base_vy_command',
                'base_yaw_command', 'torso_command', 'base_mode']
CAPABILITIES = ['load', 'scene', 'reset', 'pause', 'step', 'control', 'stop',
    'observe', 'move_tcp', 'policy_chunk', 'checkpoint', 'restore', 'replay_frame',
    'settle', 'view', 'camera', 'speed', 'run_limits', 'verify_task',
    'render_suspend', 'select_robot']


def runtime_cache():
    default = Path(Path(sys.prefix).anchor)/'EmbodiWorksCache/robocasa365' if os.name == 'nt' else Path.home()/'.cache/embodiworks/robocasa365'
    return Path(os.getenv('EMBODI_ROBOCASA_CACHE', default))/SOURCE_COMMIT[:12]


def configure_runtime():
    """Preserve upstream sources; Windows native readers need ASCII filenames."""
    folder = runtime_cache()
    if os.name == 'nt' and not str(folder).isascii():
        raise ValueError('RoboCasa365 Windows 缓存路径须为英文；请设置 EMBODI_ROBOCASA_CACHE。')
    for package in ('robocasa', 'robosuite'):
        spec = importlib.util.find_spec(package)
        if spec is None:
            raise ValueError(f'{package} 未安装。请在独立 .venv-robocasa365 安装 requirements-robocasa365.txt。')
        source = Path(spec.origin).parent
        target = folder/package
        if not (target/'.embodiworks-source-copied').is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            (target/'.embodiworks-source-copied').write_text(str(source), encoding='utf-8')
    assets = folder/'robocasa/models/assets'
    missing = [str(assets/name) for name in ('textures', 'fixtures/cabinets', 'fixtures/sinks') if not (assets/name).is_dir()]
    if missing:
        raise ValueError('RoboCasa365 厨房资产未安装：'+'、'.join(missing)+'。请按 docs/robocasa365.md 安装官方资产；无需训练数据。')
    sys.path.insert(0, str(folder))
    importlib.invalidate_caches()
    # Official robosuite macros document disabling EGL on Windows. Configure
    # them before package __init__ imports bindings; no SDK source is patched.
    if os.name == 'nt':
        macro_spec = importlib.util.spec_from_file_location('robosuite.macros', folder/'robosuite/macros.py')
        macro = importlib.util.module_from_spec(macro_spec)
        sys.modules['robosuite.macros'] = macro
        macro_spec.loader.exec_module(macro)
        macro.MUJOCO_GPU_RENDERING = False
        macro.FILE_LOGGING_LEVEL = None
        os.environ['MUJOCO_GL'] = 'glfw'
    return folder


def rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])


def json_primitive(value, depth=0):
    if depth > 20: raise TypeError('Native state nesting is too deep or cyclic')
    if isinstance(value, np.ndarray):
        if value.size > 10000 or value.dtype.kind not in 'biuf': raise TypeError('Non-numeric or large native array')
        return {'__ew_numpy__': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape), 'value': value.tolist()}
    if isinstance(value, np.generic):
        return {'__ew_numpy__': 'scalar', 'dtype': value.dtype.str, 'value': value.item()}
    if isinstance(value, tuple):
        return {'__ew_tuple__': [json_primitive(item, depth+1) for item in value]}
    if isinstance(value, list):
        return [json_primitive(item, depth+1) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            return {'__ew_mapping__': [[json_primitive(key, depth+1), json_primitive(item, depth+1)] for key, item in value.items()]}
        return {key: json_primitive(item, depth+1) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError('Native SDK handles are not checkpoint values')


def primitive_state(instance):
    """Capture JSON-valued native task/controller state, excluding SDK handles."""
    result = {}
    for key, value in vars(instance).items():
        try:
            raw = json.dumps(json_primitive(value), allow_nan=False)
        except (TypeError, ValueError):
            continue
        result[key] = json.loads(raw)
    return result


def restore_primitive(value):
    if isinstance(value, dict) and '__ew_tuple__' in value:
        return tuple(restore_primitive(item) for item in value['__ew_tuple__'])
    if isinstance(value, dict) and '__ew_mapping__' in value:
        return {restore_primitive(key): restore_primitive(item) for key, item in value['__ew_mapping__']}
    if isinstance(value, dict) and '__ew_numpy__' in value:
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'biufUS': raise ValueError('检查点 NumPy dtype 无效。')
        array = np.asarray(value['value'], dtype=dtype)
        if value['__ew_numpy__'] == 'scalar': return array[()]
        if value['__ew_numpy__'] != 'array' or dtype.kind in 'US': raise ValueError('检查点 NumPy 标记无效。')
        shape = value['shape']
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape) or math.prod(shape) > 10000:
            raise ValueError('检查点 NumPy shape 无效。')
        return array.reshape(shape)
    if isinstance(value, list): return [restore_primitive(item) for item in value]
    if isinstance(value, dict): return {key: restore_primitive(item) for key, item in value.items()}
    return deepcopy(value)


def restore_primitives(instance, values):
    for key, value in values.items():
        if key not in vars(instance): raise ValueError(f'原生状态字段不存在：{key}')
        current = vars(instance)[key]
        try: restored = restore_primitive(value)
        except (ValueError, TypeError) as exc: raise ValueError(f'原生状态 {type(instance).__name__}.{key} 无效：{exc}') from exc
        if isinstance(current, np.ndarray) and isinstance(restored, np.ndarray) and current.shape == restored.shape: current[:] = restored
        elif isinstance(current, tuple) and isinstance(restored, list): setattr(instance, key, tuple(restored))
        else: setattr(instance, key, restored)


class RoboCasaSimulation:
    def __init__(self, model_id='panda_omron', render=True, scene_id='robocasa_atomic', seed=42, task_id=None):
        task_id = task_id or 'OpenDrawer'
        self._validate(model_id, scene_id, task_id)
        self.cache = configure_runtime()
        global Rotation, Slerp
        from scipy.spatial.transform import Rotation, Slerp
        import robocasa
        import robosuite
        from robosuite.controllers import load_composite_controller_config
        self._create_env = robosuite.make
        self._controller_config = load_composite_controller_config(controller=None, robot='PandaOmron')
        self.model_id, self.scene_id, self.task_id, self.seed = model_id, scene_id, task_id, seed
        self.render_enabled, self.render_suspended = render, False
        self.generation, self.speed, self.active_robot = 1, 1., 'arm1'
        self.backend = {**backend_identity('robocasa365', robocasa.__version__),
            'capabilities': CAPABILITIES.copy(), 'source_revision': SOURCE_COMMIT,
            'physics_engine': 'mujoco', 'runtime_packages': {name: version(name) for name in ('robocasa', 'robosuite', 'mujoco', 'numpy')},
            'evaluation_protocol': 'workbench_interactive_not_leaderboard',
            'action_space': {'type': 'native_mobile_commands', 'names': ACTION_NAMES.copy(),
                             'unit': 'normalized_command', 'control_frame': 'robot_base',
                             'controller': 'robosuite.HYBRID_MOBILE_BASE / OSC_POSE',
                             'base_mode': '-1 arm mode, +1 mobile-base mode'}}
        self.flags = {'contacts': False, 'joints': False, 'collision': False}
        self.events, self.event_id = deque(maxlen=30), 0
        self.env = None
        self._load()

    @staticmethod
    def _validate(model, scene, task):
        if model not in MODELS or scene not in SCENES or task not in SCENES[scene]['tasks']:
            raise ValueError('RoboCasa365 需要官方 PandaOmron 和所选任务分类中的任务。')

    def _load(self):
        self.close()
        # Use upstream target kitchens, physical assets and default object
        # registries. Do not remove task objects/distractors to make a task fit.
        try:
            if self.task_id != 'NavigateKitchen' and not (self.cache/'robocasa/models/assets/objects/objaverse').is_dir():
                raise FileNotFoundError('该任务需要官方默认 objaverse + lightwheel 物品注册库；请运行安装器 --stage assets --asset-set complete。')
            self.env = self._create_env(env_name=self.task_id, robots='PandaOmron',
                camera_names=list(CAMERAS.values()), camera_widths=640, camera_heights=426,
                seed=self.seed, controller_configs=deepcopy(self._controller_config), has_renderer=False,
                obj_instance_split='target', layout_and_style_ids=list(zip(range(1, 11), range(1, 11))),
                use_camera_obs=False, has_offscreen_renderer=self.render_enabled,
                control_freq=20, horizon=TASKS[self.task_id]['horizon'] or 1000, ignore_done=True,
                generative_textures=False, randomize_cameras=False)
            # RoboCasa deliberately uses load_model_on_init=False. The native
            # reset constructs the robot, scene and render context first.
            self._initialize_native()
        except Exception as exc:
            # A replacement may fail after native resources were closed or
            # partially allocated. This is fatal for this worker, rather than
            # a recoverable ValueError claiming the previous scene is ready.
            self.close()
            raise RuntimeError(f'官方 RoboCasa365 {self.task_id} 创建失败，仿真进程未就绪：{type(exc).__name__}: {exc}。'
                f'请检查 {self.cache}/robocasa/models/assets 中所需 fixtures/objects/textures；未自动下载或替换资产。') from exc

    @property
    def time(self): return float(self.sim.data.time)

    def reset(self):
        # RoboCasa's hard reset recreates all native objects. Release adapter
        # references before compiling the next kitchen so old NumPy views do
        # not keep a complete MjModel alive on memory-constrained desktops.
        self._load()

    def _initialize_native(self):
        # Official Counter builds collision-geometry RGBA with legacy
        # np.random.uniform (counter.py:369,474), outside env.rng. Seed both
        # official RNG paths in this isolated worker so full XML is repeatable.
        np.random.seed(self.seed); random.seed(self.seed)
        self.env.rng = np.random.default_rng(self.seed)
        self.env.reset()
        self.robot, self.sim = self.env.robots[0], self.env.sim
        self.timestep = 1./self.env.control_freq
        self.initial_time_s = float(self.sim.data.time)
        self.model_fingerprint = hashlib.sha256(self.env.model.get_xml().encode()).hexdigest()
        self.controller = self.robot.composite_controller
        self.arm = self.controller.part_controllers['right']
        self.gripper = self.robot.gripper['right']
        self.initial_joints = self.sim.data.qpos.copy()
        self.mode = 'native'
        self.native_action = np.zeros(12)
        self.native_action[-1] = -1.
        self._mode('tcp')
        self.hold_pose = self.tcp_pose()
        self.gripper_open = 1.
        self.motion, self.pose_goal, self.playback = None, None, None
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.physics_steps, self.ever_success = 0, False
        self.task = {'status': 'idle', 'title': '等待指令', 'steps': [], 'step': -1, 'progress': 0.}
        self.set_view('perspective')
        self._evaluate()

    def _mode(self, name):
        changed = getattr(self, 'mode', None) != name
        self.mode = name
        self.arm.input_type = 'absolute' if name == 'tcp' else 'delta'
        self.arm.input_ref_frame = 'world' if name == 'tcp' else 'base'
        if changed:
            # Goals are expressed in the selected reference frame. Never carry
            # a world goal over into the hybrid controller's base-relative mode.
            self.arm.goal_pos = self.arm.goal_ori = None
            self.arm.set_goal_update_mode('achieved')

    def tcp_pose(self):
        site = self.robot.eef_site_id['right']
        q = Rotation.from_matrix(np.asarray(self.sim.data.site_xmat[site]).reshape(3, 3)).as_quat()
        return {'frame': 'world', 'position_m': np.asarray(self.sim.data.site_xpos[site]).tolist(), 'quaternion_wxyz': q[[3, 0, 1, 2]].tolist()}

    def _evaluate(self):
        raw = bool(self.env._check_success())
        finite = bool(np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all())
        success = raw and finite
        self.ever_success |= success
        values = {'native_check_success': raw, 'layout_id': int(self.env.layout_id), 'style_id': int(self.env.style_id),
                  'official_task_name': self.task_id, 'instruction': self.env.get_ep_meta().get('lang', '')}
        if self.task_id == 'NavigateKitchen':
            body = self.sim.model.body_name2id('mobilebase0_base')
            pos = np.asarray(self.sim.data.body_xpos[body])
            yaw = Rotation.from_matrix(np.asarray(self.sim.data.body_xmat[body]).reshape(3, 3)).as_euler('xyz')[2]
            values.update(base_distance_m=float(np.linalg.norm(self.env.target_pos[:2]-pos[:2])),
                          facing_cosine=float(np.cos(self.env.target_ori[2]-yaw)))
        if hasattr(self.env, 'drawer'):
            values['drawer_normalized_opening'] = {k: float(v) for k, v in self.env.drawer.get_door_state(env=self.env).items()}
        checks = [{'key': 'official_success', 'label': '官方任务状态判据', 'actual': raw, 'expected': True, 'passed': raw},
                  {'key': 'state_finite', 'label': '物理状态数值有效', 'actual': finite, 'expected': True, 'passed': finite}]
        self.evaluation = {'task_id': self.task_id, **TASKS[self.task_id],
            'instruction': values['instruction'] or TASKS[self.task_id]['instruction'], 'source': 'robocasa_official_state',
            'version': 'robocasa365-task-v1', 'success': success, 'ever_success': self.ever_success,
            'currently_satisfied': success, 'status': 'success' if success else 'unsatisfied',
            'progress': 1. if success else 0., 'evaluated_at_sim_s': self.time,
            'values': values, 'checks': checks, 'failure_reasons': [c['label'] for c in checks if not c['passed']]}

    def robot_states(self):
        indexes = np.asarray(self.arm.joint_index, dtype=int)
        q = np.asarray(self.sim.data.qpos[self.arm.qpos_index])
        v = np.asarray(self.sim.data.qvel[self.arm.qvel_index])
        joints = [{'name': self.sim.model.joint_id2name(int(j)), 'position_rad': float(q[i]),
                   'velocity_rad_s': float(v[i]), 'limits_rad': self.sim.model.jnt_range[j].tolist()} for i, j in enumerate(indexes)]
        parts = self.controller.part_controllers
        base = parts['base']
        torso = parts['torso']
        body = self.sim.model.body_name2id('mobilebase0_base')
        base_rotation = Rotation.from_matrix(np.asarray(self.sim.data.body_xmat[body]).reshape(3, 3)).as_quat()
        # Controls are explicitly native commands. Measured joint positions
        # remain separate and retain their physical units.
        actuators = [{'index': i, 'name': name, 'robot_id': 'arm1', 'is_gripper': i == 6,
                      'unit': 'normalized_command', 'control_type': 'native_mobile_command',
                      'min': -1., 'max': 1., 'target': float(self.native_action[i]),
                      'actual': None, 'position': None, 'force': None} for i, name in enumerate(ACTION_NAMES)]
        return [{'id': 'arm1', 'name': 'PandaOmron · RoboCasa365', 'model': self.model_id, 'active': True,
                 'arm_dof': 7, 'hand_dof': 1, 'tcp': self.tcp_pose(), 'joints': joints, 'actuators': actuators,
                 'gripper': {'target_open_fraction': self.gripper_open,
                     'position': float(np.abs(self.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes['right']]).mean()), 'unit': 'm_per_finger'},
                 'mobile_base': {'position_world_m': self.sim.data.body_xpos[body].tolist(),
                     'quaternion_wxyz': base_rotation[[3, 0, 1, 2]].tolist(),
                     'joint_positions': self.sim.data.qpos[base.qpos_index].tolist(),
                     'joint_velocities': self.sim.data.qvel[base.qvel_index].tolist(),
                     'control_type': 'JOINT_VELOCITY', 'action_unit': 'normalized_command'},
                 'torso': {'position_m': self.sim.data.qpos[torso.qpos_index].tolist(),
                     'control_type': 'JOINT_POSITION', 'action_unit': 'normalized_delta_command'}}]

    def state(self):
        robots = self.robot_states()
        return {'engine': 'RoboCasa365 / MuJoCo', 'version': version('robocasa'), 'backend': self.backend.copy(),
            'model': self.model_id, 'name': 'PandaOmron · RoboCasa365', 'active_robot': 'arm1', 'robots': robots,
            'generation': self.generation, 'paused': self.paused, 'speed': self.speed, 'time': self.time,
            'timestep': self.timestep, 'steps': self.physics_steps, 'contacts': int(self.sim.data.ncon),
            'initial_time_s': self.initial_time_s,
            'contact_scope': 'whole_scene', 'gravity': True, 'view': self.view, 'flags': self.flags.copy(),
            'nq': self.sim.model.nq, 'nv': self.sim.model.nv, 'nu': 12, 'meshes': self.sim.model.nmesh,
            'arm_dof': 7, 'hand_dof': 1, 'mass': float(np.sum(self.sim.model.body_mass)),
            'actuators': robots[0]['actuators'], 'tcp': self.tcp_pose()['position_m'], 'tcp_pose': self.tcp_pose(),
            'scene_id': self.scene_id, 'scene_name': SCENES[self.scene_id]['name'], 'seed': self.seed,
            'task': self.task.copy(), 'evaluation': self.evaluation.copy(), 'playback': self.playback,
            'events': list(self.events), 'blocks': {}, 'has_wrist_camera': self.render_enabled,
            'source': SOURCE_URL, 'compute_backend': 'cpu', 'render_backend': 'mujoco_offscreen',
            'model_note': '官方 RoboCasa365 厨房资产与任务判据；PandaOmron 原生 OSC、移动底盘和升降躯干控制。12 维策略输出是归一化原生命令，非关节位置。'}

    def cancel_motion(self):
        self.motion, self.pose_goal = None, None
        if self.task['status'] in ('running', 'settling'): self.task['status'] = 'stopped'
        self._mode('tcp')
        self.hold_pose = self.tcp_pose()
        # PandaGripper.current_action[0] is its native commanded opening
        # (-1 closed, +1 open). Preserve it when a native chunk ends or stops.
        self.gripper_open = float(np.clip((self.gripper.current_action[0]+1)/2, 0., 1.))
        self.native_action[:] = 0.; self.native_action[-1] = -1.

    def move_tcp(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已切换，本轮动作失效。')
        pose = PoseAction.model_validate(payload['pose'])
        if pose.status != 'move' or pose.robot_id not in (None, 'arm1') or pose.hand_joint_positions_rad is not None:
            raise ValueError('PandaOmron 需要 arm1 的末端 move 动作。')
        if self.motion is not None: raise ValueError('请等待当前动作结束或先停止。')
        before = self.tcp_pose()
        if np.linalg.norm(np.asarray(pose.position_m)-before['position_m']) > .12+1e-8: raise ValueError('单轮末端平移超过 0.12 m。')
        if (rotation(pose.quaternion_wxyz)*rotation(before['quaternion_wxyz']).inv()).magnitude() > math.radians(35)+1e-8:
            raise ValueError('单轮末端姿态变化超过 35°。')
        self._mode('tcp'); self.pose_goal = pose
        self.motion = {'type': 'tcp', 'initial': before, 'target': pose.model_dump(), 'duration': pose.duration_s, 'elapsed': 0.}
        self.paused = False
        self.task = {'status': 'running', 'title': '官方 OSC 世界坐标末端控制 · 底盘保持', 'steps': ['move'], 'step': 0, 'progress': 0.}
        return {'robot_id': 'arm1', 'controller': 'robosuite.OSC_POSE / absolute/world', 'target': pose.model_dump()}

    def policy_chunk(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已切换，策略动作失效。')
        if self.motion is not None: raise ValueError('请等待当前动作结束或先停止。')
        if payload.get('robot_id') not in (None, 'arm1') or payload['joint_names'] != ACTION_NAMES:
            raise ValueError('RoboCasa365 policy_chunk 需要公开 action_space.names 的 12 维原生命令顺序；不是关节角。')
        actions, dt = np.asarray(payload['actions'], dtype=float), payload['dt_s']
        if actions.ndim != 2 or actions.shape[1] != 12 or not 1 <= len(actions) <= 50 or not np.isfinite(actions).all():
            raise ValueError('原生动作块需为 1–50 帧，每帧 12 个有限值。')
        if np.any(np.abs(actions) > 1.+1e-8): raise ValueError('原生命令范围为 [-1,1]；超限命令不会静默改成关节位置。')
        if not math.isfinite(dt) or not .05 <= dt <= .5 or abs(dt/.05-round(dt/.05)) > 1e-7:
            raise ValueError('策略间隔须为 0.05–0.5 秒且为官方 20Hz 的整数倍。')
        self._mode('native')
        self.motion = {'type': 'native', 'targets': actions.tolist(), 'index': 0, 'duration': dt, 'elapsed': 0.}
        self.paused = False
        self.task = {'status': 'running', 'title': 'PandaOmron 原生移动策略', 'steps': ['native']*len(actions), 'step': 0, 'progress': 0.}
        return {'robot_id': 'arm1', 'frames': len(actions), 'limited_targets': 0, 'executed_targets': actions.tolist(),
                'control_type': 'native_mobile_commands', 'unit': 'normalized_command'}

    def _pack(self, action):
        return self.robot.create_action_vector({'right': action[:6], 'right_gripper': action[6:7],
            'base': action[7:10], 'torso': action[10:11], 'base_mode': action[11]})

    def step(self):
        motion = self.motion
        if motion:
            motion['elapsed'] += self.timestep
            fraction = min(1., motion['elapsed']/motion['duration'])
            if motion['type'] == 'tcp':
                start, target = motion['initial'], motion['target']
                p = np.asarray(start['position_m'])+(np.asarray(target['position_m'])-start['position_m'])*fraction
                q = Slerp([0, 1], Rotation.concatenate([rotation(start['quaternion_wxyz']), rotation(target['quaternion_wxyz'])]))([fraction])[0]
                self.hold_pose = {'position_m': p.tolist(), 'quaternion_wxyz': q.as_quat()[[3, 0, 1, 2]].tolist()}
                self.gripper_open = target['gripper_open']
            else: self.native_action = np.asarray(motion['targets'][motion['index']])
            self.task['progress'] = fraction if motion['type'] == 'tcp' else (motion['index']+fraction)/len(motion['targets'])
        if self.mode == 'tcp':
            command_open = (float(self.gripper.current_action[0])+1)/2
            error = command_open-self.gripper_open
            g = float(np.sign(error)) if abs(error) > .0025 else 0.
            action = np.r_[self.hold_pose['position_m'], rotation(self.hold_pose['quaternion_wxyz']).as_rotvec(), g, [0.]*4, -1.]
        else: action = self.native_action
        self.env.step(self._pack(action))
        self.physics_steps += 1
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            raise RuntimeError('RoboCasa365 物理状态出现非有限值；请重置并检查动作与接触。')
        if motion and fraction >= 1.:
            if motion['type'] == 'tcp':
                self.motion = None; self.task['status'] = 'settling'; self.settle_deadline = self.time+2.
            else:
                motion['index'] += 1
                if motion['index'] == len(motion['targets']):
                    self.cancel_motion(); self.task['status'] = 'complete'
                else: motion['elapsed'] = 0.; self.task['step'] = motion['index']
        if self.task['status'] == 'settling':
            current = self.tcp_pose()
            ep = float(np.linalg.norm(np.asarray(current['position_m'])-self.pose_goal.position_m))
            er = float((rotation(current['quaternion_wxyz'])*rotation(self.pose_goal.quaternion_wxyz).inv()).magnitude())
            self.task.update(position_error_m=ep, orientation_error_rad=er)
            if ep < .01 and er < .08: self.task['status'] = 'complete'
            elif self.time > self.settle_deadline: self.task.update(status='failed', message='原生 OSC 未在容差内到达目标，请检查接触或缩小动作。')
        self._evaluate()
        if (self.run_deadline is not None and self.time >= self.run_deadline-1e-8) or (self.stop_on_success and self.evaluation['success']):
            self.cancel_motion(); self.paused = True
            self.task['status'] = 'verified' if self.stop_on_success and self.evaluation['success'] else 'limit'

    def _pixels(self, camera, width=640, height=426):
        if not self.render_enabled: raise ValueError('RoboCasa365 未开启离屏渲染。')
        return np.ascontiguousarray(self.sim.render(width=width, height=height, camera_name=camera)[::-1])

    @staticmethod
    def _jpeg(rgb):
        stream = io.BytesIO(); Image.fromarray(rgb).save(stream, 'JPEG', quality=88); return stream.getvalue()

    def observe(self):
        robots, cameras = self.robot_states(), []
        for name, native in CAMERAS.items():
            i = self.sim.model.camera_name2id(native)
            cameras.append({'name': name, 'image_base64': base64.b64encode(self._jpeg(self._pixels(native))).decode(),
                'width': 640, 'height': 426, 'time': self.time, 'position_world_m': self.sim.data.cam_xpos[i].tolist(),
                'rotation_camera_to_world': self.sim.data.cam_xmat[i].reshape(3, 3).tolist(), 'fovy_deg': float(self.sim.model.cam_fovy[i])})
        result = {'backend': {key: self.backend[key] for key in ('engine_id', 'engine_version', 'control_frame')},
            'generation': self.generation, 'time': self.time, 'robot': self.model_id, 'active_robot': 'arm1', 'robots': robots,
            'arm_dof': 7, 'tcp': self.tcp_pose(), 'joints': robots[0]['joints'], 'gripper': robots[0]['gripper'],
            'actuators': robots[0]['actuators'], 'hand_joints': [], 'mobile_base': robots[0]['mobile_base'], 'torso': robots[0]['torso'],
            'task_language': self.env.get_ep_meta().get('lang', ''), 'native_action_space': self.backend['action_space'],
            'limits': {'translation_per_round_m': .12, 'rotation_per_round_deg': 35, 'position_tolerance_m': .01, 'orientation_tolerance_rad': .08},
            'cameras': cameras}
        json.dumps(result, allow_nan=False)
        return result

    def set_view(self, view):
        if view not in ('perspective', 'front', 'top', 'wrist'): raise ValueError('未知视角。')
        self.view = view
        # A fixed world azimuth can start outside a kitchen wall after the
        # mobile robot spawns with a different heading. Initialize the free
        # camera at the official agent camera instead. MuJoCo's free-camera
        # forward = [cos(el)cos(az), cos(el)sin(az), sin(el)].
        i = self.sim.model.camera_name2id(CAMERAS['front'])
        position = np.asarray(self.sim.data.cam_xpos[i])
        forward = -np.asarray(self.sim.data.cam_xmat[i]).reshape(3, 3)[:, 2]
        self.distance = 1.6
        self.lookat = position+forward*self.distance
        self.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
        self.elevation = math.degrees(math.asin(float(np.clip(forward[2], -1., 1.))))
        if view == 'top':
            self.lookat = np.asarray(self.tcp_pose()['position_m'])
            self.distance, self.elevation = 2.5, -89.
        if view == 'front': self.elevation = -10.

    def render_rgb(self):
        import mujoco
        if self.view == 'wrist': return self._pixels(CAMERAS['wrist'], 960, 640)
        context = self.sim._render_context_offscreen
        if context is None: self._pixels(CAMERAS['front']); context = self.sim._render_context_offscreen
        context.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        context.cam.lookat[:] = self.lookat
        context.cam.distance, context.cam.azimuth, context.cam.elevation = self.distance, self.azimuth, self.elevation
        return self._pixels(None, 960, 640)

    def render(self): return self._jpeg(self.render_rgb())

    def checkpoint(self):
        value = {'backend': self.backend.copy(), 'model': self.model_id, 'scene_id': self.scene_id, 'task_id': self.task_id,
            'seed': self.seed, 'model_fingerprint': self.model_fingerprint, 'time': self.time,
            'state': self.sim.get_state().flatten().tolist(), 'ctrl': self.sim.data.ctrl.tolist(),
            # Several official composite tasks encode ripeness/cooking stage
            # directly in model RGBA, which MjSimState does not include.
            'model_state': {name: getattr(self.sim.model, name).tolist() for name in ('geom_rgba', 'site_rgba', 'site_size')},
            'qacc_warmstart': self.sim.data.qacc_warmstart.tolist(), 'native_action': self.native_action.tolist(),
            'mode': self.mode, 'hold_pose': deepcopy(self.hold_pose), 'gripper_open': self.gripper_open,
            'gripper_command': self.gripper.current_action.tolist(), 'physics_steps': self.physics_steps,
            'env_state': primitive_state(self.env), 'fixture_state': {name: primitive_state(f) for name, f in self.env.fixtures.items()},
            'controller_state': {name: primitive_state(c) for name, c in self.controller.part_controllers.items()},
            'rng_state': deepcopy(self.env.rng.bit_generator.state),
            'numpy_rng_state': json_primitive(np.random.get_state()), 'python_rng_state': json_primitive(random.getstate())}
        value['state_id'] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return value

    def restore(self, value):
        if (value.get('backend') != self.backend or value.get('model_fingerprint') != self.model_fingerprint
                or value.get('task_id') != self.task_id or value.get('seed') != self.seed):
            raise ValueError('RoboCasa365 检查点后端、版本、任务、种子或模型不匹配。')
        expected = {k: v for k, v in value.items() if k != 'state_id'}
        if hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False).encode()).hexdigest() != value.get('state_id'):
            raise ValueError('RoboCasa365 检查点校验失败。')
        state = np.asarray(value['state'], dtype=float)
        if state.shape != self.sim.get_state().flatten().shape or not np.isfinite(state).all(): raise ValueError('检查点物理状态无效。')
        model_state = {}
        for name in ('geom_rgba', 'site_rgba', 'site_size'):
            array = np.asarray(value['model_state'][name], dtype=float)
            if array.shape != getattr(self.sim.model, name).shape or not np.isfinite(array).all(): raise ValueError(f'检查点模型 {name} 状态无效。')
            model_state[name] = array
        self.cancel_motion()
        self.sim.set_state_from_flattened(state); self.sim.forward()
        for name, array in model_state.items(): getattr(self.sim.model, name)[:] = array
        self._mode(value['mode'])
        restore_primitives(self.env, value['env_state'])
        for name, fields in value['fixture_state'].items(): restore_primitives(self.env.fixtures[name], fields)
        for name, fields in value['controller_state'].items(): restore_primitives(self.controller.part_controllers[name], fields)
        self.sim.data.ctrl[:] = value['ctrl']; self.sim.data.qacc_warmstart[:] = value['qacc_warmstart']
        self.native_action = np.asarray(value['native_action'], dtype=float)
        self.hold_pose, self.gripper_open = deepcopy(value['hold_pose']), value['gripper_open']
        self.gripper.current_action = np.asarray(value['gripper_command'], dtype=float)
        self.env.rng.bit_generator.state = deepcopy(value['rng_state'])
        np.random.set_state(restore_primitive(value['numpy_rng_state']))
        random.setstate(restore_primitive(value['python_rng_state']))
        self.physics_steps = value['physics_steps']; self.generation += 1
        self.paused, self.run_deadline, self.stop_on_success, self.playback = True, None, False, None
        self.task = {'status': 'idle', 'title': '检查点已恢复', 'steps': [], 'step': -1, 'progress': 0.}
        self.ever_success = False; self._evaluate()
        return {'state_id': value['state_id']}

    def trajectory_frame(self):
        return {'backend': self.backend.copy(), 'time': self.time, 'active_robot': 'arm1', 'checkpoint': self.checkpoint(),
            'qpos': self.sim.data.qpos.tolist(), 'qvel': self.sim.data.qvel.tolist(), 'ctrl': self.sim.data.ctrl.tolist(),
            'actuator_force': self.sim.data.actuator_force.tolist(), 'contacts': int(self.sim.data.ncon),
            'contact_scope': 'whole_scene', 'evaluation': self.evaluation.copy()}

    def replay_frame(self, frame):
        if frame.get('backend') != frame.get('checkpoint', {}).get('backend'): raise ValueError('轨迹后端身份与检查点不一致。')
        self.restore(frame['checkpoint'])
        self.playback = {'time': self.time, 'frame_index': frame.get('index'), 'recorded_evaluation': frame.get('evaluation')}
        return self.playback

    def verify_task(self, generation, on_step=None):
        if generation != self.generation or not self.paused or self.motion is not None: raise ValueError('请在当前场景动作结束并暂停后验收。')
        self._evaluate(); return {**self.evaluation, 'verification_sim_time_s': 0.}

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene'):
            scene = payload.get('scene_id', self.scene_id)
            task = payload.get('task_id') or (self.task_id if scene == self.scene_id else SCENES[scene]['tasks'][0])
            model, seed = payload.get('model', self.model_id), payload.get('seed', self.seed)
            self._validate(model, scene, task)
            changed = task != self.task_id or seed != self.seed
            self.model_id, self.scene_id, self.task_id, self.seed = model, scene, task, seed
            self._load() if changed else self.reset(); self.generation += 1
        elif action == 'reset': self.reset(); self.generation += 1
        elif action == 'pause': self.paused = bool(payload['value'])
        elif action == 'render_suspend': self.render_suspended = bool(payload['value']); self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停。')
            self.step()
        elif action == 'home':
            raise ValueError('移动机器人请使用重置恢复官方初态；归一化原生命令不是可归零的关节角。')
        elif action == 'stop':
            self.cancel_motion(); self.paused = True; self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits': self.run_deadline = self.time+payload['duration_s']; self.stop_on_success = payload['evaluator_stop']
        elif action == 'select_robot':
            if payload['robot_id'] != 'arm1': raise ValueError('当前只有一个 PandaOmron。')
            return {'active_robot': 'arm1', 'generation': self.generation}
        elif action == 'observe': return self.observe()
        elif action == 'move_tcp': return self.move_tcp(payload)
        elif action == 'policy_chunk': return self.policy_chunk(payload)
        elif action == 'checkpoint': return self.checkpoint()
        elif action == 'restore': return self.restore(payload['checkpoint'])
        elif action == 'replay_frame': return self.replay_frame(payload['frame'])
        elif action == 'verify_task': return self.verify_task(payload['generation'])
        elif action == 'control':
            i, value = payload['index'], payload['value']
            if type(i) is not int or not 0 <= i < 12 or not math.isfinite(value) or not -1 <= value <= 1: raise ValueError('原生命令编号或范围无效。')
            target = np.zeros(12); target[-1] = -1.; target[i] = value
            if 7 <= i <= 9: target[-1] = 1.
            return self.policy_chunk({'generation': self.generation, 'joint_names': ACTION_NAMES, 'actions': [target.tolist()], 'dt_s': .1})
        elif action == 'settle':
            self.cancel_motion()
            for _ in range(10): self.step()
            self.paused = True
        elif action == 'view': self.set_view(payload['view'])
        elif action == 'camera':
            self.view = 'custom'; self.azimuth += payload.get('dx', 0)
            self.elevation = float(np.clip(self.elevation-payload.get('dy', 0), -89, -1))
            self.distance = float(np.clip(self.distance*math.exp(payload.get('zoom', 0)), .2, 12))
            az, el = math.radians(self.azimuth), math.radians(self.elevation)
            right = np.array([math.sin(az), -math.cos(az), 0.])
            up = np.array([-math.sin(el)*math.cos(az), -math.sin(el)*math.sin(az), math.cos(el)])
            self.lookat += self.distance*(-payload.get('pan_dx', 0)*right+payload.get('pan_dy', 0)*up)
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        else: raise ValueError(f'RoboCasa365 不支持操作：{action}')
        return '操作已完成。'

    def close(self):
        environment, self.env = getattr(self, 'env', None), None
        self.robot = self.sim = self.controller = self.arm = self.gripper = None
        if environment is not None: environment.close()
        del environment
        gc.collect()
