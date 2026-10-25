"""Official LIBERO BDDL tasks and robosuite controllers in an isolated process.

LIBERO is a benchmark built on MuJoCo, not a separate physics engine. No task
geometry, goal predicates or controller equations are replaced by this adapter.
"""
from __future__ import annotations

import base64
from collections import deque
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

from ..actions import PoseAction
from ..simulators import backend_identity
from .libero_catalog import TASK_MAP, catalog

SOURCE_COMMIT = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
CAMERAS = {'front': 'agentview', 'side': 'frontview', 'wrist': 'robot0_eye_in_hand'}
JOINTS = [f'joint{i}' for i in range(1, 8)] + ['finger_joint1']
CAPABILITIES = ['load', 'scene', 'reset', 'pause', 'step', 'control', 'home', 'stop',
                'observe', 'move_tcp', 'policy_chunk', 'checkpoint', 'restore',
                'replay_frame', 'settle', 'view', 'camera', 'speed', 'run_limits',
                'verify_task', 'render_suspend', 'select_robot']


def configure_libero():
    """Create this runtime's own noninteractive config before importing LIBERO."""
    spec = importlib.util.find_spec('libero')
    if spec is None:
        raise ValueError('LIBERO 未安装；请在独立环境安装 requirements-libero.txt。')
    import yaml
    package = Path(next(iter(spec.submodule_search_locations))) / 'libero'
    if not (package/'assets').is_dir():
        raise ValueError('LIBERO 官方资产缺失；请按 requirements-libero.txt 从固定官方源码以 editable 模式安装。无需下载演示数据。')
    config_folder = Path(sys.prefix)/'libero-config'
    if os.name == 'nt':
        suite_spec, mujoco_spec = importlib.util.find_spec('robosuite'), importlib.util.find_spec('mujoco')
        suite_folder = Path(suite_spec.origin).parent
        if not str(package).isascii() or not str(suite_folder).isascii():
            # MuJoCo 3.2's Windows mesh loader cannot open these Unicode asset
            # paths. Relocate unchanged official packages to an ASCII cache,
            # preserving both SDKs' own asset resolution and parser code.
            base = Path(tempfile.gettempdir())/'ew-libero'
            if not str(base).isascii():
                raise ValueError('LIBERO Windows 原生模型解析要求英文缓存路径，请将 TEMP 设为可写的英文目录。')
            revision = source_revision(package)
            key = revision or hashlib.sha256(str(package).encode()).hexdigest()[:20]
            # Official task filenames are long; keep the cache prefix short
            # enough for Windows' native MAX_PATH mesh/file loaders as well.
            identity = f'{key}:robosuite-{version("robosuite")}:mujoco-{version("mujoco")}'
            cache = base/hashlib.sha256(identity.encode()).hexdigest()[:16]
            marker = cache/'.complete'
            if not marker.exists():
                cache.mkdir(parents=True, exist_ok=True)
                ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
                shutil.copytree(package, cache/'libero/libero', dirs_exist_ok=True, ignore=ignore)
                shutil.copytree(suite_folder, cache/'robosuite', dirs_exist_ok=True, ignore=ignore)
                (cache/'.embodiworks-source-revision').write_text(revision or '', encoding='ascii')
                marker.touch()
            sys.path.insert(0, str(cache))
            importlib.invalidate_caches()
            package, suite_folder = cache/'libero/libero', cache/'robosuite'
            config_folder = cache/'config'
        # Official robosuite Windows setup loads a DLL from its utils folder.
        # Install the matching native library from the current MuJoCo wheel;
        # no Python SDK source or controller code is patched.
        target = suite_folder/'utils/mujoco.dll'
        source = Path(mujoco_spec.origin).parent/'mujoco.dll'
        if not target.exists():
            try:
                shutil.copy2(source, target)
            except OSError as exc:
                raise ValueError(f'请按 robosuite 官方 Windows 安装说明复制 {source} 到 {target}：{exc}') from exc
        # robosuite 1.4 imports its rendering bindings in package __init__,
        # before a caller can normally set its documented global macros. Load
        # the unchanged official macro module first and configure Windows GLFW
        # instead of its Linux-only EGL default. SDK source is never edited.
        if 'robosuite.macros' not in sys.modules:
            macro_spec = importlib.util.spec_from_file_location('robosuite.macros', suite_folder/'macros.py')
            macros = importlib.util.module_from_spec(macro_spec)
            sys.modules['robosuite.macros'] = macros
            macro_spec.loader.exec_module(macros)
            macros.MUJOCO_GPU_RENDERING = False
            macros.FILE_LOGGING_LEVEL = None
            os.environ['MUJOCO_GL'] = 'glfw'
    folder = Path(os.environ.setdefault('LIBERO_CONFIG_PATH', str(config_folder)))
    config = folder/'config.yaml'
    if not config.exists():
        folder.mkdir(parents=True, exist_ok=True)
        values = {'benchmark_root': str(package), 'bddl_files': str(package/'bddl_files'),
                  'init_states': str(package/'init_files'), 'assets': str(package/'assets'),
                  'datasets': str(package.parent/'datasets')}
        config.write_text(yaml.safe_dump(values), encoding='utf-8')
    return package


def _rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])


def source_revision(package):
    """Read a checkout's actual revision, rather than label all installs alike."""
    cached = package.parents[1]/'.embodiworks-source-revision'
    if cached.is_file():
        value = cached.read_text(encoding='ascii').strip()
        return value or None
    git = package.parents[1]/'.git'
    head = git/'HEAD'
    if not head.is_file():
        return None
    value = head.read_text(encoding='utf-8').strip()
    if value.startswith('ref: '):
        ref = value[5:]
        target = git/ref
        if target.is_file():
            value = target.read_text(encoding='utf-8').strip()
        elif (git/'packed-refs').is_file():
            value = next((line.split()[0] for line in (git/'packed-refs').read_text().splitlines()
                          if line.endswith(' '+ref)), '')
    return value if len(value) == 40 and all(c in '0123456789abcdef' for c in value) else None


class LiberoSimulation:
    def __init__(self, model_id='panda', render=True, scene_id='libero_spatial', seed=42, task_id=None):
        self._validate(model_id, scene_id, task_id or f'{scene_id}:0')
        global Rotation, Slerp
        from scipy.spatial.transform import Rotation, Slerp
        package = configure_libero()
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self._environment_class, self._libero_path = OffScreenRenderEnv, get_libero_path
        self.render_enabled, self.render_suspended = render, False
        self.generation, self.speed, self.active_robot = 1, 1., 'arm1'
        self.model_id, self.scene_id, self.task_id, self.seed = model_id, scene_id, task_id or f'{scene_id}:0', seed
        self.backend = {**backend_identity('libero', version('libero')),
                        'capabilities': CAPABILITIES.copy(),
                        'source_revision': source_revision(package), 'physics_engine': 'mujoco',
                        'runtime_packages': {name: version(name) for name in ('mujoco', 'robosuite', 'numpy', 'torch')}}
        self.flags = {'contacts': False, 'joints': False, 'collision': False}
        self.events, self.event_id = deque(maxlen=30), 0
        self.env = None
        self._load()

    @staticmethod
    def _validate(model, scene, task):
        if model != 'panda' or scene not in TASK_MAP or task not in {f'{scene}:{i}' for i in range(len(TASK_MAP[scene]))}:
            raise ValueError('LIBERO 仅支持官方 Panda 和当前任务套件中的任务编号。')

    def _load(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        index = int(self.task_id.split(':')[1])
        bddl = Path(self._libero_path('bddl_files'))/self.scene_id/(TASK_MAP[self.scene_id][index]+'.bddl')
        if not bddl.is_file():
            raise ValueError(f'缺少官方 BDDL 文件：{bddl}')
        self.env = self._environment_class(bddl_file_name=str(bddl), robots=['Panda'],
            camera_names=list(CAMERAS.values()), camera_heights=426, camera_widths=640,
            use_camera_obs=False, has_offscreen_renderer=self.render_enabled,
            control_freq=20, horizon=1000000, ignore_done=True, hard_reset=False)
        self.native = self.env.env
        self.robot = self.native.robots[0]
        self.sim = self.env.sim
        self.timestep = 1./self.native.control_freq
        self.reset()

    def _mode(self, name):
        if getattr(self, 'mode', None) == name:
            return
        import robosuite
        self.robot.controller_config = robosuite.load_controller_config(default_controller=name)
        self.robot.controller_config['control_delta'] = False
        self.robot._load_controller()
        # RobotEnv caches this dimension at reset. Official OSC and joint
        # controllers have different action widths (6 vs 7 arm commands).
        from robosuite.utils.buffers import DeltaBuffer
        self.native._action_dim = sum(robot.action_dim for robot in self.native.robots)
        self.robot.recent_actions = DeltaBuffer(dim=self.robot.action_dim)
        self.mode = name

    @property
    def time(self):
        return float(self.sim.data.time)

    def reset(self):
        import torch
        self.env.seed(self.seed)
        # Call the official native reset directly so unexpected errors retain
        # their cause instead of the upstream wrapper's unbounded retry loop.
        self.native.reset()
        index = int(self.task_id.split(':')[1])
        initial = Path(self._libero_path('init_states'))/self.scene_id/(TASK_MAP[self.scene_id][index]+'.pruned_init')
        if not initial.is_file():
            raise ValueError(f'缺少官方初态：{initial}；不需要演示数据。')
        # These states come from the pinned official LIBERO source checkout.
        states = torch.load(initial, map_location='cpu', weights_only=False)
        self.init_state_index = self.seed % len(states)
        self.env.set_init_state(np.asarray(states[self.init_state_index]))
        gripper_ranges = self.sim.model.actuator_ctrlrange[self.robot._ref_joint_gripper_actuator_indexes]
        fingers = self.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes]
        self.robot.gripper.current_action = np.clip(2*(fingers-gripper_ranges[:, 0])/(gripper_ranges[:, 1]-gripper_ranges[:, 0])-1, -1, 1)
        self.mode = None
        self._mode('OSC_POSE')
        self.ctrl = np.r_[self.sim.data.qpos[self.robot._ref_joint_pos_indexes], 255.]
        self.initial_joints = self.ctrl.copy()
        self.hold_pose = self.tcp_pose()
        self.motion, self.pose_goal, self.playback = None, None, None
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.physics_steps, self.ever_success = 0, False
        self.task = {'status': 'idle', 'title': '等待指令', 'steps': [], 'step': -1, 'progress': 0.}
        self.set_view('perspective')
        self._evaluate()

    def tcp_pose(self):
        site = self.robot.eef_site_id
        quat = Rotation.from_matrix(np.asarray(self.sim.data.site_xmat[site]).reshape(3, 3)).as_quat()
        return {'frame': 'world', 'position_m': np.asarray(self.sim.data.site_xpos[site]).tolist(),
                'quaternion_wxyz': quat[[3, 0, 1, 2]].tolist()}

    def _evaluate(self):
        raw = bool(self.env.check_success())
        finite = bool(np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all())
        success = raw and finite
        self.ever_success |= success
        predicates = []
        for expression in self.native.parsed_problem['goal_state']:
            value = bool(self.native._eval_predicate(expression))
            predicates.append({'key': ' '.join(expression), 'label': ' '.join(expression),
                               'actual': value, 'expected': True, 'passed': value})
        checks = [{'key': 'state_finite', 'label': '仿真状态数值有效', 'actual': finite, 'expected': True, 'passed': finite}, *predicates]
        task = catalog()[1][self.task_id]
        self.evaluation = {'task_id': self.task_id, **task, 'source': 'libero_bddl_state',
            'version': 'libero-bddl-v1', 'success': success, 'ever_success': self.ever_success,
            'currently_satisfied': success, 'status': 'success' if success else 'unsatisfied',
            'progress': sum(c['passed'] for c in predicates)/max(1, len(predicates)),
            'evaluated_at_sim_s': self.time, 'values': {'suite': self.scene_id,
                'official_task_name': TASK_MAP[self.scene_id][int(self.task_id.split(':')[1])],
                'init_state_index': self.init_state_index, 'native_check_success': raw},
            'checks': checks, 'failure_reasons': [c['label'] for c in checks if not c['passed']]}

    def robot_states(self):
        q = np.asarray(self.sim.data.qpos[self.robot._ref_joint_pos_indexes])
        v = np.asarray(self.sim.data.qvel[self.robot._ref_joint_vel_indexes])
        limits = np.asarray(self.sim.model.jnt_range[self.robot._ref_joint_indexes])
        actuators = [{'index': i, 'name': JOINTS[i], 'robot_id': 'arm1', 'is_gripper': False,
                      'unit': 'rad', 'qpos_index': int(self.robot._ref_joint_pos_indexes[i]),
                      'dof_index': int(self.robot._ref_joint_vel_indexes[i]),
                      'min': float(limits[i, 0]), 'max': float(limits[i, 1]),
                      'position': float(q[i]), 'actual': float(q[i]), 'target': float(self.ctrl[i]),
                      'force': float(self.sim.data.actuator_force[self.robot._ref_joint_actuator_indexes[i]])} for i in range(7)]
        g = float(np.abs(np.asarray(self.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes])).mean())
        actuators.append({'index': 7, 'name': JOINTS[7], 'robot_id': 'arm1', 'is_gripper': True,
                          'unit': 'command', 'qpos_index': int(self.robot._ref_gripper_joint_pos_indexes[0]),
                          'dof_index': int(self.robot._ref_gripper_joint_vel_indexes[0]),
                          'min': 0., 'max': 255., 'position': g,
                          'actual': g, 'target': float(self.ctrl[7]), 'force': 0.})
        return [{'id': 'arm1', 'name': 'Franka Panda · LIBERO', 'model': 'panda', 'active': True,
                 'arm_dof': 7, 'hand_dof': 1, 'tcp': self.tcp_pose(), 'actuators': actuators,
                 'joints': [{'name': JOINTS[i], 'position_rad': float(q[i]), 'velocity_rad_s': float(v[i]),
                             'limits_rad': limits[i].tolist()} for i in range(7)],
                 'gripper': {'position': g, 'unit': 'm_per_finger', 'target_open_fraction': float(self.ctrl[7]/255)}}]

    def state(self):
        robots = self.robot_states()
        return {'engine': 'LIBERO / MuJoCo', 'version': version('libero'), 'backend': self.backend.copy(),
                'model': 'panda', 'name': 'Franka Panda · LIBERO', 'active_robot': 'arm1', 'robots': robots,
                'generation': self.generation, 'paused': self.paused, 'speed': self.speed,
                'time': self.time, 'timestep': self.timestep, 'steps': self.physics_steps,
                'contacts': int(self.sim.data.ncon), 'contact_scope': 'whole_scene', 'gravity': True,
                'view': self.view, 'flags': self.flags.copy(), 'nq': self.sim.model.nq,
                'nv': self.sim.model.nv, 'nu': 8, 'meshes': self.sim.model.nmesh, 'arm_dof': 7, 'hand_dof': 1,
                'mass': float(np.sum(self.sim.model.body_mass)), 'actuators': robots[0]['actuators'],
                'model_note': '官方 LIBERO 任务资产、固定初态、BDDL 成功判据；robosuite 原生绝对位姿 OSC 与关节控制器；物理由 MuJoCo 执行。',
                'scene_id': self.scene_id, 'scene_name': self.scene_id.upper(), 'seed': self.seed,
                'evaluation': self.evaluation.copy(), 'playback': self.playback,
                'tcp': self.tcp_pose()['position_m'], 'tcp_pose': self.tcp_pose(), 'task': self.task.copy(),
                'events': list(self.events), 'blocks': {}, 'has_wrist_camera': self.render_enabled,
                'source': 'https://github.com/Lifelong-Robot-Learning/LIBERO',
                'compute_backend': 'cpu', 'render_backend': 'mujoco_offscreen'}

    def cancel_motion(self):
        if self.motion is not None:
            self.task['status'] = 'cancelled'
        self.motion, self.pose_goal = None, None
        self.hold_pose = self.tcp_pose()
        self.ctrl[:7] = self.sim.data.qpos[self.robot._ref_joint_pos_indexes]

    def _start(self, mode, targets, duration, title):
        if self.motion is not None:
            raise ValueError('当前动作未结束，请等待或先停止。')
        self._mode(mode)
        if mode == 'JOINT_POSITION':
            self.ctrl[:7] = self.sim.data.qpos[self.robot._ref_joint_pos_indexes]
        self.motion = {'targets': targets, 'duration': duration, 'elapsed': 0., 'index': 0,
                       'initial': self.ctrl.copy() if mode == 'JOINT_POSITION' else self.tcp_pose()}
        self.paused = False
        self.task = {'status': 'running', 'title': title, 'steps': [title]*len(targets), 'step': 0, 'progress': 0.}

    def move_tcp(self, payload):
        if payload['generation'] != self.generation:
            raise ValueError('场景已切换，本轮动作失效。')
        pose = PoseAction.model_validate(payload['pose'])
        if pose.status != 'move' or pose.robot_id not in (None, 'arm1') or pose.hand_joint_positions_rad is not None:
            raise ValueError('LIBERO Panda 需要 arm1 的 move 末端位姿动作。')
        before = self.tcp_pose()
        if np.linalg.norm(np.asarray(pose.position_m)-before['position_m']) > .12+1e-8:
            raise ValueError('单轮末端平移超过 0.12 m。')
        if (_rotation(pose.quaternion_wxyz)*_rotation(before['quaternion_wxyz']).inv()).magnitude() > math.radians(35)+1e-8:
            raise ValueError('单轮末端姿态变化超过 35°。')
        self._start('OSC_POSE', [pose.model_dump()], pose.duration_s, '官方 OSC 绝对末端控制')
        self.pose_goal = pose
        return {'robot_id': 'arm1', 'controller': 'robosuite.OSC_POSE', 'target': pose.model_dump()}

    def policy_chunk(self, payload):
        if payload['generation'] != self.generation:
            raise ValueError('场景已切换，策略动作失效。')
        actions, dt = np.asarray(payload['actions'], dtype=float), payload['dt_s']
        if payload.get('robot_id') not in (None, 'arm1') or payload['joint_names'] != JOINTS:
            raise ValueError('LIBERO 策略需要 Panda 原生七关节和归一化夹爪顺序。')
        if actions.ndim != 2 or actions.shape[1] != 8 or not 1 <= len(actions) <= 50 or not np.isfinite(actions).all():
            raise ValueError('策略动作块需为 1–50 帧，每帧 8 个有限目标值。')
        if not .05 <= dt <= .5:
            raise ValueError('LIBERO 原生控制频率 20 Hz，策略间隔应为 0.05–0.5 秒。')
        info = self.robot_states()[0]['actuators']
        lower, upper = np.array([v['min'] for v in info]), np.array([v['max'] for v in info])
        previous, values, limited = np.r_[self.sim.data.qpos[self.robot._ref_joint_pos_indexes], self.ctrl[7]], [], 0
        for action in actions:
            value = np.clip(action, np.maximum(lower, previous-np.r_[[2.]*7, 255.]*dt),
                            np.minimum(upper, previous+np.r_[[2.]*7, 255.]*dt))
            limited += int(np.sum(np.abs(value-action) > 1e-8))
            values.append(value); previous = value
        self._start('JOINT_POSITION', values, dt, '官方关节位置控制器 · 策略动作块')
        return {'robot_id': 'arm1', 'frames': len(values), 'limited_targets': limited,
                'executed_targets': [v.tolist() for v in values]}

    def step(self):
        motion = self.motion
        if motion:
            motion['elapsed'] += self.timestep
            fraction = min(1., motion['elapsed']/motion['duration'])
            goal = motion['targets'][motion['index']]
            if self.mode == 'OSC_POSE':
                start = motion['initial']
                position = np.asarray(start['position_m'])+(np.asarray(goal['position_m'])-start['position_m'])*fraction
                orientation = Slerp([0, 1], Rotation.concatenate([_rotation(start['quaternion_wxyz']), _rotation(goal['quaternion_wxyz'])]))([fraction])[0]
                self.hold_pose = {'position_m': position.tolist(), 'quaternion_wxyz': orientation.as_quat()[[3, 0, 1, 2]].tolist()}
                self.ctrl[7] = goal['gripper_open']*255
            else:
                self.ctrl = motion['initial']+(goal-motion['initial'])*fraction
            self.task['progress'] = (motion['index']+fraction)/len(motion['targets'])
        # Official PandaGripper receives binary velocity commands. Compare its
        # native command state to the requested opening; do not reinterpret the
        # opening fraction as a velocity or replace its actuator mapping.
        commanded_open = (float(self.robot.gripper.current_action[0])+1)/2
        opening_error = commanded_open-self.ctrl[7]/255
        gripper_action = float(np.sign(opening_error)) if abs(opening_error) > .0025 else 0.
        if self.mode == 'OSC_POSE':
            action = np.r_[self.hold_pose['position_m'], _rotation(self.hold_pose['quaternion_wxyz']).as_rotvec(), gripper_action]
        else:
            # robosuite 1.4's JOINT_POSITION consumes normalized joint deltas.
            # Feed back the current position to follow our absolute public
            # targets through that official controller at its native 20 Hz.
            controller = self.robot.controller
            error = self.ctrl[:7]-self.sim.data.qpos[self.robot._ref_joint_pos_indexes]
            normalized = (error-controller.output_min)/(controller.output_max-controller.output_min)
            native_action = controller.input_min+normalized*(controller.input_max-controller.input_min)
            native_action = np.clip(native_action, controller.input_min, controller.input_max)
            action = np.r_[native_action, gripper_action]
        self.env.step(action)
        if self.mode == 'OSC_POSE':
            self.ctrl[:7] = self.sim.data.qpos[self.robot._ref_joint_pos_indexes]
        self.physics_steps += 1
        if not np.isfinite(self.sim.data.qpos).all():
            raise RuntimeError('LIBERO MuJoCo 状态出现非有限数值，请重置并检查动作。')
        if motion and fraction >= 1:
            motion['index'] += 1
            if motion['index'] >= len(motion['targets']):
                self.motion = None
                self.task['status'] = 'settling'
                self.settle_deadline = self.time+2.
            else:
                motion['elapsed'], motion['initial'] = 0., self.ctrl.copy()
                self.task['step'] = motion['index']
        if self.task['status'] == 'settling':
            qerr = np.max(np.abs(self.sim.data.qpos[self.robot._ref_joint_pos_indexes]-self.ctrl[:7]))
            reached = qerr < .03
            if self.pose_goal is not None:
                pose = self.tcp_pose()
                ep = float(np.linalg.norm(np.asarray(pose['position_m'])-self.pose_goal.position_m))
                er = float((_rotation(pose['quaternion_wxyz'])*_rotation(self.pose_goal.quaternion_wxyz).inv()).magnitude())
                self.task.update(position_error_m=ep, orientation_error_rad=er)
                reached = ep < .008 and er < .07
            if reached:
                self.task['status'] = 'complete'
            elif self.time > self.settle_deadline:
                self.task.update(status='failed', message='原生控制器未在容差内到达目标，请检查接触或缩小动作。')
        self._evaluate()
        if (self.run_deadline is not None and self.time >= self.run_deadline-1e-8) or (self.stop_on_success and self.evaluation['success']):
            self.cancel_motion(); self.paused = True
            self.task['status'] = 'verified' if self.stop_on_success and self.evaluation['success'] else 'limit'

    def _pixels(self, camera, width=640, height=426):
        if not self.render_enabled:
            raise ValueError('LIBERO 未开启原生离屏渲染。')
        return np.ascontiguousarray(self.sim.render(width=width, height=height, camera_name=camera)[::-1])

    @staticmethod
    def _jpeg(rgb):
        output = io.BytesIO(); Image.fromarray(rgb).save(output, 'JPEG', quality=88)
        return output.getvalue()

    def observe(self):
        robots = self.robot_states()
        cameras = []
        for name, native in CAMERAS.items():
            camera_id = self.sim.model.camera_name2id(native)
            cameras.append({'name': name, 'image_base64': base64.b64encode(self._jpeg(self._pixels(native))).decode(),
                            'width': 640, 'height': 426, 'time': self.time,
                            'position_world_m': self.sim.data.cam_xpos[camera_id].tolist(),
                            'rotation_camera_to_world': self.sim.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
                            'fovy_deg': float(self.sim.model.cam_fovy[camera_id])})
        return {'backend': {k: self.backend[k] for k in ('engine_id', 'engine_version', 'control_frame')},
                'generation': self.generation, 'time': self.time, 'robot': 'panda', 'active_robot': 'arm1',
                'robots': robots, 'arm_dof': 7, 'tcp': self.tcp_pose(), 'joints': robots[0]['joints'],
                'gripper': robots[0]['gripper'], 'actuators': robots[0]['actuators'], 'hand_joints': [],
                'limits': {'translation_per_round_m': .12, 'rotation_per_round_deg': 35,
                           'position_tolerance_m': .008, 'orientation_tolerance_rad': .07}, 'cameras': cameras}

    def set_view(self, view):
        if view not in ('perspective', 'front', 'top', 'wrist'):
            raise ValueError('未知视角。')
        self.view = view
        self.lookat = np.asarray(self.native.workspace_offset, dtype=float)+[0., 0., .1]
        self.distance, self.azimuth, self.elevation = 1.7, 145., -30.
        if view == 'top': self.elevation = -89.
        if view == 'front': self.azimuth, self.elevation = 180., -10.

    def render_rgb(self):
        import mujoco
        if self.view == 'wrist':
            return self._pixels(CAMERAS['wrist'], 960, 640)
        context = self.sim._render_context_offscreen
        context.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        context.cam.lookat[:] = self.lookat
        context.cam.distance, context.cam.azimuth, context.cam.elevation = self.distance, self.azimuth, self.elevation
        return self._pixels(None, 960, 640)

    def render(self):
        return self._jpeg(self.render_rgb())

    def checkpoint(self):
        state = np.asarray(self.env.get_sim_state()).tolist()
        result = {'backend': self.backend.copy(), 'model': 'panda', 'scene_id': self.scene_id,
                  'task_id': self.task_id, 'seed': self.seed, 'time': self.time, 'state': state,
                  'ctrl': self.ctrl.tolist(), 'hold_pose': self.hold_pose, 'mode': self.mode,
                  'init_state_index': self.init_state_index,
                  'gripper_command': self.robot.gripper.current_action.tolist(),
                  'native_ctrl': self.sim.data.ctrl.tolist(),
                  'qacc_warmstart': self.sim.data.qacc_warmstart.tolist(),
                  'controller_initial_joint': self.robot.controller.initial_joint.tolist(),
                  'env_timestep': self.native.timestep, 'env_cur_time': self.native.cur_time,
                  'physics_steps': self.physics_steps}
        result['state_id'] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return result

    def restore(self, value):
        if value.get('backend') != self.backend or value.get('model') != 'panda' or value.get('scene_id') != self.scene_id or value.get('task_id') != self.task_id:
            raise ValueError('LIBERO 检查点的后端、版本、场景或任务与当前环境不一致。')
        state, ctrl = np.asarray(value['state'], dtype=float), np.asarray(value['ctrl'], dtype=float)
        if state.shape != np.asarray(self.env.get_sim_state()).shape or ctrl.shape != (8,) or not np.isfinite(state).all() or not np.isfinite(ctrl).all():
            raise ValueError('LIBERO 检查点形状或数值无效。')
        hold = value.get('hold_pose', {})
        position, quat = np.asarray(hold.get('position_m'), dtype=float), np.asarray(hold.get('quaternion_wxyz'), dtype=float)
        if (position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1.) > .01
                or type(value.get('seed')) is not int or not 0 <= value['seed'] <= 4294967295
                or type(value.get('init_state_index')) is not int or value['init_state_index'] < 0
                or type(value.get('env_timestep')) is not int or value['env_timestep'] < 0
                or type(value.get('physics_steps')) is not int or value['physics_steps'] < 0
                or not isinstance(value.get('env_cur_time'), (int, float))
                or not math.isfinite(value['env_cur_time']) or value['env_cur_time'] < 0
                or not isinstance(value.get('state_id'), str)):
            raise ValueError('LIBERO 检查点的控制位姿、种子或标识无效。')
        if value['mode'] not in ('OSC_POSE', 'JOINT_POSITION'):
            raise ValueError('LIBERO 检查点控制器无效。')
        additional = {}
        for key, shape in [('gripper_command', (2,)), ('native_ctrl', self.sim.data.ctrl.shape),
                           ('qacc_warmstart', self.sim.data.qacc_warmstart.shape), ('controller_initial_joint', (7,))]:
            field = np.asarray(value[key], dtype=float)
            if field.shape != shape or not np.isfinite(field).all():
                raise ValueError(f'LIBERO 检查点字段 {key} 无效。')
            additional[key] = field
        self.env.set_init_state(state)
        self.cancel_motion()
        self.ctrl, self.seed, self.init_state_index = ctrl, value['seed'], value['init_state_index']
        self.native.timestep, self.native.cur_time = value['env_timestep'], value['env_cur_time']
        self.physics_steps = value['physics_steps']; self.mode = None
        self._mode(value['mode']); self.hold_pose = value['hold_pose']
        self.robot.gripper.current_action = additional['gripper_command']
        self.robot.controller.update_initial_joints(additional['controller_initial_joint'])
        self.sim.data.ctrl[:] = additional['native_ctrl']
        self.sim.data.qacc_warmstart[:] = additional['qacc_warmstart']
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.generation += 1; self.playback = None; self.ever_success = False
        self.task = {'status': 'idle', 'title': '检查点已恢复', 'steps': [], 'step': -1, 'progress': 0.}
        self._evaluate()
        return {'state_id': value['state_id']}

    def trajectory_frame(self):
        return {'time': self.time, 'backend': self.backend.copy(), 'active_robot': 'arm1',
                'checkpoint': self.checkpoint(), 'qpos': self.sim.data.qpos.tolist(), 'qvel': self.sim.data.qvel.tolist(),
                'ctrl': self.ctrl.tolist(), 'actuator_force': self.sim.data.actuator_force.tolist(),
                'contacts': int(self.sim.data.ncon), 'contact_scope': 'whole_scene', 'evaluation': self.evaluation.copy()}

    def replay_frame(self, frame):
        if frame.get('backend') != frame.get('checkpoint', {}).get('backend'):
            raise ValueError('轨迹后端身份与检查点不一致。')
        self.restore(frame['checkpoint'])
        self.playback = {'time': self.time, 'frame_index': frame.get('index'), 'recorded_evaluation': frame.get('evaluation')}
        return self.playback

    def verify_task(self, generation, on_step=None):
        if generation != self.generation or not self.paused or self.motion is not None:
            raise ValueError('请在当前场景动作结束并暂停后验收任务。')
        self._evaluate()
        return {**self.evaluation, 'verification_sim_time_s': 0.}

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene', 'reset'):
            model, scene = payload.get('model', self.model_id), payload.get('scene_id', self.scene_id)
            task = payload.get('task_id') or (self.task_id if scene == self.scene_id else f'{scene}:0')
            self._validate(model, scene, task)
            changed = task != self.task_id
            try:
                self.model_id, self.scene_id, self.task_id, self.seed = model, scene, task, payload.get('seed', self.seed)
                self._load() if changed else self.reset()
            except Exception as exc:
                cleanup = ''
                try: self.close()
                except Exception as cleanup_exc: cleanup = f'；资源清理失败：{cleanup_exc}'
                raise RuntimeError(f'LIBERO 场景重建失败，仿真进程未就绪：{type(exc).__name__}: {exc}{cleanup}') from exc
            self.generation += 1
        elif action == 'pause': self.paused = bool(payload['value'])
        elif action == 'render_suspend': self.render_suspended = bool(payload['value']); self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停。')
            self.step()
        elif action == 'stop': self.cancel_motion(); self.paused = True; self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits': self.run_deadline = self.time+payload['duration_s']; self.stop_on_success = payload['evaluator_stop']
        elif action == 'select_robot':
            if payload['robot_id'] != 'arm1': raise ValueError('LIBERO 当前仅支持 arm1。')
            return {'active_robot': 'arm1', 'generation': self.generation}
        elif action == 'observe': return self.observe()
        elif action == 'move_tcp': return self.move_tcp(payload)
        elif action == 'policy_chunk': return self.policy_chunk(payload)
        elif action == 'checkpoint': return self.checkpoint()
        elif action == 'restore': return self.restore(payload['checkpoint'])
        elif action == 'replay_frame': return self.replay_frame(payload['frame'])
        elif action == 'verify_task': return self.verify_task(payload['generation'])
        elif action in ('home', 'control'):
            target = self.initial_joints.copy() if action == 'home' else self.ctrl.copy()
            if action == 'control':
                index, value = payload['index'], payload['value']
                if type(index) is not int or not 0 <= index < 8: raise ValueError('执行器编号不存在。')
                info = self.robot_states()[0]['actuators'][index]
                if not math.isfinite(value) or not info['min'] <= value <= info['max']: raise ValueError('关节目标超出范围。')
                target[index] = value
            self.cancel_motion(); self._start('JOINT_POSITION', [target], 2., '原生关节位置控制')
        elif action == 'settle':
            self.cancel_motion()
            for _ in range(10): self.step()
            self.paused = True
        elif action == 'view': self.set_view(payload['view'])
        elif action == 'camera':
            self.view = 'custom'; self.azimuth += payload.get('dx', 0)
            self.elevation = float(np.clip(self.elevation-payload.get('dy', 0), -89, -1))
            self.distance = float(np.clip(self.distance*math.exp(payload.get('zoom', 0)), .2, 6))
            az, el = math.radians(self.azimuth), math.radians(self.elevation)
            right, up = np.array([math.sin(az), -math.cos(az), 0.]), np.array([-math.sin(el)*math.cos(az), -math.sin(el)*math.sin(az), math.cos(el)])
            self.lookat += self.distance*(-payload.get('pan_dx', 0)*right+payload.get('pan_dy', 0)*up)
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        else:
            raise ValueError(f'LIBERO 未实现此操作：{action}。使用 Agent 视觉闭环执行任务。')
        return '操作已完成。'

    def close(self):
        environment, self.env = self.env, None
        self.native = self.robot = self.sim = None
        if environment is not None: environment.close()
