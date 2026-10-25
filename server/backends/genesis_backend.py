"""Native Genesis CPU physics and rasterization in the isolated simulator process.

Genesis itself parses MJCF assets through its official dependency on mujoco.
Runtime dynamics, inverse kinematics and rendering use Genesis implementations.
"""
from __future__ import annotations

import base64
from collections import deque
import hashlib
import io
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import genesis as gs
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ..actions import PoseAction
from ..simulators import backend_identity

TASKS = {
    'lift_red': {'name': '抬起红色方块', 'instruction': '把红色方块抓起来，保持在桌面上方。',
                 'criterion': '红块中心高于桌面 8 cm，连续保持 0.3 秒仿真时间。'},
    'place_red': {'name': '红块放入绿盒', 'instruction': '把红色方块放入绿色收纳盒内，然后松开并移开手。',
                  'criterion': '红块完全处于盒内、高度正确、线速度低于 0.03 m/s，且脱离机械臂接触，连续保持 0.3 秒。'},
    'reach_up': {'name': '末端抬高 2 cm', 'instruction': '保持夹爪开度和当前末端姿态，将末端沿世界坐标 Z 轴抬高 2 厘米。',
                 'criterion': '末端距离初始位置上方 2 cm 的目标小于 6 mm，连续保持 0.3 秒。'},
}
HOME = np.array([0., -.4, 0., -2.1, 0., 1.7, .7854, 255.])
JOINTS = [f'joint{i}' for i in range(1, 8)] + ['finger_joint1']
STATE_FIELDS = ('qpos', 'dofs_vel', 'dofs_acc', 'links_pos', 'links_quat', 'friction_ratio')


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])


def quaternion_angle(a, b):
    return float((rotation(a) * rotation(b).inv()).magnitude())


def add_panda(scene):
    """Load official assets through an ASCII path on affected Windows installs.

    MuJoCo's MJCF mesh parser still rejects non-ASCII paths on Windows. Genesis
    1.4.5 validates MJCF.file as XML/path before its lower-level MjModel support,
    so use unchanged official files in a scoped temporary directory. Genesis
    finishes reading the meshes while add_entity resolves its description.
    """
    source = Path(gs.__file__).parent / 'assets/xml/franka_emika_panda'
    if os.name != 'nt' or str(source).isascii():
        return scene.add_entity(gs.morphs.MJCF(file=str(source/'panda.xml')))
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if not str(temporary_root).isascii():
        raise ValueError('Windows 的 Genesis 模型解析要求英文资产路径，请将 TEMP 配置为可写的英文路径后重启。')
    with tempfile.TemporaryDirectory(prefix='embodiworks-genesis-') as folder:
        target = Path(folder).resolve()
        if target.parent != temporary_root or not target.name.startswith('embodiworks-genesis-'):
            raise RuntimeError('模型临时目录超出预期位置。')
        shutil.copytree(source, target/'panda')
        return scene.add_entity(gs.morphs.MJCF(file=str(target/'panda/panda.xml')))


class GenesisTaskEvaluator:
    REQUIRED_HOLD_S = .3

    def __init__(self, sim, initial_tcp=None):
        self.initial_tcp = np.array(initial_tcp if initial_tcp is not None else sim.tcp_pose()['position_m'])
        self.success = self.ever_success = self.previous_valid = False
        self.hold_s = 0.
        self.last_time = sim.time
        self.update(sim)

    def update(self, sim):
        checks = []
        def check(key, label, actual, expected, passed, unit=None):
            checks.append(dict(key=key, label=label, actual=actual, expected=expected, passed=bool(passed), unit=unit))
        q, v = array(sim.robot.get_qpos()), array(sim.robot.get_dofs_velocity())
        finite = np.isfinite(q).all() and np.isfinite(v).all() and math.isfinite(sim.time)
        red = sim.blocks['red']
        pos, vel = array(red.get_pos()), array(red.get_vel())
        finite = bool(finite and np.isfinite(pos).all() and np.isfinite(vel).all())
        check('state_finite', '物理状态数值有效', finite, True, finite)
        if sim.task_id == 'lift_red':
            height = float(pos[2])
            check('red_height', '红块中心高于桌面', height, '> 0.08', height > .08, 'm')
            progress = (height-.024)/(.08-.024)
            values = {'red_height_m': height, 'height_above_table_m': height,
                      'table_top_world_z_m': 0., 'height_threshold_m': .08}
        elif sim.task_id == 'reach_up':
            target = self.initial_tcp + [0, 0, .02]
            error = float(np.linalg.norm(np.asarray(sim.tcp_pose()['position_m'])-target))
            check('tcp_target_error', '末端距抬高目标的误差', error, '< 0.006', error < .006, 'm')
            progress = 1-error/.02
            values = {'position_error_m': error, 'target_world_m': target.tolist(), 'position_tolerance_m': .006}
        else:
            # Axis-aligned extents come from the rotated collision geometry,
            # not merely the object's center or a model's verbal declaration.
            bounds = array(red.get_AABB())
            lower, upper = np.array([.358, .098]), np.array([.642, .402])
            contained = bool(np.all(bounds[0, :2] > lower) and np.all(bounds[1, :2] < upper))
            speed = float(np.linalg.norm(vel))
            touched = len(sim.robot.get_contacts(with_entity=red)['link_a']) > 0
            check('red_inside_bin_xy', '红块完整位于盒内', bounds[:, :2].tolist(), [lower.tolist(), upper.tolist()], contained, 'm')
            check('red_height', '红块中心高度', float(pos[2]), '0.030 < z < 0.052', .030 < pos[2] < .052, 'm')
            check('red_speed', '红块线速度', speed, '< 0.03', speed < .03, 'm/s')
            check('red_released', '红块已脱离机械臂接触', not touched, True, not touched)
            distance = float(np.linalg.norm(pos[:2]-[.5, .25]))
            progress = 1-distance/.5
            values = {'target_distance_m': distance, 'object_speed_m_s': speed}
        valid = all(item['passed'] for item in checks)
        elapsed = sim.time-self.last_time
        self.hold_s = self.hold_s + elapsed if valid and self.previous_valid and elapsed >= 0 else 0.
        self.success = bool(valid and self.hold_s + 1e-9 >= self.REQUIRED_HOLD_S)
        self.ever_success |= self.success
        self.previous_valid, self.last_time = valid, sim.time
        check('hold_duration', '物理条件连续满足的仿真时间', round(self.hold_s, 6), '>= 0.3', self.success, 's')
        self.last = {'task_id': sim.task_id, **TASKS[sim.task_id], 'version': 'genesis-task-v1',
                     'source': 'genesis_privileged_state', 'success': self.success, 'ever_success': self.ever_success,
                     'currently_satisfied': valid, 'status': 'success' if self.success else 'holding' if valid else 'unsatisfied',
                     'progress': float(np.clip(progress, 0, 1)) if math.isfinite(progress) else 0.,
                     'hold_s': round(self.hold_s, 6), 'required_hold_s': self.REQUIRED_HOLD_S,
                     'evaluated_at_sim_s': sim.time, 'values': values, 'checks': checks,
                     'failure_reasons': [item['label'] for item in checks if not item['passed']]}

    def snapshot(self):
        return self.last.copy()


class GenesisSimulation:
    def __init__(self, model_id='panda', render=True, scene_id='tabletop', seed=42, task_id=None):
        self._validate(model_id, scene_id, task_id or 'lift_red')
        gs.init(backend=gs.cpu, logging_level='warning', seed=seed)
        self.model_id, self.scene_id, self.seed, self.task_id = model_id, scene_id, seed, task_id or 'lift_red'
        self.backend = {**backend_identity('genesis', gs.__version__),
                        'runtime_packages': {name: version(name) for name in ('torch', 'quadrants', 'numpy', 'scipy', 'pillow')}}
        self.render_enabled, self.render_suspended = render, False
        self.generation, self.speed, self.active_robot = 1, 1., 'arm1'
        self.timestep, self.time, self.physics_steps = .01, 0., 0
        self.view, self.flags = 'perspective', {'contacts': False, 'joints': False, 'collision': False}
        self.events = deque(maxlen=30)
        self.event_id = 0
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self.timestep, substeps=2),
            # The official Panda's qpos0 has an out-of-range elbow. Do not let
            # that uncommanded pose permanently disable authored self contacts.
            rigid_options=gs.options.RigidOptions(enable_neutral_collision=True), show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane(pos=(0, 0, -.69)))
        self._box((1.6, 1.3, .07), (.25, 0, -.035), (.8, .85, .79), fixed=True)
        self.robot = add_panda(self.scene)
        self.blocks = {}
        self._populate(seed)
        self.cameras = {}
        if render:
            for name, pos in [('main', (1.25, -1.4, 1.05)), ('front', (.8, -1.05, .7)), ('side', (.8, .85, .65)), ('wrist', (.3, 0, .5))]:
                self.cameras[name] = self.scene.add_camera(res=(960, 640) if name == 'main' else (640, 426),
                    pos=pos, lookat=(.3, 0, .18), fov=52 if name != 'wrist' else 75, GUI=False)
        self.scene.build()
        self.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        force = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        self.robot.set_dofs_force_range(-force, force)
        self.hand = self.robot.get_link('hand')
        limits = np.stack([array(value) for value in self.robot.get_dofs_limit()])
        self.actuators = [{'index': i, 'name': name, 'robot_id': 'arm1', 'is_gripper': i == 7,
                          'min': 0. if i == 7 else float(limits[0, i]),
                          'max': 255. if i == 7 else float(limits[1, i]),
                          'qpos_index': i, 'dof_index': i, 'unit': 'command' if i == 7 else 'rad'}
                         for i, name in enumerate(JOINTS)]
        self.build_state = self.scene.get_state()
        self.reset()
        self.event('原生 Genesis CPU 物理与 Rasterizer，已加载 Panda。')

    @staticmethod
    def _validate(model, scene, task):
        if model != 'panda' or scene != 'tabletop' or task not in TASKS:
            raise ValueError('Genesis 当前支持 Panda、tabletop 场景及 lift_red / place_red / reach_up 任务。')

    def _box(self, size, pos, color, fixed=False):
        return self.scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=fixed), surface=gs.surfaces.Default(color=color))

    def _populate(self, seed):
        self._box((.3, .32, .016), (.5, .25, .007), (.4, .61, .48), True)
        for dx, dy, sx, sy in [(-.15, 0, .016, .32), (.15, 0, .016, .32), (0, -.16, .3, .016), (0, .16, .3, .016)]:
            self._box((sx, sy, .05), (.5+dx, .25+dy, .032), (.4, .61, .48), True)
        for name, pos, color in [('red', [.4, -.22, .09], (.78, .26, .21)),
                                 ('blue', [.55, -.12, .09], (.2, .43, .67)),
                                 ('yellow', [.32, -.37, .09], (.85, .65, .17))]:
            self.blocks[name] = self._box((.048,)*3, pos, color)

    def event(self, message, kind='info'):
        self.event_id += 1
        self.events.append({'id': self.event_id, 'time': round(self.time, 3), 'message': message, 'kind': kind})

    def _native_target(self, ctrl):
        return np.r_[np.asarray(ctrl)[:7], [ctrl[7]/255*.04]*2]

    def reset(self):
        self.scene.reset(state=self.build_state)
        rng = np.random.default_rng(self.seed)
        for name, nominal in [('red', [.4, -.22, .09]), ('blue', [.55, -.12, .09]), ('yellow', [.32, -.37, .09])]:
            pos = np.asarray(nominal)
            if self.seed != 42: pos[:2] += rng.uniform(-.025, .025, 2)
            self.blocks[name].set_pos(pos)
        self.ctrl = HOME.copy()
        self.robot.set_dofs_position(self._native_target(self.ctrl))
        self.robot.control_dofs_position(self._native_target(self.ctrl))
        self.time, self.physics_steps = 0., 0
        self.motion, self.pose_goal, self.playback = None, None, None
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.task = {'status': 'idle', 'title': '等待指令', 'steps': [], 'step': -1, 'progress': 0.}
        self.evaluator = GenesisTaskEvaluator(self)
        self.set_view('perspective')

    def tcp_pose(self):
        quat = array(self.hand.get_quat())
        pos = array(self.hand.get_pos()) + rotation(quat).apply([0, 0, .1034])
        return {'frame': 'world', 'position_m': pos.tolist(), 'quaternion_wxyz': quat.tolist()}

    def _actuators(self):
        q, forces = array(self.robot.get_dofs_position()), array(self.robot.get_dofs_control_force())
        return [{**a, 'position': float(q[i]), 'actual': float(q[i]), 'target': float(self.ctrl[i]),
                 'force': float(forces[i])} for i, a in enumerate(self.actuators)]

    def robot_states(self):
        q, v = array(self.robot.get_dofs_position()), array(self.robot.get_dofs_velocity())
        return [{'id': 'arm1', 'name': 'Franka Panda', 'model': 'panda', 'active': True, 'arm_dof': 7, 'hand_dof': 1,
                 'tcp': self.tcp_pose(), 'gripper': {'position': float(q[7]), 'unit': 'm_per_finger', 'target_open_fraction': float(self.ctrl[7]/255)},
                 'joints': [{'name': a['name'], 'position_rad': float(q[i]), 'velocity_rad_s': float(v[i]),
                             'limits_rad': [a['min'], a['max']]} for i, a in enumerate(self.actuators[:7])],
                 'actuators': self._actuators()}]

    def state(self):
        pose, robots = self.tcp_pose(), self.robot_states()
        return {'engine': 'Genesis', 'version': gs.__version__, 'backend': self.backend.copy(),
                'model': 'panda', 'name': 'Franka Panda', 'active_robot': 'arm1', 'robots': robots,
                'generation': self.generation, 'paused': self.paused, 'speed': self.speed, 'time': self.time,
                'timestep': self.timestep, 'steps': self.physics_steps, 'contacts': len(self.robot.get_contacts()['link_a']),
                'contact_scope': 'active_robot',
                'gravity': True, 'view': self.view, 'flags': self.flags.copy(),
                'nq': self.scene.rigid_solver.n_qs, 'nv': self.scene.rigid_solver.n_dofs, 'nu': 8,
                'meshes': self.robot.n_geoms, 'arm_dof': 7, 'hand_dof': 1, 'mass': float(self.robot.get_mass()),
                'actuators': robots[0]['actuators'], 'model_note': 'Genesis 原生 CPU 物理、原生 IK、Rasterizer；桌面场景。SDK 将 Panda tendon 近似为关节执行器，本平台控制双指位置；不等同 MuJoCo 动力学。',
                'scene_id': self.scene_id, 'scene_name': '桌面抓取', 'seed': self.seed,
                'evaluation': self.evaluator.snapshot(), 'playback': self.playback,
                'tcp': pose['position_m'], 'tcp_pose': pose, 'task': self.task.copy(), 'events': list(self.events),
                'blocks': {name: array(box.get_pos()).tolist() for name, box in self.blocks.items()},
                'has_wrist_camera': bool(self.cameras), 'source': 'https://github.com/Genesis-Embodied-AI/Genesis',
                'compute_backend': 'cpu', 'render_backend': 'rasterizer'}

    def cancel_motion(self):
        if self.motion is not None or self.task['status'] == 'settling': self.task['status'] = 'cancelled'
        self.motion, self.pose_goal = None, None

    def start_motion(self, poses, title, pose_goal=None, policy=False):
        if self.motion is not None or self.task['status'] == 'settling': raise ValueError('当前动作尚未完成，请先等待或停止。')
        self.motion = {'poses': poses, 'index': 0, 'elapsed': 0., 'from': self.ctrl.copy(), 'policy': policy}
        self.pose_goal = pose_goal
        self.task = {'status': 'running', 'title': title, 'steps': [p['name'] for p in poses], 'step': 0, 'progress': 0.}
        self.paused = False

    def step(self):
        if self.motion is not None:
            motion = self.motion; pose = motion['poses'][motion['index']]
            motion['elapsed'] += self.timestep
            fraction = min(motion['elapsed']/pose['duration'], 1.)
            smooth = fraction*fraction*(3-2*fraction)
            self.ctrl = motion['from']+(pose['ctrl']-motion['from'])*smooth
            self.task['progress'] = (motion['index']+fraction)/len(motion['poses'])
            if fraction >= 1:
                motion['index'] += 1
                if motion['index'] == len(motion['poses']):
                    self.motion = None
                    self.task['status'] = 'complete' if motion['policy'] else 'settling'
                    self.settle_deadline = self.time+3.
                else:
                    motion['elapsed'], motion['from'] = 0., self.ctrl.copy()
                    self.task['step'] = motion['index']
        self.robot.control_dofs_position(self._native_target(self.ctrl))
        self.scene.step(update_visualizer=False)
        self.physics_steps += 1; self.time += self.timestep
        if not np.isfinite(array(self.robot.get_qpos())).all(): raise RuntimeError('Genesis 物理状态出现非有限数值，请重置并检查控制目标。')
        self.evaluator.update(self)
        if self.task['status'] == 'settling':
            q, v = array(self.robot.get_dofs_position()), array(self.robot.get_dofs_velocity())
            reached = np.max(np.abs(q[:7]-self.ctrl[:7])) < .025
            if self.pose_goal:
                pose = self.tcp_pose()
                ep = float(np.linalg.norm(np.asarray(pose['position_m'])-self.pose_goal.position_m))
                er = quaternion_angle(pose['quaternion_wxyz'], self.pose_goal.quaternion_wxyz)
                self.task.update(position_error_m=ep, orientation_error_rad=er)
                reached = ep < .006 and er < .05
            if reached and np.max(np.abs(v[:7])) < .05: self.task['status'] = 'complete'
            elif self.time > self.settle_deadline:
                self.task.update(status='failed', message='Genesis 动作未收敛到目标容差，请缩小动作并检查接触或关节限制。')
        if (self.run_deadline is not None and self.time >= self.run_deadline-1e-8) or (self.stop_on_success and self.evaluator.success):
            self.cancel_motion(); self.paused = True
            self.task['status'] = 'verified' if self.stop_on_success and self.evaluator.success else 'limit'

    def move_tcp(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已经切换，本轮动作已失效。')
        action = PoseAction.model_validate(payload['pose'])
        if action.status != 'move': raise ValueError('只有 move 动作可以进入 IK。')
        if action.robot_id not in (None, 'arm1'): raise ValueError('Genesis 当前仅支持 arm1。')
        if action.hand_joint_positions_rad is not None: raise ValueError('Panda 不支持灵巧手独立关节控制。')
        current = self.tcp_pose()
        if np.linalg.norm(np.asarray(action.position_m)-current['position_m']) > .12+1e-8: raise ValueError('单轮末端平移超过 0.12 m。')
        if action.position_m[2] < .008: raise ValueError('目标末端位于桌面或桌面以下。')
        if quaternion_angle(action.quaternion_wxyz, current['quaternion_wxyz']) > math.radians(35): raise ValueError('单轮姿态变化超过 35°。')
        result, error = self.robot.inverse_kinematics(link=self.hand, pos=np.array(action.position_m),
            quat=np.array(action.quaternion_wxyz), local_point=np.array([0, 0, .1034]), return_error=True,
            init_qpos=self.robot.get_qpos(), max_samples=1, max_solver_iters=100)
        joints = array(result)[:7]
        if not np.isfinite(joints).all() or np.max(np.abs(array(error)[:3])) > .0015 or np.max(np.abs(array(error)[3:])) > .02:
            raise ValueError(f'Genesis IK 未找到可达位姿；原生求解误差：{array(error).tolist()}')
        q = array(self.robot.get_dofs_position())[:7]
        if np.max(np.abs(joints-q)) > .65: raise ValueError('IK 所需关节变化超过单轮 0.65 rad，请缩小目标动作。')
        ctrl = np.r_[joints, action.gripper_open*255]
        self.start_motion([{'name': 'Genesis IK → PD 控制 → 末端反馈', 'ctrl': ctrl, 'duration': action.duration_s}], 'Agent 末端动作', action)
        return {'robot_id': 'arm1', 'joint_targets_rad': joints.tolist(), 'target': action.model_dump()}

    def policy_chunk(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已变化，策略动作块失效。')
        if payload.get('robot_id') not in (None, 'arm1'): raise ValueError('Genesis 当前仅支持 arm1。')
        targets, dt = np.asarray(payload['actions'], dtype=float), float(payload['dt_s'])
        if targets.ndim != 2 or targets.shape[1] != 8 or not 1 <= len(targets) <= 50 or not np.isfinite(targets).all(): raise ValueError('动作块必须为 1–50 帧、每帧 8 个有限执行器目标。')
        if payload['joint_names'] != JOINTS: raise ValueError('策略关节顺序与当前机器人不一致。')
        if not .02 <= dt <= .5: raise ValueError('策略动作间隔应在 0.02–0.5 秒。')
        lower, upper = np.array([a['min'] for a in self.actuators]), np.array([a['max'] for a in self.actuators])
        current, clipped, poses = self.ctrl.copy(), 0, []
        rate = np.array([2.]*7+[255.])
        for target in targets:
            bounded = np.clip(target, np.maximum(lower, current-rate*dt), np.minimum(upper, current+rate*dt))
            clipped += int(np.sum(np.abs(bounded-target) > 1e-8)); current = bounded
            poses.append({'name': '策略动作块', 'ctrl': bounded, 'duration': dt})
        self.start_motion(poses, '策略模型动作块', policy=True)
        return {'robot_id': 'arm1', 'frames': len(poses), 'limited_targets': clipped, 'executed_targets': [p['ctrl'].tolist() for p in poses]}

    def _rigid_state(self):
        state = self.scene.get_state()
        return state, state.solvers_state[self.scene.solvers.index(self.scene.rigid_solver)]

    def checkpoint(self):
        _, rigid = self._rigid_state()
        values = {key: array(getattr(rigid, key)).tolist() for key in STATE_FIELDS}
        result = {'backend': self.backend.copy(), 'model': self.model_id, 'scene_id': self.scene_id,
                  'seed': self.seed, 'task_id': self.task_id, 'active_robot': 'arm1', 'state': values,
                  'ctrl': self.ctrl.tolist(), 'time': self.time, 'initial_tcp': self.evaluator.initial_tcp.tolist()}
        result['state_id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
        return result

    def restore(self, checkpoint):
        identity = backend_identity('genesis', gs.__version__)
        recorded = checkpoint.get('backend', {})
        if any(recorded.get(key) != identity[key] for key in ('engine_id', 'engine_version', 'state_schema')):
            raise ValueError('检查点的仿真引擎、版本或状态格式与当前 Genesis 环境不一致。')
        self._validate(checkpoint['model'], checkpoint['scene_id'], checkpoint['task_id'])
        state, rigid = self._rigid_state()
        for key in STATE_FIELDS:
            tensor, value = getattr(rigid, key), np.asarray(checkpoint['state'][key])
            if tuple(value.shape) != tuple(tensor.shape) or not np.isfinite(value).all(): raise ValueError('记录状态的维度或数值与当前模型不符。')
        ctrl = np.asarray(checkpoint['ctrl'], dtype=float)
        if ctrl.shape != (8,) or not np.isfinite(ctrl).all(): raise ValueError('检查点控制目标无效。')
        initial_tcp = np.asarray(checkpoint['initial_tcp'], dtype=float)
        recorded_time, seed = checkpoint['time'], checkpoint['seed']
        if (initial_tcp.shape != (3,) or not np.isfinite(initial_tcp).all()
                or isinstance(recorded_time, bool) or not isinstance(recorded_time, (int, float))
                or not math.isfinite(recorded_time) or recorded_time < 0
                or type(seed) is not int or not 0 <= seed <= 4294967295):
            raise ValueError('检查点的初始末端、时间或布局种子无效。')
        import torch
        for key in STATE_FIELDS:
            tensor = getattr(rigid, key)
            setattr(rigid, key, torch.as_tensor(checkpoint['state'][key], dtype=tensor.dtype, device=tensor.device))
        self.scene.reset(state=state)
        self.ctrl, self.time, self.task_id, self.seed = ctrl, float(checkpoint['time']), checkpoint['task_id'], checkpoint['seed']
        # Existing checkpoints store elapsed fixed-step time rather than a
        # separate counter. Restore both public clocks to that recorded point.
        self.physics_steps = round(self.time / self.timestep)
        self.robot.control_dofs_position(self._native_target(ctrl))
        self.cancel_motion(); self.paused = True; self.generation += 1
        self.run_deadline, self.stop_on_success = None, False
        self.task = {'status': 'idle', 'title': '检查点状态已恢复', 'steps': [], 'step': -1, 'progress': 0.}
        self.evaluator = GenesisTaskEvaluator(self, initial_tcp)
        self.playback = None
        return {'state_id': checkpoint['state_id']}

    def trajectory_frame(self):
        checkpoint = self.checkpoint()
        return {'time': self.time, 'backend': checkpoint['backend'], 'active_robot': 'arm1', 'checkpoint': checkpoint,
                'qpos': array(self.robot.get_qpos()).tolist(), 'qvel': array(self.robot.get_dofs_velocity()).tolist(),
                'ctrl': self.ctrl.tolist(), 'actuator_force': array(self.robot.get_dofs_control_force()).tolist(),
                'contacts': len(self.robot.get_contacts()['link_a']), 'contact_scope': 'active_robot', 'evaluation': self.evaluator.snapshot()}

    def replay_frame(self, frame):
        if frame.get('backend') != frame.get('checkpoint', {}).get('backend'):
            raise ValueError('轨迹帧的后端身份与检查点不一致。')
        self.restore(frame['checkpoint'])
        self.playback = {'time': self.time, 'frame_index': frame.get('index'), 'recorded_evaluation': frame.get('evaluation')}
        self.task['title'] = '记录状态回放 · 未重新推理或执行动作'
        return self.playback

    def verify_task(self, generation, on_step=None):
        if generation != self.generation: raise ValueError('场景已切换，完成声明未被验收。')
        if not self.paused or self.motion is not None: raise ValueError('请在动作结束并暂停后验收任务。')
        self.evaluator.update(self)
        started = self.time; deadline = started+self.evaluator.REQUIRED_HOLD_S
        if self.run_deadline is not None: deadline = min(deadline, self.run_deadline)
        try:
            while not self.evaluator.success and self.time+self.timestep <= deadline+1e-9:
                self.step()
                if on_step is not None: on_step()
        finally: self.paused = True
        return {**self.evaluator.snapshot(), 'verification_sim_time_s': self.time-started}

    def _wrist(self):
        quat = array(self.hand.get_quat()); mat = rotation(quat)
        pos = array(self.hand.get_pos())+mat.apply([.045, 0, .035])
        self.cameras['wrist'].set_pose(pos=pos, lookat=pos+mat.apply([0, 0, 1]), up=mat.apply([0, -1, 0]))

    @staticmethod
    def _jpeg(pixels, quality=86):
        output = io.BytesIO(); Image.fromarray(pixels).save(output, 'JPEG', quality=quality)
        return output.getvalue()

    def observe(self):
        if not self.cameras: raise ValueError('三路视觉观察需要原生渲染器。')
        self._wrist(); robots = self.robot_states()
        observations = []
        for name in ('front', 'side', 'wrist'):
            cam = self.cameras[name]; rgb, *_ = cam.render()
            observations.append({'name': name, 'image_base64': base64.b64encode(self._jpeg(rgb, 88)).decode(),
                                 'width': 640, 'height': 426, 'time': self.time,
                                 'position_world_m': np.asarray(cam.pos).tolist(),
                                 'rotation_camera_to_world': np.asarray(cam.transform)[:3, :3].tolist(), 'fovy_deg': float(cam.fov)})
        return {'backend': {'engine_id': 'genesis', 'engine_version': gs.__version__, 'control_frame': 'world_z_up_m_wxyz'},
                'generation': self.generation, 'time': self.time, 'robot': 'panda', 'active_robot': 'arm1', 'robots': robots,
                'arm_dof': 7, 'tcp': self.tcp_pose(), 'joints': robots[0]['joints'], 'gripper': robots[0]['gripper'],
                'limits': {'translation_per_round_m': .12, 'rotation_per_round_deg': 35, 'position_tolerance_m': .006, 'orientation_tolerance_rad': .05},
                'actuators': robots[0]['actuators'], 'hand_joints': [], 'cameras': observations}

    def set_view(self, view):
        if view not in ('perspective', 'top', 'front', 'wrist'): raise ValueError('未知视角。')
        self.view = view
        self.lookat, self.distance = np.array([.24, 0, .3]), 2.15
        self.azimuth = {'perspective': -48, 'top': -90, 'front': -90, 'wrist': 0}[view]
        self.elevation = {'perspective': 26, 'top': 89, 'front': 8, 'wrist': 0}[view]

    def render_rgb(self):
        if not self.cameras: raise ValueError('Genesis 渲染器未开启。')
        if self.view == 'wrist': self._wrist(); cam = self.cameras['wrist']
        else:
            cam = self.cameras['main']
            az, el = math.radians(self.azimuth), math.radians(self.elevation)
            offset = self.distance*np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
            cam.set_pose(pos=self.lookat+offset, lookat=self.lookat)
        rgb, *_ = cam.render()
        return np.asarray(Image.fromarray(rgb).resize((960, 640))) if rgb.shape[:2] != (640, 960) else rgb

    def render(self):
        return self._jpeg(self.render_rgb())

    def instruction(self, text):
        clean = re.sub(r'[\s，。！!]', '', text).removeprefix('请')
        if clean in ('机械臂归位', '归位', '回到初始姿态'): self.command({'action': 'home'})
        elif clean in ('张开夹爪', '打开夹爪', '闭合夹爪', '关闭夹爪'):
            self.cancel_motion(); self.ctrl[7] = 255. if clean.startswith(('张开', '打开')) else 0.
        elif clean in ('暂停', '暂停仿真'): self.paused = True
        elif clean in ('继续', '继续仿真', '开始仿真'): self.paused = False
        else:
            match = re.fullmatch(r'(?:把)?关节([1-7])(?:转到|设为)([-+]?\d+(?:\.\d+)?)(?:度|°)', clean)
            if not match: raise ValueError('快捷命令支持归位、开闭夹爪、暂停/继续及“关节1转到30度”。视觉任务请使用 Agent 接管。')
            self.command({'action': 'control', 'index': int(match[1])-1, 'value': math.radians(float(match[2]))})
        return '指令已交给 Genesis 原生控制接口。'

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene'):
            model, scene, task = payload.get('model', self.model_id), payload.get('scene_id', self.scene_id), payload.get('task_id') or self.task_id
            self._validate(model, scene, task)
            self.seed, self.task_id = payload.get('seed', self.seed), task
            self.reset(); self.generation += 1
        elif action == 'reset': self.reset(); self.generation += 1
        elif action == 'pause': self.paused = bool(payload['value'])
        elif action == 'render_suspend': self.render_suspended = bool(payload['value']); self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停仿真，再执行单步。')
            self.step()
        elif action == 'control':
            index, value = payload['index'], payload['value']
            if not 0 <= index < 8: raise ValueError('执行器编号不存在。')
            item = self.actuators[index]
            if not math.isfinite(value) or not item['min'] <= value <= item['max']: raise ValueError('控制目标超出关节范围。')
            self.cancel_motion(); self.ctrl[index] = value
        elif action == 'home': self.start_motion([{'name': '回到初始关节姿态', 'ctrl': HOME.copy(), 'duration': 2.5}], '机械臂归位')
        elif action == 'stop': self.cancel_motion(); self.paused = True; self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits':
            self.evaluator = GenesisTaskEvaluator(self); self.run_deadline = self.time+payload['duration_s']; self.stop_on_success = payload['evaluator_stop']
        elif action == 'select_robot':
            if payload['robot_id'] != 'arm1': raise ValueError('Genesis 当前仅支持 arm1。')
            return {'active_robot': 'arm1', 'generation': self.generation}
        elif action == 'instruction': return self.instruction(payload['text'])
        elif action == 'observe': return self.observe()
        elif action == 'move_tcp': return self.move_tcp(payload)
        elif action == 'policy_chunk': return self.policy_chunk(payload)
        elif action == 'checkpoint': return self.checkpoint()
        elif action == 'restore': return self.restore(payload['checkpoint'])
        elif action == 'replay_frame': return self.replay_frame(payload['frame'])
        elif action == 'verify_task': return self.verify_task(payload['generation'])
        elif action == 'settle':
            self.cancel_motion()
            for _ in range(100): self.step()
            self.paused = True; self.evaluator = GenesisTaskEvaluator(self)
        elif action == 'view': self.set_view(payload['view'])
        elif action == 'camera':
            self.view = 'custom'
            self.azimuth += payload.get('dx', 0); self.elevation = float(np.clip(self.elevation-payload.get('dy', 0), 3, 89))
            self.distance = float(np.clip(self.distance*math.exp(payload.get('zoom', 0)), .5, 5))
            az, el = math.radians(self.azimuth), math.radians(self.elevation)
            right = np.array([-math.sin(az), math.cos(az), 0.])
            up = np.array([-math.sin(el)*math.cos(az), -math.sin(el)*math.sin(az), math.cos(el)])
            self.lookat += self.distance*(-payload.get('pan_dx', 0)*right+payload.get('pan_dy', 0)*up)
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        else: raise ValueError(f'Genesis 未实现此操作：{action}')
        return '操作已完成。'

    def close(self):
        gs.destroy()
