"""Official ManiSkill tasks using SAPIEN CPU PhysX and native Vulkan cameras."""
from __future__ import annotations

import base64
from collections import deque
import hashlib
import io
from importlib.metadata import version
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

from ..actions import PoseAction
from ..simulators import backend_identity

TASKS = {
    'pick_cube': {'env_id': 'PickCube-v1', 'name': '抓取方块到目标',
                  'instruction': '用 Panda 抓住红色方块并移动到绿色目标球，保持机械臂稳定。',
                  'criterion': '官方 PickCube-v1 evaluate()：方块距目标不超过 2.5 cm，机械臂静止；连续保持 0.3 秒。'},
    'push_cube': {'env_id': 'PushCube-v1', 'name': '推方块到目标区',
                  'instruction': '用 Panda 将蓝色方块推入红白圆形目标区，方块保持在桌面上。',
                  'criterion': '官方 PushCube-v1 evaluate()：方块位于目标半径内且留在桌面；连续保持 0.3 秒。'},
}
MODELS = {'panda': {'name': 'Franka Panda', 'arm_dof': 7, 'hand_dof': 1,
                     'folder': 'mani_skill/assets/robots/panda', 'file': 'panda_v2.urdf', 'image': 'panda.png',
                     'source': 'https://github.com/mani-skill/ManiSkill/tree/main/mani_skill/assets/robots/panda'}}
SCENES = {key: {'name': value['name'], 'description': value['instruction'], 'tasks': [key]} for key, value in TASKS.items()}
JOINTS = [f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1']


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def serial(value):
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    return array(value).tolist()


def rotation(quat):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])


def quaternion_angle(a, b):
    return float((rotation(a) * rotation(b).inv()).magnitude())


class ManiSkillEvaluator:
    REQUIRED_HOLD_S = .3

    def __init__(self, sim):
        self.success = self.ever_success = self.previous_valid = False
        self.hold_s, self.last_time = 0., sim.time
        self.update(sim)

    def update(self, sim):
        native = {key: bool(array(value).reshape(-1)[0]) for key, value in sim.env.evaluate().items()}
        finite = bool(np.isfinite(array(sim.robot.get_qpos())).all() and np.isfinite(array(sim.robot.get_qvel())).all())
        valid = native.get('success', False) and finite
        elapsed = sim.time - self.last_time
        self.hold_s = self.hold_s + elapsed if valid and self.previous_valid and elapsed >= 0 else 0.
        self.success = bool(valid and self.hold_s + 1e-8 >= self.REQUIRED_HOLD_S)
        self.ever_success |= self.success
        self.previous_valid, self.last_time = valid, sim.time
        checks = [{'key': 'official_success', 'label': '官方任务成功判据', 'actual': native.get('success', False), 'expected': True, 'passed': valid},
                  {'key': 'state_finite', 'label': '物理状态数值有效', 'actual': finite, 'expected': True, 'passed': finite},
                  {'key': 'hold_duration', 'label': '成功状态持续时间', 'actual': self.hold_s, 'expected': '>= 0.3', 'passed': self.success, 'unit': 's'}]
        self.last = {'task_id': sim.task_id, **TASKS[sim.task_id], 'source': 'maniskill_official_evaluate',
                     'version': 'maniskill-task-v1', 'success': self.success, 'ever_success': self.ever_success,
                     'currently_satisfied': valid, 'hold_s': self.hold_s, 'required_hold_s': self.REQUIRED_HOLD_S,
                     'status': 'success' if self.success else 'holding' if valid else 'unsatisfied',
                     'progress': 1. if self.success else .8 if valid else 0., 'evaluated_at_sim_s': sim.time,
                     'values': {'official_env_id': TASKS[sim.task_id]['env_id'], 'official_evaluation': native},
                     'checks': checks, 'failure_reasons': [c['label'] for c in checks if not c['passed']]}

    def snapshot(self): return self.last.copy()


class ManiSkillSimulation:
    def __init__(self, model_id='panda', render=True, scene_id='pick_cube', seed=42, task_id=None):
        self._validate(model_id, scene_id, task_id or scene_id)
        # Catalog import in the main service must not import the optional SDK.
        global gym, Camera, CameraConfig, sapien_utils, Pose, torch, pk
        self._sdk_directory = None
        spec = importlib.util.find_spec('mani_skill')
        if spec is None: raise ImportError('请在独立 .venv-maniskill 中安装 requirements-maniskill.txt。')
        # SAPIEN's native asset reader rejects existing Unicode paths on
        # Windows. Load unchanged official package assets and Python modules
        # from one scoped ASCII directory, without patching installed files.
        if os.name == 'nt' and not str(spec.origin).isascii():
            parent = Path(tempfile.gettempdir()).resolve()
            if not str(parent).isascii(): raise ValueError('SAPIEN Windows 资源加载需要英文路径，请将 TEMP 设为可写英文目录。')
            self._sdk_directory = tempfile.TemporaryDirectory(prefix='embodiworks-maniskill-')
            target = Path(self._sdk_directory.name).resolve()
            if target.parent != parent or not target.name.startswith('embodiworks-maniskill-'): raise RuntimeError('SDK 临时目录超出预期位置。')
            shutil.copytree(Path(spec.origin).parent, target/'mani_skill', ignore=shutil.ignore_patterns('__pycache__'))
            sys.path.insert(0, str(target))
            import sapien
            # The same native Unicode failure affects SAPIEN's 279 KB shader
            # directory. Use its own package initialization setter for the
            # unchanged ASCII copy, before ManiSkill creates any cameras.
            shutil.copytree(Path(sapien.__file__).parent/'vulkan_shader', target/'vulkan_shader')
            sapien.pysapien.render._internal_set_shader_search_path(str(target/'vulkan_shader'))
        import gymnasium as gym
        import mani_skill.envs
        from mani_skill.sensors.camera import Camera, CameraConfig
        from mani_skill.utils import sapien_utils
        from mani_skill.utils.structs import Pose
        import torch
        import pytorch_kinematics as pk
        self.model_id, self.scene_id, self.seed, self.task_id = model_id, scene_id, seed, task_id or scene_id
        self.backend = {**backend_identity('maniskill', version('mani_skill')),
                        'runtime_packages': {p: version(p) for p in ('sapien', 'torch', 'pytorch_kinematics', 'numpy', 'scipy', 'pillow')}}
        self.render_enabled, self.render_suspended = render, False
        self.generation, self.speed, self.active_robot = 1, 1., 'arm1'
        self.flags = {'contacts': False, 'joints': False, 'collision': False}
        self.events, self.event_id = deque(maxlen=30), 0
        self.env = None
        self._load()
        self.event('已加载官方 ManiSkill Panda 任务，使用 SAPIEN CPU PhysX 与 Vulkan 相机。')

    @staticmethod
    def _validate(model, scene, task):
        if model != 'panda' or scene not in TASKS or task != scene:
            raise ValueError('ManiSkill 当前支持 Panda、pick_cube / push_cube 场景及同名官方任务。')

    def _load(self):
        if self.env is not None:
            self.cameras = {}; self.kinematics = None; self.robot = None
            self.env.close()
            self.env = None
        self.env = gym.make(TASKS[self.task_id]['env_id'], robot_uids='panda', num_envs=1,
            obs_mode='none', reward_mode='none', control_mode='pd_joint_pos',
            sim_backend='physx_cpu', render_backend='gpu' if self.render_enabled else 'none',
            render_mode='rgb_array' if self.render_enabled else None).unwrapped
        self.robot = self.env.agent.robot
        self.kinematics = pk.build_serial_chain_from_urdf(Path(self.env.agent.urdf_path).read_text(),
                                                        self.env.agent.ee_link_name).to(dtype=torch.float32, device='cpu')
        self.timestep = 1. / self.env.control_freq
        limits = array(self.robot.get_qlimits())[0]
        self.actuators = [{'index': i, 'name': name, 'robot_id': 'arm1', 'is_gripper': i == 7,
                          'min': 0. if i == 7 else float(limits[i, 0]), 'max': 255. if i == 7 else float(limits[i, 1]),
                          'qpos_index': i, 'dof_index': i, 'unit': 'command' if i == 7 else 'rad'} for i, name in enumerate(JOINTS)]
        self.cameras = {}
        if self.render_enabled:
            for name, eye, target in [('main', (1.1, -1.3, 1.), (-.15, 0, .2)),
                                      ('front', (.45, -.7, .55), (0, 0, .1)),
                                      ('side', (.1, .8, .65), (-.15, 0, .15)),
                                      ('wrist', (.1, -.2, .6), (0, 0, 0))]:
                self.cameras[name] = Camera(CameraConfig('embodiworks_'+name,
                    sapien_utils.look_at(eye, target), 960 if name == 'main' else 640,
                    640 if name == 'main' else 426, math.radians(52 if name != 'wrist' else 75), shader_pack='minimal'), self.env.scene)
        self.reset()

    def reset(self):
        self.env.reset(seed=self.seed)
        if self.env.agent.control_mode != 'pd_joint_pos': self.env.agent.set_control_mode('pd_joint_pos')
        self.env.agent.controller.reset()
        self.ctrl = self._public_ctrl(array(self.robot.get_qpos())[0])
        self.home = self.ctrl.copy()
        self.time, self.physics_steps = 0., 0
        self.motion, self.pose_goal, self.playback = None, None, None
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.task = {'status': 'idle', 'title': '等待指令', 'steps': [], 'step': -1, 'progress': 0.}
        self.evaluator = ManiSkillEvaluator(self)
        self.set_view('perspective')

    @staticmethod
    def _public_ctrl(q): return np.r_[q[:7], np.clip(q[7]/.04, 0, 1)*255]

    @staticmethod
    def _native_ctrl(q): return np.r_[q[:7], (q[7]/255*.04+.01)/.05*2-1].astype(np.float32)

    def event(self, message, kind='info'):
        self.event_id += 1
        self.events.append({'id': self.event_id, 'time': self.time, 'message': message, 'kind': kind})

    def tcp_pose(self):
        pose = array(self.env.agent.tcp_pose.raw_pose)[0]
        return {'frame': 'world', 'position_m': pose[:3].tolist(), 'quaternion_wxyz': pose[3:].tolist()}

    def _actuators(self):
        q, f = array(self.robot.get_qpos())[0], array(self.robot.get_qf())[0]
        return [{**a, 'position': float(q[i]), 'actual': float(q[i]), 'target': float(self.ctrl[i]), 'force': float(f[i])} for i, a in enumerate(self.actuators)]

    def robot_states(self):
        q, v = array(self.robot.get_qpos())[0], array(self.robot.get_qvel())[0]
        return [{'id': 'arm1', 'name': 'Franka Panda', 'model': 'panda', 'active': True, 'arm_dof': 7, 'hand_dof': 1,
                 'tcp': self.tcp_pose(), 'gripper': {'position': float(q[7]), 'unit': 'm_per_finger', 'target_open_fraction': float(self.ctrl[7]/255)},
                 'joints': [{'name': a['name'], 'position_rad': float(q[i]), 'velocity_rad_s': float(v[i]), 'limits_rad': [a['min'], a['max']]} for i, a in enumerate(self.actuators[:7])],
                 'actuators': self._actuators()}]

    def state(self):
        pose, robots = self.tcp_pose(), self.robot_states()
        return {'engine': 'ManiSkill / SAPIEN', 'version': version('mani_skill'), 'backend': self.backend.copy(),
                'model': 'panda', 'name': 'Franka Panda', 'active_robot': 'arm1', 'robots': robots,
                'generation': self.generation, 'paused': self.paused, 'speed': self.speed, 'time': self.time,
                'timestep': self.timestep, 'steps': self.physics_steps, 'contacts': len(self.env.scene.get_contacts()),
                'contact_scope': 'scene', 'gravity': True, 'view': self.view, 'flags': self.flags.copy(),
                'nq': int(self.robot.dof[0]), 'nv': int(self.robot.dof[0]), 'nu': 8, 'meshes': 0,
                'arm_dof': 7, 'hand_dof': 1, 'actuators': robots[0]['actuators'],
                'model_note': '官方 ManiSkill Panda；SAPIEN CPU PhysX、Vulkan、PyTorch Kinematics CPU IK 与官方 PD 控制器。每次步进为一个原生控制周期。',
                'scene_id': self.scene_id, 'scene_name': TASKS[self.scene_id]['name'], 'seed': self.seed,
                'evaluation': self.evaluator.snapshot(), 'playback': self.playback, 'tcp': pose['position_m'], 'tcp_pose': pose,
                'task': self.task.copy(), 'events': list(self.events), 'has_wrist_camera': bool(self.cameras),
                'source': 'https://github.com/mani-skill/ManiSkill', 'compute_backend': 'physx_cpu', 'render_backend': 'vulkan'}

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
            if fraction >= 1-1e-8:
                motion['index'] += 1
                if motion['index'] == len(motion['poses']):
                    self.motion = None
                    self.task['status'] = 'complete' if motion['policy'] else 'settling'
                    self.settle_deadline = self.time+3.
                else:
                    motion['elapsed'], motion['from'] = 0., self.ctrl.copy()
                    self.task['step'] = motion['index']
        self.env.step(self._native_ctrl(self.ctrl))
        self.physics_steps += 1; self.time = self.physics_steps*self.timestep
        q, v = array(self.robot.get_qpos())[0], array(self.robot.get_qvel())[0]
        if not np.isfinite(q).all() or not np.isfinite(v).all(): raise RuntimeError('SAPIEN 物理状态出现非有限数值，请重置并检查控制目标。')
        self.evaluator.update(self)
        if self.task['status'] == 'settling':
            reached = np.max(np.abs(q[:7]-self.ctrl[:7])) < .025
            if self.pose_goal:
                pose = self.tcp_pose()
                ep = float(np.linalg.norm(np.asarray(pose['position_m'])-self.pose_goal.position_m))
                er = quaternion_angle(pose['quaternion_wxyz'], self.pose_goal.quaternion_wxyz)
                self.task.update(position_error_m=ep, orientation_error_rad=er)
                reached = ep < .006 and er < .05
            if reached and np.max(np.abs(v[:7])) < .05: self.task['status'] = 'complete'
            elif self.time > self.settle_deadline: self.task.update(status='failed', message='SAPIEN 动作未收敛到目标容差，请缩小动作并检查碰撞与关节范围。')
        if (self.run_deadline is not None and self.time >= self.run_deadline-1e-8) or (self.stop_on_success and self.evaluator.success):
            self.cancel_motion(); self.paused = True
            self.task['status'] = 'verified' if self.stop_on_success and self.evaluator.success else 'limit'

    def move_tcp(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已变化，本轮动作失效。')
        action = PoseAction.model_validate(payload['pose'])
        if action.status != 'move': raise ValueError('只有 move 动作可以进入 IK。')
        if action.robot_id not in (None, 'arm1') or action.hand_joint_positions_rad is not None: raise ValueError('ManiSkill Panda 支持 arm1 与双指开合。')
        current = self.tcp_pose()
        if np.linalg.norm(np.asarray(action.position_m)-current['position_m']) > .12+1e-8: raise ValueError('单轮末端平移超过 0.12 m。')
        if action.position_m[2] < .008: raise ValueError('目标末端位于桌面或桌面以下。')
        if quaternion_angle(action.quaternion_wxyz, current['quaternion_wxyz']) > math.radians(35): raise ValueError('单轮姿态变化超过 35°。')
        target = self.robot.pose.inv()*Pose.create_from_pq(action.position_m, action.quaternion_wxyz)
        solver = pk.PseudoInverseIK(self.kinematics, retry_configs=self.robot.get_qpos()[:, :7].clone(),
            joint_limits=torch.as_tensor(self.kinematics.get_joint_limits(), dtype=torch.float32).T,
            max_iterations=100, pos_tolerance=.001, rot_tolerance=.015, lr=.3,
            early_stopping_any_converged=True, early_stopping_no_improvement=None)
        result = solver.solve(pk.Transform3d(matrix=target.to_transformation_matrix()))
        if not bool(result.converged[0, 0]): raise ValueError('PyTorch Kinematics IK 未找到可达解，请缩小目标位姿。')
        joints = array(result.solutions)[0, 0]
        if not np.isfinite(joints).all() or np.max(np.abs(joints-array(self.robot.get_qpos())[0, :7])) > .65: raise ValueError('IK 关节变化超过单轮 0.65 rad，请缩小目标动作。')
        if any(not item['min'] <= joints[i] <= item['max'] for i, item in enumerate(self.actuators[:7])): raise ValueError('IK 解超出 Panda 关节范围，请调整目标位姿。')
        self.start_motion([{'name': 'PyTorch Kinematics IK → PhysX PD → 末端反馈', 'ctrl': np.r_[joints, action.gripper_open*255], 'duration': action.duration_s}], 'Agent 末端动作', action)
        return {'robot_id': 'arm1', 'joint_targets_rad': joints.tolist(), 'target': action.model_dump()}

    def policy_chunk(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已变化，策略动作块失效。')
        if payload.get('robot_id') not in (None, 'arm1'): raise ValueError('ManiSkill 当前仅支持 arm1。')
        targets, dt = np.asarray(payload['actions'], dtype=float), float(payload['dt_s'])
        if targets.ndim != 2 or targets.shape[1] != 8 or not 1 <= len(targets) <= 50 or not np.isfinite(targets).all(): raise ValueError('动作块须为 1–50 帧、每帧 8 个有限执行器目标。')
        if payload['joint_names'] != JOINTS: raise ValueError('策略关节顺序与当前机器人不一致。')
        if not self.timestep <= dt <= .5: raise ValueError(f'动作间隔须为 {self.timestep}–0.5 秒。')
        lower, upper = np.array([a['min'] for a in self.actuators]), np.array([a['max'] for a in self.actuators])
        current, clipped, poses = self.ctrl.copy(), 0, []
        for target in targets:
            rate = np.array([2.]*7+[255.])*dt
            bounded = np.clip(target, np.maximum(lower, current-rate), np.minimum(upper, current+rate))
            clipped += int(np.sum(np.abs(bounded-target) > 1e-8)); current = bounded
            poses.append({'name': '策略动作块', 'ctrl': bounded, 'duration': dt})
        self.start_motion(poses, '策略模型动作块', policy=True)
        return {'robot_id': 'arm1', 'frames': len(poses), 'limited_targets': clipped, 'executed_targets': [p['ctrl'].tolist() for p in poses]}

    def checkpoint(self):
        result = {'backend': self.backend.copy(), 'model': self.model_id, 'scene_id': self.scene_id, 'task_id': self.task_id,
                  'seed': self.seed, 'state': serial(self.env.get_state_dict()), 'ctrl': self.ctrl.tolist(), 'home': self.home.tolist(),
                  'time': self.time, 'steps': self.physics_steps, 'active_robot': 'arm1'}
        result['state_id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
        return result

    def restore(self, checkpoint):
        recorded = checkpoint.get('backend', {})
        if any(recorded.get(key) != self.backend[key] for key in ('engine_id', 'engine_version', 'state_schema', 'runtime_packages')): raise ValueError('检查点的引擎、SDK 版本或状态格式与当前 ManiSkill 不一致。')
        self._validate(checkpoint['model'], checkpoint['scene_id'], checkpoint['task_id'])
        ctrl = np.asarray(checkpoint['ctrl'], dtype=float)
        if ctrl.shape != (8,) or not np.isfinite(ctrl).all(): raise ValueError('检查点控制目标无效。')
        home = np.asarray(checkpoint['home'], dtype=float)
        if home.shape != (8,) or not np.isfinite(home).all(): raise ValueError('检查点归位姿态无效。')
        def tensors(value, template):
            if isinstance(template, dict):
                if not isinstance(value, dict) or value.keys() != template.keys(): raise ValueError('检查点状态字段不匹配。')
                return {k: tensors(value[k], v) for k, v in template.items()}
            data = np.asarray(value)
            if tuple(data.shape) != tuple(template.shape) or not np.isfinite(data).all(): raise ValueError('检查点状态维度或数值无效。')
            return torch.as_tensor(data, dtype=template.dtype, device=template.device)
        # Validate before mutating or changing tasks.
        if checkpoint['scene_id'] != self.scene_id: raise ValueError('请先加载检查点对应的 ManiSkill 场景再恢复。')
        restored = tensors(checkpoint['state'], self.env.get_state_dict())
        if (type(checkpoint['seed']) is not int or not 0 <= checkpoint['seed'] <= 4294967295
            or type(checkpoint['steps']) is not int or checkpoint['steps'] < 0
            or not isinstance(checkpoint['time'], (int, float)) or not math.isfinite(checkpoint['time'])
            or abs(checkpoint['time']-checkpoint['steps']*self.timestep) > 1e-6): raise ValueError('检查点种子或时间无效。')
        self.env.set_state_dict(restored)
        self.env.agent.controller.reset()
        self.ctrl, self.home, self.time, self.physics_steps, self.seed = ctrl, home, checkpoint['time'], checkpoint['steps'], checkpoint['seed']
        self.cancel_motion(); self.paused = True; self.generation += 1
        self.run_deadline, self.stop_on_success, self.playback = None, False, None
        self.task = {'status': 'idle', 'title': '检查点状态已恢复', 'steps': [], 'step': -1, 'progress': 0.}
        self.evaluator = ManiSkillEvaluator(self)
        return {'state_id': checkpoint['state_id']}

    def trajectory_frame(self):
        return {'time': self.time, 'backend': self.backend.copy(), 'active_robot': 'arm1', 'checkpoint': self.checkpoint(),
                'qpos': array(self.robot.get_qpos())[0].tolist(), 'qvel': array(self.robot.get_qvel())[0].tolist(),
                'ctrl': self.ctrl.tolist(), 'actuator_force': array(self.robot.get_qf())[0].tolist(),
                'contacts': len(self.env.scene.get_contacts()), 'contact_scope': 'scene', 'evaluation': self.evaluator.snapshot()}

    def replay_frame(self, frame):
        if frame.get('backend') != frame.get('checkpoint', {}).get('backend'): raise ValueError('轨迹帧后端身份与检查点不一致。')
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
            while not self.evaluator.success and self.time+self.timestep <= deadline+1e-8:
                self.step()
                if on_step is not None: on_step()
        finally: self.paused = True
        return {**self.evaluator.snapshot(), 'verification_sim_time_s': self.time-started}

    @staticmethod
    def _jpeg(rgb, quality=86):
        output = io.BytesIO(); Image.fromarray(rgb).save(output, 'JPEG', quality=quality)
        return output.getvalue()

    def _wrist(self):
        tcp = self.tcp_pose(); mat = rotation(tcp['quaternion_wxyz'])
        pos = np.asarray(tcp['position_m'])+mat.apply([.045, 0, -.04])
        self.cameras['wrist'].camera.set_local_pose(sapien_utils.look_at(pos, pos+mat.apply([0, 0, 1]), up=mat.apply([0, -1, 0])).sp)

    def _camera_rgb(self, cam):
        self.env.scene.update_render()
        cam.capture()
        # CPU PhysX uses CPU tensors. Read the native Vulkan camera into host
        # memory instead of requesting CUDA tensor interoperability, which the
        # Windows SAPIEN wheel and CPU PyTorch do not provide.
        pixels = cam.camera._render_cameras[0].get_picture('Color')
        return np.asarray(pixels)[..., :3].astype(np.uint8)

    def observe(self):
        if not self.cameras: raise ValueError('三路视觉观察需要原生 Vulkan 渲染器。')
        self._wrist(); cameras = []
        for name in ('front', 'side', 'wrist'):
            cam = self.cameras[name]; rgb = self._camera_rgb(cam)
            cameras.append({'name': name, 'image_base64': base64.b64encode(self._jpeg(rgb, 88)).decode(),
                            'width': 640, 'height': 426, 'time': self.time, 'calibration': serial(cam.get_params())})
        robots = self.robot_states()
        return {'backend': {'engine_id': 'maniskill', 'engine_version': version('mani_skill'), 'control_frame': 'world_z_up_m_wxyz'},
                'generation': self.generation, 'time': self.time, 'robot': 'panda', 'active_robot': 'arm1', 'robots': robots,
                'arm_dof': 7, 'tcp': self.tcp_pose(), 'joints': robots[0]['joints'], 'gripper': robots[0]['gripper'],
                'limits': {'translation_per_round_m': .12, 'rotation_per_round_deg': 35, 'position_tolerance_m': .006, 'orientation_tolerance_rad': .05},
                'actuators': robots[0]['actuators'], 'hand_joints': [], 'cameras': cameras}

    def set_view(self, view):
        if view not in ('perspective', 'top', 'front', 'wrist'): raise ValueError('未知视角。')
        self.view, self.lookat, self.distance = view, np.array([-.15, 0, .25]), 1.8
        self.azimuth = {'perspective': -48, 'top': -90, 'front': -90, 'wrist': 0}[view]
        self.elevation = {'perspective': 26, 'top': 89, 'front': 8, 'wrist': 0}[view]

    def render_rgb(self):
        if not self.cameras: raise ValueError('ManiSkill 原生渲染器未开启。')
        if self.view == 'wrist': self._wrist(); cam = self.cameras['wrist']
        else:
            cam = self.cameras['main']; az, el = math.radians(self.azimuth), math.radians(self.elevation)
            offset = self.distance*np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
            cam.camera.set_local_pose(sapien_utils.look_at(self.lookat+offset, self.lookat).sp)
        rgb = self._camera_rgb(cam)
        return np.asarray(Image.fromarray(rgb).resize((960, 640))) if rgb.shape[:2] != (640, 960) else rgb

    def render(self): return self._jpeg(self.render_rgb())

    def instruction(self, text):
        clean = re.sub(r'[\s，。！!]', '', text).removeprefix('请')
        if clean in ('机械臂归位', '归位', '回到初始姿态'): self.command({'action': 'home'})
        elif clean in ('张开夹爪', '打开夹爪', '闭合夹爪', '关闭夹爪'):
            self.cancel_motion(); self.ctrl[7] = 255. if clean.startswith(('张开', '打开')) else 0.
        elif clean in ('暂停', '暂停仿真'): self.paused = True
        elif clean in ('继续', '继续仿真', '开始仿真'): self.paused = False
        else: raise ValueError('快捷命令支持归位、开闭夹爪、暂停/继续。视觉任务请使用 Agent 接管。')
        return '指令已交给 SAPIEN 原生控制接口。'

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene', 'reset'):
            model = payload.get('model', self.model_id); scene = payload.get('scene_id', self.scene_id)
            task = payload.get('task_id') or scene
            self._validate(model, scene, task)
            old = self.scene_id
            try:
                self.model_id, self.scene_id, self.task_id, self.seed = model, scene, task, payload.get('seed', self.seed)
                if old != scene: self._load()
                else: self.reset()
            except Exception as exc:
                cleanup = ''
                try: self.close()
                except Exception as cleanup_exc: cleanup = f'；资源清理失败：{cleanup_exc}'
                raise RuntimeError(f'ManiSkill 场景重建失败，仿真进程未就绪：{type(exc).__name__}: {exc}{cleanup}') from exc
            self.generation += 1
        elif action == 'pause': self.paused = bool(payload['value'])
        elif action == 'render_suspend': self.render_suspended = bool(payload['value']); self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停，再执行单步。')
            self.step()
        elif action == 'control':
            index, value = payload['index'], payload['value']
            if not 0 <= index < 8 or not math.isfinite(value): raise ValueError('执行器编号或控制数值无效。')
            item = self.actuators[index]
            if not item['min'] <= value <= item['max']: raise ValueError('控制目标超出关节范围。')
            self.cancel_motion(); self.ctrl[index] = value
        elif action == 'home': self.start_motion([{'name': '回到初始关节姿态', 'ctrl': self.home.copy(), 'duration': 2.5}], '机械臂归位')
        elif action == 'stop': self.cancel_motion(); self.paused = True; self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits': self.evaluator = ManiSkillEvaluator(self); self.run_deadline = self.time+payload['duration_s']; self.stop_on_success = payload['evaluator_stop']
        elif action == 'select_robot':
            if payload['robot_id'] != 'arm1': raise ValueError('ManiSkill 当前仅支持 arm1。')
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
            for _ in range(round(1./self.timestep)): self.step()
            self.paused = True; self.evaluator = ManiSkillEvaluator(self)
        elif action == 'view': self.set_view(payload['view'])
        elif action == 'camera':
            self.view = 'custom'; self.azimuth += payload.get('dx', 0)
            self.elevation = float(np.clip(self.elevation-payload.get('dy', 0), 3, 89))
            self.distance = float(np.clip(self.distance*math.exp(payload.get('zoom', 0)), .4, 5))
            az, el = math.radians(self.azimuth), math.radians(self.elevation)
            right = np.array([-math.sin(az), math.cos(az), 0.])
            up = np.array([-math.sin(el)*math.cos(az), -math.sin(el)*math.sin(az), math.cos(el)])
            self.lookat += self.distance*(-payload.get('pan_dx', 0)*right+payload.get('pan_dy', 0)*up)
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        else: raise ValueError(f'ManiSkill 未实现此操作：{action}')
        return '操作已完成。'

    def close(self):
        environment, self.env = self.env, None
        self.cameras = {}; self.kinematics = None; self.robot = None
        if environment is not None: environment.close()
        if self._sdk_directory is not None:
            folder = self._sdk_directory.name
            if folder in sys.path: sys.path.remove(folder)
            self._sdk_directory.cleanup(); self._sdk_directory = None
