"""Native BEHAVIOR v3.9.4 session. Imported SDKs live only in the worker process.

Reference APIs are pinned to SOURCE_COMMIT in omnigibson_catalog. In particular,
this release returns lists / length-one tensors even for num_envs=1. Do not
silently reinterpret a different upstream version's observation/action contract.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .omnigibson_catalog import SOURCE_COMMIT, SOURCE_REF, resolve_selection

MANUAL_EPISODE_MAX_STEPS = 108000


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def single(value, label):
    if len(value) != 1:
        raise RuntimeError(f'OmniGibson {label} 必须来自单个原生环境。')
    return value[0]


def camera_alias(name):
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', name)
    # Hash the native name, not an index that changes when cameras are added.
    return safe[:46] + '_' + hashlib.sha256(name.encode()).hexdigest()[:10]


def json_value(value):
    if isinstance(value, dict): return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [json_value(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, 'detach') or isinstance(value, np.ndarray): return array(value).tolist()
    return value


class NativeSession:
    def __init__(self, model_id, scene_id, task_id, seed, render=True):
        from ..omnigibson import runtime_config
        config = runtime_config() or {}
        self.selection = resolve_selection(model_id, scene_id, task_id, seed, config.get('data_path'))
        data_path = config.get('data_path')
        if not data_path:
            raise ValueError('OmniGibson 数据目录未配置，请按 docs/omnigibson.md 配置独立运行环境。')
        os.environ['OMNIGIBSON_DATA_PATH'] = str(Path(data_path).resolve())
        os.environ['OMNIGIBSON_HEADLESS'] = 'True'
        # Importing the SDK cannot download assets or accept a license here.
        import torch
        import yaml
        import omnigibson as og
        from omnigibson.macros import gm
        from omnigibson.metrics.task_metric import compute_q_score
        from omnigibson.controllers.controller_view import ControllerView
        self.og, self.torch, self.ControllerView, self.compute_q_score = og, torch, ControllerView, compute_q_score
        self.env, self.closed, self.render_enabled = None, False, render
        with gm.unlocked():
            gm.HEADLESS = True
            gm.ENABLE_OBJECT_STATES = True
            gm.USE_GPU_DYNAMICS = True
            gm.ENABLE_HQ_RENDERING = False
        try:
            template = Path(og.example_config_path) / 'r1pro_behavior.yaml'
            cfg = yaml.safe_load(template.read_text(encoding='utf-8'))
            cfg['env'].update(num_envs=1, automatic_reset=False, flatten_action_space=True,
                flatten_obs_space=False, external_sensors=None, action_frequency=30,
                rendering_frequency=30, physics_frequency=120, num_rerenders_on_reset=3)
            cfg['render'].update(viewer_width=640, viewer_height=480)
            cfg['scene'].update(scene_model=scene_id, load_room_types=None, load_room_instances=None, include_robots=False)
            cfg['robots'] = [cfg['robots'][0]]
            cfg['robots'][0].pop('type', None)
            cfg['robots'][0].pop('self_collision', None)
            cfg['robots'][0].update(model='r1pro', name='robot0', self_collisions=False, obs_modalities=['rgb', 'depth', 'proprio'],
                grasping_mode='physical', action_normalize=True)
            cfg['robots'][0]['sensor_config']['VisionSensor']['sensor_kwargs'].update(image_width=320, image_height=240)
            cfg['task'] = {key: value for key, value in cfg['task'].items()
                           if key not in ('predefined_problem', 'debug_object_sampling')}
            cfg['task'].update(type='BehaviorTask', activity_name=self.selection['activity_name'],
                activity_definition_id=self.selection['activity_definition_id'],
                activity_instance_id=self.selection['activity_instance_id'], online_object_sampling=False,
                use_presampled_robot_pose=True, randomize_presampled_pose=False,
                highlight_task_relevant_objects=False, include_obs=False,
                termination_config={'max_steps': MANUAL_EPISODE_MAX_STEPS})
            torch.manual_seed(seed)
            self.env = og.Environment(configs=cfg)
            if self.env.num_envs != 1:
                raise RuntimeError('工作台交互会话必须使用 num_envs=1。')
            self.robot = self.env.scene.robots[0]
            if len(self.env.scene.robots) != 1 or self.robot.model != 'r1pro' or self.robot.grasping_mode != 'physical':
                raise RuntimeError('场景必须加载一台 R1Pro 并使用 physical 抓取，不允许静默采用其他机器人或辅助抓取。')
            self.timestep = 1. / cfg['env']['action_frequency']
            self.robot_config = cfg['robots'][0]
            self.action_spec = self._action_spec()
            self.reset()
        except BaseException as exc:
            try: self.close()
            except Exception as cleanup: exc.add_note(f'OmniGibson 资源清理失败：{cleanup}')
            raise

    def _action_spec(self):
        low, high = array(self.robot.action_space.low), array(self.robot.action_space.high)
        if low.ndim != 1 or low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all():
            raise RuntimeError('原生动作空间必须是有界有限的一维连续空间。')
        names = [''] * len(low)
        groups = []
        for group, indices in self.robot.controller_action_idx.items():
            idx = array(indices).astype(int).tolist()
            key, _ = self.robot.controllers[group]
            controller = self.ControllerView.get_controller_type_str(key)
            settings = self.robot_config['controller_config'].get(group, {})
            for offset, i in enumerate(idx):
                names[i] = f'{group}_{offset}'
            groups.append({'name': group, 'indices': idx, 'controller': controller,
                'command_semantics': 'normalized native controller input',
                'configuration': deepcopy(settings)})
        if not all(names) or len(set(names)) != len(names):
            raise RuntimeError('原生控制器动作索引没有完整覆盖动作空间。')
        return {'type': 'native', 'robot_id': 'robot0', 'names': names, 'low': low.tolist(), 'high': high.tolist(),
                'dimension': len(names), 'dt_s': self.timestep, 'unit': 'normalized_controller_command',
                'control_groups': groups, 'grasping_mode': 'physical', 'max_repeat_steps': 120}

    def reset(self):
        # A new manual episode must not inherit the preceding experiment's
        # short budget. A subsequent run_limits configures its own budget.
        self.env.task._termination_conditions['timeout']._max_steps = MANUAL_EPISODE_MAX_STEPS
        result, _ = self.env.reset()
        self.latest = single(result, 'reset observation')
        self.elapsed, self.steps, self.reward = 0., 0, 0.
        self.terminated = self.truncated = False
        self.last_info = {}
        self.initial_options = deepcopy(self.env.task.get_goal_option_satisfaction(0))
        self.home_action = self.hold_action()
        self.camera_map = {}
        for name, sensor in self.robot.sensors.items():
            if 'rgb' in sensor.modalities:
                alias = camera_alias(name)
                if alias in self.camera_map:
                    raise RuntimeError('原生相机名称重复。')
                self.camera_map[alias] = (name, sensor)
        if not self.camera_map:
            raise RuntimeError('R1Pro 没有返回原生 RGB 相机，不能开始视觉实验。')
        self.initial_state = {**self.selection, 'source_ref': SOURCE_REF, 'source_commit': SOURCE_COMMIT,
            'initial_goal_options': deepcopy(self.initial_options), 'native_version': self.og.__version__,
            'simulation_mode': 'turn_based', 'physics_frequency': 120, 'action_frequency': 30,
            'gpu_dynamics': True, 'state_restore_verified': False}
        self.native_step_limit = self.env.task._termination_conditions['timeout']._max_steps
        self._setup_viewer()

    def set_budget(self, duration_s):
        # The interactive track uses the workbench's declared episode budget.
        # Preserve the upstream Timeout implementation; only configure its
        # threshold, rather than allowing an unrelated fixed cap to cut a run.
        self.native_step_limit = self.steps + math.ceil(duration_s / self.timestep)
        self.env.task._termination_conditions['timeout']._max_steps = self.native_step_limit

    def _setup_viewer(self):
        # Start from a real robot camera, avoiding a free-camera spawn inside
        # walls. Subsequent panning moves in camera image-plane axes.
        sensor = next(iter(self.camera_map.values()))[1]
        pos, quat = sensor.get_position_orientation()
        self.og.sim.viewer_camera.set_position_orientation(position=pos, orientation=quat)

    def hold_action(self):
        return np.concatenate([array(self.ControllerView.compute_no_op_action(key, idx))
                               for key, idx in self.robot.controllers.values()]).tolist()

    def advance(self, action):
        if self.terminated or self.truncated:
            raise ValueError('原生任务已终止或截断，请重置后继续。')
        obs, reward, terminated, truncated, info = self.env.step(self.torch.tensor(action, dtype=self.torch.float32))
        self.latest = single(obs, 'step observation')
        self.reward = float(array(reward).reshape(-1)[0])
        self.terminated = bool(array(terminated).reshape(-1)[0])
        self.truncated = bool(array(truncated).reshape(-1)[0])
        self.last_info = single(info, 'step info')
        self.steps += 1
        self.elapsed = self.steps * self.timestep

    def evaluate(self):
        task = self.env.task
        success, status = task.compiled_task.check_goal(
            lambda predicate, *entities: task._evaluate_predicate(0, predicate, *entities))
        options = task.get_goal_option_satisfaction(0)
        q = float(self.compute_q_score(bool(success), options, self.initial_options))
        return {'success': bool(success), 'terminated': bool(self.terminated or success),
            'truncated': self.truncated, 'q_score': q, 'reward': self.reward,
            'goal_status': {'satisfied': list(status['satisfied']), 'unsatisfied': list(status['unsatisfied'])},
            'goal_option_satisfaction': options, 'steps': self.steps,
            'native_step_limit': self.native_step_limit,
            'termination_conditions': json_value(self.last_info.get('done', {}).get('termination_conditions', {}))}

    def instruction(self):
        goals = self.env.task.activity_natural_language_goal_conditions
        if isinstance(goals, str): return goals
        if isinstance(goals, (list, tuple)): return '\n'.join(str(item) for item in goals)
        return self.selection['activity_name'].replace('_', ' ')

    def robot_state(self):
        qpos, qvel = array(self.robot.get_joint_positions()), array(self.robot.get_joint_velocities())
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise RuntimeError('OmniGibson 机器人状态包含非有限值，请停止实验并检查原生物理日志。')
        names = self.robot.dof_names_ordered
        if len(names) != len(qpos) or len(qpos) != len(qvel):
            raise RuntimeError('原生机器人 DOF 名称与关节状态维度不一致。')
        joints = []
        for i, name in enumerate(names):
            revolute = self.robot.joints[name].is_revolute
            item = {'name': name, 'position': float(qpos[i]), 'velocity': float(qvel[i]),
                    'unit': 'rad' if revolute else 'm', 'velocity_unit': 'rad/s' if revolute else 'm/s'}
            item.update({'position_rad': float(qpos[i]), 'velocity_rad_s': float(qvel[i])} if revolute else
                        {'position_m': float(qpos[i]), 'velocity_m_s': float(qvel[i])})
            joints.append(item)
        ends = {}
        for arm in self.robot.arm_names:
            p, q = self.robot.get_eef_pose(arm)
            p, q = array(p), array(q)
            ends[arm] = {'frame': 'world', 'position_m': p.tolist(), 'quaternion_wxyz': q[[3, 0, 1, 2]].tolist()}
        proprio, _ = self.robot.get_proprioception()
        return {'id': 'robot0', 'name': 'R1Pro 移动双臂', 'model': 'r1pro', 'active': True,
            'joints': joints, 'end_effectors': ends, 'tcp': next(iter(ends.values())),
            'proprioception': array(proprio).tolist(), 'gripper': {'unit': 'native_controller_command'},
            'arm_dof': 6, 'hand_dof': 1}

    def images(self):
        observation, _ = self.env.get_obs()
        robot_obs = single(observation, 'observation').get(self.robot.name, {})
        result = {}
        for alias, (name, sensor) in self.camera_map.items():
            frame = robot_obs.get(name, {})
            rgb = array(frame.get('rgb'))
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise RuntimeError(f'原生相机 {name} 没有返回 uint8 RGB。')
            depth = array(frame.get('depth'))
            if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
                raise RuntimeError(f'原生相机 {name} 没有返回与 RGB 同尺寸的米制深度。')
            invalid = ~np.isfinite(depth) | (depth <= 0)
            depth = np.where(invalid, 0, depth).astype(np.float32)
            pos, quat = sensor.get_position_orientation()
            result[alias] = {'rgb': np.ascontiguousarray(rgb[..., :3]), 'depth': depth,
                'native_name': name, 'intrinsics': array(sensor.intrinsic_matrix).tolist(),
                'position_world_m': array(pos).tolist(), 'quaternion_xyzw': array(quat).tolist(),
                'depth_measurement': 'distance_to_camera', 'invalid_depth_value': 0.}
        return result

    def viewer_rgb(self):
        self.og.sim.render()
        raw = array(self.og.sim.viewer_camera.get_obs()[0]['rgb'])
        if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] not in (3, 4):
            raise RuntimeError('OmniGibson 自由相机没有返回有效 RGB。')
        return np.ascontiguousarray(raw[..., :3])

    def camera(self, payload):
        from scipy.spatial.transform import Rotation
        pos, quat = self.og.sim.viewer_camera.get_position_orientation()
        pos, orientation = array(pos).astype(float), Rotation.from_quat(array(quat))
        yaw = Rotation.from_rotvec([0., 0., -np.deg2rad(payload.get('dx', 0.))])
        pitch = Rotation.from_rotvec([-np.deg2rad(payload.get('dy', 0.)), 0., 0.])
        orientation = yaw * orientation * pitch
        basis = orientation.as_matrix()
        pos += -2. * payload.get('pan_dx', 0.) * basis[:, 0] + 2. * payload.get('pan_dy', 0.) * basis[:, 1]
        pos += 2. * payload.get('zoom', 0.) * basis[:, 2]
        self.og.sim.viewer_camera.set_position_orientation(
            position=self.torch.tensor(pos, dtype=self.torch.float32),
            orientation=self.torch.tensor(orientation.as_quat(), dtype=self.torch.float32))

    def close(self):
        if self.closed: return
        self.closed = True
        self.env = None
        self.og.shutdown(due_to_signal=True)
