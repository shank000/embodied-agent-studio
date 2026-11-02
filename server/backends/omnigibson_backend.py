"""BEHAVIOR interactive research adapter; real physical R1Pro control only."""
from collections import deque
from copy import deepcopy
import base64
import hashlib
import io
import json
import math

import numpy as np
from PIL import Image

from ..simulators import backend_identity
from .omnigibson_catalog import (MODELS, SCENES, TASKS, DEFAULT_SCENE, DEFAULT_TASK,
                                SOURCE_REF, SOURCE_COMMIT, UPSTREAM, CRITERION, catalog, resolve_selection)
from .omnigibson_native import NativeSession

CAPABILITIES = ['load', 'scene', 'reset', 'pause', 'step', 'stop', 'control', 'observe',
    'native_action', 'policy_chunk', 'checkpoint', 'verify_task', 'view', 'camera',
    'speed', 'run_limits', 'render_suspend', 'select_robot']


class OmniGibsonSimulation:
    def __init__(self, model_id='r1pro', render=True, scene_id=None, seed=0, task_id=None):
        _, scenes, tasks = catalog()
        if scene_id is None:
            installed = [(s, t) for s, v in scenes.items() for t in v['tasks'] if v.get('instances', {}).get(t)]
            scene_id, task_id = ((DEFAULT_SCENE, DEFAULT_TASK) if (DEFAULT_SCENE, DEFAULT_TASK) in installed
                                 else installed[0] if installed else (DEFAULT_SCENE, DEFAULT_TASK))
        task_id = task_id or scenes.get(scene_id, {}).get('tasks', [DEFAULT_TASK])[0]
        self.session = None
        self.render_enabled, self.render_suspended = render, False
        self.model_id, self.scene_id, self.task_id, self.seed = model_id, scene_id, task_id, seed
        self.generation, self.speed, self.active_robot = 1, 1., 'robot0'
        self.events, self.event_id = deque(maxlen=30), 0
        try:
            self.session = NativeSession(model_id, scene_id, task_id, seed, render=render)
            self._reset_state()
        except BaseException as exc:
            try: self.close()
            except Exception as cleanup: exc.add_note(f'OmniGibson 初始化清理失败：{cleanup}')
            raise
        self.event('已加载 BEHAVIOR 官方预采样实例；R1Pro 物理控制、BDDL 验收与 Q 分数。原生执行尚需本机验收。')

    def _reset_state(self):
        _, scenes, _ = catalog()
        self.seed_max = max(0, len(scenes.get(self.scene_id, {}).get('instances', {}).get(self.task_id, []))-1)
        self.timestep = self.session.timestep
        self.time, self.physics_steps = 0., 0
        self.pending, self.total_motion_steps = deque(), 0
        self.held_action = self.session.hold_action()
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.view = 'perspective'
        self.action_space = deepcopy(self.session.action_spec)
        self.backend = {**backend_identity('omnigibson', SOURCE_REF), 'source_commit': SOURCE_COMMIT,
            'action_mode': 'native', 'control_frame': 'native_controller_groups',
            'camera_names': list(self.session.camera_map), 'observation_cameras': list(self.session.camera_map),
            'action_space': deepcopy(self.action_space), 'evaluation_track': 'interactive_research',
            'grasping_mode': 'physical', 'inference_timing': 'turn_based', 'native_acceptance': 'pending'}
        self.task = {'status': 'idle', 'title': '等待全身原生动作', 'steps': [], 'step': -1, 'progress': 0.}
        self._evaluation = self._evaluate()

    def event(self, message, kind='info'):
        self.event_id += 1
        self.events.append({'id': self.event_id, 'time': self.time, 'kind': kind, 'message': message})

    def _evaluate(self):
        native = self.session.evaluate()
        success, terminated, truncated = native['success'], native['terminated'], native['truncated']
        return {'task_id': self.task_id, 'source': 'omnigibson_bddl_state', 'version': 'omnigibson-bddl-v1',
            'name': self.task_id.replace('_', ' '), 'instruction': self.session.instruction(), 'criterion': CRITERION,
            'success': success, 'currently_satisfied': success, 'terminated': terminated, 'truncated': truncated,
            'status': 'success' if success else 'truncated' if truncated else 'failed' if terminated else 'unsatisfied',
            'termination_reason': 'success' if success else 'native_timeout' if truncated else 'native_failure' if terminated else None,
            'progress': native['q_score'], 'score': native['q_score'], 'q_score': native['q_score'],
            'native_reward': native['reward'], 'evaluated_at_sim_s': self.time,
            'values': native, 'initial_state': deepcopy(self.session.initial_state),
            'checks': [{'key': 'bddl_goal', 'label': '全部 BDDL 目标逻辑', 'actual': success, 'expected': True, 'passed': success}],
            'failure_reasons': [] if success else ['原生任务步数预算耗尽'] if truncated else
                               ['原生环境已失败终止'] if terminated else ['官方 BDDL 目标尚未全部满足']}

    def _actuators(self):
        return [{'index': i, 'name': name, 'robot_id': 'robot0', 'unit': 'normalized_controller_command',
            'min': self.action_space['low'][i], 'max': self.action_space['high'][i],
            'position': self.held_action[i], 'target': self.held_action[i], 'is_gripper': name.startswith('gripper_')}
            for i, name in enumerate(self.action_space['names'])]

    def robot_states(self):
        r = self.session.robot_state()
        r['actuators'] = self._actuators()
        return [r]

    def state(self):
        robots = self.robot_states()
        r = robots[0]
        return {'ready': True, 'engine': 'OmniGibson / Isaac Sim', 'version': SOURCE_REF,
            'backend': deepcopy(self.backend), 'model': self.model_id, 'name': 'R1Pro 移动双臂',
            'active_robot': 'robot0', 'robots': robots, 'generation': self.generation, 'paused': self.paused,
            'speed': self.speed, 'time': self.time, 'steps': self.physics_steps, 'timestep': self.timestep,
            'contacts': None, 'contact_scope': 'not_reported', 'gravity': True, 'flags': {}, 'playback': None,
            'view': self.view, 'view_presets': [{'id': 'perspective', 'label': '自由视角'}] +
                [{'id': name, 'label': name} for name in self.backend['camera_names']],
            'observation_cameras': self.backend['camera_names'], 'arm_dof': 6, 'hand_dof': 1,
            'nu': len(self.action_space['names']), 'actuators': self._actuators(),
            'action_space': deepcopy(self.action_space), 'native_action_spec': deepcopy(self.action_space),
            'model_note': 'R1Pro 全身控制器归一化命令，包含底盘/躯干/双臂/夹爪；数值不是关节角。物理抓取，不使用符号动作。',
            'scene_id': self.scene_id, 'scene_name': self.scene_id, 'seed': self.seed,
            'seed_range': {'min': 0, 'max': self.seed_max, 'meaning': 'official_presampled_instance_index'},
            'evaluation': deepcopy(self._evaluation), 'tcp': r['tcp']['position_m'], 'tcp_pose': r['tcp'],
            'task': deepcopy(self.task), 'events': list(self.events), 'blocks': {}, 'has_wrist_camera': True,
            'source': UPSTREAM, 'compute_backend': 'isaacsim_physx', 'render_backend': 'isaacsim_rtx'}

    def _check_generation(self, payload):
        if payload.get('generation') != self.generation:
            raise ValueError('场景已切换，动作已失效，请重新观察。')

    def _ensure_action(self):
        if self.pending: raise ValueError('当前动作尚未完成，请先等待或停止。')
        if self._evaluation['terminated'] or self._evaluation['truncated']:
            raise ValueError('原生任务已终止或截断，请重置后继续。')
        if self.run_deadline is not None and self.time >= self.run_deadline - 1e-8:
            raise ValueError('已达到实验仿真时间预算，请停止当前实验。')

    def _validate_rows(self, actions):
        if not isinstance(actions, list) or any(isinstance(v, bool) for row in actions if isinstance(row, list) for v in row):
            raise ValueError('原生动作必须为有限数值向量，不能使用布尔值。')
        try: rows = np.asarray(actions, dtype=float)
        except (ValueError, TypeError) as exc: raise ValueError('原生动作向量格式无效。') from exc
        if rows.ndim != 2 or rows.shape[1] != len(self.action_space['names']) or not 1 <= len(rows) <= 120 or not np.isfinite(rows).all():
            raise ValueError('原生动作维度必须与 action_space.names 一致，且每个值有限。')
        if np.any(rows < np.asarray(self.action_space['low'])) or np.any(rows > np.asarray(self.action_space['high'])):
            raise ValueError('原生命令超出 SDK 动作空间范围；不会裁剪或转换成关节角。')
        return rows.tolist()

    def _schedule(self, actions, title):
        self._ensure_action()
        self.pending = deque(actions)
        self.total_motion_steps = len(actions)
        self.task = {'status': 'running', 'title': title, 'steps': [title], 'step': 0, 'progress': 0.}
        self.paused = False

    def native_action(self, payload):
        self._check_generation(payload)
        value = payload.get('native_action') or {}
        if value.get('status') != 'move': raise ValueError('native_action 仅执行 move；done 由独立任务验收处理。')
        steps = value.get('steps')
        if type(steps) is not int or not 1 <= steps <= 120: raise ValueError('单轮原生动作须重复 1–120 步。')
        row = self._validate_rows([value.get('action')])[0]
        self._schedule([row] * steps, value.get('explanation', 'R1Pro 原生全身动作'))
        return {'native_steps': steps, 'duration_s': steps*self.timestep, 'control_type': 'native', 'robot_id': 'robot0'}

    def policy_chunk(self, payload):
        self._check_generation(payload)
        if payload.get('action_space') != 'native' or payload.get('action_names') != self.action_space['names']:
            raise ValueError('OmniGibson 策略须声明 action_space=native，action_names 与观察完全一致。')
        if payload.get('joint_names') or payload.get('robot_id') not in (None, 'robot0'):
            raise ValueError('原生全身策略不接受关节角标签或其他机器人 ID。')
        rows = self._validate_rows(payload.get('actions'))
        if len(rows) > 50: raise ValueError('策略动作块最多 50 帧。')
        dt = payload.get('dt_s', self.timestep)
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or not self.timestep-1e-8 <= dt <= .5:
            raise ValueError('动作间隔须为一个原生周期至 0.5 秒。')
        repeats = round(dt/self.timestep)
        if abs(dt/self.timestep-repeats) > 1e-6: raise ValueError('动作间隔必须为原生周期的整数倍。')
        self._schedule([row for row in rows for _ in range(repeats)], '原生全身策略动作块')
        return {'frames': len(rows), 'dt_s': dt, 'executed_targets': rows, 'limited_targets': 0,
                'control_type': 'native', 'robot_id': 'robot0'}

    def step(self):
        if self._evaluation['terminated'] or self._evaluation['truncated']:
            self.paused = True
            return
        action = self.pending[0] if self.pending else self.session.hold_action()
        self.session.advance(action)
        self.held_action = list(action)
        self.time, self.physics_steps = self.session.elapsed, self.session.steps
        if self.pending:
            self.pending.popleft()
            self.task['progress'] = 1-len(self.pending)/self.total_motion_steps
            if not self.pending:
                self.task.update(status='complete', progress=1.)
                self.paused = True
                self.held_action = self.session.hold_action()
        self._evaluation = self._evaluate()
        if self._evaluation['terminated'] or self._evaluation['truncated']:
            self.pending.clear(); self.paused = True
            self.task.update(status='verified' if self._evaluation['success'] else 'limit' if self._evaluation['truncated'] else 'failed',
                             title='官方 BDDL 任务成功' if self._evaluation['success'] else self._evaluation['failure_reasons'][0])
        elif self.run_deadline is not None and self.time >= self.run_deadline-1e-8:
            self.pending.clear(); self.paused = True
            self.task.update(status='limit', title='实验仿真时间预算耗尽')

    def cancel_motion(self):
        self.pending.clear()
        if self.task['status'] == 'running': self.task['status'] = 'cancelled'
        self.held_action = self.session.hold_action()

    def verify_task(self, generation, on_step=None):
        self._check_generation({'generation': generation})
        if self.pending or not self.paused: raise ValueError('请等待当前动作完成并暂停后再验收。')
        self._evaluation = self._evaluate()
        return {**deepcopy(self._evaluation), 'verification_sim_time_s': 0.}

    @staticmethod
    def _jpeg(rgb):
        buffer = io.BytesIO(); Image.fromarray(rgb).save(buffer, 'JPEG', quality=88)
        return buffer.getvalue()

    def observe(self):
        if not self.render_enabled: raise ValueError('原生视觉传感器未启用。')
        cameras = []
        for name, image in self.session.images().items():
            rgb, depth = image['rgb'], image['depth']
            output = io.BytesIO(); np.save(output, depth, allow_pickle=False)
            cameras.append({'name': name, 'native_name': image['native_name'], 'time': self.time,
                'width': rgb.shape[1], 'height': rgb.shape[0],
                'image_base64': base64.b64encode(self._jpeg(rgb)).decode('ascii'),
                'intrinsics': image['intrinsics'], 'position_world_m': image['position_world_m'],
                'quaternion_xyzw': image['quaternion_xyzw'],
                'depth': {'encoding': 'npy-float32', 'unit': 'm', 'shape': list(depth.shape),
                    'data_base64': base64.b64encode(output.getvalue()).decode('ascii'),
                    'measurement': image['depth_measurement'], 'validity': '0 denotes invalid/no return'}})
        r = self.robot_states()[0]
        return {'backend': deepcopy(self.backend), 'generation': self.generation, 'time': self.time,
            'robot': self.model_id, 'active_robot': 'robot0', 'robots': [r], 'tcp': r['tcp'],
            'joints': r['joints'], 'gripper': r['gripper'], 'actuators': r['actuators'], 'hand_joints': [],
            'proprioception': r['proprioception'], 'end_effectors': r['end_effectors'],
            'action_space': deepcopy(self.action_space), 'native_action_spec': deepcopy(self.action_space),
            'task_instruction': self.session.instruction(), 'cameras': cameras}

    def render_rgb(self):
        if not self.render_enabled: raise ValueError('原生相机未启用。')
        return self.session.viewer_rgb() if self.view == 'perspective' else self.session.images()[self.view]['rgb']

    def render(self): return self._jpeg(self.render_rgb())

    def checkpoint(self):
        value = {'backend': deepcopy(self.backend), 'model': self.model_id, 'scene_id': self.scene_id,
            'task_id': self.task_id, 'seed': self.seed, 'generation': self.generation, 'time': self.time,
            'initial_state': deepcopy(self.session.initial_state), 'restorable': False,
            'snapshot_kind': 'native_observation_metadata', 'robots': self.robot_states()}
        value['state_id'] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        return value

    def trajectory_frame(self):
        r = self.robot_states()[0]
        return {'time': self.time, 'backend': deepcopy(self.backend), 'active_robot': 'robot0',
            'qpos': [j['position'] for j in r['joints']], 'qvel': [j['velocity'] for j in r['joints']],
            'joint_names': [j['name'] for j in r['joints']], 'joint_units': [j['unit'] for j in r['joints']],
            'ctrl': list(self.held_action), 'evaluation': deepcopy(self._evaluation), 'replay_supported': False,
            'contacts': None, 'contact_scope': 'not_reported'}

    def replay_frame(self, frame):
        raise ValueError('OmniGibson 当前保留原生视频和动作轨迹；未验收完整物理状态恢复，不能物理回放。')

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene', 'reset'):
            model = payload.get('model', self.model_id)
            scene = payload.get('scene_id', self.scene_id)
            task = payload.get('task_id', self.task_id)
            seed = payload.get('seed', self.seed)
            resolve_selection(model, scene, task, seed)
            try:
                if (model, scene, task, seed) == (self.model_id, self.scene_id, self.task_id, self.seed):
                    self.session.reset()
                else:
                    # Clear the whole native stage before loading a different task;
                    # scene-changing tasks cannot be restored from poses alone.
                    self.session.og.clear()
                    self.session.closed = True
                    self.session = NativeSession(model, scene, task, seed, render=self.render_enabled)
                self.model_id, self.scene_id, self.task_id, self.seed = model, scene, task, seed
                self._reset_state(); self.generation += 1
            except Exception as exc:
                try: self.close()
                except Exception as cleanup: exc.add_note(f'OmniGibson 重建清理失败：{cleanup}')
                raise RuntimeError(f'OmniGibson 原生场景重建失败，必须重新加载环境：{type(exc).__name__}: {exc}') from exc
        elif action == 'pause':
            if not payload['value']: self._ensure_action() if not self.pending else None
            self.paused = bool(payload['value'])
        elif action == 'render_suspend': self.render_suspended = bool(payload['value']); self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停再单步。')
            self.step()
        elif action == 'stop':
            self.cancel_motion(); self.paused = True; self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits':
            duration = payload.get('duration_s')
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('实验仿真时间预算须为有限正数。')
            self.run_deadline = self.time + duration
            self.session.set_budget(duration)
            self.stop_on_success = bool(payload['evaluator_stop'])
        elif action == 'control':
            index, value = payload.get('index'), payload.get('value')
            if type(index) is not int or not 0 <= index < len(self.held_action): raise ValueError('原生控制编号无效。')
            row = self.session.hold_action(); row[index] = value
            self._schedule(self._validate_rows([row]), '手动原生控制一步')
        elif action == 'select_robot':
            if payload['robot_id'] != 'robot0': raise ValueError('当前会话为一台 R1Pro，双臂由同一原生动作控制。')
            return {'active_robot': 'robot0', 'generation': self.generation}
        elif action == 'observe': return self.observe()
        elif action == 'native_action': return self.native_action(payload)
        elif action == 'policy_chunk': return self.policy_chunk(payload)
        elif action == 'checkpoint': return self.checkpoint()
        elif action in ('restore', 'replay_frame'): return self.replay_frame(payload)
        elif action == 'verify_task': return self.verify_task(payload['generation'])
        elif action == 'view':
            if payload['view'] not in ('perspective', *self.backend['camera_names']): raise ValueError('请选择自由视角或当前原生相机。')
            self.view = payload['view']
        elif action == 'camera':
            values = [payload.get(k, 0.) for k in ('dx', 'dy', 'zoom', 'pan_dx', 'pan_dy')]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError('相机位移须为有限数值。')
            self.session.camera(payload); self.view = 'perspective'
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        else: raise ValueError(f'OmniGibson 不支持操作：{action}；请使用原生全身动作接口。')
        return '操作已完成。'

    def close(self):
        session, self.session = self.session, None
        if session is not None: session.close()
