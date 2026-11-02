"""Interactive RoboDojo adapter over the pinned official Isaac Sim EvalEnv.

This is a host-controlled research session, not an official leaderboard run.
Images, actions, layouts and terminal rewards come from the native environment.
"""
from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
import hashlib
import io
import json
import math
import re

import numpy as np
from PIL import Image

from ..actions import PoseAction
from ..robodojo_catalog import catalog, REVISION, VERSION, UPSTREAM
from ..simulators import backend_identity
from .robodojo_native import NativeSession, seed_parts


TASKS = {t['id']: {**t, 'instruction': '任务指令由官方场景布局和描述管理器提供。',
                  'criterion': '官方任务 reward_manager 与 is_episode_end 判定；模型的完成声明不作为成功证据。'}
         for t in catalog()['tasks']}
SCENES = {key: {'name': item['name'], 'description': item['dimension'] + ' · ' + item['criterion'],
                'tasks': [key]} for key, item in TASKS.items()}
MODELS = {'arx_x5': {'name': '双 ARX X5', 'arm_dof': 6, 'hand_dof': 1, 'robot_count': 2,
                     'folder': 'env_cfg/robot', 'file': 'dual_x5.yml', 'source': UPSTREAM}}
CAMERAS = {'head': 'cam_head', 'left_wrist': 'cam_left_wrist', 'right_wrist': 'cam_right_wrist'}
CAPABILITIES = ['load', 'scene', 'reset', 'pause', 'step', 'control', 'home', 'stop', 'instruction',
                'observe', 'move_tcp', 'policy_chunk', 'checkpoint', 'settle', 'view', 'speed',
                'run_limits', 'verify_task', 'render_suspend', 'select_robot']
VIEWS = [{'id': 'head', 'label': '头部相机'}, {'id': 'left_wrist', 'label': '左腕相机'},
         {'id': 'right_wrist', 'label': '右腕相机'}]


def pose_dict(pose):
    return {'frame': 'world', 'position_m': pose[:3], 'quaternion_wxyz': pose[3:]}


class RoboDojoSimulation:
    def __init__(self, model_id='arx_x5', render=True, scene_id='stack_bowls', seed=0, task_id=None):
        self._validate(model_id, scene_id, task_id or scene_id, seed)
        self.session = None
        self.model_id, self.scene_id, self.task_id, self.seed = model_id, scene_id, task_id or scene_id, seed
        self.backend = {**backend_identity('robodojo', VERSION), 'upstream_revision': REVISION,
                        'evaluation_track': 'interactive_research', 'observation_cameras': list(CAMERAS)}
        self.generation, self.speed, self.active_robot = 1, 1., 'arm1'
        self.events, self.event_id = deque(maxlen=30), 0
        self.render_enabled, self.render_suspended = render, False
        try:
            self.session = NativeSession(self.task_id, seed)
            self._reset_state()
        except BaseException as exc:
            try: self.close()
            except Exception as cleanup: exc.add_note(f'RoboDojo 初始化清理失败：{cleanup}')
            raise
        self.event('已加载官方 RoboDojo 场景、双 X5、三路原生相机及任务判定。此会话为交互研究模式。')

    @staticmethod
    def _validate(model, scene, task, seed):
        if model != 'arx_x5' or scene not in TASKS or task != scene:
            raise ValueError('RoboDojo 支持双 ARX X5；场景与任务必须选择同一个官方任务 ID。')
        _, layout = seed_parts(seed)
        if layout >= TASKS[task]['native_episodes']:
            raise ValueError(f'此任务的布局种子范围为 0–{3 * TASKS[task]["native_episodes"] - 1}；'
                             'seed % 3 选择官方评测种子，seed // 3 选择布局，不循环复用初态。')

    def _reset_state(self):
        self.timestep = self.session.timestep
        self.time, self.physics_steps = 0., 0
        self.pending, self.pose_goal = deque(), None
        self.total_motion_steps = 0
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.view, self.playback = 'head', None
        self.held_action = deepcopy(self.session.home_action)
        self.task = {'status': 'idle', 'title': '等待控制指令', 'steps': [], 'step': -1, 'progress': 0.}
        self._evaluation = self._evaluate()

    def event(self, message, kind='info'):
        self.event_id += 1
        self.events.append({'id': self.event_id, 'time': self.time, 'kind': kind, 'message': message})

    def _evaluate(self):
        native = self.session.evaluate()
        terminal, success = bool(native['terminal']), bool(native['success'])
        return {'task_id': self.task_id, 'source': 'robodojo_official_state', 'version': 'robodojo-task-v1',
                'name': TASKS[self.task_id]['name'], 'criterion': TASKS[self.task_id]['criterion'],
                'instruction': native.get('instruction', ''),
                'success': success, 'ever_success': success, 'terminated': terminal,
                'currently_satisfied': success, 'status': 'success' if success else 'failed' if terminal else 'unsatisfied',
                'progress': 1. if success else 0., 'score': native['score'],
                'evaluated_at_sim_s': self.time, 'values': native,
                'initial_state': self.session.initial_state.copy(),
                'checks': [{'key': 'official_terminal_success', 'label': '官方终局成功判定',
                            'actual': success, 'expected': True, 'passed': success}],
                'failure_reasons': [] if success else ['官方布局不稳定，不能计入有效评测'] if native['unstable']
                                   else ['官方任务已终止且未成功'] if terminal else ['官方任务尚未满足成功判据']}

    def robot_states(self):
        robots = []
        for robot_index, r in enumerate(self.session.robot_states()):
            actuators = []
            for i, name in enumerate(r['joint_names']):
                actuators.append({'index': robot_index * 7 + i, 'name': r['side'] + '/' + name,
                    'native_name': name, 'robot_id': r['id'], 'is_gripper': False, 'unit': 'rad',
                    'min': r['limits'][i][0], 'max': r['limits'][i][1],
                    'position': r['qpos'][i], 'actual': r['qpos'][i],
                    'target': self.held_action[r['side'] + '_arm_joint_state'][i]})
            actuators.append({'index': robot_index * 7 + 6, 'name': r['side'] + '/gripper',
                'robot_id': r['id'], 'is_gripper': True, 'min': 0., 'max': 1., 'unit': 'open_fraction',
                'position': r['gripper_open'], 'actual': r['gripper_open'], 'target': r['gripper_target']})
            robots.append({'id': r['id'], 'name': ('左' if r['side'] == 'left' else '右') + ' ARX X5',
                'model': 'arx_x5', 'active': r['id'] == self.active_robot, 'arm_dof': 6, 'hand_dof': 1,
                'tcp': pose_dict(r['pose']), 'gripper': {'position': r['gripper_open'], 'unit': 'open_fraction',
                    'target_open_fraction': r['gripper_target']},
                'joints': [{'name': a['name'], 'position_rad': r['qpos'][i], 'velocity_rad_s': r['qvel'][i],
                            'limits_rad': r['limits'][i]} for i, a in enumerate(actuators[:6])], 'actuators': actuators})
        return robots

    def state(self):
        robots = self.robot_states(); active = next(r for r in robots if r['active'])
        return {'ready': True, 'engine': 'RoboDojo / Isaac Sim', 'version': VERSION, 'backend': self.backend.copy(),
                'model': self.model_id, 'name': '双 ARX X5', 'active_robot': self.active_robot, 'robots': robots,
                'generation': self.generation, 'paused': self.paused, 'speed': self.speed, 'time': self.time,
                'steps': self.physics_steps, 'timestep': self.timestep,
                'contacts': None, 'contact_scope': 'not_reported', 'gravity': True,
                'view': self.view, 'view_presets': deepcopy(VIEWS), 'observation_cameras': list(CAMERAS),
                'flags': {}, 'arm_dof': 6, 'hand_dof': 1, 'nu': 14,
                'actuators': [a for r in robots for a in r['actuators']],
                'model_note': '官方双 X5、CuRobo IK 与控制插值。夹爪为 0–1 开度；原生相机视角，不支持自由相机。',
                'scene_id': self.scene_id, 'scene_name': TASKS[self.scene_id]['name'], 'seed': self.seed,
                'evaluation': deepcopy(self._evaluation), 'playback': None, 'tcp': active['tcp']['position_m'],
                'tcp_pose': active['tcp'], 'task': deepcopy(self.task), 'events': list(self.events),
                'has_wrist_camera': True, 'source': UPSTREAM, 'compute_backend': 'isaacsim_physx', 'render_backend': 'isaacsim_rtx'}

    def _robot(self, robot_id):
        selected = robot_id or self.active_robot
        if selected not in ('arm1', 'arm2'): raise ValueError('请选择 arm1（左臂）或 arm2（右臂）。')
        return next(r for r in self.session.robot_states() if r['id'] == selected)

    def _ensure_action(self):
        if self.pending: raise ValueError('当前动作尚未完成，请先等待或停止。')
        if self._evaluation['terminated']: raise ValueError('官方任务已终止，请先重置后再执行动作。')

    def _schedule(self, actions, title, goal=None):
        self._ensure_action()
        self.pending = deque(actions)
        self.total_motion_steps, self.pose_goal = len(actions), goal
        self.task = {'status': 'running', 'title': title, 'steps': [title], 'step': 0, 'progress': 0.}
        self.paused = False

    def move_tcp(self, payload):
        if payload.get('generation') != self.generation: raise ValueError('场景已改变，请重新观察。')
        action = PoseAction.model_validate(payload['pose'])
        if action.status != 'move': raise ValueError('move_tcp 仅接受 move；完成声明请交给 verify_task。')
        if action.hand_joint_positions_rad is not None: raise ValueError('双 X5 使用夹爪开度，不支持灵巧手关节动作。')
        selected = self._robot(action.robot_id)
        if np.linalg.norm(np.asarray(action.position_m) - selected['pose'][:3]) > .12 + 1e-8:
            raise ValueError('单轮末端平移超过 0.12 m，请分段执行。')
        orientation_delta = 2 * math.acos(min(1., abs(float(np.dot(action.quaternion_wxyz, selected['pose'][3:])))))
        if orientation_delta > math.radians(35) + 1e-8:
            raise ValueError('单轮末端姿态变化超过 35°，请分段执行。')
        target = action.position_m + action.quaternion_wxyz
        count = math.ceil(action.duration_s / self.timestep - 1e-8)
        start = np.asarray(selected['pose']); end = np.asarray(target)
        # Interpolate over the requested duration, then let the native CuRobo
        # IK and controller handle each 25 Hz target. Other arms hold joints.
        if np.dot(start[3:], end[3:]) < 0: end[3:] *= -1
        cosine = float(np.clip(np.dot(start[3:], end[3:]), -1, 1)); angle = math.acos(cosine)
        actions = []
        for i in range(count):
            fraction = (i + 1) / count
            pose = start + fraction * (end - start)
            if cosine < .9995:
                pose[3:] = (math.sin((1 - fraction) * angle) * start[3:] + math.sin(fraction * angle) * end[3:]) / math.sin(angle)
            else: pose[3:] /= np.linalg.norm(pose[3:])
            command = deepcopy(self.held_action)
            command.pop(selected['side'] + '_arm_joint_state')
            command[selected['side'] + '_ee_pose'] = pose.tolist()
            command[selected['side'] + '_ee_joint_state'] = [action.gripper_open]
            actions.append(command)
        self._schedule(actions, action.explanation, {'robot_id': selected['id'], 'pose': target})
        return {'robot_id': selected['id'], 'duration_s': count * self.timestep, 'native_control_steps': count}

    def policy_chunk(self, payload):
        if payload.get('generation') != self.generation: raise ValueError('场景已改变，请重新观察。')
        selected = self._robot(payload.get('robot_id')); side = selected['side']
        robot = next(r for r in self.robot_states() if r['id'] == selected['id'])
        if payload.get('joint_names') != [a['name'] for a in robot['actuators']]:
            raise ValueError('策略关节名称须与所选机器人 observe.actuators 完全一致（6 个弧度关节与 1 个归一化夹爪）。')
        dt = float(payload.get('dt_s', self.timestep))
        if not math.isfinite(dt) or dt < self.timestep - 1e-8 or dt > .5 or abs(dt / self.timestep - round(dt / self.timestep)) > 1e-6:
            raise ValueError(f'RoboDojo 动作 dt_s 必须为原生周期 {self.timestep:g} 秒的正整数倍，且不超过 0.5 秒。')
        rows = np.asarray(payload['actions'], dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 7 or not 1 <= len(rows) <= 50 or not np.isfinite(rows).all():
            raise ValueError('策略动作块须为 1–50 帧，每帧 6 个关节弧度与 1 个夹爪开度。')
        limits = np.asarray([[a['min'], a['max']] for a in robot['actuators']])
        if np.any(rows < limits[:, 0]) or np.any(rows > limits[:, 1]): raise ValueError('策略动作超出真实关节范围或夹爪 0–1 范围。')
        actions = []
        for row in rows:
            action = deepcopy(self.held_action)
            action[side + '_arm_joint_state'] = row[:6].tolist()
            action[side + '_ee_joint_state'] = [float(row[6])]
            actions.extend([action] * round(dt / self.timestep))
        self._schedule(actions, '策略模型动作块')
        return {'robot_id': selected['id'], 'frames': len(rows), 'dt_s': dt, 'limited_targets': 0, 'executed_targets': rows.tolist()}

    def step(self):
        if self._evaluation['terminated']:
            self.paused = True
            return
        target = self.pending[0] if self.pending else self.held_action
        try:
            self.session.advance(target)
        except ValueError as exc:
            self.cancel_motion(); self.paused = True
            self.task.update(status='failed', title=str(exc))
            self.event(str(exc), 'error')
            return
        self.time = self.session.elapsed
        self.physics_steps = round(self.time / self.timestep)
        self.held_action = self.session.hold_action() if any(k.endswith('_ee_pose') for k in target) else deepcopy(target)
        if self.pending:
            self.pending.popleft()
            self.task['progress'] = 1. - len(self.pending) / self.total_motion_steps
            if not self.pending:
                goal = self.pose_goal
                self.task.update(status='complete', progress=1.)
                if goal:
                    actual = self._robot(goal['robot_id'])['pose']
                    position_error = float(np.linalg.norm(np.asarray(actual[:3]) - goal['pose'][:3]))
                    dot = min(1., abs(float(np.dot(actual[3:], goal['pose'][3:]))))
                    orientation_error = 2 * math.acos(dot)
                    self.task['motion_result'] = {'position_error_m': position_error, 'orientation_error_rad': orientation_error}
                    if position_error > .02 or orientation_error > .15:
                        self.task.update(status='failed', title='末端未到达目标；请根据最新观察调整动作。')
                self.pose_goal = None
                self.paused = True
        self._evaluation = self._evaluate()
        if self._evaluation['terminated']:
            self.pending.clear(); self.pose_goal = None; self.paused = True
            self.task.update(status='complete' if self._evaluation['success'] else 'failed',
                             title='官方任务成功' if self._evaluation['success'] else self._evaluation['failure_reasons'][0])
        elif self.run_deadline is not None and self.time + 1e-8 >= self.run_deadline:
            self.cancel_motion(); self.paused = True
            self.task.update(status='failed', title='已达到本次实验仿真时间预算；官方任务未完成。')

    def cancel_motion(self):
        self.pending.clear(); self.pose_goal = None
        if self.task['status'] in ('running', 'settling'): self.task['status'] = 'cancelled'

    def checkpoint(self):
        result = {'backend': self.backend.copy(), 'model': self.model_id, 'scene_id': self.scene_id,
                  'task_id': self.task_id, 'seed': self.seed, 'generation': self.generation,
                  'time': self.time, 'steps': self.physics_steps, 'active_robot': self.active_robot,
                  'initial_state': self.session.initial_state.copy(), 'restorable': False,
                  'snapshot_kind': 'observation_metadata', 'robots': self.robot_states()}
        result['state_id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
        return result

    def trajectory_frame(self):
        native = self.session.robot_states()
        return {'time': self.time, 'backend': self.backend.copy(), 'active_robot': self.active_robot,
                'checkpoint': self.checkpoint(), 'qpos': [v for r in native for v in r['qpos'] + [r['gripper_open']]],
                'qvel': [v for r in native for v in r['qvel'] + [None]],
                'ctrl': [a['target'] for r in self.robot_states() for a in r['actuators']],
                'contacts': None, 'contact_scope': 'not_reported', 'evaluation': deepcopy(self._evaluation),
                'replay_supported': False}

    def replay_frame(self, frame):
        raise ValueError('RoboDojo 轨迹可导出分析与视频回放；不支持原生物理状态回放。')

    def verify_task(self, generation, on_step=None):
        if generation != self.generation: raise ValueError('场景已切换，完成声明未被验收。')
        if self.pending or not self.paused: raise ValueError('请在动作完成并暂停后验收任务。')
        self._evaluation = self._evaluate()
        return {**deepcopy(self._evaluation), 'verification_sim_time_s': 0.}

    @staticmethod
    def _jpeg(rgb):
        buffer = io.BytesIO(); Image.fromarray(rgb).save(buffer, 'JPEG', quality=88)
        return buffer.getvalue()

    def _images(self):
        if not self.render_enabled: raise ValueError('RoboDojo 原生相机未开启。')
        observation = self.session.observe()
        images = {}
        for name, native_name in CAMERAS.items():
            color = observation.get('vision', {}).get(native_name, {}).get('color')
            if color is None: raise RuntimeError(f'官方相机 {native_name} 未返回 RGB；请检查 Isaac 相机与任务配置。')
            rgb = np.asarray(color)
            if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or rgb.dtype != np.uint8:
                raise RuntimeError(f'官方相机 {native_name} 的 RGB 格式不符合 uint8 H×W×3/4。')
            images[name] = np.ascontiguousarray(rgb[..., :3])
        return observation, images

    def observe(self):
        native, images = self._images()
        robots = self.robot_states(); active = next(r for r in robots if r['active'])
        cameras = [{'name': name, 'native_name': CAMERAS[name], 'image_base64': base64.b64encode(self._jpeg(rgb)).decode(),
                    'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]), 'time': self.time} for name, rgb in images.items()]
        return {'backend': {'engine_id': 'robodojo', 'engine_version': VERSION, 'control_frame': 'world_z_up_m_wxyz'},
                'generation': self.generation, 'time': self.time, 'robot': self.model_id, 'active_robot': self.active_robot,
                'robots': robots, 'arm_dof': 6, 'tcp': active['tcp'], 'joints': active['joints'], 'gripper': active['gripper'],
                'limits': {'translation_per_round_m': .12, 'rotation_per_round_deg': 35.,
                           'position_tolerance_m': .02, 'orientation_tolerance_rad': .15, 'control_timestep_s': self.timestep},
                'actuators': active['actuators'], 'hand_joints': [], 'cameras': cameras,
                'task_instruction': native.get('instruction', ''), 'observation_cameras': list(CAMERAS)}

    def render_rgb(self): return self._images()[1][self.view]
    def render(self): return self._jpeg(self.render_rgb())

    def instruction(self, text):
        clean = re.sub(r'[\s，。！!]', '', text).removeprefix('请')
        if clean in ('归位', '机械臂归位', '回到初始姿态'): self.command({'action': 'home'})
        elif clean in ('张开夹爪', '打开夹爪', '闭合夹爪', '关闭夹爪'):
            selected = self._robot(None); index = (0 if selected['id'] == 'arm1' else 7) + 6
            self.command({'action': 'control', 'index': index, 'value': 1. if clean.startswith(('张开', '打开')) else 0.})
        elif clean in ('暂停', '暂停仿真'): self.paused = True
        elif clean in ('继续', '继续仿真', '开始仿真'): self.command({'action': 'pause', 'value': False})
        else: raise ValueError('快捷命令支持归位、开闭夹爪、暂停/继续；完整视觉任务请使用 Agent 或策略。')
        return '指令已交给 RoboDojo 原生控制接口。'

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene', 'reset'):
            model, scene, seed = payload.get('model', self.model_id), payload.get('scene_id', self.scene_id), payload.get('seed', self.seed)
            task = payload.get('task_id') or scene
            self._validate(model, scene, task, seed)
            try:
                self.session.reset(task, seed)
                self.model_id, self.scene_id, self.task_id, self.seed = model, scene, task, seed
                self._reset_state(); self.generation += 1
            except Exception as exc:
                try: self.close()
                except Exception as cleanup: exc.add_note(f'资源清理失败：{cleanup}')
                raise RuntimeError(f'RoboDojo 场景重建失败，原生进程未就绪：{type(exc).__name__}: {exc}') from exc
        elif action == 'pause':
            if not payload['value'] and self._evaluation['terminated']: raise ValueError('官方任务已终止，请重置。')
            self.paused = bool(payload['value'])
        elif action == 'render_suspend': self.render_suspended = bool(payload['value']); self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停，再执行单步。')
            self.step()
        elif action == 'control':
            self._ensure_action()
            index, value = payload['index'], payload['value']
            actuators = [a for r in self.robot_states() for a in r['actuators']]
            if type(index) is not int or not 0 <= index < 14 or not math.isfinite(value): raise ValueError('执行器编号或数值无效。')
            a = actuators[index]
            if not a['min'] <= value <= a['max']: raise ValueError('控制目标超出关节范围。')
            side, local_index = ('left' if index < 7 else 'right'), index % 7
            self.held_action[side + ('_ee_joint_state' if local_index == 6 else '_arm_joint_state')][0 if local_index == 6 else local_index] = value
        elif action == 'home': self._schedule([deepcopy(self.session.home_action)] * math.ceil(2.5 / self.timestep), '双臂回到初始关节姿态')
        elif action == 'stop': self.cancel_motion(); self.paused = True; self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits':
            duration = float(payload['duration_s'])
            if not math.isfinite(duration) or duration <= 0: raise ValueError('实验时长须为有限正数。')
            self.run_deadline = self.time + duration; self.stop_on_success = bool(payload['evaluator_stop'])
        elif action == 'select_robot':
            selected = self._robot(payload['robot_id']); self.active_robot = selected['id']
            return {'active_robot': self.active_robot, 'generation': self.generation}
        elif action == 'instruction': return self.instruction(payload['text'])
        elif action == 'observe': return self.observe()
        elif action == 'move_tcp': return self.move_tcp(payload)
        elif action == 'policy_chunk': return self.policy_chunk(payload)
        elif action == 'checkpoint': return self.checkpoint()
        elif action in ('restore', 'replay_frame'): raise ValueError('RoboDojo 检查点仅记录观察元数据；原生任务内部状态尚不支持精确恢复或物理状态回放。')
        elif action == 'verify_task': return self.verify_task(payload['generation'])
        elif action == 'settle':
            self.cancel_motion()
            for _ in range(math.ceil(1. / self.timestep)):
                if self._evaluation['terminated']: break
                self.step()
            self.paused = True
        elif action == 'view':
            if payload['view'] not in CAMERAS: raise ValueError('请选择 head、left_wrist 或 right_wrist 原生相机。')
            self.view = payload['view']
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        else: raise ValueError(f'RoboDojo 未实现此操作：{action}。原生相机不支持自由视角拖拽。')
        return '操作已完成。'

    def close(self):
        session, self.session = self.session, None
        if session is not None: session.close()
