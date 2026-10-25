"""ALFWorld's official TextWorld/PDDL runtime, with native discrete transitions.

The JPEG is a readable rendering of the current text observation, not a 3D
camera. Actor observations never include PDDL facts, plans or success flags.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import io
from importlib.metadata import version
import json
import os
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
MAX_LAYOUT_SEED = 4294967295
CONFIGURED_DATA = os.environ.get('ALFWORLD_DATA')
TASKS = {
    'pick_and_place_simple': {'name': '取物并放置', 'instruction': '完成 ALFWorld 当前回合提供的取放任务。'},
    'look_at_obj_in_light': {'name': '灯光下检查物品', 'instruction': '完成 ALFWorld 当前回合提供的灯光检查任务。'},
    'pick_clean_then_place_in_recep': {'name': '清洁后放置', 'instruction': '完成 ALFWorld 当前回合提供的清洁取放任务。'},
    'pick_heat_then_place_in_recep': {'name': '加热后放置', 'instruction': '完成 ALFWorld 当前回合提供的加热取放任务。'},
    'pick_cool_then_place_in_recep': {'name': '冷却后放置', 'instruction': '完成 ALFWorld 当前回合提供的冷却取放任务。'},
    'pick_two_obj_and_place': {'name': '放置两件物品', 'instruction': '完成 ALFWorld 当前回合提供的双物体取放任务。'},
}
for _value in TASKS.values():
    _value['criterion'] = '官方 TextWorld/PDDL 环境 GameState.won 为 true；模型声明不作为依据。'
SCENES = {'alfworld_unseen': {'name': 'ALFWorld 未见家庭任务', 'description': '官方 valid_unseen PDDL 回合，原生文字观察与可执行命令。', 'tasks': list(TASKS)}}
MODELS = {'alfworld_agent': {'name': 'ALFWorld 文字智能体', 'arm_dof': 0, 'hand_dof': 0, 'image': None}}


class DiscreteEnvironment:
    """Workbench contract for turn-based SDKs; physics controls stay unsupported."""
    engine_id = None
    engine_name = None
    package = None
    action_mode = 'text'
    evaluator_source = None
    evaluator_version = None
    timestep = 1.
    time_unit = 'environment_step'

    def _initialize(self, model, scene, seed, task):
        self._validate(model, scene, task)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_LAYOUT_SEED:
            raise ValueError(f'布局种子须为 0–{MAX_LAYOUT_SEED} 的整数。')
        self.model_id, self.scene_id, self.seed, self.task_id = model, scene, seed, task
        self.generation, self.time, self.physics_steps = 1, 0., 0
        self.paused, self.render_suspended, self.speed = True, False, 1.
        self.view, self.flags, self.motion, self.playback = 'front', {}, None, False
        self.events, self._event_id, self.actions = deque(maxlen=30), 0, []
        self.task = self.task_status(self.tasks[task]['name'])
        self.run_deadline, self.stop_on_success = None, False
        self.backend = {'engine_id': self.engine_id, 'engine_version': version(self.package),
                        'evaluator_source': self.evaluator_source, 'evaluator_version': self.evaluator_version,
                        'state_schema': self.engine_id+'-integration-v1', 'action_mode': self.action_mode,
                        'control_frame': None, 'time_unit': self.time_unit, 'capabilities': self.capabilities()}
        self.reset()

    def capabilities(self):
        return ['load', 'scene', 'reset', 'pause', 'stop', 'instruction', 'environment_action',
                'observe', 'verify_task', 'render_suspend', 'run_limits', 'checkpoint', 'restore', 'replay_frame']

    def _validate(self, model, scene, task):
        if model not in self.models: raise ValueError(f'{self.engine_name} 不支持机器人 {model}。')
        if scene not in self.scenes: raise ValueError(f'{self.engine_name} 不支持场景 {scene}。')
        if task not in self.scenes[scene]['tasks']: raise ValueError('当前环境不支持此任务。')

    def event(self, message, kind='info'):
        self._event_id += 1
        self.events.append({'id': self._event_id, 'message': message, 'kind': kind, 'time': self.time})

    @staticmethod
    def task_status(title):
        return {'status': 'idle', 'name': title, 'title': title, 'steps': [], 'step': -1, 'progress': 0.}

    def environment(self):
        return {'observation_text': self.observation_text, 'admissible_commands': self.admissible_commands(),
                'task_name': self.task_name, 'action_mode': self.action_mode, 'step_count': self.physics_steps}

    def evaluation(self):
        success = self.native_success()
        task = self.tasks[self.task_id]
        return {'source': self.evaluator_source, 'version': self.evaluator_version,
                'success': success, 'ever_success': self.ever_success or success,
                'progress': float(success), 'task_id': self.task_id, 'task_name': self.task_name,
                'name': task['name'], 'instruction': task['instruction'], 'criterion': task['criterion'],
                'checks': [{'key': 'native_goal', 'label': self.native_criterion,
                            'actual': success, 'expected': True, 'passed': success}]}

    def state(self):
        return {'ready': True, 'engine': self.engine_name, 'version': self.backend['engine_version'],
                'backend': deepcopy(self.backend), 'model': self.model_id, 'name': self.models[self.model_id]['name'],
                'generation': self.generation, 'paused': True, 'speed': 1., 'time': self.time,
                'timestep': 1., 'time_unit': self.time_unit, 'steps': self.physics_steps, 'view': self.view, 'flags': {},
                'nq': 0, 'nv': 0, 'nu': 0, 'arm_dof': 0, 'hand_dof': 0, 'actuators': [], 'robots': [],
                'active_robot': None, 'tcp': [0., 0., 0.], 'tcp_pose': None, 'blocks': {},
                'scene_id': self.scene_id, 'scene_name': self.scenes[self.scene_id]['name'], 'seed': self.seed,
                'evaluation': self.evaluation(), 'task': self.task.copy(), 'events': list(self.events),
                'playback': self.playback, 'environment': self.environment(), 'has_wrist_camera': False,
                'model_note': self.model_note, 'source': self.source}

    def observe(self):
        return {'backend': {k: self.backend[k] for k in ('engine_id', 'engine_version', 'action_mode', 'control_frame', 'time_unit')},
                'generation': self.generation, 'time': self.time, 'time_unit': self.time_unit, 'robot': self.model_id,
                'robots': [], 'actuators': [], 'joints': [], 'tcp': None,
                'environment': self.environment(), 'cameras': self.camera_observations()}

    def camera_observations(self): return []

    def step(self):
        raise ValueError(f'{self.engine_name} 按离散动作推进，请选择环境动作；不支持无动作物理单步。')

    def verify_task(self, generation, on_step=None):
        if generation != self.generation: raise ValueError('环境已切换，不能验收旧回合。')
        return {**self.evaluation(), 'verification_sim_time_s': 0.}

    def execute(self, command):
        if not isinstance(command, str) or not command.strip(): raise ValueError('环境动作必须为非空字符串。')
        if self.terminated: raise ValueError('当前回合已结束，请重置环境。')
        if self.run_deadline is not None and self.time >= self.run_deadline:
            self.task.update(status='timeout'); raise ValueError('当前回合已达到环境步数预算。')
        result = self._execute(command.strip())
        self.actions.append(command.strip())
        self.physics_steps += 1; self.time = float(self.physics_steps)
        self.ever_success = self.ever_success or self.native_success()
        success = self.native_success()
        self.task.update(status='completed' if success else 'idle', name=self.task_name,
                         title=self.task_name, progress=float(success))
        return {'environment_action': command.strip(), 'step_count': self.physics_steps, **result}

    def checkpoint(self):
        return {'backend': deepcopy(self.backend), 'model': self.model_id, 'scene_id': self.scene_id,
                'task_id': self.task_id, 'seed': self.seed, 'episode_id': self.episode_id,
                'actions': self.actions.copy(), 'time': self.time, 'time_unit': self.time_unit, 'environment': self.environment(),
                'native_success': self.native_success()}

    def restore(self, checkpoint):
        if not isinstance(checkpoint, dict): raise ValueError('回合检查点必须为对象。')
        identity = checkpoint.get('backend')
        if not isinstance(identity, dict): raise ValueError('检查点后端身份必须为对象。')
        for key in ('engine_id', 'engine_version', 'state_schema'):
            if identity.get(key) != self.backend[key]: raise ValueError('检查点引擎、版本或格式与当前环境不一致。')
        model, scene, task = (checkpoint.get(k) for k in ('model', 'scene_id', 'task_id'))
        self._validate(model, scene, task)
        seed, actions = checkpoint.get('seed'), checkpoint.get('actions')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_LAYOUT_SEED or not isinstance(actions, list) or not all(isinstance(a, str) and a.strip() for a in actions):
            raise ValueError('检查点种子或动作序列无效。')
        if not isinstance(checkpoint.get('episode_id'),str) or not checkpoint['episode_id'] or not isinstance(checkpoint.get('environment'),dict) or type(checkpoint.get('native_success')) is not bool:
            raise ValueError('检查点缺少有效的回合、观察或原生成功状态。')
        self.model_id, self.scene_id, self.task_id, self.seed = model, scene, task, seed
        self.generation += 1
        try:
            self.reset()
            if self.episode_id != checkpoint.get('episode_id'): raise ValueError('官方环境回合与检查点不一致，请恢复对应版本的资产。')
            for command in actions: self.execute(command)
            if self.native_success() != checkpoint.get('native_success') or self.environment() != checkpoint.get('environment'):
                raise ValueError('原生环境重新执行动作后与检查点不一致，未将回放声明为精确恢复。')
        except Exception as exc:
            # Reset/replay has already replaced or advanced the native world.
            # A recoverable protocol error would advertise this partial state
            # as ready while the caller still expects the previous episode.
            raise RuntimeError(f'{self.engine_name} 检查点恢复失败，当前回合未通过一致性验证；请重新加载环境：{exc}') from exc
        return {'restored': True, 'generation': self.generation, 'restore_mode': 'native_action_replay'}

    def trajectory_frame(self):
        return {**self.checkpoint(), 'qpos': [], 'qvel': [], 'ctrl': [], 'generation': self.generation}

    def replay_frame(self, frame):
        result = self.restore(frame); self.playback = True
        return result

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene'):
            model, scene = payload.get('model', self.model_id), payload.get('scene_id', self.scene_id)
            task = payload.get('task_id') or self.task_id
            self._validate(model, scene, task)
            seed = payload.get('seed', self.seed)
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_LAYOUT_SEED: raise ValueError('布局种子无效。')
            self.model_id, self.scene_id, self.task_id, self.seed = model, scene, task, seed
            try: self.reset()
            except Exception as exc:
                raise RuntimeError(f'{self.engine_name} 回合加载失败，当前原生状态未通过初始化，请重新加载环境：{exc}') from exc
            self.generation += 1
        elif action == 'reset':
            try: self.reset()
            except Exception as exc:
                raise RuntimeError(f'{self.engine_name} 重置失败，请重新加载环境：{exc}') from exc
            self.generation += 1
        elif action == 'pause':
            if not payload['value']: raise ValueError('该环境按动作逐轮推进，请使用环境动作或 Agent 接管。')
        elif action == 'stop': self.paused = True; self.run_deadline = None
        elif action == 'render_suspend': self.render_suspended = bool(payload['value'])
        elif action in ('instruction', 'environment_action'):
            if payload.get('generation', self.generation) != self.generation: raise ValueError('环境已切换，旧动作已拒绝。')
            return self.execute(payload.get('environment_action') if action == 'environment_action' else payload.get('text'))
        elif action == 'observe': return self.observe()
        elif action == 'verify_task': return self.verify_task(payload['generation'])
        elif action == 'checkpoint': return self.checkpoint()
        elif action == 'restore': return self.restore(payload['checkpoint'])
        elif action == 'replay_frame': return self.replay_frame(payload['frame'])
        elif action == 'run_limits':
            self.run_deadline = self.time+payload['duration_s']; self.stop_on_success = bool(payload['evaluator_stop'])
        else: raise ValueError(f'{self.engine_name} 不支持机械臂控制操作：{action}')
        return '环境操作已完成。'

    @staticmethod
    def _jpeg(pixels):
        from PIL import Image
        output = io.BytesIO(); Image.fromarray(pixels).save(output, 'JPEG', quality=88)
        return output.getvalue()

    def render(self): return self._jpeg(self.render_rgb())


class ALFWorldSimulation(DiscreteEnvironment):
    engine_id, engine_name, package = 'alfworld', 'ALFWorld', 'alfworld'
    evaluator_source, evaluator_version = 'alfworld_native_task', 'alfworld-task-v1'
    models, scenes, tasks = MODELS, SCENES, TASKS
    native_criterion = '官方 TextWorld GameState.won'
    source = 'https://github.com/alfworld/alfworld'
    model_note = 'ALFWorld 官方 TextWorld/PDDL 文字环境，无机械臂、TCP 或三路相机。图像为当前原生文字观察的排版；时间单位为环境步。'

    def __init__(self, model_id='alfworld_agent', render=True, scene_id='alfworld_unseen', seed=42,
                 task_id='pick_and_place_simple', data_path=None):
        # alfworld.info populates ALFWORLD_DATA with its default during import;
        # capture an actual user override before loading the SDK.
        self.data_path = Path(data_path or CONFIGURED_DATA or ROOT/'vendor/alfworld_data')
        self.env = None
        try: self._initialize(model_id, scene_id, seed, task_id)
        except Exception: self.close(); raise

    def _games(self):
        # Official game files embed PDDL, grammar and solvability. No training
        # demonstrations, model checkpoints or expert plans are needed.
        candidates = sorted(path for path in self.data_path.rglob('game.tw-pddl')
                            if '/valid_unseen/' in path.as_posix() and path.parent.parent.name.startswith(self.task_id+'-')
                            and 'movable' not in path.as_posix() and 'Sliced' not in path.as_posix())
        return [path for path in candidates if json.loads(path.read_text(encoding='utf-8')).get('solvable') is True]

    def reset(self):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        games = self._games()
        if not games: raise ValueError(f'缺少 {self.task_id} 官方 valid_unseen game.tw-pddl。请按 docs/simulators.md 安装 ALFWorld 任务资产到 {self.data_path}。')
        game = games[self.seed % len(games)]
        new_env = textworld.start(str(game), textworld.EnvInfos(won=True, admissible_commands=True), wrappers=[AlfredDemangler])
        try: state = new_env.reset()
        except Exception: new_env.close(); raise
        if self.env is not None: self.env.close()
        self.env, self.native_state = new_env, state
        self.episode_id = game.relative_to(self.data_path).as_posix()
        self.observation_text = str(state.feedback)
        marker = 'Your task is to:'
        self.task_name = self.observation_text.split(marker, 1)[-1].strip() if marker in self.observation_text else self.tasks[self.task_id]['instruction']
        self.actions, self.time, self.physics_steps = [], 0., 0
        self.terminated, self.ever_success, self.playback = False, bool(state.won), False
        self.run_deadline = None
        self.task = self.task_status(self.task_name)
        self.event('已加载官方 ALFWorld 回合：'+self.episode_id)

    def admissible_commands(self): return list(self.native_state.admissible_commands or [])
    def native_success(self): return bool(self.native_state.won)

    def _execute(self, command):
        if command not in self.admissible_commands(): raise ValueError('命令不是当前官方环境允许的动作，请从可执行动作列表选择。')
        state, reward, done = self.env.step(command)
        self.native_state, self.terminated = state, bool(done)
        self.observation_text = str(state.feedback)
        return {'observation_text': self.observation_text, 'reward': float(reward), 'terminated': bool(done)}

    def render_rgb(self):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new('RGB', (960, 640), '#111927'); draw = ImageDraw.Draw(image)
        try: font = ImageFont.truetype('DejaVuSans.ttf', 19)
        except OSError: font = ImageFont.load_default(size=19)
        header = f'ALFWorld | official text observation | step {self.physics_steps}'
        draw.text((30, 24), header, fill='#75d8c3', font=font)
        text = '\n'.join(textwrap.fill(line, 78) for line in self.observation_text.splitlines())
        # Preserve the full observation in state/logs even when the preview is long.
        draw.multiline_text((30, 70), text[:3400], fill='#e4e9f1', font=font, spacing=8)
        return np.asarray(image)

    def close(self):
        if self.env is not None: self.env.close(); self.env = None
