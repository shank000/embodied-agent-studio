"""Official ALFWorld/ALFRED tasks in their AI2-THOR 2.1 visual world.

Policies receive RGB and visible-object metadata through opaque handles. The
official ThorEnv owns scene restoration, clean/heat/cool semantics and task
verification. Oracle navigation, expert plans and full-scene state are never
policy observations.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

from .alfworld_backend import CONFIGURED_DATA, ROOT, TASKS as TEXT_TASKS
from .thor_backend import INTERACTIONS, ThorSimulation, decode_action


TASKS = deepcopy(TEXT_TASKS)
for _task in TASKS.values():
    _task['criterion'] = '官方 ALFWorld ThorEnv.get_goal_satisfied() 为 true；由 ALFRED 原生任务状态验收。'
SCENES = {'alfworld_thor_unseen': {'name': 'ALFWorld 未见家庭任务 · 三维视觉',
    'description': '与文字模式对应的官方 valid_unseen 回合；真实 Unity RGB、导航与物体交互。',
    'tasks': list(TASKS)}}
MODELS = {'alfworld_thor_agent': {'name': 'ALFWorld 视觉智能体', 'arm_dof': 0, 'hand_dof': 0, 'image': None}}


class ALFWorldThorSimulation(ThorSimulation):
    engine_id, engine_name, package = 'alfworld_thor', 'ALFWorld · 三维视觉', 'alfworld'
    action_mode = 'discrete'
    evaluator_source, evaluator_version = 'alfworld_thor_native_task', 'alfworld-thor-task-v1'
    models, scenes, tasks = MODELS, SCENES, TASKS
    source = 'https://github.com/alfworld/alfworld'
    model_note = ('ALFWorld 官方 ALFRED 任务与 AI2-THOR 2.1 三维世界；'
                  '策略观察包含第一人称 RGB 与原生可见物体状态，动作使用不透明句柄。'
                  '使用低层导航与交互，不包含 Oracle 瞬移导航、专家解法或三路机械臂相机。')
    native_criterion = '官方 ALFRED ThorEnv.get_goal_satisfied'

    def __init__(self, model_id='alfworld_thor_agent', render=True,
                 scene_id='alfworld_thor_unseen', seed=42,
                 task_id='pick_and_place_simple', data_path=None, controller=None):
        self.data_path = Path(data_path or CONFIGURED_DATA or ROOT/'vendor/alfworld_data')
        self.controller = controller
        self._owns_controller = controller is None
        self._display = None
        self._id_to_handle, self._handle_to_id = {}, {}
        try:
            self._initialize(model_id, scene_id, seed, task_id)
            self.backend.update(state_schema='alfworld-thor-lowlevel-v1',
                                native_engine_version=version('ai2thor'),
                                observation_contract='rgb_and_visible_object_state')
        except Exception:
            self.close()
            raise

    def _episode(self):
        # Use the same ordered, solvable games as the text adapter: a shared
        # task family and seed select the same official episode in both modes.
        games = sorted(path for path in self.data_path.rglob('game.tw-pddl')
                       if '/valid_unseen/' in path.as_posix()
                       and path.parent.parent.name.startswith(self.task_id+'-')
                       and 'movable' not in path.as_posix() and 'Sliced' not in path.as_posix())
        games = [path for path in games if json.loads(path.read_text(encoding='utf-8')).get('solvable') is True]
        if not games:
            raise ValueError(f'缺少 {self.task_id} 官方 valid_unseen game.tw-pddl，请运行 ALFWorld 视觉环境安装工具。')
        game = games[self.seed % len(games)]
        path = game.with_name('traj_data.json')
        if not path.is_file():
            raise ValueError(f'此回合缺少官方视觉场景文件 {path}。请安装 json_2.1.1_json.zip 中对应的 traj_data.json；文字文件不能生成三维房间。')
        trajectory = json.loads(path.read_text(encoding='utf-8'))
        if trajectory.get('task_type') != self.task_id:
            raise ValueError('官方视觉回合的 task_type 与所选任务不一致。')
        scene = trajectory.get('scene')
        if not isinstance(scene, dict) or any(key not in scene for key in
                ('scene_num', 'object_poses', 'object_toggles', 'dirty_and_empty', 'init_action')):
            raise ValueError('官方视觉回合缺少完整的场景重建与初态信息。')
        if not isinstance(scene['init_action'], dict) or scene['init_action'].get('action') != 'TeleportFull':
            raise ValueError('官方视觉回合缺少有效的 TeleportFull 初始位姿。')
        return game, trajectory

    @staticmethod
    def _instruction(trajectory):
        annotations = trajectory.get('turk_annotations', {}).get('anns', [])
        if annotations and isinstance(annotations[0].get('task_desc'), str) and annotations[0]['task_desc'].strip():
            return annotations[0]['task_desc'].strip()
        # This is the upstream template library; importing agents.utils.misc
        # would also load unrelated learning dependencies such as torch.
        from alfworld.gen.goal_library import gdict
        params = trajectory['pddl_params']
        goal = trajectory['task_type'] + ('_slice' if params.get('object_sliced') else '')
        return gdict[goal]['templates'][0].format(
            obj=params['object_target'].lower(), recep=params['parent_target'].lower(),
            toggle=params['toggle_target'].lower(), mrecep=params['mrecep_target'].lower())

    def _create_controller(self):
        if version('ai2thor') != '2.1.0':
            raise RuntimeError('ALFWorld 0.4.2 视觉任务需要独立 ai2thor==2.1.0 环境，不能使用 AI2-THOR 5.x 房间替代。')
        from .alfworld_thor_assets import executable_path
        build_path = executable_path()
        self._start_display()
        from alfworld.env.thor_env import ThorEnv
        return ThorEnv(player_screen_height=480, player_screen_width=640,
                       smooth_nav=False, save_frames_to_disk=False, build_path=str(build_path))

    def _require_native_success(self, operation):
        metadata = self.controller.last_event.metadata
        if not metadata.get('lastActionSuccess', False):
            raise RuntimeError(f'ALFWorld 视觉回合{operation}失败：{metadata.get("errorMessage") or "原生操作未成功"}')

    def reset(self):
        game, trajectory = self._episode()
        instruction = self._instruction(trajectory)
        if self.controller is None:
            self.controller = self._create_controller()
        scene = trajectory['scene']
        self.controller.reset('FloorPlan%d' % int(scene['scene_num']))
        self._require_native_success('重置房间')
        self.controller.restore_scene(scene['object_poses'], scene['object_toggles'], scene['dirty_and_empty'])
        self._require_native_success('恢复物体位置与状态')
        self.controller.step(dict(scene['init_action']))
        self._require_native_success('设置初始位姿')
        import alfworld
        reward_config = str(Path(alfworld.__file__).parent/'agents/config/rewards.json')
        self.controller.set_task(trajectory, SimpleNamespace(reward_config=reward_config), reward_type='sparse')
        self.native_event = self.controller.last_event
        self._official_instruction = instruction
        self.task_name = instruction
        self.episode_id = game.relative_to(self.data_path).as_posix()
        self.trajectory_data = trajectory
        self._id_to_handle, self._handle_to_id = {}, {}
        self._register_object_handles()
        self.actions, self.time, self.physics_steps = [], 0., 0
        self.terminated, self.ever_success, self.playback = False, False, False
        self.run_deadline = None
        self.task = self.task_status(self.task_name)
        self._update_observation()
        if self.native_success():
            raise RuntimeError('官方 ALFWorld 视觉回合在初始化后已满足目标；请检查场景资产，当前回合未接受为有效实验。')
        self.event('已加载 ALFWorld 官方视觉回合：'+self.episode_id)

    def _goal_text(self):
        return self._official_instruction

    def environment(self):
        return {**super().environment(), 'observation_source': 'RGB and native visible-object state',
                'action_protocol': 'AI2-THOR 2.1 low-level navigation and interaction'}

    def evaluation(self):
        result = super().evaluation()
        met, total = self.controller.get_goal_conditions_met()
        return {**result, 'instruction': self.task_name,
                'progress': float(met)/float(total) if total else float(result['success']),
                'values': {'goal_conditions_met': int(met), 'goal_conditions_total': int(total)}}

    def native_success(self):
        return bool(self.controller.get_goal_satisfied())

    def _execute(self, command):
        action = decode_action(command)
        if 'objectId' in action:
            handle = action['objectId']
            if handle not in self._handle_to_id:
                raise ValueError('未知或失效的物体句柄，请使用当前原生观察提供的动作。')
            target = next((obj for obj in self._visible_objects() if obj['objectId'] == handle), None)
            if target is None or not target.get(INTERACTIONS[action['action']]):
                raise ValueError('目标不是当前可见且支持此操作的原生物体。')
            native_id = self._handle_to_id[handle]
            if action['action'] == 'PutObject':
                inventory = self.native_event.metadata.get('inventoryObjects', [])
                if len(inventory) != 1:
                    raise ValueError('PutObject 需要当前手中持有一个物体。')
                # AI2-THOR 2.1 takes both held object and target receptacle IDs.
                action = {'action': 'PutObject', 'objectId': inventory[0]['objectId'],
                          'receptacleObjectId': native_id, 'placeStationary': True}
            else:
                action['objectId'] = native_id
        try:
            self.controller.step(action)
            # ThorEnv's official faucet postconditions can execute extra native
            # steps; the returned event can precede those state changes.
            self.native_event = self.controller.last_event
            self._update_observation()
        except Exception as exc:
            raise RuntimeError(f'ALFWorld 原生动作执行中断，当前状态需重新加载：{self._public_error(exc)}') from exc
        metadata = self.native_event.metadata
        return {'observation_text': self.observation_text,
                'last_action_success': bool(metadata.get('lastActionSuccess')),
                'error': self._public_error(metadata.get('errorMessage', '')), 'terminated': False}

    def camera_observations(self):
        cameras = super().camera_observations()
        cameras[0]['source'] = 'alfworld.ThorEnv.last_event.frame'
        return cameras
