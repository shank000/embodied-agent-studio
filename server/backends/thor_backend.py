"""AI2-THOR Unity rooms with native navigation and object interaction actions.

The task evaluator reads the Unity object metadata/inventory independently of
the model. No synthetic room, TCP, privileged full-scene policy input, or extra
camera is substituted for the official environment.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import sys

from .alfworld_backend import DiscreteEnvironment

TASKS = {
    'thor_open_object': {'name': '打开容器', 'instruction': '找到并打开环境指定的容器。',
                         'criterion': 'AI2-THOR 原生目标 objectId 的 isOpen 为 true。'},
    'thor_pickup_object': {'name': '拾取物品', 'instruction': '找到并拿起环境指定的物品。',
                           'criterion': 'AI2-THOR 原生 inventoryObjects 包含目标 objectId。'},
    'thor_toggle_object': {'name': '开启设备', 'instruction': '找到并开启环境指定的设备。',
                           'criterion': 'AI2-THOR 原生目标 objectId 的 isToggled 为 true。'},
}
SCENES = {name: {'name': f'{label} · {name}', 'description': 'AI2-THOR 官方 iTHOR Unity 房间；原生导航和物体交互。', 'tasks': list(TASKS)}
          for name, label in [('FloorPlan1', '厨房'), ('FloorPlan201', '客厅'), ('FloorPlan301', '卧室'), ('FloorPlan401', '浴室')]}
MODELS = {'thor_agent': {'name': 'AI2-THOR 导航与交互智能体', 'arm_dof': 0, 'hand_dof': 0, 'image': None}}
NAVIGATION = ('MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown')
INTERACTIONS = {'OpenObject': 'openable', 'CloseObject': 'openable', 'PickupObject': 'pickupable',
                'ToggleObjectOn': 'toggleable', 'ToggleObjectOff': 'toggleable', 'PutObject': 'receptacle'}


def decode_action(command):
    """Allow task actions, never arbitrary Unity debug, teleport or force actions."""
    if command in NAVIGATION or command == 'DropHandObject': return {'action': command}
    try: action = json.loads(command)
    except (TypeError, ValueError) as exc: raise ValueError('交互动作须使用列表中的 JSON 命令；导航可直接填写 MoveAhead 等动作名。') from exc
    if not isinstance(action, dict) or set(action) != {'action', 'objectId'} or action.get('action') not in INTERACTIONS:
        raise ValueError('只支持导航及 Open/Close/Pickup/Put/ToggleObject 原生动作，不接受调试、瞬移或 forceAction 参数。')
    if not isinstance(action['objectId'], str) or not re.fullmatch(r'object_[0-9a-f]{24}', action['objectId']):
        raise ValueError('交互动作需要当前观察提供的不透明物体句柄；不接受原生 objectId。')
    return action


class ThorSimulation(DiscreteEnvironment):
    engine_id, engine_name, package = 'ai2thor', 'AI2-THOR', 'ai2thor'
    action_mode = 'discrete'
    evaluator_source, evaluator_version = 'ai2thor_native_task', 'ai2thor-task-v1'
    models, scenes, tasks = MODELS, SCENES, TASKS
    source = 'https://ai2thor.allenai.org/ithor/documentation/'
    model_note = 'AI2-THOR 官方 Unity 房间、原生第一人称相机和离散动作。环境时间按动作步计；自由视角、机械臂 TCP 和关节控制不适用。'

    def __init__(self, model_id='thor_agent', render=True, scene_id='FloorPlan1', seed=42,
                 task_id='thor_open_object', controller=None):
        self.controller = controller
        self._owns_controller = controller is None
        self._display = None
        self._id_to_handle, self._handle_to_id = {}, {}
        try:
            self._initialize(model_id, scene_id, seed, task_id)
            # Version 1 accepted coordinate-bearing SDK object IDs in actions.
            # Its action histories must not be restored under the opaque schema.
            self.backend['state_schema'] = 'ai2thor-integration-v2'
        except Exception: self.close(); raise

    def reset(self):
        if self.controller is None:
            self._start_display()
            from ai2thor.controller import Controller
            self.controller = Controller(scene=self.scene_id, agentMode='default',
                                         width=640, height=480, gridSize=.25,
                                         rotateStepDegrees=90, visibilityDistance=1.5,
                                         snapToGrid=True, renderDepthImage=False,
                                         renderInstanceSegmentation=False)
        else:
            self.controller.reset(scene=self.scene_id)
        self.native_event = self.controller.last_event
        metadata = self.native_event.metadata
        if not metadata.get('lastActionSuccess', False): raise RuntimeError(metadata.get('errorMessage') or 'AI2-THOR 重置失败。')
        self._id_to_handle, self._handle_to_id = {}, {}
        self._register_object_handles()
        attribute = {'thor_open_object': 'openable', 'thor_pickup_object': 'pickupable', 'thor_toggle_object': 'toggleable'}[self.task_id]
        already = {'thor_open_object': 'isOpen', 'thor_pickup_object': 'isPickedUp', 'thor_toggle_object': 'isToggled'}[self.task_id]
        targets = sorted((obj for obj in metadata['objects'] if obj.get(attribute) and not obj.get(already)),
                         key=lambda obj: (float(obj.get('distance', math.inf)), obj['objectId']))
        if not targets: raise ValueError(f'{self.scene_id} 当前没有适用于 {self.task_id} 的原生目标，请选择其他房间或任务。')
        target = targets[self.seed % min(3, len(targets))]
        self.target_id, self.target_type = target['objectId'], target['objectType']
        self.task_name = self._goal_text()
        self.native_criterion = self.tasks[self.task_id]['criterion']
        self.episode_id = f'{self.scene_id}:{self.task_id}:{self.seed}:{self._object_handle(self.target_id)}'
        self.actions, self.time, self.physics_steps = [], 0., 0
        self.terminated, self.ever_success, self.playback = False, False, False
        self.run_deadline = None
        self.task = self.task_status(self.task_name)
        self._update_observation()
        self.event('已加载 AI2-THOR 官方 Unity 房间：'+self.scene_id)

    def _start_display(self):
        # WSLg is tied to the desktop/RDP session and can leave a live socket
        # which rejects X11 handshakes. Each Unity worker owns its display so
        # disconnecting the Windows desktop cannot interrupt a rollout.
        if sys.platform != 'linux' or not (os.getenv('WSL_INTEROP') or not os.getenv('DISPLAY')):
            return
        if self._display is not None:
            return
        if not shutil.which('Xvfb'):
            raise RuntimeError('Unity 后台显示缺少 Xvfb。请在运行仿真的 Linux/WSL 中安装 xvfb xauth，再加载环境。')
        try:
            from pyvirtualdisplay import Display
        except ImportError as exc:
            raise RuntimeError('独立仿真 Python 缺少 PyVirtualDisplay，请重新安装对应 requirements-thor.txt 或 requirements-alfworld-thor.txt。') from exc
        # WSLg's :0 socket is shared across namespaces. Xvfb -displayfd can
        # claim that number and hang; the library's numbered allocator starts
        # above desktop displays and retries occupied numbers itself.
        previous = os.environ.get('PYVIRTUALDISPLAY_DISPLAYFD')
        if os.getenv('WSL_INTEROP'): os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'
        try:
            self._display = Display(backend='xvfb', size=(1280, 720), color_depth=24, use_xauth=True)
        finally:
            if previous is None: os.environ.pop('PYVIRTUALDISPLAY_DISPLAYFD', None)
            else: os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = previous
        self._display.start()
        self.backend['display_mode'] = 'isolated_xvfb'
        self.model_note += ' 使用独立后台显示，生命周期随此仿真进程管理。'

    def _register_object_handles(self):
        metadata = self.native_event.metadata
        for obj in [*metadata.get('objects', []), *metadata.get('inventoryObjects', [])]:
            native_id = obj.get('objectId')
            if not isinstance(native_id, str) or not native_id or native_id in self._id_to_handle: continue
            handle = 'object_'+hashlib.sha256((self.scene_id+'\0'+native_id).encode()).hexdigest()[:24]
            if handle in self._handle_to_id and self._handle_to_id[handle] != native_id:
                raise RuntimeError('原生物体句柄冲突，当前回合不能继续。')
            self._id_to_handle[native_id], self._handle_to_id[handle] = handle, native_id

    def _object_handle(self, native_id):
        # The native ID can encode the object's initial world coordinates.
        # It is used only inside the SDK bridge and independent evaluator.
        return self._id_to_handle[native_id]

    def _goal_text(self):
        verb = {'thor_open_object': 'Open', 'thor_pickup_object': 'Pick up', 'thor_toggle_object': 'Turn on'}[self.task_id]
        return f'{verb} the {self.target_type} ({self._object_handle(self.target_id)}).'

    def _public_error(self, value):
        text = str(value or '')
        for native_id in sorted(self._id_to_handle, key=len, reverse=True):
            text = text.replace(native_id, self._id_to_handle[native_id])
        return text

    def _visible_objects(self):
        # SDK visibility is a normal agent observation, not full scene ground truth.
        fields = ('objectType', 'openable', 'isOpen', 'pickupable',
                  'isPickedUp', 'receptacle', 'toggleable', 'isToggled')
        return [{'objectId': self._object_handle(obj['objectId']), **{key: obj[key] for key in fields if key in obj}}
                for obj in self.native_event.metadata['objects'] if obj.get('visible')]

    def _update_observation(self):
        metadata = self.native_event.metadata
        self._register_object_handles()
        self.task_name = self._goal_text()
        observation = {'task': self.task_name, 'visible_objects': self._visible_objects(),
                       'inventory': [{'objectId': self._object_handle(obj['objectId']),
                                      'objectType': obj.get('objectType')}
                                     for obj in metadata.get('inventoryObjects', [])],
                       'last_action': metadata.get('lastAction'),
                       'last_action_success': bool(metadata.get('lastActionSuccess')),
                       'error': self._public_error(metadata.get('errorMessage', ''))}
        self.observation_text = json.dumps(observation, ensure_ascii=False)

    def admissible_commands(self):
        commands = list(NAVIGATION)
        inventory = self.native_event.metadata.get('inventoryObjects', [])
        if inventory: commands.append('DropHandObject')
        for obj in self._visible_objects():
            for action, flag in INTERACTIONS.items():
                if not obj.get(flag): continue
                if action == 'OpenObject' and obj.get('isOpen'): continue
                if action == 'CloseObject' and not obj.get('isOpen'): continue
                if action == 'PickupObject' and (inventory or obj.get('isPickedUp')): continue
                if action == 'PutObject' and not inventory: continue
                if action == 'ToggleObjectOn' and obj.get('isToggled'): continue
                if action == 'ToggleObjectOff' and not obj.get('isToggled'): continue
                commands.append(json.dumps({'action': action, 'objectId': obj['objectId']}, ensure_ascii=False))
        return commands

    def native_success(self):
        metadata = self.native_event.metadata
        if self.task_id == 'thor_pickup_object':
            return any(obj.get('objectId') == self.target_id for obj in metadata.get('inventoryObjects', []))
        target = next((obj for obj in metadata.get('objects', []) if obj.get('objectId') == self.target_id), None)
        if target is None: return False
        return bool(target.get('isOpen' if self.task_id == 'thor_open_object' else 'isToggled'))

    def _execute(self, command):
        action = decode_action(command)
        if 'objectId' in action:
            if action['objectId'] not in self._handle_to_id:
                raise ValueError('未知或失效的物体句柄，请使用当前原生观察提供的动作。')
            visible = {obj['objectId']: obj for obj in self._visible_objects()}
            target = visible.get(action['objectId'])
            if target is None or not target.get(INTERACTIONS[action['action']]):
                raise ValueError('目标不是当前可见且支持此操作的原生物体。')
            action['objectId'] = self._handle_to_id[action['objectId']]
        self.native_event = self.controller.step(**action)
        self._update_observation()
        metadata = self.native_event.metadata
        # A rejected interaction is a real completed environment step; preserve
        # its native error and let the next agent round correct the action.
        return {'observation_text': self.observation_text,
                'last_action_success': bool(metadata.get('lastActionSuccess')),
                'error': self._public_error(metadata.get('errorMessage', '')), 'terminated': False}

    def render_rgb(self):
        return self.native_event.frame.copy()

    def camera_observations(self):
        image = self.render_rgb()
        return [{'name': 'front', 'image_base64': base64.b64encode(self._jpeg(image)).decode(),
                 'width': int(image.shape[1]), 'height': int(image.shape[0]), 'time': self.time,
                 'source': 'ai2thor.Event.frame'}]

    def close(self):
        try:
            if self.controller is not None and self._owns_controller:
                controller, self.controller = self.controller, None
                controller.stop()
        finally:
            if getattr(self, '_display', None) is not None:
                display, self._display = self._display, None
                display.stop()
