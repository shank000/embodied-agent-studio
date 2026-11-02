"""BEHAVIOR task catalog from installed official pre-sampled scene instances.

No SDK import, asset download or task sampling happens during catalog discovery.
The bundled entry is the upstream r1pro_behavior.yaml example, not a claim that
all 1,000 activities have an installed, robot-compatible initial state.
"""
from pathlib import Path
import re

UPSTREAM = 'https://github.com/StanfordVL/BEHAVIOR-1K'
SOURCE_REF = 'v3.9.4'
SOURCE_COMMIT = 'fe66cc74538f7eba3c57255da0ee88e8ce113c87'
MODELS = {'r1pro': {'name': 'R1Pro 移动双臂', 'arm_dof': 6, 'hand_dof': 1,
    'robot_count': 1, 'source': UPSTREAM, 'description': '官方 R1Pro 全身原生控制；物理抓取。'}}
DEFAULT_SCENE = 'house_double_floor_lower'
DEFAULT_TASK = 'picking_up_trash'
CRITERION = '官方 BDDL 目标逻辑验收；并列记录完全成功与官方 Q 分数，不接受模型自报成功。'


def configured_data_path():
    from ..omnigibson import runtime_config
    config = runtime_config() or {}
    value = config.get('data_path')
    return Path(value) if value else None


def task_key(activity, definition):
    return activity if definition == 0 else f'{activity}__definition_{definition}'


def catalog(data_path=None):
    root = Path(data_path) if data_path else configured_data_path()
    tasks = {DEFAULT_TASK: {'name': '捡拾垃圾 · picking_up_trash', 'instruction': 'Pick up trash.',
        'criterion': CRITERION, 'activity_name': DEFAULT_TASK, 'activity_definition_id': 0,
        'source': UPSTREAM, 'catalog_evidence': 'upstream_example', 'native_validated': False}}
    scenes = {DEFAULT_SCENE: {'name': '双层住宅下层 · house_double_floor_lower',
        'description': '官方 R1Pro BEHAVIOR 示例；须安装对应资产和预采样实例。',
        'tasks': [DEFAULT_TASK], 'instances': {DEFAULT_TASK: []}}}
    if root is None:
        return MODELS.copy(), scenes, tasks
    # v3.9.4 get_task_instance_path resolves this exact training directory.
    # Old asset templates and hidden/public evaluation splits are not silently
    # reinterpreted as compatible research task instances.
    base = root / '2026-challenge-task-instances' / 'scenes'
    for file in sorted(base.glob('*/json/*_template.json')):
        scene = file.parent.parent.name
        prefix = scene + '_task_'
        if not re.fullmatch(r'[A-Za-z0-9_-]+', scene) or not file.stem.startswith(prefix):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_]+)_(\d+)_(\d+)_template', file.stem[len(prefix):])
        if not match:
            continue
        activity, definition, instance = match.group(1), int(match.group(2)), int(match.group(3))
        key = task_key(activity, definition)
        tasks[key] = {'name': activity.replace('_', ' '), 'instruction': activity.replace('_', ' ') + '.',
            'criterion': CRITERION, 'activity_name': activity, 'activity_definition_id': definition,
            'source': UPSTREAM, 'catalog_evidence': 'installed_presampled_template', 'native_validated': False}
        item = scenes.setdefault(scene, {'name': scene, 'description': '已发现官方预采样任务文件；不代表原生验收通过。',
                                        'tasks': [], 'instances': {}})
        if key not in item['tasks']:
            item['tasks'].append(key)
        instances = item['instances'].setdefault(key, [])
        if instance not in instances:
            instances.append(instance)
            instances.sort()
    return MODELS.copy(), scenes, tasks


def resolve_selection(model, scene, task, seed, data_path=None):
    models, scenes, tasks = catalog(data_path)
    if model not in models or scene not in scenes or task not in scenes[scene]['tasks']:
        raise ValueError('请选择已列出的 R1Pro、官方场景和对应 BEHAVIOR 任务。')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('布局种子须为非负整数；它选择预采样实例索引，不会循环复用初态。')
    instances = scenes[scene]['instances'].get(task, [])
    if not instances:
        raise ValueError('未找到此任务的官方预采样实例。请安装匹配版本的 BEHAVIOR 资产和任务实例后刷新目录。')
    if seed >= len(instances):
        raise ValueError(f'当前场景任务有 {len(instances)} 个预采样实例，布局种子范围为 0–{len(instances)-1}。')
    return {'scene_model': scene, 'activity_name': tasks[task]['activity_name'],
            'activity_definition_id': tasks[task]['activity_definition_id'],
            'activity_instance_id': instances[seed], 'seed': seed, 'task_id': task,
            'sampling': 'official_presampled_train', 'control_mode': 'physical'}


# SDK-free defaults for importers. Runtime catalog() always re-reads installed
# files so that adding assets is visible after refreshing the workbench.
SCENES = {DEFAULT_SCENE: {'name': '双层住宅下层', 'description': '官方 R1Pro 示例，需准备资产。', 'tasks': [DEFAULT_TASK]}}
TASKS = {DEFAULT_TASK: {'name': '捡拾垃圾', 'instruction': 'Pick up trash.', 'criterion': CRITERION}}
