"""Static RoboDojo metadata; importing this module never loads a simulator.

Task grouping and native episode counts are derived from the official
scripts/internal/task_inventory.py and task/RoboDojo/config/_task.yml at
f57e1633ffed59761bc8e568cfe7530c4b33381c.

MIT License
Copyright (c) 2026 RoboDojo Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from copy import deepcopy


BENCHMARK_ID = 'robodojo'
REVISION = 'f57e1633ffed59761bc8e568cfe7530c4b33381c'
VERSION = '0.2.0'
UPSTREAM = 'https://github.com/RoboDojo-Benchmark/RoboDojo'

DIMENSION_TASKS = {
    'generalization': (
        'stack_bowls', 'push_T', 'pack_objects_into_box', 'fold_clothes',
        'hang_mugs', 'sweep_blocks', 'pour_liquid_into_cup', 'make_toast',
        'arrange_largest_number', 'sort_nesting_dolls_by_size',
        'store_laptop_and_headphones', 'stack_blocks',
    ),
    'memory': (
        'cover_blocks', 'match_and_pick_from_conveyor', 'swap_blocks', 'swap_T',
        'press_by_number', 'imitate_sorting_sequence',
    ),
    'precision': (
        'fasten_screws', 'plug_in_charger', 'insert_tubes', 'pour_balls_into_vase',
        'play_Xylophone', 'deposit_coin', 'insert_key', 'build_tower',
    ),
    'long-horizon': (
        'fill_pen_holder', 'classify_objects', 'put_bottles_into_dustbin',
        'play_tic_tac_toe', 'fill_egg_holder', 'organize_table', 'make_kong',
        'play_stacking_toy',
    ),
    'open': (
        'align_blocks', 'general_pickup', 'solve_equation', 'stack_blocks_by_language',
        'classify_objects_by_language', 'pick_from_conveyor_by_image',
        'store_tools_in_toolbox', 'pour_by_language',
    ),
}

DIMENSIONS = [
    {'id': 'generalization', 'name': '泛化 · Generalization'},
    {'id': 'memory', 'name': '记忆 · Memory'},
    {'id': 'precision', 'name': '精度 · Precision'},
    {'id': 'long-horizon', 'name': '长程任务 · Long-Horizon'},
    {'id': 'open', 'name': '开放任务 · Open'},
]
EMBODIMENTS = [{'id': 'arx_x5', 'name': '双 ARX X5 · 官方 arx_x5 仿真配置'}]


def catalog():
    """Return the 42 canonical tasks and their 12 official random variants."""
    tasks = []
    for dimension, names in DIMENSION_TASKS.items():
        for base in names:
            variants = ('standard', 'random') if dimension == 'generalization' else ('standard',)
            for variant in variants:
                task_id = base+'_random' if variant == 'random' else base
                tasks.append({'id': task_id, 'name': task_id.replace('_', ' '),
                              'dimension': dimension, 'base_task': base, 'variant': variant,
                              'native_episodes': 25 if dimension == 'generalization' else 50})
    return {'benchmark_id': BENCHMARK_ID, 'revision': REVISION, 'version': VERSION,
            'dimensions': deepcopy(DIMENSIONS), 'tasks': tasks, 'embodiments': deepcopy(EMBODIMENTS)}


def task_metadata(task_id):
    task = next((value for value in catalog()['tasks'] if value['id'] == task_id), None)
    if task is None:
        raise ValueError('未知 RoboDojo 任务；请从固定版本的官方任务目录选择。')
    return task
