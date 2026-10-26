"""Export measured transitions without inventing observations or dense rewards."""
import json
from .rounds import timeline
from .runlog import run_folder


def export_transitions(run_id):
    folder = run_folder(run_id)
    summary_file = folder/'summary.json'
    if not summary_file.exists(): raise ValueError('请在实验结束后导出训练数据。')
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    if summary.get('kind') in ('comparison', 'research_experiment', 'benchmark', 'repair'):
        raise ValueError('请选择实际执行机器人动作的子实验；批量汇总或外部基准没有本地逐轮训练转移。')
    trace = timeline(run_id)
    transitions = []
    seen_terminal = False
    for index, row in enumerate(trace['rounds']):
        events = row['events']
        before = next((event['data'].get('observation') for event in events if event['event'] == 'observation'), None)
        after = next((event['data'].get('observation') for event in events if event['event'] == 'observation_after'), None)
        actions = [event['data'] for event in events if event['event'] in ('action', 'policy_action', 'selection')]
        missing = [name for name, value in (('observation', before), ('next_observation', after),
                    ('action', actions), ('reward', row.get('reward'))) if value is None or value == []]
        terminal = (row.get('after') or {}).get('success')
        eligible = not missing and not seen_terminal and not row.get('feedback_error') and row['status'] not in ('error', 'cancelled', 'interrupted')
        transitions.append({'round': row['round'], 'observation': before, 'action': actions,
            'next_observation': after, 'reward': row.get('reward'), 'reward_version': row.get('reward_version'),
            'terminated': terminal, 'truncated': index == len(trace['rounds'])-1 and terminal is not True,
            'termination_reason': summary.get('status') if index == len(trace['rounds'])-1 else None,
            'elapsed_s': row.get('elapsed_s'), 'phase_s': row.get('phase_s'),
            'privileged_evaluation': {'before': row.get('before'), 'after': row.get('after')},
            'calls': row['calls'], 'events': events, 'training_eligible': eligible,
            'after_terminal': seen_terminal, 'missing': missing, 'feedback_error': row.get('feedback_error')})
        seen_terminal = seen_terminal or terminal is True
    return {'schema_version': 'embodied-transitions-v1', 'run_id': run_id,
            'task': summary.get('task'), 'backend': summary.get('backend'),
            'robot': summary.get('robot'), 'scene_id': summary.get('scene_id'), 'seed': summary.get('seed'),
            'method_id': summary.get('method_id'), 'status': summary.get('status'),
            'reward_contract': 'native-success-transition-v1',
            'privileged_data_usage': '仅用于评价、奖励或训练 critic；不可直接作为同条件视觉策略的输入。',
            'assets': '相机 image_ref 相对于本次实验 bundle；下载完整 episode 获取对应 JPEG。',
            'eligible_transitions': sum(row['training_eligible'] for row in transitions),
            'transition_count': len(transitions), 'issues': trace['issues'], 'transitions': transitions}
