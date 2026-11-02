"""Export measured transitions without inventing observations or dense rewards."""
import json
from .rounds import timeline, terminal_evidence
from .runlog import run_folder
from .costing import aggregate_costs


def export_transitions(run_id):
    folder = run_folder(run_id)
    summary_file = folder/'summary.json'
    if not summary_file.exists(): raise ValueError('请在实验结束后导出训练数据。')
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    if summary.get('kind') in ('comparison', 'research_experiment', 'benchmark', 'repair', 'harnesspai'):
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
        after_evaluation = row.get('after') or {}
        # Re-read trusted native evidence as older round records omitted native failures.
        ending = terminal_evidence(after_evaluation) if after_evaluation.get('evaluation_valid') is True else row
        terminal = ending.get('terminated', after_evaluation.get('success'))
        last = index == len(trace['rounds'])-1
        truncated = (False if terminal is True else
                     True if ending.get('truncated') is True else
                     bool(last) if terminal is False else None)
        eligible = (not missing and not seen_terminal and type(terminal) is bool
                    and not row.get('feedback_error') and row['status'] not in ('error', 'cancelled', 'interrupted'))
        routes = [{key: call.get(key) for key in ('call_id', 'parent_call_id', 'trace_id', 'decision_id',
                   'requested_model', 'model', 'routing', 'status')} for call in row['calls']
                  if (call.get('routing') or {}).get('selected_capability')]
        routing_eligible = eligible and bool(routes) and all(
            (route.get('routing') or {}).get('router_policy_version') for route in routes)
        propensity_eligible = bool(routing_eligible and all(
            isinstance((p := route['routing'].get('selection_probability')), (int, float))
            and not isinstance(p, bool) and 0 < p <= 1 for route in routes))
        transitions.append({'round': row['round'], 'decision_id': row.get('decision_id'),
            'observation': before, 'action': actions,
            'next_observation': after, 'reward': row.get('reward'), 'reward_version': row.get('reward_version'),
            'terminated': terminal, 'truncated': truncated,
            'termination_reason': ending.get('termination_reason') or (summary.get('status') if last else None),
            'elapsed_s': row.get('elapsed_s'), 'phase_s': row.get('phase_s'),
            'clock_mode': row.get('clock_mode'), 'duration_unit': 'wall_seconds',
            'reward_components': row.get('reward_components'),
            'reward_components_version': row.get('reward_components_version'),
            'stage_events': row.get('stage_events', []),
            'call_cost': aggregate_costs(call['cost'] if isinstance(call.get('cost'), dict) else
                {'total_calls': 1, 'missing_calls': 1, 'call_count_incomplete': True,
                 'warnings': ['此调用没有保存费用证据。']} for call in row['calls']),
            'routing_decisions': routes, 'routing_training_eligible': bool(routing_eligible),
            'propensity_evaluation_eligible': propensity_eligible,
            'privileged_evaluation': {'before': row.get('before'), 'after': row.get('after')},
            'calls': row['calls'], 'events': events, 'training_eligible': eligible,
            'after_terminal': seen_terminal, 'missing': missing, 'feedback_error': row.get('feedback_error')})
        seen_terminal = seen_terminal or terminal is True
    return {'schema_version': 'embodied-transitions-v2', 'run_id': run_id,
            'task': summary.get('task'), 'backend': summary.get('backend'),
            'robot': summary.get('robot'), 'scene_id': summary.get('scene_id'), 'seed': summary.get('seed'),
            'method_id': summary.get('method_id'), 'status': summary.get('status'),
            'reward_contract': 'native-success-transition-v1',
            'decision_contract': 'robot-decision-measurements-v1',
            'cost_scope': '每轮已记录模型/工具调用费用；其他资源费用在实验账本中。缺失调用费用不代表零成本。',
            'privileged_data_usage': '仅用于评价、奖励或训练 critic；不可直接作为同条件视觉策略的输入。',
            'assets': '相机 image_ref 与 depth.depth_ref 相对于本次实验 bundle；下载完整 episode 获取 JPEG 和米制 float32 深度 NPY（如有）。',
            'eligible_transitions': sum(row['training_eligible'] for row in transitions),
            'transition_count': len(transitions), 'issues': trace['issues'], 'transitions': transitions}
