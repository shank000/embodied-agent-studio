"""Small evaluator-only task graphs over measured physical predicates."""
from copy import deepcopy
import math


TASK_GRAPHS = {
    'order_fulfillment': {
        'version': 'order-fulfillment-v1',
        'recommended_budget': {'sim_seconds': 180, 'max_actions': 100},
        'stages': [
            {'id': 'red_delivered', 'label': '红块先入绿盒并释放', 'after': [], 'hold_s': .3},
            {'id': 'blue_delivered', 'label': '蓝块再入蓝盒并释放', 'after': ['red_delivered'], 'hold_s': .3},
            {'id': 'withdrawn', 'label': '机械臂撤离交付物', 'after': ['red_delivered', 'blue_delivered'], 'hold_s': .3},
        ],
        'rules': ['先完成红块再交付蓝块；提前交付蓝块判本次任务失败。',
                  '已交付物必须保持在盒内；黄色非目标方块水平位移不得超过3 cm（随场景缩放）。'],
    },
    'drawer_cycle': {
        'version': 'drawer-cycle-v1',
        'recommended_budget': {'sim_seconds': 180, 'max_actions': 100},
        'stages': [
            {'id': 'opened', 'label': '拉开抽屉并保持', 'after': [], 'hold_s': .3, 'retain': True},
            {'id': 'closed', 'label': '重新关闭抽屉', 'after': ['opened'], 'hold_s': .3},
            {'id': 'released', 'label': '松手撤离抽屉', 'after': ['closed'], 'hold_s': .3},
        ],
        'rules': ['先拉开超过12 cm并保持0.3秒，再关至2 cm以内（随场景缩放）。',
                  '最终抽屉静止、无手部接触，机械臂末端离把手至少8 cm（随场景缩放）。'],
    },
}


class TaskGraph:
    """No observation-only progress; dependencies use the prior physical state.

    Maintained stages revoke on predicate/dependency loss. Retained stages are
    explicitly historical requirements (a drawer must open before it closes).
    First achievements are counted once; current potential may decrease.
    """
    def __init__(self, task_id):
        self.spec = deepcopy(TASK_GRAPHS[task_id])
        self.stages = {s['id']: {**s, 'status': 'pending', 'satisfied': False,
                                'hold_elapsed_s': 0., 'since': None, 'achieved_at': None,
                                'ever_achieved': False} for s in self.spec['stages']}
        self.events = []
        self.terminal_failure = False
        self.failure_reason = None
        self.last_time = None

    def _event(self, kind, stage, now, **extra):
        self.events.append({'id': len(self.events) + 1, 'type': kind,
                            'stage_id': stage, 'time': now, **extra})

    def update(self, now, predicates, violation=None):
        now = float(now)
        if not math.isfinite(now):
            violation = violation or '物理状态数值无效'
            now = self.last_time if self.last_time is not None else 0.
        if self.last_time is not None and now < self.last_time:
            raise ValueError('任务图时间倒退；恢复物理检查点时必须同时恢复任务评价状态。')
        previous = {key: s['satisfied'] for key, s in self.stages.items()}
        if violation and not self.terminal_failure:
            self.terminal_failure = True
            self.failure_reason = str(violation)
            self._event('violation', None, now, reason=self.failure_reason)
        for key, stage in self.stages.items():
            ready = all(previous[parent] and self.stages[parent]['satisfied'] for parent in stage['after'])
            valid = bool(predicates.get(key)) and ready and not self.terminal_failure
            if stage.get('retain') and stage['satisfied'] and not self.terminal_failure:
                continue
            if valid:
                if stage['since'] is None: stage['since'] = now
                stage['hold_elapsed_s'] = now - stage['since']
                stage['satisfied'] = stage['hold_elapsed_s'] + 1e-9 >= stage['hold_s']
                stage['status'] = 'complete' if stage['satisfied'] else 'holding'
                if stage['satisfied'] and not previous[key]:
                    first = not stage['ever_achieved']
                    stage['ever_achieved'] = True
                    stage['achieved_at'] = now
                    self._event('achieved', key, now, first_achievement=first)
            else:
                if previous[key]: self._event('revoked', key, now)
                stage.update(satisfied=False, since=None, hold_elapsed_s=0.,
                             status='failed' if self.terminal_failure else 'blocked' if not ready else 'pending')
        self.last_time = now

    def snapshot(self):
        stages = list(self.stages.values())
        return {'version': self.spec['version'], 'stages': deepcopy(stages),
                'events': deepcopy(self.events), 'terminal_failure': self.terminal_failure,
                'failure_reason': self.failure_reason,
                'potential': sum(s['satisfied'] for s in stages) / len(stages),
                'first_achievements': sum(s['ever_achieved'] for s in stages),
                'complete': not self.terminal_failure and all(s['satisfied'] for s in stages)}

    def restore(self, value):
        if value['version'] != self.spec['version']:
            raise ValueError('任务图版本与检查点不匹配。')
        if [s['id'] for s in value['stages']] != list(self.stages):
            raise ValueError('任务图阶段与检查点不匹配。')
        self.stages = {s['id']: deepcopy(s) for s in value['stages']}
        self.events = deepcopy(value['events'])
        self.terminal_failure = value['terminal_failure']
        self.failure_reason = value['failure_reason']
