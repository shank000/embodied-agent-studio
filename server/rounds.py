"""Measured decision rounds, shared by live inspection and offline training export."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
import time

from .task_verification import environment_success, trusted_evaluation


def evaluation_state(worker):
    with worker.lock:
        snapshot = worker.snapshot
        evaluation = deepcopy(snapshot.get('evaluation') or {})
        valid = snapshot.get('ready') is True and trusted_evaluation(evaluation, snapshot)
        return {'time': snapshot.get('time'), 'steps': snapshot.get('steps'),
                'evaluation': evaluation, 'evaluation_valid': valid,
                'success': environment_success(snapshot) if valid else None}


class RoundClock:
    def __init__(self, number):
        self.number = number
        self.started = self.changed = time.monotonic()
        self.phase = 'observation'
        self.durations = {}

    def set_phase(self, phase):
        now = time.monotonic()
        self.durations[self.phase] = self.durations.get(self.phase, 0.) + now-self.changed
        self.changed, self.phase = now, phase

    def snapshot(self):
        now = time.monotonic()
        phases = dict(self.durations)
        phases[self.phase] = phases.get(self.phase, 0.) + now-self.changed
        return {'round': self.number, 'elapsed_s': now-self.started, 'phase_s': phases, 'phase': self.phase}


def round_phase(log, phase):
    clock = getattr(log, '_round_clock', None)
    if clock:
        clock.set_phase(phase)


@asynccontextmanager
async def measured_round(log, worker, number, state):
    before = evaluation_state(worker)
    clock = RoundClock(number)
    log.current_round = number
    log._round_clock = clock
    log.write('round_start', {'round': number, 'before': before})
    error = None
    status = 'complete'
    try:
        yield
    except BaseException as exc:
        status = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'error'
        error = str(getattr(exc, 'detail', exc)) or type(exc).__name__
        raise
    finally:
        feedback_error = None
        feedback_cancel = None
        if error is None:
            clock.set_phase('feedback')
            try:
                await worker.execute({'action': 'pause', 'value': True})
                observation = await worker.execute({'action': 'observe'})
                saved = log.save_observation(observation, number, phase='after')
                log.write('observation_after', {'round': number, 'observation': saved})
            except asyncio.CancelledError as exc:
                status, error = 'cancelled', '执行后观察采集已取消。'
                feedback_cancel = exc
            except Exception as exc:
                feedback_error = str(getattr(exc, 'detail', exc)) or type(exc).__name__
        after = evaluation_state(worker)
        if error is None and state.get('status') in ('verified', 'blocked', 'limit', 'replan', 'unverified'):
            status = state['status']
        valid = before['evaluation_valid'] and after['evaluation_valid']
        log.write('round_end', {**clock.snapshot(), 'status': status, 'error': error,
            'after': after, 'feedback_error': feedback_error,
            'reward': float(after['success'] is True and before['success'] is not True) if valid else None,
            'reward_version': 'native-success-transition-v1',
            'reward_semantics': '仅原生判据从未成功到成功时奖励 1；其他有效状态为 0；判据不可用为 null。'})
        log._round_clock = None
        if feedback_cancel is not None: raise feedback_cancel


def timeline(run_id):
    from .analysis import _events
    from .runlog import run_folder, LIVE_RUNS
    events, issues = _events(run_folder(run_id))
    rounds = {}
    current = 0
    excluded = {'start', 'finish', 'initial_state', 'video_finalized', 'video_failed', 'resources'}
    calls = {}
    for event in events:
        data = event.get('data') or {}
        if not isinstance(data, dict): continue
        if event['event'] in ('round_start', 'observation'):
            current = data.get('round', data.get('step', current))
        number = data.get('round', data.get('step', current))
        if not isinstance(number, int) or number < 1 or event['event'] in excluded: continue
        row = rounds.setdefault(number, {'round': number, 'status': 'unmeasured', 'elapsed_s': None,
                                         'phase_s': {}, 'calls': [], 'events': []})
        kind = event['event']
        if kind == 'round_start': row.update(started_at=event['timestamp'], before=data.get('before'), status='running')
        elif kind == 'round_end': row.update(data, ended_at=event['timestamp'])
        elif kind in ('inference_start', 'inference'):
            calls[data['call_id']] = {**data, 'round': number}
        elif kind not in ('model_output',): row['events'].append(event)
        elif not any(call['round'] == number for call in calls.values()): row['events'].append(event)
    log = LIVE_RUNS.get(run_id)
    clock = getattr(log, '_round_clock', None) if log else None
    if clock and clock.number in rounds: rounds[clock.number].update(clock.snapshot())
    active = bool(log and not log._finished)
    for call in calls.values():
        if call.get('status') == 'running':
            call['status'] = 'running' if active else 'interrupted'
            if active:
                call['current_wait_s'] = max(0., (datetime.now(timezone.utc)-datetime.fromisoformat(call['started_at'])).total_seconds())
        rounds[call['round']]['calls'].append(call)
    for row in rounds.values():
        if row['status'] == 'running' and not active: row['status'] = 'interrupted'
    return {'run_id': run_id, 'active': active, 'rounds': list(rounds.values()), 'issues': issues}
