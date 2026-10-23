"""Versioned, paired research episodes with frozen evaluation protocols."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import get_method, verify_method
from .runlog import RunLog, run_folder, list_runs
from .run_management import primary_models
from .telemetry import batch_timing
from .scenes import SCENES, TASKS
from .simulation import MODELS

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / 'protocols'
REGISTRY = ROOT / 'methods' / 'registry.json'
PROTOCOL_ID = re.compile(r'^protocol-[0-9a-f]{24}$')


def evaluation_contract():
    """Evaluator and physical environment are outside the candidate search space."""
    import mujoco
    paths = ['server/scenes.py', 'server/simulation.py', 'server/task_verification.py',
             'server/robot_assets.py', 'server/multiarm.py']
    return {'mujoco': mujoco.__version__, 'files': {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
        'input_condition': 'three_rgb_proprioception',
        'inference_timing': 'physics_paused', 'success_denominator': 'all_assigned_episodes'}


class ResearchProtocol(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, validate_default=True)
    name: str = Field(min_length=1, max_length=120)
    robot: str = 'so101'
    scene_id: str = 'tabletop'
    task_id: str = 'lift_red'
    development_seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=100)
    validation_seeds: list[int] = Field(default_factory=lambda: [43, 44, 45], min_length=1, max_length=100)
    test_seeds: list[int] = Field(default_factory=lambda: [46, 47, 48], min_length=1, max_length=100)
    max_sim_time_s: float = Field(default=10, ge=1, le=300)
    max_wall_time_s: float = Field(default=120, ge=1, le=1800)
    max_steps: int = Field(default=20, ge=1, le=600)
    schedule_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    min_success_delta: float = Field(default=0, ge=0, le=1)
    max_error_increase: int = Field(default=0, ge=0, le=100)
    min_pairs: int = Field(default=3, ge=2, le=100)

    @model_validator(mode='after')
    def validate_protocol(self):
        if self.robot not in MODELS or self.scene_id not in SCENES:
            raise ValueError('未知机器人或场景。')
        if self.task_id not in SCENES[self.scene_id]['tasks']:
            raise ValueError('任务不属于所选场景。')
        seen = set()
        for name in ('development_seeds', 'validation_seeds', 'test_seeds'):
            seeds = getattr(self, name)
            if len(set(seeds)) != len(seeds) or any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
                raise ValueError('种子必须是不重复的 0–4294967295 整数。')
            if seen.intersection(seeds):
                raise ValueError('开发、验证和测试种子必须互不重叠。')
            seen.update(seeds)
        return self


def register_protocol(spec):
    spec = spec if isinstance(spec, ResearchProtocol) else ResearchProtocol.model_validate(spec)
    content = {'spec': spec.model_dump(), 'contract': evaluation_contract(),
               'task': TASKS[spec.task_id], 'schema_version': 'embodied-protocol-v1'}
    encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    protocol_id = 'protocol-' + hashlib.sha256(encoded).hexdigest()[:24]
    PROTOCOLS.mkdir(parents=True, exist_ok=True)
    path = PROTOCOLS / (protocol_id + '.json')
    if path.exists():
        return get_protocol(protocol_id)
    result = {**content, 'protocol_id': protocol_id, 'created_at': datetime.now(timezone.utc).isoformat()}
    with path.open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def get_protocol(protocol_id):
    if not PROTOCOL_ID.fullmatch(protocol_id):
        raise ValueError('无效的评测协议编号。')
    path = PROTOCOLS / (protocol_id + '.json')
    if not path.is_file():
        raise ValueError('评测协议不存在。')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    spec = ResearchProtocol.model_validate(manifest['spec']).model_dump()
    content = {'spec': spec, 'contract': manifest['contract'], 'task': manifest['task'],
               'schema_version': manifest['schema_version']}
    encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    identity = 'protocol-' + hashlib.sha256(encoded).hexdigest()[:24]
    if manifest.get('protocol_id') != protocol_id or identity != protocol_id:
        raise ValueError('评测协议内容与登记编号不一致，请登记新协议。')
    return manifest


def list_protocols():
    if not PROTOCOLS.exists():
        return []
    return [get_protocol(path.stem) for path in sorted(PROTOCOLS.glob('protocol-*.json'))]


class ResearchExperiment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    protocol_id: str
    method_ids: list[str] = Field(min_length=2, max_length=8)
    split: str = 'development'
    record_video: bool = False

    @model_validator(mode='after')
    def validate_selection(self):
        if self.split not in ('development', 'validation', 'test'):
            raise ValueError('请选择 development、validation 或 test。')
        if len(set(self.method_ids)) != len(self.method_ids):
            raise ValueError('不能重复选择同一版本。')
        return self


def aggregate_rows(rows, method_ids):
    result = []
    for method_id in method_ids:
        group = [row for row in rows if row['method_id'] == method_id]
        completed = [row for row in group if row['status'] not in ('queued', 'running')]
        successes = sum(bool((row.get('metrics') or {}).get('success')) and
                        row['status'] not in ('error', 'stopped', 'queued', 'running') for row in group)
        result.append({'method_id': method_id, 'name': group[0]['method_name'] if group else method_id,
                       'assigned': len(group), 'completed': len(completed), 'successes': successes,
                       'success_rate': successes / len(group) if group else 0,
                       'errors': sum(row['status'] == 'error' for row in group),
                       'stopped': sum(row['status'] == 'stopped' for row in group),
                       'wall_time_s': sum((row.get('metrics') or {}).get('wall_time_s', 0) for row in group)})
    return result


class ResearchController:
    def __init__(self, worker, runner):
        self.worker, self.runner = worker, runner
        self.task = None
        self._active_trial = None
        self.state = {'active': False, 'status': 'idle', 'message': '登记方法版本后开始配对评测。', 'rows': [], 'aggregates': []}

    def snapshot(self):
        children = []
        trial = self._active_trial
        if trial and self.runner.task is not None and self.runner.task is not trial['previous_task']:
            child = self.runner.snapshot()
            if child.get('run_id') and child['run_id'] != trial['previous_run_id']:
                trial['row']['run_id'] = child['run_id']
                children.append(child)
        return batch_timing(self.state, getattr(self, 'run_log', None), children)

    async def start(self, request):
        if self.state['active'] or self.runner.snapshot().get('active'):
            raise ValueError('请先停止正在运行的研究任务。')
        request = request if isinstance(request, ResearchExperiment) else ResearchExperiment.model_validate(request)
        protocol = get_protocol(request.protocol_id)
        if protocol['contract'] != evaluation_contract():
            raise ValueError('物理环境或评价器已变化，请登记新的评测协议。')
        methods = {key: verify_method(key) for key in request.method_ids}
        spec = protocol['spec']
        seeds = spec[request.split + '_seeds']
        rows = [{'method_id': method_id, 'method_name': methods[method_id]['recipe']['name'],
                 'seed': seed, 'status': 'queued', 'message': '等待运行。', 'metrics': None}
                for seed in seeds for method_id in request.method_ids]
        self.state = {'active': True, 'status': 'preparing', 'message': '按冻结协议准备配对初态。',
                      'request': request.model_dump(), 'protocol_id': request.protocol_id, 'split': request.split,
                      'rows': rows, 'aggregates': aggregate_rows(rows, request.method_ids)}
        self.task = asyncio.create_task(self.run(request, protocol))
        return self.state

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None

    async def run(self, request, protocol):
        log = RunLog('research_experiment', {'task': TASKS[protocol['spec']['task_id']]['instruction'],
                     'protocol': protocol, 'settings': request.model_dump(), 'rows': self.state['rows'],
                     'primary_models': [model for method_id in request.method_ids
                                        for model in primary_models(get_method(method_id))]})
        self.run_log = log
        log.snapshot_provider = self.snapshot
        self.state.update(run_id=log.id, experiment_id=log.id)
        spec = protocol['spec']
        rng = random.Random(spec['schedule_seed'])
        started = time.monotonic()
        try:
            seeds = list(spec[request.split + '_seeds'])
            rng.shuffle(seeds)
            for seed in seeds:
                await self.worker.execute({'action': 'load', 'model': spec['robot'], 'scene_id': spec['scene_id'],
                                           'task_id': spec['task_id'], 'seed': seed})
                await self.worker.execute({'action': 'settle'})
                checkpoint = await self.worker.execute({'action': 'checkpoint'})
                log.write('initial_state', checkpoint)
                order = list(request.method_ids)
                rng.shuffle(order)
                for method_id in order:
                    row = next(row for row in self.state['rows'] if row['seed'] == seed and row['method_id'] == method_id)
                    row.update(status='running', initial_state_id=checkpoint['state_id'])
                    self.state.update(status='running', message=f'种子 {seed} · {row["method_name"]}')
                    previous_run_id = self.runner.snapshot().get('run_id')
                    self._active_trial = {'row': row, 'previous_task': self.runner.task,
                                          'previous_run_id': previous_run_id}
                    try:
                        await self.worker.execute({'action': 'restore', 'checkpoint': checkpoint})
                        await self.runner.start({'method_id': method_id, 'task': TASKS[spec['task_id']]['instruction'],
                            'max_sim_time_s': spec['max_sim_time_s'], 'max_wall_time_s': spec['max_wall_time_s'],
                            'max_steps': spec['max_steps'], 'one_step': False, 'record_video': request.record_video})
                        await self.runner.task
                        outcome = self.runner.snapshot()
                        row.update(status=outcome['status'], message=outcome.get('message', ''),
                                   run_id=outcome.get('run_id'), metrics=outcome.get('metrics'), video=outcome.get('video'))
                    except asyncio.CancelledError:
                        await self.runner.stop()
                        outcome = self.runner.snapshot()
                        if outcome.get('run_id') and outcome['run_id'] != previous_run_id:
                            row.update(status='stopped', message=outcome.get('message', '用户停止。'),
                                       run_id=outcome['run_id'], metrics=outcome.get('metrics'), video=outcome.get('video'))
                            log.write('trial_result', row)
                        raise
                    except Exception as exc:
                        row.update(status='error', message=str(getattr(exc, 'detail', exc)), metrics={'success': False})
                    log.write('trial_result', row)
                    self._active_trial = None
                    self.state['aggregates'] = aggregate_rows(self.state['rows'], request.method_ids)
            self.state.update(status='complete', message='全部分配试验已结束；成功率分母包含错误、停止和未达成试验。')
        except asyncio.CancelledError:
            self.state.update(status='stopped', message='实验已停止，所有已分配试验保留在分母中。')
            raise
        except Exception as exc:
            self.state.update(status='error', message=str(getattr(exc, 'detail', exc)))
            log.write('error', {'message': self.state['message']})
        finally:
            await self.runner.stop()
            for row in self.state['rows']:
                if row['status'] in ('queued', 'running'):
                    row.update(status='stopped' if self.state['status'] == 'stopped' else 'error',
                               message='实验中断；未完成该次试验。')
            self.state['active'] = False
            self._active_trial = None
            self.state['aggregates'] = aggregate_rows(self.state['rows'], request.method_ids)
            log.finish({**self.state, 'timing': self.snapshot()['timing'],
                        'wall_time_s': round(time.monotonic() - started, 3)})


def read_registry():
    if not REGISTRY.exists():
        return {'active_method_id': None, 'history': []}
    return json.loads(REGISTRY.read_text(encoding='utf-8'))


def write_registry(registry):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY.with_suffix('.tmp')
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(REGISTRY)
    return registry


def experiment_summaries():
    return [run for run in list_runs(1000) if run.get('kind') == 'research_experiment']


def promote(experiment_id, candidate_id, parent_id):
    candidate, parent = get_method(candidate_id), get_method(parent_id)
    if candidate['recipe'].get('parent_id') != parent_id:
        raise ValueError('候选必须登记为所选父代的直接子版本。')
    path = run_folder(experiment_id) / 'summary.json'
    if not path.is_file():
        raise ValueError('实验尚未结束。')
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('kind') != 'research_experiment' or result.get('status') != 'complete' or result.get('split') != 'validation':
        raise ValueError('只有完整结束的验证集配对实验可用于晋升；测试集结果不得用于选版本。')
    protocol = get_protocol(result['protocol_id'])
    if protocol['contract'] != evaluation_contract():
        raise ValueError('评价环境已变化，旧实验不能用于当前晋升。')
    rows = result['rows']
    groups = {key: [row for row in rows if row['method_id'] == key] for key in (parent_id, candidate_id)}
    if any(len(group) != len({row['seed'] for row in group}) for group in groups.values()):
        raise ValueError('验证结果含重复的版本／种子配对，不能晋升。')
    selected = {key: {row['seed']: row for row in group} for key, group in groups.items()}
    seeds = protocol['spec']['validation_seeds']
    if len(seeds) < protocol['spec']['min_pairs'] or any(set(selected[key]) != set(seeds) for key in selected):
        raise ValueError('验证集完整配对数不足，不能晋升。')
    if any(not selected[parent_id][seed].get('initial_state_id') or
           selected[parent_id][seed]['initial_state_id'] != selected[candidate_id][seed].get('initial_state_id')
           for seed in seeds):
        raise ValueError('父代与子代未共享相同物理初态，不能晋升。')
    if any(row['status'] in ('queued', 'running', 'stopped') for group in selected.values() for row in group.values()):
        raise ValueError('验证配对有未完成试验，不能晋升。')
    def succeeded(row):
        return row['status'] != 'error' and bool((row.get('metrics') or {}).get('success'))
    deltas = np.asarray([int(succeeded(selected[candidate_id][seed])) - int(succeeded(selected[parent_id][seed])) for seed in seeds])
    delta = float(deltas.mean())
    errors = {key: sum(row['status'] == 'error' for row in group.values()) for key, group in selected.items()}
    if delta <= protocol['spec']['min_success_delta'] or errors[candidate_id] - errors[parent_id] > protocol['spec']['max_error_increase']:
        raise ValueError(f'未达到预先登记的晋升门槛：成功率差 {delta:+.3f}，运行错误 {errors[candidate_id]} / 父代 {errors[parent_id]}。')
    rng = np.random.default_rng(protocol['spec']['schedule_seed'])
    bootstrap = rng.choice(deltas, size=(2000, len(deltas)), replace=True).mean(axis=1)
    registry = read_registry()
    record = {'action': 'promote', 'method_id': candidate_id, 'parent_id': parent_id,
              'previous_method_id': registry['active_method_id'], 'experiment_id': experiment_id,
              'protocol_id': result['protocol_id'], 'paired_success_delta': delta,
              'paired_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
              'acceptance_status': 'exploratory', 'message': '达到登记的经验门槛；置信区间不作为独立确认性检验。',
              'timestamp': datetime.now(timezone.utc).isoformat()}
    registry['history'].append(record)
    registry['active_method_id'] = candidate_id
    with (run_folder(experiment_id) / 'selection.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    return write_registry(registry)


def rollback():
    registry = read_registry()
    current = registry['active_method_id']
    if not current:
        raise ValueError('尚无已晋升版本。')
    promotion = next((item for item in reversed(registry['history'])
                      if item['action'] == 'promote' and item['method_id'] == current), None)
    if not promotion:
        raise ValueError('找不到可回退的晋升记录。')
    target = promotion.get('previous_method_id') or promotion['parent_id']
    get_method(target)
    registry['history'].append({'action': 'rollback', 'method_id': target, 'previous_method_id': current,
                                'timestamp': datetime.now(timezone.utc).isoformat()})
    registry['active_method_id'] = target
    return write_registry(registry)
