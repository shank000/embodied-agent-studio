"""Run the pinned upstream BEHAVIOR evaluator; retain its metrics and provenance.

No SDK import occurs here. The external WebSocket policy is not instrumented by
the workbench, so missing usage and costs remain unknown. Local v3.9.4 runs are
not certified challenge leaderboard submissions.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import csv
import ctypes
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import omnigibson
from .runlog import RunLog, list_runs

RELEASE = omnigibson.RELEASE
REVISION = omnigibson.REVISION
BENCHMARK_ID = 'behavior'
IDENTIFIER = r'^[a-z][a-z0-9_]{0,159}$'
WARNINGS = [
    '使用固定 v3.9.4 官方评估器；官网挑战说明的 v3.9.3 与当前标签存在差异，本地结果不等于官方认证榜单成绩。',
    '只汇总所选任务实例的原始 Q 与成功标志；不据此报告整个挑战的官方总分。',
    '策略服务须实现上游 BEHAVIOR WebSocket 协议；填写普通大模型 HTTP API 地址不能代替该服务。',
    '固定官方 R1Pro 评测配置使用 assisted 抓取，与工作台交互环境的 physical 抓取分别记录和比较。',
    '外部策略的调用用量、推理耗时和费用未上报时保持未知，不按零计算。',
]
UNREPORTED = {'status': 'unavailable', 'total_calls': 0, 'missing_calls': 1,
    'call_count_incomplete': True, 'warnings': [WARNINGS[-1]]}


class BehaviorBenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    task_name: str = Field(default='turning_on_radio', pattern=IDENTIFIER)
    instance_indices: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=1000)
    mode: Literal['train', 'public_test'] = 'public_test'
    policy_host: str = Field(default='127.0.0.1', min_length=1, max_length=253)
    policy_port: int = Field(default=8001, ge=1, le=65535, strict=True)
    policy_scheme: Literal['ws', 'wss'] = 'ws'
    policy_name: str = Field(default='external-policy', min_length=1, max_length=120)
    num_rollouts: int = Field(default=1, ge=1, strict=True)
    max_steps: int | None = Field(default=None, ge=1, strict=True)
    record_video: bool = Field(default=True, strict=True)
    max_wall_time_s: int = Field(default=3600, ge=1, strict=True)

    @field_validator('policy_host')
    @classmethod
    def host_only(cls, value):
        try:
            ipaddress.ip_address(value)
        except ValueError:
            if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', value):
                raise ValueError('策略主机须为 IP 或 DNS 名；协议和端口使用单独字段。')
        return value

    @field_validator('instance_indices', mode='before')
    @classmethod
    def integer_indices(cls, value):
        if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
            raise ValueError('实例索引须为非负整数列表。')
        return value

    @model_validator(mode='after')
    def instance_split(self):
        if len(set(self.instance_indices)) != len(self.instance_indices):
            raise ValueError('实例索引不能重复。')
        if self.mode == 'public_test' and any(index >= 20 for index in self.instance_indices):
            raise ValueError('固定 v3.9.4 的 public_test 索引范围为 0–19，对应官方实例 301–320。')
        return self


class ResultImport(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    payload: dict | list[dict]
    policy_name: str = Field(default='external-policy', min_length=1, max_length=120)
    mode: Literal['train', 'public_test'] = 'public_test'
    source_revision: str = Field(default=REVISION, pattern=r'^[0-9a-f]{40}$')


def _task_names(config):
    if not config:
        return []
    path = Path(config['data_path'])/'2026-challenge-task-instances/metadata/B100_task_misc.csv'
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return sorted({row['Task'] for row in csv.DictReader(stream)
                       if re.fullmatch(IDENTIFIER, row.get('Task', ''))})


def readiness(config=None):
    if config is None:
        try:
            config = omnigibson.runtime_config()
        except (ValueError, OSError):
            config = None
    status = omnigibson.readiness(config)
    checks = list(status['checks'])
    for relative, label in (
        ('OmniGibson/omnigibson/eval/eval.py', '固定版本官方评估器'),
        ('OmniGibson/omnigibson/eval/r1pro.yaml', '官方 R1Pro 评测配置')):
        exists = bool(config and (Path(config['source_root'])/relative).is_file())
        checks.append({'id': relative, 'label': label, 'status': 'passed' if exists else 'missing',
                       'detail': relative})
    for filename in ('B100_task_misc.csv', 'task.jsonl'):
        path = Path(config['data_path'])/'2026-challenge-task-instances/metadata'/filename if config else None
        checks.append({'id': filename, 'label': '2026 任务元数据 ' + filename,
                       'status': 'passed' if path and path.is_file() else 'missing',
                       'detail': str(path) if path else '请安装官方 2026 challenge task instances。'})
    ready = all(item['status'] == 'passed' for item in checks)
    return {**status, 'ready': ready, 'checks': checks,
            'message': '静态检查通过；评估器依赖、策略协议和原生任务以实际运行结果验收。' if ready else
                       'BEHAVIOR 评测运行环境尚未就绪，请完成缺失项；仍可预览计划和导入结果。'}


def _expected_instance(spec, index):
    # Pinned upstream eval_utils.py: TEST_INSTANCE_IDS=range(301,341), public first 20.
    return index + 301 if spec.mode == 'public_test' else index


def _argv(spec, config, index, output):
    python = config['python'] if config else '<configured-omnigibson-python>'
    args = [python, '-u', '-m', 'omnigibson.eval.eval', '--task-name', spec.task_name,
            '--host', spec.policy_host, '--port', str(spec.policy_port), '--policy-scheme', spec.policy_scheme,
            '--instance-indices', str(index), '--mode', spec.mode, '--num-rollouts', str(spec.num_rollouts),
            '--num-envs', '1', '--output-dir', str(output), '--headless',
            '--write-video' if spec.record_video else '--no-write-video']
    if spec.max_steps is not None:
        args.extend(['--max-steps', str(spec.max_steps)])
    return args


def plan(spec):
    spec = BehaviorBenchmarkSpec.model_validate(spec)
    try:
        config = omnigibson.runtime_config()
    except (ValueError, OSError):
        config = None
    status = readiness(config)
    tasks = _task_names(config)
    if tasks and spec.task_name not in tasks:
        raise ValueError('该任务不在已安装的官方挑战任务目录中。')
    commands = [{'instance_index': index, 'instance_id': _expected_instance(spec, index),
                 'argv': _argv(spec, config, index, f'<run-directory>/benchmark/instance-{index}'),
                 'cwd': config['source_root'] if config else '<configured-source-root>'}
                for index in spec.instance_indices]
    return {'benchmark_id': BENCHMARK_ID, 'release': RELEASE, 'revision': REVISION,
            'settings': spec.model_dump(), 'commands': commands, 'readiness': status, 'ready': status['ready'],
            'robot': 'upstream_default_r1pro', 'grasping_mode': 'assisted',
            'robot_config': 'OmniGibson/omnigibson/eval/r1pro.yaml',
            'expected_results': len(spec.instance_indices)*spec.num_rollouts,
            'official_leaderboard_certified': False, 'warnings': WARNINGS.copy()}


def parse_result(payload):
    """Validate upstream per-rollout JSON, preserving metrics without recomputing Q."""
    if not isinstance(payload, dict):
        raise ValueError('须提供官方逐回合结果 JSON 对象。')
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('官方结果须为有限数值的 JSON。') from exc
    if len(encoded) > 10*1024**2:
        raise ValueError('单份结果超过 10 MiB，请按回合导入。')
    if not isinstance(payload.get('task'), str) or not re.fullmatch(IDENTIFIER, payload['task']):
        raise ValueError('缺少合法的官方 task 名称。')
    if any(type(payload.get(key)) is not int or payload[key] < 0 for key in ('instance_id', 'rollout_id', 'steps')):
        raise ValueError('官方结果须包含非负整数 instance_id、rollout_id 和 steps。')
    if type(payload.get('success')) is not bool:
        raise ValueError('官方 success 须为布尔值，不能由 Q 分数推断。')
    q = (payload.get('q_score') or {}).get('final') if isinstance(payload.get('q_score'), dict) else None
    if type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('官方 q_score.final 须为 0–1 的有限数值。')
    timing = payload.get('time')
    seconds = timing.get('simulator_time') if isinstance(timing, dict) else None
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('官方 time.simulator_time 须为非负有限数值。')
    return deepcopy(payload)


def _metrics(rows, expected=None):
    raw = [row['result'] for row in rows]
    return {'success': None, 'episodes': len(raw), 'expected_episodes': expected,
            'success_rate': sum(item['success'] for item in raw)/len(raw) if raw else None,
            'mean_reported_q': sum(item['q_score']['final'] for item in raw)/len(raw) if raw else None,
            'sim_time_s': sum(item['time']['simulator_time'] for item in raw) if raw else None,
            'inference_count': None, 'inference_s': None, 'benchmark_scope': 'selected_instance_rollouts'}


def _new_log(policy, task, origin, *, revision=REVISION, settings=None):
    log = RunLog('benchmark', {'model': policy, 'task': task, 'robot': 'r1pro',
        'backend': {'engine_id': 'omnigibson', 'engine_version': RELEASE,
                    'evaluator_source': 'behavior_official_result', 'evaluator_version': revision,
                    'grasping_mode': 'assisted' if origin == 'native_job' else None},
        'benchmark': {'benchmark_id': BENCHMARK_ID, 'release': RELEASE, 'revision': revision,
                       'official_leaderboard_certified': False, 'scope': 'selected_instance_rollouts'},
        'settings': settings, 'record_origin': origin, 'external_telemetry': True})
    log.metadata['telemetry_schema_version'] = 'behavior-external-policy-v1'
    log._call_costs.append(deepcopy(UNREPORTED))
    return log


def import_result(request):
    request = ResultImport.model_validate(request)
    payloads = request.payload if isinstance(request.payload, list) else [request.payload]
    if not payloads:
        raise ValueError('不能导入空结果列表。')
    results = [parse_result(item) for item in payloads]
    keys = [(item['task'], item['instance_id'], item['rollout_id']) for item in results]
    if len(set(keys)) != len(keys):
        raise ValueError('同任务、实例和回合的结果不能重复导入到同一记录。')
    if request.mode == 'public_test' and any(not 301 <= item['instance_id'] <= 320 for item in results):
        raise ValueError('public_test 结果应使用原生实例 ID 301–320；train 请显式选择训练模式。')
    log = _new_log(request.policy_name, ','.join(sorted({item['task'] for item in results})), 'external_import',
                   revision=request.source_revision, settings={'mode': request.mode})
    rows = []
    try:
        folder = log.folder/'benchmark/json'
        folder.mkdir(parents=True)
        for item in results:
            relative = f'benchmark/json/{item["task"]}_{item["instance_id"]}_{item["rollout_id"]}.json'
            raw = json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
            (log.folder/relative).write_bytes(raw)
            row = {'result': item, 'result_ref': relative, 'sha256': hashlib.sha256(raw).hexdigest()}
            rows.append(row)
            log.write('benchmark_result', row)
        metrics = _metrics(rows)
        log.finish({'status': 'imported', 'message': '已导入官方格式结果；未验证其执行来源，不作为认证榜单成绩。',
                    'rows': rows, 'metrics': metrics})
        return {'run_id': log.id, 'rows': rows, 'metrics': metrics, 'warnings': WARNINGS.copy()}
    except Exception as exc:
        log.finish({'status': 'error', 'message': str(exc), 'rows': rows, 'metrics': _metrics(rows)})
        raise


class _WindowsJob:
    """Own the evaluator and all descendants before letting its bootstrap proceed."""
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                        ('flags', w.DWORD), ('min_working', ctypes.c_size_t), ('max_working', ctypes.c_size_t),
                        ('active_limit', w.DWORD), ('affinity', ctypes.c_size_t),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('read_ops', 'write_ops', 'other_ops', 'read', 'write', 'other')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self):
        if self.handle and not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class _OwnedProcess:
    def __init__(self, args, *, cwd, env, stdout):
        self.job = None
        self.process = None
        try:
            if sys.platform == 'win32':
                self.job = _WindowsJob()
                # No SDK code or child process starts before the ownership barrier.
                bootstrap = "import subprocess,sys; token=sys.stdin.buffer.read(1); sys.exit(125) if token!=b'\\n' else None; sys.exit(subprocess.call([sys.executable,*sys.argv[1:]],stdin=subprocess.DEVNULL))"
                self.process = subprocess.Popen([args[0], '-u', '-c', bootstrap, *args[1:]], cwd=cwd, env=env,
                    stdin=subprocess.PIPE, stdout=stdout, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                self.job.attach(self.process)
                self.process.stdin.write(b'\n')
                self.process.stdin.flush()
                self.process.stdin.close()
            else:
                self.process = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            if self.process is not None:
                self.process.kill()
                self.process.wait(timeout=5)
            if self.job is not None:
                self.job.close()
            raise

    async def wait(self, timeout):
        return await asyncio.to_thread(self.process.wait, timeout=timeout)

    async def close(self):
        try:
            if self.job is not None:
                self.job.terminate()
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                await asyncio.to_thread(self.process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                if self.job is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await asyncio.to_thread(self.process.wait, timeout=5)
            if self.job is None:
                # A launcher may exit before a child; the private process group is still ours.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            if self.job is not None:
                self.job.close()


def _check_runtime():
    config = omnigibson.runtime_config()
    status = readiness(config)
    if not config or not status['ready']:
        reasons = '; '.join(item['label'] + ': ' + item['detail'] for item in status['checks'] if item['status'] != 'passed')
        raise ValueError('BEHAVIOR 评测尚未就绪：' + reasons)
    fresh = {**config, 'probe': omnigibson.probe_runtime(config)}
    status = readiness(fresh)
    if not status['ready']:
        raise ValueError('BEHAVIOR 启动前检查未通过：' + '; '.join(item['detail'] for item in status['checks'] if item['status'] != 'passed'))
    script = "import importlib.util,json; print(json.dumps([n for n in ('hydra','omegaconf','gello','websockets','msgpack','cv2','av') if importlib.util.find_spec(n) is None]))"
    checked = subprocess.run([config['python'], '-I', '-c', script], capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=25,
                              **({'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}))
    if checked.returncode:
        raise ValueError('官方评测依赖检查失败：' + checked.stderr[-1500:])
    missing = json.loads(checked.stdout)
    if missing:
        raise ValueError('缺少官方评测依赖：' + ', '.join(missing) + '。请按固定版本上游 Eval / JoyLo 安装说明补齐。')
    return config


class BehaviorBenchmarkController:
    def __init__(self):
        self.state = {'active': False, 'status': 'idle', 'message': '准备 BEHAVIOR 官方评估器计划，或导入逐回合结果。'}
        self.task = None
        self.process = None
        self.log = None
        self._generation = 0

    def snapshot(self):
        return {**deepcopy(self.state), **(self.log.timing_snapshot() if self.log else {})}

    def info(self):
        try:
            config = omnigibson.runtime_config()
            tasks = _task_names(config)
        except (OSError, ValueError, KeyError):
            tasks = []
        history = []
        for row in list_runs(limit=None):
            if row.get('kind') == 'benchmark' and (row.get('benchmark') or {}).get('benchmark_id') == BENCHMARK_ID:
                history.append({key: row.get(key) for key in ('run_id', 'display_name', 'status', 'task', 'model', 'metrics', 'message')})
                if len(history) >= 100:
                    break
        return {'benchmark_id': BENCHMARK_ID, 'name': 'BEHAVIOR 官方评估器', 'release': RELEASE, 'revision': REVISION,
                'defaults': BehaviorBenchmarkSpec().model_dump(), 'schema': BehaviorBenchmarkSpec.model_json_schema(),
                'tasks': tasks, 'readiness': readiness(), 'state': self.snapshot(), 'history': history,
                'warnings': WARNINGS.copy(), 'docs': 'https://behavior.stanford.edu/challenge/evaluation.html'}

    async def start(self, spec):
        if self.state['active']:
            raise ValueError('已有 BEHAVIOR 评测正在运行。')
        spec = BehaviorBenchmarkSpec.model_validate(spec)
        self._generation += 1
        generation = self._generation
        self.log = self.task = None
        self.state = {'active': True, 'status': 'preparing', 'message': '检查独立运行环境与官方评测依赖。'}
        try:
            proposed = plan(spec)
            config = await asyncio.to_thread(_check_runtime)
            if generation != self._generation:
                return self.snapshot()
            self.log = _new_log(spec.policy_name, spec.task_name, 'native_job', settings=spec.model_dump())
            folder = self.log.folder/'benchmark'
            folder.mkdir()
            (folder/'plan.json').write_text(json.dumps(proposed, ensure_ascii=False, indent=2), encoding='utf-8')
            self.state = {'active': True, 'status': 'running', 'message': '正在运行固定版本的官方 BEHAVIOR 评估器。',
                          'run_id': self.log.id, 'rows': [], 'results': [], 'native_videos': [],
                          'completed_jobs': 0, 'total_jobs': len(spec.instance_indices),
                          'expected_results': proposed['expected_results'], 'current_instance_index': None,
                          'plan': proposed, 'warnings': WARNINGS.copy()}
            self.task = asyncio.create_task(self._run(spec, config))
            return self.snapshot()
        except BaseException as exc:
            if generation == self._generation:
                self.state.update(active=False, status='error', message=str(exc) or '启动检查已取消。')
                if self.log:
                    self.log.finish({'status': 'error', 'message': self.state['message']})
            raise

    def _collect(self, spec, index, folder):
        collected = []
        issues = []
        for file in sorted((folder/'json').glob('*.json')):
            try:
                if file.is_symlink() or not file.resolve().is_relative_to(folder.resolve()):
                    raise ValueError('结果文件超出本次运行目录。')
                item = parse_result(json.loads(file.read_text(encoding='utf-8')))
                if item['task'] != spec.task_name or item['instance_id'] != _expected_instance(spec, index):
                    raise ValueError('官方结果中的任务/实例与启动计划不一致。')
                if not 0 <= item['rollout_id'] < spec.num_rollouts or any(
                        existing['result']['rollout_id'] == item['rollout_id'] for existing in collected):
                    raise ValueError('官方结果回合编号越界或重复。')
                row = {'instance_index': index, 'result': item,
                       'result_ref': file.relative_to(self.log.folder).as_posix(),
                       'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
                collected.append(row)
                self.log.write('benchmark_result', row)
            except (ValueError, OSError) as exc:
                issues.append(f'{file.name}: {exc}')
        self.state['rows'].extend(collected)
        self.state['results'] = self.state['rows']
        for file in sorted((folder/'videos').glob('*.mp4')):
            if file.is_symlink() or not file.resolve().is_relative_to(folder.resolve()):
                issues.append(f'{file.name}: 视频路径不在本次目录内。')
                continue
            relative = file.relative_to(self.log.folder).as_posix()
            self.state['native_videos'].append({'name': file.name, 'path': relative,
                'url': f'/api/runs/{self.log.id}/artifact/{relative}', 'camera': 'combined'})
        self.log.artifacts['native_videos'] = self.state['native_videos']
        if issues:
            self.log.write('archive_error', {'instance_index': index, 'issues': issues})
        return len(collected), issues

    async def _run(self, spec, config):
        started = time.monotonic()
        try:
            for index in spec.instance_indices:
                remaining = spec.max_wall_time_s - (time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError('评测墙钟预算已耗尽。')
                folder = self.log.folder/'benchmark'/f'instance-{index}'
                folder.mkdir()
                self.state.update(current_instance_index=index, message=f'正在评测 {spec.task_name} · 实例索引 {index}。')
                args = _argv(spec, config, index, folder)
                self.log.write('benchmark_command', {'argv': args, 'cwd': config['source_root'], 'instance_index': index})
                env = {**os.environ, 'OMNIGIBSON_DATA_PATH': config['data_path'], 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1'}
                error = None
                count, issues = 0, []
                try:
                    with (folder/'stdout.log').open('wb') as stream:
                        self.process = _OwnedProcess(args, cwd=config['source_root'], env=env, stdout=stream)
                        code = await self.process.wait(remaining)
                        if code:
                            raise RuntimeError(f'官方评估器退出 {code}；详见 benchmark/instance-{index}/stdout.log。')
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    try:
                        if self.process:
                            await self.process.close()
                    finally:
                        self.process = None
                        count, issues = self._collect(spec, index, folder)
                if error is None and (issues or count != spec.num_rollouts):
                    raise RuntimeError('官方评估器未生成全部有效回合结果：' + '; '.join(issues or [f'{count}/{spec.num_rollouts}']))
                self.state['completed_jobs'] += 1
            self.state.update(status='complete', message='所选实例评估结束；Q 和成功标志来自上游逐回合结果。')
        except asyncio.CancelledError:
            self.state.update(status='stopped', message='评测已停止；已生成结果、视频和诊断日志已保留。')
            raise
        except (TimeoutError, subprocess.TimeoutExpired):
            self.state.update(status='limit', message=f'达到 {spec.max_wall_time_s} 秒墙钟上限；已生成结果和日志保留。')
        except Exception as exc:
            self.state.update(status='error', message=str(exc))
            self.log.write('error', {'message': str(exc)})
        finally:
            if self.process:
                await self.process.close()
                self.process = None
            self.state['active'] = False
            metrics = _metrics(self.state['rows'], self.state['expected_results'])
            self.state['metrics'] = metrics
            self.log.finish({'status': self.state['status'], 'message': self.state['message'],
                             'rows': self.state['rows'], 'metrics': metrics})

    async def stop(self):
        self._generation += 1
        if self.state.get('status') == 'preparing':
            self.state.update(active=False, status='stopped', message='启动检查已取消，未运行仿真。')
        task = self.task
        if task and not task.done():
            if not task.cancelling():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.state.get('active'):
            if self.process:
                await self.process.close()
                self.process = None
            self.state.update(active=False, status='stopped', message='评测已停止，官方进程尚未开始执行。')
            if self.log:
                self.log.finish({'status': 'stopped', 'message': self.state['message'],
                                 'rows': self.state.get('rows', []), 'metrics': _metrics([])})
        self.task = None
        return self.snapshot()
