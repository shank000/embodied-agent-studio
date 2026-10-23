"""Direct action-chunk policies, independent of the language-agent control loop."""
import asyncio
import base64
import os
from pathlib import Path
import subprocess
import time
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from . import recovery as recovery_module
from .control import PoseAction
from .runlog import RunLog, CURRENT_LOG, model_output
from .telemetry import track_inference, controller_timing

ROOT = Path(__file__).resolve().parents[1]
POLICY_KINDS = ('smolvla', 'vla', 'wam')
_local_process = None


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['smolvla', 'vla', 'wam'] = 'smolvla'
    model: str = Field(default='lerobot/smolvla_base', min_length=1, max_length=300)
    endpoint: str = ''
    api_key: SecretStr = SecretStr('')
    device: Literal['cuda', 'cpu'] = 'cpu'
    horizon: int = Field(default=10, ge=1, le=50)
    dt_s: float = Field(default=.05, ge=.02, le=.5)
    timeout_s: int = Field(default=120, ge=15, le=600)
    max_chunks: int = Field(default=60, ge=1, le=600)

    @field_validator('endpoint')
    @classmethod
    def valid_endpoint(cls, value):
        if not value: return ''
        parts = urlsplit(value)
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('策略接口须为不含账号、密钥或参数的 HTTP(S) URL。')
        if parts.scheme == 'http' and parts.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('远程策略接口请使用 HTTPS。')
        return value.rstrip('/')


class PolicyStart(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['smolvla', 'vla', 'wam'] = 'smolvla'
    task: str = Field(min_length=1, max_length=2000)
    one_chunk: bool = False
    record_video: bool = False
    max_sim_time_s: float = Field(default=60, ge=1, le=600)


class PolicyOutput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    protocol: Literal['embodied-policy-v1']
    action_type: Literal['actuator_targets', 'tcp_pose']
    robot_id: str | None = Field(default=None, min_length=1, max_length=40)
    joint_names: list[str] = Field(default_factory=list, max_length=64)
    actions: list[list[float]] = Field(default_factory=list, max_length=50)
    dt_s: float = Field(default=.05, ge=.02, le=.5)
    pose: PoseAction | None = None
    model: str | None = None
    inference_s: float | None = None
    summary: str = Field(default='', max_length=2000)
    model_output: list | dict | str | None = None
    mapping: dict | None = None
    usage: dict | None = None
    reasoning_content: str | None = None
    thinking: str | None = None

    @model_validator(mode='after')
    def required_action(self):
        if self.action_type == 'actuator_targets' and (not self.actions or not self.joint_names):
            raise ValueError('动作块需要非空 actions 与明确的 joint_names。')
        if self.action_type == 'tcp_pose' and self.pose is None: raise ValueError('TCP 策略输出必须包含 pose。')
        if self.pose is not None and self.robot_id:
            if self.pose.robot_id and self.pose.robot_id != self.robot_id: raise ValueError('策略外层与 TCP 动作的 robot_id 不一致。')
            self.pose = self.pose.model_copy(update={'robot_id':self.robot_id})
        return self


async def ensure_local_service():
    global _local_process
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        try:
            response = await client.get('http://127.0.0.1:8010/health')
            response.raise_for_status()
            if response.json().get('service') != 'embodied-smolvla': raise ValueError('8010 端口被其他服务占用。')
            return
        except (httpx.ConnectError, httpx.ConnectTimeout): pass
        python = ROOT / '.venv-policy' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not python.exists(): raise ValueError('缺少本地策略环境；请按 docs/INSTALL.md 安装 .venv-policy。')
        with (ROOT / 'policy-server.log').open('ab') as out, (ROOT / 'policy-server-error.log').open('ab') as err:
            _local_process = subprocess.Popen([str(python), '-m', 'uvicorn', 'server.policy_service:app', '--host', '127.0.0.1', '--port', '8010', '--no-access-log'],
                cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        for _ in range(40):
            await asyncio.sleep(.25)
            try:
                response = await client.get('http://127.0.0.1:8010/health')
                if response.is_success and response.json().get('service') == 'embodied-smolvla': return
            except (httpx.ConnectError, httpx.ConnectTimeout): pass
    raise RuntimeError('SmolVLA 服务未能启动，请查看 policy-server-error.log。')


async def release_local_service():
    """Release only the sidecar process tree launched by this workbench."""
    global _local_process
    process = _local_process
    if process is None: return False
    if process.poll() is None:
        if os.name == 'nt':
            # Windows venv python.exe is a launcher with a child interpreter.
            # Terminating the launcher alone leaves the model and its RAM alive.
            await asyncio.to_thread(subprocess.run,
                ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process.terminate()
    await asyncio.to_thread(process.wait, timeout=10)
    _local_process = None
    return True


def http_result(response):
    if not response.is_success:
        try: detail = response.json().get('detail', response.text)
        except ValueError: detail = response.text
        raise ValueError(f'策略服务 HTTP {response.status_code}：{detail}')
    return response.json()


@track_inference('policy_inference')
async def predict_policy(config, body, client=None):
    """Inference only. The episode owner controls observation and execution."""
    endpoint = 'http://127.0.0.1:8010/predict' if config.kind == 'smolvla' else config.endpoint
    headers = {'Authorization': 'Bearer ' + config.api_key.get_secret_value()} if config.api_key.get_secret_value() else {}
    if client is not None:
        response = await client.post(endpoint, json=body, headers=headers, timeout=config.timeout_s)
    else:
        async with httpx.AsyncClient(timeout=config.timeout_s, trust_env=False) as session:
            response = await session.post(endpoint, json=body, headers=headers)
    try: raw = response.json()
    except ValueError: raw = {'content': response.text}
    model_output({'provider': config.kind, 'model': config.model, 'response': raw})
    return http_result(response)


async def warmup_policy(worker, config):
    await ensure_local_service()
    await worker.execute({'action':'render_suspend', 'value':True})
    try:
        async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
            return http_result(await client.post('http://127.0.0.1:8010/load', json={'model': config.model, 'device': config.device}))
    finally:
        if not worker.failure:
            await worker.execute({'action':'render_suspend', 'value':False})


class PolicyController:
    def __init__(self, worker):
        self.worker = worker
        self.configs = {kind: PolicyConfig(kind=kind, model='lerobot/smolvla_base' if kind == 'smolvla' else kind.upper() + ' 自定义模型') for kind in POLICY_KINDS}
        self.state = {'active': False, 'status': 'idle', 'round': 0, 'message': '选择策略模型后开始。'}
        self.observation = None; self.observation_id = 0; self.task = None
        self._accepting = False
        self.retain_local_service = False

    def configure(self, config):
        if self.state['active'] and self.state.get('status') != 'repair_paused': raise ValueError('请先停止策略运行。')
        previous = self.configs[config.kind]
        if previous.endpoint == config.endpoint and not config.api_key.get_secret_value(): config.api_key = previous.api_key
        self.configs[config.kind] = config
        return self.public_config(config.kind)

    def public_config(self, kind):
        config = self.configs[kind]
        return {**config.model_dump(exclude={'api_key'}), 'has_api_key': bool(config.api_key.get_secret_value())}

    def snapshot(self):
        return {**controller_timing(self.state, getattr(self, 'run_log', None)), 'observation_id': self.observation_id,
                'observation_time': self.observation['time'] if self.observation else None,
                'observation_robot': self.observation['robot'] if self.observation else None}

    def reset_observation(self):
        self.observation = None; self.observation_id += 1

    def validate_start(self, kind, robot):
        config = self.configs[kind]
        if kind == 'smolvla' and robot != 'so101': raise ValueError('默认 SmolVLA 当前适配 SO-101。其他机器人请接入对应 VLA/WAM 策略。')
        if kind != 'smolvla' and not config.endpoint: raise ValueError(f'请先配置 {kind.upper()} 推理接口。')

    async def warmup(self):
        return await warmup_policy(self.worker, self.configs['smolvla'])

    async def start(self, request):
        if self.state['active']: raise ValueError('已有策略任务在运行。')
        with self.worker.lock: robot = self.worker.snapshot['model']
        self.validate_start(request.kind, robot)
        await self.worker.execute({'action': 'stop'})
        config = self.configs[request.kind]
        self.state = {'active': True, 'status': 'loading', 'round': 0, 'kind': request.kind, 'model': config.model,
                      'max_rounds': 1 if request.one_chunk else config.max_chunks,
                      'message': '准备策略模型。', 'task': request.task, 'history': []}
        self.task = asyncio.create_task(self.run(request, config.model_copy(deep=True)))
        self._accepting = True
        return self.snapshot()

    async def stop(self):
        self._accepting = False
        if self.task and not self.task.done():
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
        self.task = None

    async def run(self, request, config):
        started = time.monotonic(); latencies = []; frames = 0; limited = 0
        with self.worker.lock: initial = self.worker.snapshot.copy()
        log = RunLog('policy', {'model': config.model, 'provider': config.kind, 'robot': initial['model'],
            'scene_id': initial.get('scene_id'), 'seed': initial.get('seed'), 'task': request.task,
            'settings': {**config.model_dump(exclude={'api_key'}), 'max_sim_time_s':request.max_sim_time_s,
                         'one_chunk':request.one_chunk, 'record_video':request.record_video}},
                         [config.api_key.get_secret_value()], record_video=request.record_video)
        self.state['run_id'] = log.id
        self.run_log = log
        context = CURRENT_LOG.set(log)
        recording = False
        try:
            log.save_checkpoint(await self.worker.execute({'action': 'checkpoint'}))
            recording = True
            await self.worker.execute({'action': 'record_start', 'log': log})
            await self.worker.execute({'action':'run_limits', 'duration_s':request.max_sim_time_s, 'evaluator_stop':True})
            if config.kind == 'smolvla':
                self.state['message'] = '加载官方 SmolVLA 权重；首次下载可能需要数分钟。'
                await self.warmup()
            async with httpx.AsyncClient(timeout=config.timeout_s, trust_env=False) as client:
                for number in range(1, self.state['max_rounds']+1):
                    with self.worker.lock: current = self.worker.snapshot.copy()
                    remaining = request.max_sim_time_s - (current['time']-initial['time'])
                    if current.get('evaluation', {}).get('success'):
                        self.state.update(status='verified', message='场景评价器已确认任务成功。'); return
                    if remaining < config.dt_s:
                        self.state.update(status='limit', message='已达到仿真时间上限。'); return
                    self.state.update(status='observing', round=number, message='采集三路相机和实际关节状态。')
                    await self.worker.execute({'action': 'pause', 'value': True})
                    self.observation = await self.worker.execute({'action': 'observe'}); self.observation_id += 1
                    obs = self.observation
                    body = {'protocol': 'embodied-policy-v1', 'model': config.model, 'device': config.device,
                            'task': request.task, 'observation': obs, 'seed': initial.get('seed',42)+number-1,
                            'reset': number == 1, 'horizon': min(config.horizon, max(1, int(remaining/config.dt_s))), 'dt_s': config.dt_s}
                    saved = log.save_observation(obs, number)
                    log.write('observation', {'round': number, 'request': body, 'observation': saved})
                    self.state.update(status='thinking', message=f'{config.model} 正在预测动作块。')
                    async def infer(feedback, effective_config):
                        attempt_body = {**body, 'model': effective_config.model, 'device': effective_config.device,
                            'horizon': min(effective_config.horizon, max(1, int(remaining/effective_config.dt_s))),
                            'dt_s': effective_config.dt_s, **({'recovery_feedback': feedback} if feedback else {})}
                        inference_start = time.monotonic()
                        raw = await predict_policy(effective_config, attempt_body, client)
                        output = PolicyOutput.model_validate(raw)
                        return raw, output, time.monotonic()-inference_start, effective_config
                    raw, output, latency, config = await recovery_module.invoke_with_recovery(
                        self, 'policy_inference', infer, obs, config)
                    if not self._accepting or asyncio.current_task().cancelling(): raise asyncio.CancelledError
                    latencies.append(latency)
                    log.write('model_output', {'round': number, 'response': raw, 'latency_s': latencies[-1]})
                    self.state.update(status='executing', message=output.summary or '执行策略动作块。', latency_s=round(latencies[-1],3))
                    if output.action_type == 'tcp_pose':
                        if output.pose.status != 'move':
                            self.state.update(status='model_done' if output.pose.status == 'done' else 'blocked', message=output.pose.explanation); return
                        result = await self.worker.execute({'action': 'move_tcp', 'generation': obs['generation'], 'pose': output.pose.model_dump()})
                        frames += 1
                    else:
                        result = await self.worker.execute({'action': 'policy_chunk', 'generation': obs['generation'],
                            'actions': output.actions, 'joint_names': output.joint_names, 'dt_s': output.dt_s, 'robot_id': output.robot_id})
                        frames += result['frames']; limited += result['limited_targets']
                    deadline = time.monotonic()+60
                    while time.monotonic() < deadline:
                        await asyncio.sleep(.04)
                        with self.worker.lock: current = self.worker.snapshot.copy()
                        if not current.get('ready'): raise RuntimeError(current.get('error','MuJoCo 未就绪。'))
                        if current['generation'] != obs['generation']: raise RuntimeError('场景已切换，策略运行结束。')
                        if current['time']-initial['time'] >= request.max_sim_time_s or current.get('evaluation',{}).get('success'):
                            await self.worker.execute({'action':'stop'}); break
                        if current['task']['status'] not in ('running','settling'): break
                    else: raise TimeoutError('策略动作执行超过 60 秒。')
                    await self.worker.execute({'action':'pause', 'value':True})
                    record = {'round': number, 'latency_s': latencies[-1], 'execution': result, 'evaluation': current.get('evaluation')}
                    log.write('execution', record)
                    self.state['history'].append({'round':number, 'frames': result.get('frames',1), 'limited_targets':result.get('limited_targets',0), 'latency_s':round(latencies[-1],3)})
                    self.state['history'] = self.state['history'][-30:]
                    if current.get('evaluation',{}).get('success'):
                        self.state.update(status='verified', message='场景评价器已确认任务成功。'); return
                self.state.update(status='round_complete' if request.one_chunk else 'limit', message='本次策略运行结束；任务结果以场景评价器为准。')
        except asyncio.CancelledError:
            self.state.update(status='stopped', message='策略已停止，迟到的推理结果不会执行。'); raise
        except Exception as exc:
            message = str(getattr(exc, 'detail', exc)) or type(exc).__name__
            for key in log.secrets:
                if key: message = message.replace(key, '[已隐藏密钥]')
            self.state.update(status='error', message=message[:2400]); log.write('error', {'message':message[:2400]})
        finally:
            try: await self.worker.execute({'action':'stop'})
            except Exception: pass
            if recording:
                try: await self.worker.execute({'action': 'record_stop'})
                except Exception as exc: log.write('cleanup_error', {'phase': 'recording', 'message': str(getattr(exc, 'detail', exc))})
            with self.worker.lock: final = self.worker.snapshot.copy()
            self.state['metrics'] = {'success': final.get('evaluation',{}).get('success',False),
                'engine_ready':final.get('ready',False),
                'wall_time_s':round(time.monotonic()-started,3), 'sim_time_s':round(final.get('time',0)-initial['time'],3),
                'inference_s': log.timing_snapshot()['timing']['inference_elapsed_s'],
                'inference_count': log._inference_count,
                'mean_inference_s': log.timing_snapshot()['timing']['inference_elapsed_s']/log._inference_count if log._inference_count else None,
                'action_frames':frames, 'limited_targets':limited, 'evaluation':final.get('evaluation')}
            try:
                if config.kind == 'smolvla' and not self.retain_local_service:
                    released = await release_local_service()
                    log.write('resources', {'local_service_released':released})
            except Exception as exc:
                self.state.update(status='error', message=f'本地策略进程未能释放：{exc}')
                log.write('error', {'phase':'cleanup', 'message':self.state['message']})
            finally:
                self.state['active'] = False
                self._accepting = False
                try:
                    log.finish({'status':self.state['status'], 'message':self.state['message'], 'metrics':self.state['metrics']})
                    self.state['video'] = log.artifacts.get('video', {}).copy()
                finally: CURRENT_LOG.reset(context)
