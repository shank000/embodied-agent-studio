"""Local MuJoCo service: native rendering, state and serialized controls."""
from __future__ import annotations

import asyncio
import base64
from collections import deque
import csv
import io
import json
import math
import tempfile
import zipfile
from itertools import islice
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
import traceback
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.background import BackgroundTask

from .simulation import Simulation, MODELS, VENDOR, ROOT
from .simulators import catalog as simulator_catalog, supported_catalog, EVALUATORS, RUNTIMES
from .agent import AgentController, AgentConfig, AgentStart
from .policy import PolicyController, PolicyConfig, PolicyStart, release_local_service
from .comparison import ComparisonController, ComparisonStart
from .scenes import SCENES, TASKS
from .runlog import list_runs, run_folder
from .analysis import analyze_run, analyze_experiments
from .robodojo import RoboDojoController, EvaluationPlan, ResultImport, plan as benchmark_plan, import_result
from .run_management import rename_run, delete_run, restore_run, batch_delete_runs, RunInUseError
from .model_capabilities import discover_model_capabilities
from .pricing import catalog as price_catalog
from .repair import RepairController, RepairConfig, RepairStart, secret_values
from .recovery import RecoveryCoordinator
from .artifacts import MethodRecipe, register_method, list_methods, get_method, verify_method
from .composition import CompositionController, CompositionStart
from .research import (ResearchProtocol, ResearchExperiment, ResearchController, register_protocol,
                       list_protocols, experiment_summaries, read_registry, promote, rollback)


class Command(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    action: Literal['load', 'scene', 'reset', 'pause', 'step', 'control', 'home', 'demo', 'stop', 'instruction', 'environment_action', 'view', 'camera', 'speed', 'flags', 'gravity', 'drop', 'select_robot']
    model: str | None = Field(default=None, min_length=1, max_length=64)
    robot_id: str | None = Field(default=None, min_length=1, max_length=40)
    scene_id: str | None = None
    task_id: str | None = None
    seed: int | None = Field(default=None, ge=0, le=4294967295)
    value: float | bool | None = None
    index: int | None = Field(default=None, ge=0, le=63)
    text: str | None = Field(default=None, max_length=160)
    environment_action: str | None = Field(default=None, min_length=1, max_length=1000)
    generation: int | None = Field(default=None, ge=0)
    view: Literal['perspective', 'top', 'front', 'wrist'] | None = None
    dx: float = Field(default=0, ge=-180, le=180)
    dy: float = Field(default=0, ge=-180, le=180)
    zoom: float = Field(default=0, ge=-1, le=1)
    pan_dx: float = Field(default=0, ge=-2, le=2)
    pan_dy: float = Field(default=0, ge=-2, le=2)
    flags: dict[Literal['contacts', 'joints', 'collision'], bool] | None = None

    @model_validator(mode='after')
    def required_fields(self):
        if (self.action == 'load' and self.model is not None
                and not any(self.model in supported_catalog(engine)[0] for engine in EVALUATORS)):
            raise ValueError('未知机器人型号。')
        required = {'select_robot':['robot_id'], 'load': ['model'], 'scene': ['scene_id'], 'pause': ['value'], 'control': ['index', 'value'], 'instruction': ['text'], 'environment_action': ['environment_action','generation'],
                    'view': ['view'], 'speed': ['value'], 'flags': ['flags'], 'gravity': ['value']}
        for name in required.get(self.action, []):
            if getattr(self, name) is None:
                raise ValueError(f'{self.action} 缺少 {name}')
        if self.action in ('pause', 'gravity') and not isinstance(self.value, bool):
            raise ValueError('value 必须是布尔值')
        if self.action in ('control', 'speed') and isinstance(self.value, bool):
            raise ValueError('value 必须是数值')
        return self


class Worker:
    def __init__(self):
        self.queue = Queue()
        self.lock = Lock()
        self.ready, self.shutdown = Event(), Event()
        self.snapshot, self.jpeg, self.frame = {}, b'', 0
        self.failure = None
        self.recorder = None
        self.record_frames = 0
        self.engine_id = 'mujoco'
        self.initial_engine_id = 'mujoco'
        self.session_id = uuid4().hex
        self.switching = False
        self.generation_base = 0
        self.thread = Thread(target=self.run, name='simulation', daemon=True)

    @staticmethod
    def create_backend(engine_id, generation_base):
        if engine_id == 'mujoco':
            simulation = Simulation()
            simulation.generation += generation_base
            return simulation
        if engine_id in RUNTIMES:
            from .backends.process_backend import ProcessBackend
            return ProcessBackend(engine_id=engine_id, generation_base=generation_base)
        raise ValueError('该仿真后端尚未完成适配。')

    @staticmethod
    def validate_backend(simulation, state, engine_id):
        discrete = engine_id in ('alfworld', 'ai2thor')
        return (state.get('ready') is True
                and (isinstance(state.get('environment'), dict) if discrete else bool(state.get('actuators')))
                and state.get('backend', {}).get('engine_id') == engine_id
                and math.isfinite(simulation.timestep) and simulation.timestep > 0)

    def run(self):
        simulation = None
        processed = []
        try:
            simulation = self.create_backend(self.initial_engine_id, self.generation_base)
            self.engine_id = self.initial_engine_id
            initial_state = simulation.state()
            if not self.validate_backend(simulation, initial_state, self.engine_id):
                raise ValueError('仿真后端未能提供有效模型状态。')
            self.jpeg = simulation.render()
            if not self.jpeg:
                raise ValueError('仿真后端未能提供有效画面。')
            self.frame += 1
            initializing = True
            accumulator = 0.
            previous = time.perf_counter()
            fps_start, frame_count, fps = previous, 0, 0.
            while not self.shutdown.is_set():
                start = time.perf_counter()
                elapsed = min(start - previous, .1)
                previous = start
                processed = []
                for _ in range(40):
                    try:
                        payload, future = self.queue.get_nowait()
                    except Empty:
                        break
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        if payload['action'] == 'backend_select':
                            candidate = None
                            try:
                                if self.recorder is not None: raise ValueError('请先结束正在录制的实验。')
                                engine_id = payload['engine_id']
                                if engine_id == self.engine_id:
                                    result = {'backend': simulation.state()['backend'], 'session_id': self.session_id}
                                else:
                                    next_base = max(self.generation_base + 1000000, simulation.generation + 1)
                                    candidate = self.create_backend(engine_id, next_base)
                                    first_state, first_jpeg = candidate.state(), candidate.render()
                                    if (not first_jpeg or not self.validate_backend(candidate, first_state, engine_id)
                                            or first_state.get('generation', -1) <= simulation.generation):
                                        raise ValueError('新后端未能提供有效模型与画面。')
                                    old, simulation = simulation, candidate
                                    candidate = None
                                    self.engine_id, self.generation_base = engine_id, next_base
                                    self.session_id = uuid4().hex
                                    self.jpeg = first_jpeg
                                    try: old.close()
                                    except Exception: traceback.print_exc()
                                    result = {'backend': first_state['backend'], 'session_id': self.session_id}
                                    accumulator = elapsed = 0.
                            except Exception as exc:
                                cleanup = ''
                                if candidate is not None:
                                    try: candidate.close()
                                    except Exception as cleanup_exc: cleanup = f'；候选资源清理失败：{cleanup_exc}'
                                raise ValueError(f'切换失败，当前仿真保持不变：{exc}{cleanup}') from exc
                            finally:
                                self.switching = False
                        elif payload['action'] == 'record_start':
                            if self.recorder is not None:
                                raise ValueError('已有任务正在记录轨迹。')
                            self.recorder = payload['log']
                            self.record_frames = 0
                            existing = self.recorder.artifacts.get('checkpoints', {}).get('initial')
                            reference = existing['path'] if existing else self.recorder.save_checkpoint(simulation.checkpoint())
                            self.recorder.write('recording_started', {'initial_checkpoint': reference,
                                'timestep': float(simulation.timestep), 'model': simulation.model_id,
                                'backend': simulation.state().get('backend'), 'session_id': self.session_id})
                            self.record_frame(simulation)
                            result = {'initial_checkpoint': reference}
                        elif payload['action'] == 'record_stop':
                            recorder = self.recorder
                            try:
                                if recorder is not None: recorder.flush_trajectory()
                            finally:
                                try:
                                    if recorder is not None: recorder.finalize_video()
                                finally:
                                    self.recorder = None
                            result = {'trajectory_frames': self.record_frames,
                                      'video': recorder.artifacts.get('video') if recorder else None}
                        elif payload['action'] == 'verify_task':
                            extra = {'recording': self.recorder.video_sampling() if self.recorder else None} if hasattr(simulation, 'advance') else {}
                            result = simulation.verify_task(payload['generation'],
                                on_step=lambda: self.record_frame(simulation), **extra)
                        else:
                            result = simulation.command(payload)
                            if (payload['action'] in ('environment_action','instruction')
                                    and simulation.state().get('backend', {}).get('action_mode') in ('text','discrete')):
                                self.record_frame(simulation)
                        processed.append((future, result))
                        if payload['action'] in ('reset', 'load', 'scene', 'restore', 'settle', 'pause', 'step', 'verify_task'):
                            accumulator = 0.
                            elapsed = 0.
                    except ValueError as exc:
                        future.set_exception(exc)
                    except Exception as exc:
                        future.set_exception(exc)
                        raise
                physics_advanced = False
                if simulation.paused:
                    accumulator = 0.
                else:
                    accumulator += elapsed * simulation.speed
                    if hasattr(simulation, 'advance'):
                        steps = int(accumulator / simulation.timestep)
                        if steps:
                            advanced = simulation.advance(steps, on_step=lambda: self.record_frame(simulation),
                                recording=self.recorder.video_sampling() if self.recorder else None)
                            physics_advanced = advanced > 0
                            accumulator = max(0., accumulator - advanced * simulation.timestep)
                    else:
                        while accumulator >= simulation.timestep and not simulation.paused:
                            simulation.step()
                            self.record_frame(simulation)
                            physics_advanced = True
                            accumulator -= simulation.timestep
                # A paused, unchanged scene already has a valid frame. Avoid
                # repeatedly allocating/flipping full-size RGB images while idle.
                redraw = not self.jpeg or (not simulation.render_suspended and (processed or physics_advanced))
                jpeg = simulation.render() if redraw else self.jpeg
                if redraw: frame_count += 1
                if start - fps_start >= 1:
                    fps = frame_count / (start - fps_start)
                    frame_count, fps_start = 0, start
                state = simulation.state()
                if state.get('ready') is not True:
                    raise RuntimeError(state.get('error') or '仿真后端当前未就绪。')
                state.update({'fps': round(fps, 1), 'render_ms': round((time.perf_counter()-start)*1000, 1),
                              'ready': True, 'session_id': self.session_id})
                with self.lock:
                    self.snapshot, self.jpeg = state, jpeg
                    if redraw: self.frame += 1
                if initializing:
                    self.switching = False
                    initializing = False
                self.ready.set()
                for future, result in processed:
                    if not future.done(): future.set_result(result)
                self.shutdown.wait(max(0, 1/30 - (time.perf_counter() - start)))
        except Exception as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
            with self.lock:
                self.snapshot = {**self.snapshot, 'ready': False, 'error': self.failure}
            self.switching = False
            self.ready.set()
            for future, _ in processed:
                if not future.done(): future.set_exception(RuntimeError(self.failure))
            while not self.queue.empty():
                _, future = self.queue.get_nowait()
                if not future.done(): future.set_exception(RuntimeError(self.failure))
        finally:
            try:
                if self.recorder is not None:
                    try: self.recorder.flush_trajectory()
                    finally: self.recorder.finalize_video()
            finally:
                self.recorder = None
                if simulation: simulation.close()

    async def restart(self, engine_id):
        """Explicitly reload a failed worker; the caller holds the idle-controller lock."""
        from .simulators import backend_identity
        if engine_id not in EVALUATORS:
            raise HTTPException(400, '该仿真后端尚未完成适配。')
        if self.thread.is_alive() and not self.failure:
            raise HTTPException(409, '仿真服务仍在运行，请使用正常环境切换。')
        if self.thread.ident is not None:
            await asyncio.to_thread(self.thread.join, 5)
        if self.thread.is_alive():
            self.switching = False
            raise HTTPException(409, '故障后端仍在清理资源，请稍后再重新加载。')
        if self.recorder is not None:
            raise HTTPException(409, '故障实验的轨迹和视频尚未完成归档，请等待清理结束。')
        with self.lock:
            self.generation_base = max(self.generation_base, self.snapshot.get('generation', 0)) + 1000000
            self.session_id = uuid4().hex
            self.snapshot = {**self.snapshot, 'ready': False, 'error': None,
                             'generation': self.generation_base, 'session_id': self.session_id,
                             'backend': backend_identity(engine_id)}
            self.jpeg = b''
        # Resolve any commands abandoned by a normal shutdown before replacing
        # the queue. A failed run already resolves its pending commands.
        while True:
            try:
                _, pending = self.queue.get_nowait()
            except Empty:
                break
            if not pending.done(): pending.set_exception(RuntimeError('旧仿真会话已结束，命令未执行。'))
        self.queue = Queue()
        self.ready, self.shutdown = Event(), Event()
        self.failure = None
        self.record_frames = 0
        self.initial_engine_id = self.engine_id = engine_id
        self.switching = True
        self.thread = Thread(target=self.run, name='simulation', daemon=True)
        try:
            self.thread.start()
        except RuntimeError as exc:
            self.failure = str(exc)
            self.switching = False
            with self.lock: self.snapshot.update(ready=False, error=self.failure)
            raise HTTPException(503, self.failure) from exc
        ready = await asyncio.to_thread(self.ready.wait, 660)
        if not ready:
            raise HTTPException(504, '重新加载超过 660 秒；原生初始化仍在处理中，请等待状态更新后再操作。')
        if self.failure:
            raise HTTPException(503, f'重新加载失败：{self.failure}')
        with self.lock:
            return {'backend': self.snapshot['backend'], 'session_id': self.session_id,
                    'restarted': True}

    def record_frame(self, simulation):
        if self.recorder is not None:
            frame = simulation.trajectory_frame()
            frame['index'] = self.record_frames
            self.recorder.write_trajectory(frame)
            self.record_frames += 1
            if self.recorder.video_due(frame['time']):
                try:
                    self.recorder.capture_video_frame(simulation.render_rgb(), frame['time'])
                except Exception as exc:
                    self.recorder.fail_video(exc)

    async def execute(self, command):
        if self.failure or not self.thread.is_alive():
            raise HTTPException(503, self.failure or '仿真服务未运行。')
        future = Future()
        self.queue.put((command, future))
        try:
            timeout = 660 if command['action'] == 'backend_select' else 60
            # A backend load owns native resources. Let it finish its transactional
            # commit or rollback even if the requesting browser disconnects.
            wrapped = asyncio.wrap_future(future)
            return await asyncio.wait_for(asyncio.shield(wrapped) if command['action'] == 'backend_select' else wrapped, timeout=timeout)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except TimeoutError as exc:
            future.cancel()
            raise HTTPException(504, '操作耗时过长；请查看当前模型状态后再操作。') from exc
        except Exception as exc:
            raise HTTPException(500, f'{type(exc).__name__}: {exc}') from exc


worker = Worker()
agent = AgentController(worker)
policy = PolicyController(worker)
comparison = ComparisonController(worker, agent, policy)


def resolve_research_method(method_id):
    manifest = verify_method(method_id)
    recipe = manifest['recipe']
    credentials = composition.runtime_credentials.get(method_id, {}).copy()
    declared_agent = recipe.get('agent') or {}
    if (declared_agent.get('provider') == agent.config.provider and
            declared_agent.get('base_url') == agent.config.base_url):
        key = agent.config.api_key.get_secret_value()
        if key: credentials.setdefault('agent_api_key', key)
    declared_policy = recipe.get('policy') or {}
    live_policy = policy.configs.get(declared_policy.get('kind'))
    if live_policy is not None and declared_policy.get('endpoint') == live_policy.endpoint:
        key = live_policy.api_key.get_secret_value()
        if key: credentials.setdefault('policy_api_key', key)
    composition.configure_runtime(method_id, credentials)
    return manifest


composition = CompositionController(worker, method_resolver=resolve_research_method)
research = ResearchController(worker, composition)
repair = RepairController()
benchmark = RoboDojoController()


def repair_secrets():
    return [agent.config.api_key.get_secret_value(),
            *(config.api_key.get_secret_value() for config in policy.configs.values()),
            *secret_values(composition.runtime_credentials)]


recovery = RecoveryCoordinator(repair, secrets=repair_secrets)
for controller in (agent, policy, composition): controller.recovery = recovery


def research_active():
    return composition.snapshot().get('active') or research.state.get('active') or benchmark.state.get('active')


async def stop_research():
    await benchmark.stop()
    await research.stop()
    await composition.stop()


@asynccontextmanager
async def lifespan(app):
    worker.thread.start()
    ready = await asyncio.to_thread(worker.ready.wait, 60)
    if not ready or worker.failure:
        worker.shutdown.set()
        raise RuntimeError(worker.failure or '仿真初始化超时。')
    await price_catalog.start()
    try: yield
    finally:
        await price_catalog.close()
        await repair.stop()
    await stop_research()
    await comparison.stop()
    await policy.stop()
    await release_local_service()
    await agent.stop()
    worker.shutdown.set()
    await asyncio.to_thread(worker.thread.join, 5)


app = FastAPI(title='EmbodiWorks · 具身工坊', lifespan=lifespan)


def local_request(request):
    origin = request.headers.get('origin')
    allowed = {'http://127.0.0.1:5173', 'http://localhost:5173', 'http://127.0.0.1:8000', 'http://localhost:8000'}
    if request.url.hostname in ('127.0.0.1', 'localhost'):
        allowed.add(f'{request.url.scheme}://{request.url.netloc}')
    if origin and origin not in allowed:
        raise HTTPException(403, '控制接口仅允许来自本地工作台页面。')


@app.get('/api/state')
async def state():
    with worker.lock:
        snapshot = worker.snapshot.copy()
    snapshot['agent'] = agent.snapshot()
    snapshot['policy'] = policy.snapshot()
    snapshot['comparison'] = comparison.snapshot()
    snapshot['research'] = composition.snapshot()
    snapshot['experiment'] = research.snapshot()
    snapshot['repair'] = repair.snapshot()
    snapshot['benchmark'] = benchmark.snapshot()
    snapshot['registry'] = read_registry()
    snapshot['backend_switching'] = worker.switching
    return snapshot


class SimulatorSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    engine_id: Literal['mujoco', 'genesis', 'maniskill', 'libero', 'alfworld', 'ai2thor', 'isaac', 'isaacsim', 'robocasa365']


@app.get('/api/simulators')
async def simulators():
    return simulator_catalog(worker.engine_id, worker.session_id)


@app.post('/api/simulators/select')
async def select_simulator(body: SimulatorSelection, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        if worker.switching or repair.snapshot().get('active'):
            raise HTTPException(409, '请等待当前切换或故障修复结束。')
        recovering = bool(getattr(worker, 'failure', None) or not worker.thread.is_alive())
        with worker.lock:
            if not recovering and worker.snapshot.get('task', {}).get('status') in ('running', 'settling'):
                raise HTTPException(409, '请先停止当前机械臂动作。')
        worker.switching = True
        try:
            if recovering:
                result = await asyncio.shield(worker.restart(body.engine_id))
            else:
                result = await worker.execute({'action': 'backend_select', 'engine_id': body.engine_id})
            agent.reset_observation()
            policy.reset_observation()
            return result
        finally:
            # A timeout may leave the native load running; the worker clears the
            # flag after its final commit/rollback.
            if not worker.thread.is_alive(): worker.switching = False


@app.get('/api/models')
async def models():
    model_catalog, _, _ = supported_catalog(worker.engine_id)
    return [{'id': key, 'name': item['name'], 'arm_dof': item['arm_dof'],
             'robot_count': item.get('robot_count', 1),
             'hand_dof': item.get('hand_dof',1), 'note': item.get('model_note',''),
             'image': f'/api/models/{key}/image' if key in MODELS else '/brand/embodiworks-mark.svg',
             'source': item.get('source') or f"{item.get('folder','')}/{item.get('file','')}"} for key, item in model_catalog.items()]


@app.get('/api/scenes')
async def scenes():
    _, scene_catalog, task_catalog = supported_catalog(worker.engine_id)
    return {'scenes':[{'id':key,**value} for key,value in scene_catalog.items()], 'tasks':task_catalog,
            'engine_id': worker.engine_id}


@app.get('/api/models/{model_id}/image')
async def model_image(model_id: str):
    if model_id not in MODELS: raise HTTPException(404)
    item = MODELS[model_id]
    if model_id == 'rm65_gaia': return FileResponse(ROOT/'assets'/'rm65_gaia.png')
    return FileResponse(VENDOR / item['folder'] / item['image'])


@app.post('/api/command')
async def command(body: Command, request: Request):
    local_request(request)
    async with agent.lock:
        if worker.switching: raise HTTPException(409, '正在切换仿真后端，请等待加载完成。')
        if body.action in ('stop', 'reset', 'load', 'scene', 'pause'):
            await stop_research()
            await comparison.stop()
            await policy.stop()
            await agent.stop()
        elif (agent.state['active'] or policy.state['active'] or comparison.state['active'] or research_active()) and body.action not in ('view', 'camera', 'flags'):
            raise HTTPException(409, '自动控制正在运行，请先停止后再手动操作。')
        message = await worker.execute(body.model_dump(exclude_none=True))
        if body.action in ('reset', 'load', 'scene', 'select_robot'): agent.reset_observation(); policy.reset_observation()
    return {'ok': True, 'message': message}


@app.get('/api/agent/config')
async def agent_config():
    return agent.public_config()


@app.get('/api/repair/config')
async def repair_config():
    return repair.public_config()


@app.post('/api/repair/config')
async def configure_repair(body: RepairConfig, request: Request):
    local_request(request)
    async with repair.lock:
        try: return repair.configure(body)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.get('/api/repair/state')
async def repair_state():
    return repair.snapshot()


@app.post('/api/repair/model-capabilities')
async def repair_model_capabilities(body: RepairConfig, request: Request):
    local_request(request)
    if body.provider == repair.config.provider and body.base_url == repair.config.base_url and not body.api_key.get_secret_value():
        body.api_key = repair.config.api_key
    return await discover_model_capabilities(body)


@app.get('/api/repair/sources')
async def repair_sources():
    rows = [row for row in list_runs(limit=None) if row.get('kind') != 'repair' and row.get('status') in ('error', 'interrupted', 'blocked')]
    if recovery.current:
        owner = recovery.current['owner']
        rows.insert(0, {'run_id':owner.state['run_id'], 'model':owner.state.get('model'),
                       'status':'repair_paused','display_name':'当前故障暂停实验'})
    return rows[:100]


@app.post('/api/repair/start')
async def start_repair(body: RepairStart, request: Request):
    local_request(request)
    async with repair.lock:
        if recovery.current:
            if body.source_run_id == recovery.current['run_id']: return repair.snapshot()
            raise HTTPException(409, '已有实验处于故障恢复流程，请先继续或停止该实验。')
        try: return await repair.start(body, secrets=repair_secrets())
        except (ValueError, FileNotFoundError) as exc: raise HTTPException(400, str(exc)) from exc


class RecoveryResume(BaseModel):
    model_config = ConfigDict(extra='forbid')
    failure_id: str = Field(min_length=1,max_length=64)


@app.post('/api/repair/resume')
async def resume_repair(body: RecoveryResume, request: Request):
    local_request(request)
    async with repair.lock:
        try: return recovery.resume(body.failure_id)
        except ValueError as exc: raise HTTPException(409,str(exc)) from exc


@app.post('/api/repair/stop')
async def stop_repair(request: Request):
    local_request(request)
    async with repair.lock: return await repair.stop()


def ensure_idle_for_repair():
    if worker.switching: raise HTTPException(409, '正在切换仿真后端，请等待加载完成。')
    if agent.state['active'] or policy.state['active'] or comparison.state['active'] or research_active():
        raise HTTPException(409, '请先停止正在运行的机械臂实验，再验证或应用代码修复。')


def ensure_no_repair_validation():
    if worker.switching: raise HTTPException(409, '正在切换仿真后端，请等待加载完成。')
    if repair.state.get('status') == 'validating' and repair.state.get('active'):
        raise HTTPException(409, '正在验证模型接入代码，请等待验证完成或停止验证后再运行实验。')


class RepairValidation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    allow_local_code_execution: Literal[True]


@app.post('/api/repair/validate')
async def validate_repair(body: RepairValidation, request: Request):
    local_request(request)
    async with agent.lock, repair.lock:
        ensure_idle_for_repair()
        try: return await repair.validate(allow_local_code_execution=body.allow_local_code_execution)
        except (ValueError, OSError) as exc: raise HTTPException(400, str(exc)) from exc


@app.post('/api/repair/apply')
async def apply_repair(request: Request):
    local_request(request)
    async with agent.lock, repair.lock:
        ensure_idle_for_repair()
        try: return repair.apply()
        except (ValueError, OSError) as exc: raise HTTPException(400, str(exc)) from exc


@app.post('/api/repair/rollback')
async def rollback_repair(request: Request):
    local_request(request)
    async with agent.lock, repair.lock:
        ensure_idle_for_repair()
        try: return repair.rollback()
        except (ValueError, OSError) as exc: raise HTTPException(400, str(exc)) from exc


@app.post('/api/agent/config')
async def configure_agent(body: AgentConfig, request: Request):
    local_request(request)
    async with agent.lock:
        if comparison.state['active'] or research_active(): raise HTTPException(409, '实验运行中不能修改配置。')
        try: return agent.configure(body)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.post('/api/agent/model-capabilities')
async def agent_model_capabilities(body: AgentConfig, request: Request):
    local_request(request)
    if body.provider == agent.config.provider and body.base_url == agent.config.base_url and not body.api_key.get_secret_value():
        body.api_key = agent.config.api_key
    return await discover_model_capabilities(body)


@app.post('/api/agent/start')
async def start_agent(body: AgentStart, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_no_repair_validation()
        if comparison.state['active'] or policy.state['active'] or research_active(): raise HTTPException(409, '请先停止正在运行的策略或实验任务。')
        try: return await agent.start(body)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.post('/api/agent/stop')
async def stop_agent(request: Request):
    local_request(request)
    async with agent.lock:
        await stop_research()
        if comparison.state['active']: await comparison.stop()
        await agent.stop()
        physics = await stop_physics_if_available()
        return {**agent.snapshot(), **physics}


async def stop_physics_if_available():
    """Cancelling inference remains possible after a native process has failed."""
    if worker.failure or not worker.thread.is_alive():
        return {'engine_stopped': False, 'engine_error': worker.failure or '仿真服务未运行。'}
    try:
        await worker.execute({'action': 'stop'})
        return {'engine_stopped': True}
    except HTTPException as exc:
        if not worker.failure and worker.thread.is_alive(): raise
        return {'engine_stopped': False, 'engine_error': worker.failure or str(exc.detail)}


@app.post('/api/agent/observe')
async def observe_agent(request: Request):
    local_request(request)
    async with agent.lock:
        if agent.state['active'] or policy.state['active'] or comparison.state['active'] or research_active(): raise HTTPException(409, '自动控制中将自动更新观察。')
        return await agent.capture()


@app.get('/api/agent/camera/{name}')
async def agent_camera(name: Literal['front', 'side', 'wrist']):
    if not agent.observation: raise HTTPException(404, '请先采集一次当前环境观察。')
    camera = next((c for c in agent.observation['cameras'] if c['name'] == name), None)
    if camera is None: raise HTTPException(404, '当前环境观察没有提供该相机。')
    return Response(base64.b64decode(camera['image_base64']), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.get('/api/policy/config')
async def policy_config():
    return {kind:policy.public_config(kind) for kind in policy.configs}


@app.post('/api/policy/config')
async def configure_policy(body: PolicyConfig, request: Request):
    local_request(request)
    async with agent.lock:
        if comparison.state['active'] or research_active(): raise HTTPException(409, '实验运行中不能修改策略。')
        try: return policy.configure(body)
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc


@app.post('/api/policy/start')
async def start_policy(body: PolicyStart, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_no_repair_validation()
        if agent.state['active'] or comparison.state['active'] or research_active(): raise HTTPException(409,'请先停止 Agent 或实验任务。')
        try: return await policy.start(body)
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc


@app.post('/api/policy/stop')
async def stop_policy(request: Request):
    local_request(request)
    async with agent.lock:
        await stop_research()
        if comparison.state['active']: await comparison.stop()
        await policy.stop()
        physics = await stop_physics_if_available()
        return {**policy.snapshot(), **physics}


@app.get('/api/policy/camera/{name}')
async def policy_camera(name: Literal['front','side','wrist']):
    if not policy.observation: raise HTTPException(404,'尚未采集策略观察。')
    camera = next((c for c in policy.observation['cameras'] if c['name']==name), None)
    if camera is None: raise HTTPException(404, '当前环境观察没有提供该相机。')
    return Response(base64.b64decode(camera['image_base64']),media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.post('/api/comparison/start')
async def start_comparison(body: ComparisonStart, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_no_repair_validation()
        if agent.state['active'] or policy.state['active'] or research_active(): raise HTTPException(409,'请先停止当前自动控制。')
        try: return await comparison.start(body)
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc


@app.post('/api/comparison/stop')
async def stop_comparison(request: Request):
    local_request(request)
    async with agent.lock: await comparison.stop()
    return comparison.state


@app.get('/api/comparison/export.csv')
async def comparison_csv():
    out=io.StringIO(); writer=csv.writer(out)
    writer.writerow(['method','seed','initial_state_id','status','success','wall_time_s','sim_time_s','inference_count','mean_inference_s','run_id'])
    for row in comparison.state['rows']:
        metrics=row.get('metrics') or {}
        writer.writerow([row.get(k,'') for k in ('method','seed','initial_state_id','status')]+[metrics.get(k,'') for k in ('success','wall_time_s','sim_time_s','inference_count','mean_inference_s')]+[row.get('run_id','')])
    return Response(out.getvalue().encode('utf-8-sig'),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="comparison.csv"'})


@app.get('/api/runs')
async def runs(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), deleted: bool = False):
    return list_runs(limit=limit, offset=offset, deleted=deleted)


@app.get('/api/pricing')
async def pricing_status():
    return price_catalog.status()


@app.post('/api/pricing/refresh')
async def refresh_pricing(request: Request):
    local_request(request)
    return await price_catalog.refresh(force=True)


class RunRename(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(max_length=160)


class RunBatchDelete(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_ids: list[str] = Field(min_length=1, max_length=1000)


@app.post('/api/runs/batch-delete')
async def batch_delete_run_records(body: RunBatchDelete, request: Request):
    local_request(request)
    try: return batch_delete_runs(body.run_ids)
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc


def manage_run(operation, run_id, *args):
    try: return operation(run_id, *args)
    except RunInUseError as exc: raise HTTPException(409, str(exc)) from exc
    except FileNotFoundError as exc: raise HTTPException(404, str(exc)) from exc
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.patch('/api/runs/{run_id}')
async def rename_run_record(run_id: str, body: RunRename, request: Request):
    local_request(request)
    return manage_run(rename_run, run_id, body.name)


@app.delete('/api/runs/{run_id}')
async def delete_run_record(run_id: str, request: Request):
    local_request(request)
    return manage_run(delete_run, run_id)


@app.post('/api/runs/{run_id}/restore')
async def restore_run_record(run_id: str, request: Request):
    local_request(request)
    return manage_run(restore_run, run_id)


@app.get('/api/runs/{run_id}/analysis')
async def run_analysis(run_id: str):
    try: return analyze_run(run_id)
    except (ValueError, FileNotFoundError, KeyError) as exc: raise HTTPException(404, str(exc)) from exc


@app.get('/api/experiments/analysis')
async def experiments_analysis(task: str | None = None, model: str | None = None,
                               reasoning: str | None = None, kind: str | None = None,
                               limit: int = Query(100, ge=1, le=1000)):
    return analyze_experiments(task=task, model=model, reasoning=reasoning, kind=kind, limit=limit)


def get_run(run_id):
    try: return run_folder(run_id)
    except (ValueError,FileNotFoundError) as exc: raise HTTPException(404,str(exc)) from exc


@app.get('/api/runs/{run_id}/download')
async def run_download(run_id: str):
    return FileResponse(get_run(run_id)/'events.jsonl',media_type='application/x-ndjson',filename=run_id+'.jsonl')


@app.get('/api/runs/{run_id}/events')
async def run_events(run_id: str):
    with (get_run(run_id)/'events.jsonl').open(encoding='utf-8') as file: lines=deque(file,maxlen=50)
    return [json.loads(line) for line in lines]


@app.get('/api/runs/{run_id}/frame')
async def run_frame(run_id: str):
    path=get_run(run_id)/'final.jpg'
    if not path.exists(): raise HTTPException(404,'本次运行没有结果截图。')
    return FileResponse(path,media_type='image/jpeg')


class MethodRegistration(BaseModel):
    model_config = ConfigDict(extra='forbid')
    recipe: dict
    credentials: dict = Field(default_factory=dict)


class PromotionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    experiment_id: str
    candidate_id: str
    parent_id: str


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_id: str
    frame_index: int = Field(default=0, ge=0)


def research_error(exc):
    return HTTPException(400, str(getattr(exc, 'detail', exc)))


def ensure_research_idle():
    if worker.switching: raise HTTPException(409, '正在切换仿真后端，请等待加载完成。')
    if agent.state['active'] or policy.state['active'] or comparison.state['active'] or research_active():
        raise HTTPException(409, '请先停止正在运行的控制或实验任务。')


@app.get('/api/benchmarks/robodojo')
async def benchmark_info():
    return await asyncio.to_thread(benchmark.info)


@app.post('/api/benchmarks/robodojo/plan')
async def prepare_benchmark(body: EvaluationPlan, request: Request):
    local_request(request)
    return await asyncio.to_thread(benchmark_plan, body)


@app.post('/api/benchmarks/robodojo/start')
async def start_benchmark(body: EvaluationPlan, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        if repair.snapshot().get('active'): raise HTTPException(409, '请先等待故障修复结束。')
        with worker.lock:
            if worker.snapshot.get('task', {}).get('status') in ('running', 'settling'):
                raise HTTPException(409, '请先停止当前机械臂动作。')
        try: return await benchmark.start(body.model_dump())
        except (ValueError, OSError) as exc: raise research_error(exc)


@app.post('/api/benchmarks/robodojo/stop')
async def stop_benchmark(request: Request):
    local_request(request)
    await benchmark.stop()
    return benchmark.snapshot()


@app.post('/api/benchmarks/robodojo/import')
async def import_benchmark(body: ResultImport, request: Request):
    local_request(request)
    try: return await asyncio.to_thread(import_result, body)
    except (ValueError, OSError) as exc: raise research_error(exc)


@app.get('/api/research/defaults')
async def research_defaults():
    return {'agent': agent.config.model_dump(exclude={'api_key'}),
            'policy': {kind: config.model_dump(exclude={'api_key'}) for kind, config in policy.configs.items()},
            'decision': {'kind': 'typesafe', 'url': 'https://api.typesafe.ai/v1/systemone',
                         'model': 'jev-latest', 'vision': False, 'question_name': 'next_action', 'timeout_s': 30}}


@app.get('/api/research/methods')
async def research_methods():
    return list_methods()


@app.post('/api/research/methods')
async def save_method(body: MethodRegistration, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        recipe = body.recipe.copy()
        if recipe.get('mode') in ('agent', 'hybrid') and not recipe.get('agent'):
            recipe['agent'] = agent.config.model_dump(exclude={'api_key'})
        if recipe.get('mode') == 'policy' and not recipe.get('policy'):
            recipe['policy'] = policy.configs['smolvla'].model_dump(exclude={'api_key'})
        try:
            # Validate runtime fields before creating any persistent version.
            allowed = {'agent_api_key', 'policy_api_key', 'decision_api_key', 'decision_api_keys'}
            if set(body.credentials) - allowed:
                raise ValueError('运行时凭据仅接受 Agent、策略和决策接口的 API Key。')
            manifest = register_method(MethodRecipe.model_validate(recipe))
            method_id = manifest.get('method_id', manifest['id'])
            credentials = composition.runtime_credentials.get(method_id, {}).copy()
            credentials.update({key: value for key, value in body.credentials.items() if value})
            composition.configure_runtime(method_id, credentials)
            return manifest
        except (ValueError, FileNotFoundError) as exc:
            raise research_error(exc) from exc


@app.post('/api/research/methods/{method_id}/credentials')
async def method_credentials(method_id: str, body: dict, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        try:
            get_method(method_id)
            credentials = composition.runtime_credentials.get(method_id, {}).copy()
            credentials.update({key: value for key, value in body.items() if value})
            return composition.configure_runtime(method_id, credentials)
        except (ValueError, FileNotFoundError) as exc:
            raise research_error(exc) from exc


@app.post('/api/research/start')
async def start_research(body: CompositionStart, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_no_repair_validation()
        ensure_research_idle()
        try:
            return await composition.start(body.model_dump())
        except (ValueError, FileNotFoundError) as exc:
            raise research_error(exc) from exc


@app.post('/api/research/stop')
async def end_research(request: Request):
    local_request(request)
    async with agent.lock:
        await stop_research()
        return composition.snapshot()


@app.get('/api/research/protocols')
async def research_protocols():
    return list_protocols()


@app.post('/api/research/protocols')
async def save_protocol(body: ResearchProtocol, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        try:
            return register_protocol(body)
        except ValueError as exc:
            raise research_error(exc) from exc


@app.get('/api/research/experiments')
async def research_experiments():
    return experiment_summaries()


@app.post('/api/research/experiments')
async def start_research_experiment(body: ResearchExperiment, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_no_repair_validation()
        ensure_research_idle()
        try:
            # Preflight version identity before allocating a batch.
            for method_id in body.method_ids:
                resolve_research_method(method_id)
            return await research.start(body)
        except (ValueError, FileNotFoundError) as exc:
            raise research_error(exc) from exc


@app.post('/api/research/experiments/stop')
async def stop_research_experiment(request: Request):
    local_request(request)
    async with agent.lock:
        await research.stop()
        return research.state


@app.post('/api/research/promote')
async def promote_method(body: PromotionRequest, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        try:
            verify_method(body.candidate_id)
            return promote(body.experiment_id, body.candidate_id, body.parent_id)
        except (ValueError, FileNotFoundError) as exc:
            raise research_error(exc) from exc


@app.post('/api/research/rollback')
async def rollback_method(request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        try:
            return rollback()
        except (ValueError, FileNotFoundError) as exc:
            raise research_error(exc) from exc


def artifact_path(run_id, relative_path):
    folder = get_run(run_id).resolve()
    path = (folder / relative_path).resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise HTTPException(404, '记录附件不存在。')
    return path


@app.get('/api/runs/{run_id}/artifact/{relative_path:path}')
async def run_artifact(run_id: str, relative_path: str):
    return FileResponse(artifact_path(run_id, relative_path))


@app.get('/api/runs/{run_id}/episode')
async def run_episode(run_id: str):
    folder = get_run(run_id)
    summary_path = folder / 'summary.json'
    if not summary_path.exists():
        return {'available': False, 'message': '任务尚未结束，结束后可回放已记录的状态。', 'observations': [], 'trajectory_frames': 0}
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    artifacts = summary.get('artifacts', {})
    observations = []
    manifest = folder / 'observations' / 'manifest.jsonl'
    if manifest.exists():
        observations = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line]
    checkpoint = artifacts.get('checkpoints', {}).get('initial', {})
    return {'available': bool(artifacts.get('trajectory') and checkpoint), 'run_id': run_id,
            'message': '记录状态回放不会重新调用模型，也不代表重新执行动力学后得到相同结果。',
            'method_id': summary.get('method_id'), 'observations': observations,
            'trajectory_frames': artifacts.get('trajectory_frames', 0),
            'initial_checkpoint': checkpoint.get('path'), 'metrics': summary.get('metrics'),
            'video': artifacts.get('video'),
            'artifacts': artifacts}


@app.post('/api/research/replay')
async def replay_research(body: ReplayRequest, request: Request):
    local_request(request)
    async with agent.lock:
        ensure_research_idle()
        try:
            episode = await run_episode(body.run_id)
            if not episode['available']:
                raise ValueError('此记录没有完整状态轨迹，无法在原生视图中回放。')
            if body.frame_index >= episode['trajectory_frames']:
                raise ValueError('回放帧超出记录范围。')
            checkpoint = json.loads(artifact_path(body.run_id, episode['initial_checkpoint']).read_text(encoding='utf-8'))
            with artifact_path(body.run_id, 'trajectory.jsonl').open(encoding='utf-8') as file:
                line = next(islice(file, body.frame_index, body.frame_index + 1), None)
            if line is None:
                raise ValueError('记录轨迹不完整。')
            recorded = json.loads(line)
            await worker.execute({'action': 'stop'})
            await worker.execute({'action': 'restore', 'checkpoint': checkpoint})
            value = await worker.execute({'action': 'replay_frame', 'frame': recorded})
            agent.reset_observation()
            policy.reset_observation()
            return {'ok': True, 'run_id': body.run_id, 'playback': value}
        except (ValueError, KeyError) as exc:
            raise research_error(exc) from exc


@app.get('/api/runs/{run_id}/bundle')
async def run_bundle(run_id: str):
    folder = get_run(run_id).resolve()
    if not (folder / 'summary.json').is_file():
        raise HTTPException(409, '运行尚未结束，请结束后下载完整记录。')
    temporary = tempfile.NamedTemporaryFile(prefix='embodied-episode-', suffix='.zip', delete=False)
    path = Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for file in folder.rglob('*'):
                if file.is_file() and file.resolve().is_relative_to(folder):
                    archive.write(file, file.relative_to(folder).as_posix())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return FileResponse(path, media_type='application/zip', filename=run_id + '.zip',
                        background=BackgroundTask(path.unlink, missing_ok=True))


@app.get('/api/frame.jpg')
async def frame():
    if worker.failure: raise HTTPException(503, worker.failure)
    with worker.lock: jpeg = worker.jpeg
    return Response(jpeg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.get('/api/stream')
async def stream(request: Request):
    async def frames():
        last = -1
        part_prefix = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        yield part_prefix
        while not await request.is_disconnected() and not worker.failure:
            with worker.lock:
                frame_id, jpeg = worker.frame, worker.jpeg
            if frame_id != last and jpeg:
                last = frame_id
                # Chromium publishes the previous image after the next part's
                # complete headers arrive, even if physics stays paused here.
                yield jpeg + b'\r\n' + part_prefix
            await asyncio.sleep(1/30)
    return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})


if (ROOT / 'dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'dist', html=True), name='workspace')
