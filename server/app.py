"""Local MuJoCo service: native rendering, state and serialized controls."""
from __future__ import annotations

import asyncio
import base64
from collections import deque
import csv
import io
import json
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

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.background import BackgroundTask

from .simulation import Simulation, MODELS, VENDOR, ROOT
from .agent import AgentController, AgentConfig, AgentStart
from .policy import PolicyController, PolicyConfig, PolicyStart, release_local_service
from .comparison import ComparisonController, ComparisonStart
from .scenes import SCENES, TASKS
from .runlog import list_runs, run_folder
from .analysis import analyze_run, analyze_experiments
from .run_management import rename_run, delete_run, restore_run, RunInUseError
from .artifacts import MethodRecipe, register_method, list_methods, get_method, verify_method
from .composition import CompositionController, CompositionStart
from .research import (ResearchProtocol, ResearchExperiment, ResearchController, register_protocol,
                       list_protocols, experiment_summaries, read_registry, promote, rollback)


class Command(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    action: Literal['load', 'scene', 'reset', 'pause', 'step', 'control', 'home', 'demo', 'stop', 'instruction', 'view', 'camera', 'speed', 'flags', 'gravity', 'drop']
    model: Literal['panda', 'so101', 'rm65_gaia'] | None = None
    scene_id: str | None = None
    task_id: str | None = None
    seed: int | None = Field(default=None, ge=0, le=4294967295)
    value: float | bool | None = None
    index: int | None = Field(default=None, ge=0, le=63)
    text: str | None = Field(default=None, max_length=160)
    view: Literal['perspective', 'top', 'front', 'wrist'] | None = None
    dx: float = Field(default=0, ge=-180, le=180)
    dy: float = Field(default=0, ge=-180, le=180)
    zoom: float = Field(default=0, ge=-1, le=1)
    pan_dx: float = Field(default=0, ge=-2, le=2)
    pan_dy: float = Field(default=0, ge=-2, le=2)
    flags: dict[Literal['contacts', 'joints', 'collision'], bool] | None = None

    @model_validator(mode='after')
    def required_fields(self):
        required = {'load': ['model'], 'scene': ['scene_id'], 'pause': ['value'], 'control': ['index', 'value'], 'instruction': ['text'],
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
        self.thread = Thread(target=self.run, name='MuJoCo', daemon=True)

    def run(self):
        simulation = None
        processed = []
        try:
            simulation = Simulation()
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
                        if payload['action'] == 'record_start':
                            if self.recorder is not None:
                                raise ValueError('已有任务正在记录轨迹。')
                            self.recorder = payload['log']
                            self.record_frames = 0
                            existing = self.recorder.artifacts.get('checkpoints', {}).get('initial')
                            reference = existing['path'] if existing else self.recorder.save_checkpoint(simulation.checkpoint())
                            self.recorder.write('recording_started', {'initial_checkpoint': reference,
                                'timestep': float(simulation.model.opt.timestep), 'model': simulation.model_id})
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
                        else:
                            result = simulation.command(payload)
                        processed.append((future, result))
                        if payload['action'] in ('reset', 'load', 'scene', 'restore', 'settle', 'pause', 'step'):
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
                    while accumulator >= simulation.model.opt.timestep and not simulation.paused:
                        simulation.step()
                        self.record_frame(simulation)
                        physics_advanced = True
                        accumulator -= simulation.model.opt.timestep
                # A paused, unchanged scene already has a valid frame. Avoid
                # repeatedly allocating/flipping full-size RGB images while idle.
                redraw = not self.jpeg or (not simulation.render_suspended and (processed or physics_advanced))
                jpeg = simulation.render() if redraw else self.jpeg
                if redraw: frame_count += 1
                if start - fps_start >= 1:
                    fps = frame_count / (start - fps_start)
                    frame_count, fps_start = 0, start
                state = simulation.state()
                state.update({'fps': round(fps, 1), 'render_ms': round((time.perf_counter()-start)*1000, 1), 'ready': True})
                with self.lock:
                    self.snapshot, self.jpeg = state, jpeg
                    if redraw: self.frame += 1
                self.ready.set()
                for future, result in processed:
                    if not future.done(): future.set_result(result)
                self.shutdown.wait(max(0, 1/30 - (time.perf_counter() - start)))
        except Exception as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
            with self.lock:
                self.snapshot = {**self.snapshot, 'ready': False, 'error': self.failure}
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
            raise HTTPException(503, self.failure or 'MuJoCo 服务未运行。')
        future = Future()
        self.queue.put((command, future))
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=60)
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


def research_active():
    return composition.snapshot().get('active') or research.state.get('active')


async def stop_research():
    await research.stop()
    await composition.stop()


@asynccontextmanager
async def lifespan(app):
    worker.thread.start()
    ready = await asyncio.to_thread(worker.ready.wait, 60)
    if not ready or worker.failure:
        worker.shutdown.set()
        raise RuntimeError(worker.failure or 'MuJoCo 初始化超时。')
    yield
    await stop_research()
    await comparison.stop()
    await policy.stop()
    await release_local_service()
    await agent.stop()
    worker.shutdown.set()
    await asyncio.to_thread(worker.thread.join, 5)


app = FastAPI(title='具身工坊 · MuJoCo', lifespan=lifespan)


def local_request(request):
    origin = request.headers.get('origin')
    if origin and origin not in ('http://127.0.0.1:5173', 'http://localhost:5173', 'http://127.0.0.1:8000', 'http://localhost:8000'):
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
    snapshot['registry'] = read_registry()
    return snapshot


@app.get('/api/models')
async def models():
    return [{'id': key, 'name': item['name'], 'arm_dof': item['arm_dof'],
             'hand_dof': item.get('hand_dof',1), 'note': item.get('model_note',''),
             'image': f'/api/models/{key}/image', 'source': f"{item['folder']}/{item['file']}"} for key, item in MODELS.items()]


@app.get('/api/scenes')
async def scenes():
    return {'scenes':[{'id':key,**value} for key,value in SCENES.items()], 'tasks':TASKS}


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
        if body.action in ('stop', 'reset', 'load', 'scene', 'pause'):
            await stop_research()
            await comparison.stop()
            await policy.stop()
            await agent.stop()
        elif (agent.state['active'] or policy.state['active'] or comparison.state['active'] or research_active()) and body.action not in ('view', 'camera', 'flags'):
            raise HTTPException(409, '自动控制正在运行，请先停止后再手动操作。')
        message = await worker.execute(body.model_dump(exclude_none=True))
        if body.action in ('reset', 'load', 'scene'): agent.reset_observation(); policy.reset_observation()
    return {'ok': True, 'message': message}


@app.get('/api/agent/config')
async def agent_config():
    return agent.public_config()


@app.post('/api/agent/config')
async def configure_agent(body: AgentConfig, request: Request):
    local_request(request)
    async with agent.lock:
        if comparison.state['active'] or research_active(): raise HTTPException(409, '实验运行中不能修改配置。')
        try: return agent.configure(body)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.post('/api/agent/start')
async def start_agent(body: AgentStart, request: Request):
    local_request(request)
    async with agent.lock:
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
        await worker.execute({'action': 'stop'})
        return agent.snapshot()


@app.post('/api/agent/observe')
async def observe_agent(request: Request):
    local_request(request)
    async with agent.lock:
        if agent.state['active'] or policy.state['active'] or comparison.state['active'] or research_active(): raise HTTPException(409, '自动控制中将自动更新观察。')
        return await agent.capture()


@app.get('/api/agent/camera/{name}')
async def agent_camera(name: Literal['front', 'side', 'wrist']):
    if not agent.observation: raise HTTPException(404, '请先采集一次三路观察。')
    camera = next(c for c in agent.observation['cameras'] if c['name'] == name)
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
        if agent.state['active'] or comparison.state['active'] or research_active(): raise HTTPException(409,'请先停止 Agent 或实验任务。')
        try: return await policy.start(body)
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc


@app.post('/api/policy/stop')
async def stop_policy(request: Request):
    local_request(request)
    async with agent.lock:
        await stop_research()
        if comparison.state['active']: await comparison.stop()
        await policy.stop(); await worker.execute({'action':'stop'})
        return policy.snapshot()


@app.get('/api/policy/camera/{name}')
async def policy_camera(name: Literal['front','side','wrist']):
    if not policy.observation: raise HTTPException(404,'尚未采集策略观察。')
    camera = next(c for c in policy.observation['cameras'] if c['name']==name)
    return Response(base64.b64decode(camera['image_base64']),media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.post('/api/comparison/start')
async def start_comparison(body: ComparisonStart, request: Request):
    local_request(request)
    async with agent.lock:
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


class RunRename(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(max_length=160)


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
    if agent.state['active'] or policy.state['active'] or comparison.state['active'] or research_active():
        raise HTTPException(409, '请先停止正在运行的控制或实验任务。')


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
        while not await request.is_disconnected() and not worker.failure:
            with worker.lock:
                frame_id, jpeg = worker.frame, worker.jpeg
            if frame_id != last and jpeg:
                last = frame_id
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n'
            await asyncio.sleep(1/30)
    return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})


if (ROOT / 'dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'dist', html=True), name='workspace')
