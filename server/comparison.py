"""Paired method trials, restoring the same MuJoCo integration state per seed."""
import asyncio
import time
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent import AgentStart, validate_agent_access
from .policy import PolicyStart, release_local_service
from .runlog import RunLog, run_folder
from .telemetry import batch_timing
from .scenes import SCENES, TASKS
from .simulation import MODELS
from .simulators import validate_selection, supported_catalog


class ComparisonStart(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    robot: str = 'so101'
    engine_id: str = 'mujoco'
    scene_id: str = 'tabletop'
    task_id: str = 'lift_red'
    seeds: list[int] = Field(default_factory=lambda:[42], min_length=1, max_length=10)
    methods: list[Literal['agent','smolvla','vla','wam']] = Field(default_factory=lambda:['agent','smolvla'], min_length=2, max_length=4)
    max_sim_time_s: float = Field(default=30, ge=1, le=300)
    record_video: bool = False

    @model_validator(mode='after')
    def valid_trial(self):
        validate_selection(self.engine_id, self.robot, self.scene_id, self.task_id)
        if len(set(self.methods)) != len(self.methods): raise ValueError('对比方法不能重复。')
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 or seed > 2**32-1 for seed in self.seeds): raise ValueError('种子应为不重复的 0–4294967295 整数。')
        return self


class ComparisonController:
    def __init__(self, worker, agent, policy):
        self.worker, self.agent, self.policy = worker, agent, policy
        self.task = None
        self._active_trial = None
        self.state = {'active':False, 'status':'idle', 'message':'选择同一任务下要比较的方法。', 'rows':[]}

    def snapshot(self):
        children = []
        trial = self._active_trial
        if trial and trial['controller'].task is not None and trial['controller'].task is not trial['previous_task']:
            child = trial['controller'].snapshot()
            if child.get('run_id') and child['run_id'] != trial['previous_run_id']:
                trial['row']['run_id'] = child['run_id']
                children.append(child)
        return batch_timing(self.state, getattr(self, 'run_log', None), children)

    async def start(self, request):
        if self.state['active']: raise ValueError('对比实验正在运行。')
        with self.worker.lock: engine = (self.worker.snapshot.get('backend') or {}).get('engine_id', 'mujoco')
        if engine in ('alfworld','ai2thor'):
            raise ValueError('当前方法对比使用连续机械臂动作协议；文字和离散环境请先使用 Agent 单次实验。')
        if engine != request.engine_id: raise ValueError('请先切换到本次对比指定的仿真后端。')
        for method in request.methods:
            if method != 'agent': self.policy.validate_start(method, request.robot)
            else: validate_agent_access(self.agent.config)
        rows = [{'method': method, 'seed': seed, 'initial_state_id': '', 'status': 'queued',
                 'message': '等待运行。', 'run_id': None, 'metrics': None}
                for seed in request.seeds for method in request.methods]
        self.state = {'active':True, 'status':'preparing', 'message':'准备共享初始场景。', 'rows':rows, 'request':request.model_dump()}
        self.task = asyncio.create_task(self.run(request))
        return self.state

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
        self.task = None

    async def run(self, request):
        with self.worker.lock: backend = self.worker.snapshot.get('backend')
        instruction = supported_catalog(request.engine_id)[2][request.task_id]['instruction']
        log = RunLog('comparison', {'task':instruction, 'settings':request.model_dump(),
                                   'backend': backend,
                                   'rows': self.state['rows'], 'primary_models': [
                                       self.agent.config.model if method == 'agent' else self.policy.configs[method].model
                                       for method in request.methods]})
        self.run_log = log
        log.snapshot_provider = self.snapshot
        self.state['run_id'] = log.id; started = time.monotonic()
        self.policy.retain_local_service = 'smolvla' in request.methods
        active_trial = None
        try:
            await self.worker.execute({'action':'stop'})
            if 'smolvla' in request.methods:
                self.state['message'] = '预加载 SmolVLA；加载时间不计入各方法的试验耗时。'
                await self.policy.warmup()
            for seed in request.seeds:
                await self.worker.execute({'action':'load','model':request.robot,'scene_id':request.scene_id,'task_id':request.task_id,'seed':seed})
                await self.worker.execute({'action':'settle'})
                checkpoint = await self.worker.execute({'action':'checkpoint'})
                log.write('initial_state', checkpoint)
                for method in request.methods:
                    await self.worker.execute({'action':'restore','checkpoint':checkpoint})
                    self.agent.reset_observation(); self.policy.reset_observation()
                    row = next(row for row in self.state['rows'] if row['method'] == method and row['seed'] == seed)
                    row.update(initial_state_id=checkpoint['state_id'], status='running')
                    self.state.update(status='running',message=f'种子 {seed} · 正在运行 {method}。')
                    child = self.agent if method == 'agent' else self.policy
                    active_trial = {'row': row, 'controller': child, 'previous_task': child.task,
                                    'previous_run_id': child.snapshot().get('run_id')}
                    self._active_trial = active_trial
                    if method == 'agent':
                        await self.agent.start(AgentStart(task=instruction,
                            max_sim_time_s=request.max_sim_time_s,evaluator_stop=True,record_video=request.record_video))
                        await self.agent.task
                        result = self.agent.snapshot()
                    else:
                        await self.policy.start(PolicyStart(kind=method,task=instruction,
                            max_sim_time_s=request.max_sim_time_s,record_video=request.record_video))
                        await self.policy.task
                        result = self.policy.snapshot()
                    row.update(status=result['status'],message=result['message'],run_id=result.get('run_id'),
                               metrics=result.get('metrics'),video=result.get('video'))
                    if row['run_id']:
                        with self.worker.lock: jpeg = self.worker.jpeg
                        (run_folder(row['run_id'])/'final.jpg').write_bytes(jpeg)
                    log.write('trial_result', row)
                    active_trial = None
                    self._active_trial = None
            self.state.update(status='complete',message='对比运行完成；成功由同一场景评价器判定。')
        except asyncio.CancelledError:
            self.state.update(status='stopped',message='对比已停止，已完成的试验结果保留。')
            for row in self.state['rows']:
                if row['status'] == 'running': row.update(status='stopped',message='用户停止。')
            raise
        except Exception as exc:
            message = str(getattr(exc,'detail',exc)) or type(exc).__name__
            self.state.update(status='error',message=message[:2000]); log.write('error',{'message':message[:2000]})
            for row in self.state['rows']:
                if row['status'] == 'running': row.update(status='error',message=message[:2000])
        finally:
            # start() may be cancelled at its first await while the controller still
            # exposes the previous trial. Bind recovery to a newly created task.
            cancelled_trial = active_trial if self.state['status'] == 'stopped' else None
            recover_child = (cancelled_trial is not None and cancelled_trial['controller'].task is not None and
                             cancelled_trial['controller'].task is not cancelled_trial['previous_task'])
            await self.agent.stop(); await self.policy.stop()
            if recover_child:
                result = cancelled_trial['controller'].snapshot()
                run_id = result.get('run_id')
                if run_id and run_id != cancelled_trial['previous_run_id']:
                    row = cancelled_trial['row']
                    row.update(status=result['status'], message=result['message'], run_id=run_id,
                               metrics=result.get('metrics'), video=result.get('video'))
                    with self.worker.lock: jpeg = self.worker.jpeg
                    (run_folder(run_id)/'final.jpg').write_bytes(jpeg)
                    log.write('trial_result', row)
            for row in self.state['rows']:
                if row['status'] in ('queued', 'running'):
                    row.update(status='stopped' if self.state['status'] == 'stopped' else 'error', message='实验未完成该次试验。')
            self.policy.retain_local_service = False
            try:
                released = await release_local_service()
                log.write('resources', {'local_service_released':released})
            except Exception as exc:
                self.state.update(status='error',message=f'本地策略进程未能释放：{exc}')
                log.write('error',{'phase':'cleanup','message':self.state['message']})
            try: await self.worker.execute({'action':'stop'})
            except Exception: pass
            self.state['active'] = False
            self._active_trial = None
            log.finish({'status':self.state['status'],'message':self.state['message'],'rows':self.state['rows'],
                        'timing': self.snapshot()['timing'],
                        'wall_time_s':round(time.monotonic()-started,3)})
