"""Bounded recovery at inference boundaries; never replay physical commands."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from uuid import uuid4

from .telemetry import RECOVERY_CONTEXT


async def invoke_with_recovery(owner, phase, operation, observation, config):
    coordinator = getattr(owner, 'recovery', None)
    if coordinator is None: return await operation('', config)
    return await coordinator.call(owner, phase, operation, observation, config)


class RecoveryCoordinator:
    def __init__(self, repair, *, secrets=lambda: ()):
        self.repair = repair
        self.secrets = secrets
        self.current = None
        self.view = None
        self._mutex = asyncio.Lock()
        repair.recovery_snapshot = self.snapshot

    def snapshot(self): return deepcopy(self.view)

    @staticmethod
    def _check(owner, observation, run_id):
        if asyncio.current_task().cancelling() or not getattr(owner,'_accepting',True):
            raise asyncio.CancelledError('实验已请求停止，迟到的恢复或模型输出不会执行。')
        with owner.worker.lock: state = owner.worker.snapshot.copy()
        if not owner.state.get('active') or owner.state.get('run_id') != run_id:
            raise asyncio.CancelledError('恢复所属实验已停止或被替换。')
        if not state.get('ready'):
            raise RuntimeError(state.get('error', 'MuJoCo 未就绪，无法恢复当前轮次。'))
        if (state.get('generation') != observation.get('generation') or
                abs(state.get('time', -1)-observation.get('time', -2)) > 1e-7 or
                state.get('active_robot', 'arm1') != observation.get('active_robot', 'arm1')):
            raise RuntimeError('故障暂停期间场景或机械臂状态已改变，未执行旧观察的动作。')

    def resume(self, failure_id):
        current = self.current
        if not current or current['id'] != failure_id or not self.view.get('can_resume'):
            raise ValueError('该故障已结束或当前不能恢复，请刷新恢复状态。')
        self._check(current['owner'], current['observation'], current['run_id'])
        if self.repair.state.get('restart_required'):
            raise ValueError('源码修复尚未重启生效，不能用旧适配器假装恢复。')
        self.view.update(can_resume=False, message='已接受恢复请求，准备重试当前阶段。')
        current['resume'].set()
        return self.repair.snapshot()

    @staticmethod
    def _settings(config, plan):
        changes = (plan.get('settings') or {})
        changes = {key:value for key,value in changes.items() if value is not None}
        if not changes: return config, {}
        if config is None: raise ValueError('该阶段不支持修改运行参数。')
        unknown = changes.keys()-type(config).model_fields.keys()
        if unknown: raise ValueError('当前策略不支持这些恢复参数：'+', '.join(sorted(unknown)))
        updated = type(config).model_validate({**config.model_dump(), **changes})
        if getattr(updated, 'reasoning_native', False) and getattr(updated, 'reasoning', None) is not None:
            from .model_capabilities import cached_model_capabilities
            capability = cached_model_capabilities(updated)
            if capability.get('levels') is not None and updated.reasoning not in capability['levels']:
                raise ValueError('恢复方案的推理等级不属于该模型的原生等级。')
        return updated, changes

    async def call(self, owner, phase, operation, observation, config):
        scope = {'locked':False}
        try:
            return await self._call(owner,phase,operation,observation,config,scope)
        except BaseException as exc:
            if self.current and self.current['task'] is asyncio.current_task():
                if isinstance(exc,asyncio.CancelledError):
                    self.view.update(status='stopped',can_resume=False,message='实验已停止，故障恢复已取消。')
                elif self.view['status'] != 'exhausted':
                    from .repair import redact
                    self.view.update(status='failed',can_resume=False,
                        message=redact(str(exc) or type(exc).__name__,self.secrets()))
            raise
        finally:
            if self.current and self.current['task'] is asyncio.current_task(): self.current=None
            if scope['locked']: self._mutex.release()

    async def _call(self, owner, phase, operation, observation, config, scope):
        from .repair import RepairStart, redact
        log = getattr(owner, 'run_log', None) or getattr(owner, 'log', None)
        run_id = owner.state['run_id']
        if getattr(owner, '_recovery_config_run', None) != run_id:
            owner._recovery_config_run = run_id
            owner._recovery_configs = {}
        config = owner._recovery_configs.get(phase,config)
        feedback = ''
        attempt = 0
        recovery_id = None
        while True:
            self._check(owner, observation, run_id)
            token = RECOVERY_CONTEXT.set({'attempt':attempt, 'recovery_id':recovery_id})
            try:
                result = await operation(feedback, config)
                self._check(owner, observation, run_id)
                if recovery_id:
                    self.view.update(status='recovered', can_resume=False, message='当前阶段已恢复，实验继续执行。')
                    owner.state['recovery'] = self.snapshot()
                    log.write('recovery_complete', self.view)
                return result
            except asyncio.CancelledError:
                if recovery_id and self.view:
                    self.view.update(status='stopped', can_resume=False, message='实验已停止，不再重试或执行迟到输出。')
                raise
            except Exception as exc:
                self._check(owner, observation, run_id)
                if not self.repair.config.auto_analyze: raise
                if not scope['locked']:
                    await self._mutex.acquire()
                    scope['locked'] = True
                    self._check(owner, observation, run_id)
                used = owner.state.get('recovery_attempts', 0)
                budget = owner.state.setdefault('recovery_budget', self.repair.config.max_recovery_attempts)
                if used >= budget:
                    if self.view:
                        self.view.update(status='exhausted', can_resume=False, message='本次实验恢复预算已耗尽。')
                    log.write('recovery_exhausted', {'phase':phase,'attempts_used':used,'max_attempts':budget})
                    detail=redact(str(getattr(exc,'detail',exc)) or type(exc).__name__,(*self.secrets(),*log.secrets))
                    raise RuntimeError(f'本次实验恢复预算已耗尽（{used}/{budget}）；最后错误：{detail}') from exc
                if self.current is not None and self.current['owner'] is not owner:
                    raise RuntimeError('另一个修复任务仍占用恢复流程。') from exc
                await owner.worker.execute({'action':'pause','value':True})
                self._check(owner, observation, run_id)
                recovery_id = uuid4().hex
                used += 1
                owner.state['recovery_attempts'] = used
                message = redact(str(getattr(exc,'detail',exc)) or type(exc).__name__, self.secrets())[:6000]
                round_number = owner.state.get('round', owner.state.get('step', 0))
                self.view = {'failure_id':recovery_id, 'owner':log.metadata.get('kind'), 'source_run_id':run_id,
                    'round':round_number, 'phase':phase, 'status':'diagnosing', 'attempts_used':used,
                    'max_attempts':budget, 'can_resume':False, 'message':'当前轮次已暂停，正在诊断：'+message}
                self.current = {'id':recovery_id,'owner':owner,'task':asyncio.current_task(),
                                'observation':observation,'run_id':run_id,
                                'resume':asyncio.Event()}
                live_before = getattr(owner,'config',None)
                if live_before is None and hasattr(owner,'configs') and config is not None:
                    live_before = owner.configs.get(getattr(config,'kind',None))
                live_before = live_before.model_dump() if live_before is not None else None
                RECOVERY_CONTEXT.set({'attempt':attempt+1,'recovery_id':recovery_id,'recovery_stage':'diagnosis'})
                owner.state.update(status='repairing',message=self.view['message'],recovery=self.snapshot())
                log.metadata['recovery_assisted'] = True
                log.write('recoverable_error', {**self.view,'error':message,'attempt':attempt,
                    'observation_time':observation['time'],'action_submitted':False})
                plan = None
                try:
                    try:
                        async with self.repair.lock:
                            await self.repair.start(RepairStart(source_run_id=run_id, description=
                                f'Live experiment paused before submitting an action. Failure phase: {phase}. '
                                f'Round: {round_number}. Authorized remaining recovery attempts: {budget-used+1}. '
                                'Return a structured recovery plan. This is a new inference, not a replay of a motion. '
                                'The current stage has not submitted a physical action. Original error: '+message),
                                secrets=self.secrets(), parent_log=log)
                        try: await self.repair.task
                        except asyncio.CancelledError:
                            if asyncio.current_task().cancelling(): raise
                        proposal = self.repair.state.get('proposal') or {}
                        plan = proposal.get('recovery')
                        if proposal.get('edits'):
                            plan = None
                            notice = '已提出源码候选；请先停止实验，验证并应用修复、重启后端。当前适配器尚未更新，未自动重试。'
                        elif plan and plan.get('action') == 'retry':
                            config, changes = self._settings(config, plan)
                            notice = '诊断已给出恢复方案：'+plan.get('instruction','')
                            if changes: notice += ' 本次参数修正：'+json.dumps(changes,ensure_ascii=False)
                            log.write('recovery_plan',{'failure_id':recovery_id,'instruction':plan.get('instruction'),
                                'settings_delta':changes,'effective_settings':config.model_dump(exclude={'api_key'}) if config else None})
                        else:
                            notice = '诊断未给出可自动执行的恢复方案。请查看建议、修正配置后手动重试，或停止实验。'
                    except asyncio.CancelledError: raise
                    except Exception as repair_error:
                        plan = None
                        notice = '修复分析失败：'+redact(str(repair_error) or type(repair_error).__name__,self.secrets())
                        log.write('recovery_diagnosis_error',{'failure_id':recovery_id,'message':notice})
                    self._check(owner, observation, run_id)
                    if not plan or plan.get('action') != 'retry' or not self.repair.config.auto_recover:
                        has_edits = bool((self.repair.state.get('proposal') or {}).get('edits'))
                        self.view.update(status='waiting',can_resume=not has_edits,message=notice)
                        owner.state.update(status='repair_paused',message=notice,recovery=self.snapshot())
                        log.write('recovery_wait',self.view)
                        await self.current['resume'].wait()
                        # Explicit configuration changes made by the user while
                        # paused affect standalone controllers, not immutable recipes.
                        live = getattr(owner,'config',None)
                        if live is None and hasattr(owner,'configs') and config is not None:
                            live = owner.configs.get(getattr(config,'kind',None))
                        if live is not None and live.model_dump() != live_before:
                            config = live.model_copy(deep=True)
                            key = config.api_key.get_secret_value()
                            if key and key not in log.secrets: log.secrets.append(key)
                        log.write('recovery_manual_settings',config.model_dump(exclude={'api_key'}) if config else {})
                    self._check(owner, observation, run_id)
                    if self.repair.state.get('restart_required'): raise RuntimeError('源码变更尚未重启生效，未重试旧适配器。')
                    feedback = ('Previous attempt failed before action submission: '+message+
                        '\nCorrection: '+((plan or {}).get('instruction') or 'Return a valid result matching the exact provided schema; do not repeat an invalid or empty output.'))
                    attempt += 1
                    owner._recovery_configs[phase] = config
                    self.view.update(status='retrying',can_resume=False,message='正在重试当前轮次的失败阶段，不重复提交旧动作。')
                    owner.state.update(status='retrying',message=self.view['message'],recovery=self.snapshot())
                    log.write('recovery_retry',{**self.view,'attempt':attempt})
                finally:
                    if asyncio.current_task().cancelling():
                        self.view.update(status='stopped',can_resume=False,message='实验已停止，故障恢复已取消。')
            finally: RECOVERY_CONTEXT.reset(token)
