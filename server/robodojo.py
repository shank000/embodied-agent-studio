"""Official RoboDojo evaluation jobs and explicit external-result imports.

This benchmark owns its Isaac Sim lifecycle. It never runs in the Newton or
MuJoCo workbench, and preparing a plan does not claim a physical rollout.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .robodojo_catalog import catalog
from .robodojo_results import parse_result
from .runlog import RunLog, list_runs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/'vendor/robodojo'
RUNTIME_FILE = ROOT/'.venv-robodojo/runtime.json'
REVISION = 'f57e1633ffed59761bc8e568cfe7530c4b33381c'
SUBMODULES = {'XPolicyLab': '5d0be1499634d3176258c8f2e31f5fe816497b0a',
              'third_party/IsaacLab': 'afca7b09d60d8beb9c1cb28b43066499940b969b',
              'third_party/curobo': 'd17b54ce32cba095c0b000c4c58777075d11de0e'}
IDENTIFIER = r'^[A-Za-z][A-Za-z0-9_]{0,63}$'
UNREPORTED = {'status':'unavailable', 'total_calls':0, 'missing_calls':1,
    'call_count_incomplete':True,
    'warnings':['外部 XPolicyLab 策略未向工作台上报调用用量、推理耗时或费用；不按零费用计算。']}


class EvaluationPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    task_ids: list[str] = Field(default_factory=lambda:['stack_bowls'], min_length=1, max_length=54)
    seeds: list[int] = Field(default_factory=lambda:[0], min_length=1, max_length=3)
    embodiment: str = 'arx_x5'
    policy_name: str = Field(default='demo_policy', pattern=IDENTIFIER)
    policy_host: str = Field(default='127.0.0.1', min_length=1, max_length=253)
    policy_port: int = Field(default=9999, ge=1, le=65535, strict=True)
    checkpoint: str = Field(default='external', pattern=r'^[A-Za-z0-9_.-]{1,120}$')
    action_type: str = 'ee'
    episodes: int | str = 1
    mode: str = 'smoke'
    copy_videos: bool = False
    max_wall_time_s: int = Field(default=3600, ge=10, le=86400, strict=True)

    @field_validator('policy_host')
    @classmethod
    def valid_host(cls, value):
        try: ipaddress.ip_address(value)
        except ValueError:
            if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', value):
                raise ValueError('策略主机须为 IP 或 DNS 名，不接受 URL、命令或凭据。')
        return value

    @field_validator('seeds', 'episodes', mode='before')
    @classmethod
    def reject_coerced_counts(cls, value):
        if isinstance(value, list):
            if any(type(item) is not int for item in value): raise ValueError('种子须为整数。')
        elif value != 'native' and type(value) is not int: raise ValueError('回合数须为整数或 native。')
        return value

    @model_validator(mode='after')
    def supported_plan(self):
        known = {task['id'] for task in catalog()['tasks']}
        if len(set(self.task_ids)) != len(self.task_ids) or any(task not in known for task in self.task_ids):
            raise ValueError('请选择不重复的官方 RoboDojo 任务。')
        if len(set(self.seeds)) != len(self.seeds) or any(type(seed) is not int or seed not in (0,1,2) for seed in self.seeds):
            raise ValueError('官方评测种子为 0、1、2，请勿重复。')
        if self.embodiment != 'arx_x5' or self.action_type not in ('ee','joint') or self.mode not in ('smoke','benchmark'):
            raise ValueError('当前官方配置为 arx_x5，动作支持 ee / joint，运行模式支持 smoke / benchmark。')
        if self.episodes != 'native' and (type(self.episodes) is not int or not 1 <= self.episodes <= 50):
            raise ValueError('回合数须为 1–50 的整数，或 native。')
        return self


class ResultImport(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    task_id: str
    seed: int = Field(ge=0, le=2, strict=True)
    policy: str = Field(min_length=1, max_length=120)
    embodiment: str = 'arx_x5'
    payload: dict
    artifact_dir: str | None = None
    copy_videos: bool = False


def git_revision(path):
    if not (Path(path)/'.git').exists(): return None
    try:
        result = subprocess.run(['git','-C',str(path),'rev-parse','HEAD'], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired): return None


def runtime_config():
    if not RUNTIME_FILE.is_file(): return None
    config = json.loads(RUNTIME_FILE.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or set(config)-{'kind','source_root','python'}:
        raise ValueError('RoboDojo runtime.json 仅接受 kind、source_root、python。')
    if config.get('kind') != 'native': raise ValueError('当前执行器使用原生 Linux 环境；远程服务器请部署工作台或导出评测计划。')
    for key in ('source_root','python'):
        if not isinstance(config.get(key),str) or not Path(config[key]).is_absolute():
            raise ValueError('RoboDojo 运行目录与 Python 须为本机绝对路径。')
    return config


def readiness():
    checks=[]; config=None
    def add(key,label,status,detail): checks.append({'id':key,'label':label,'status':status,'detail':detail})
    try: config=runtime_config()
    except (ValueError,OSError) as exc: add('runtime','独立运行配置','missing',str(exc))
    source=Path(config['source_root']) if config else DEFAULT_SOURCE
    revision=git_revision(source)
    add('source','官方源码','passed' if revision==REVISION else 'missing', revision or '运行 tools/setup-robodojo.py 下载固定版本源码。')
    add('runtime','独立 RoboDojo 运行时','passed' if config else 'missing',str(RUNTIME_FILE) if config else '尚未配置完整 Isaac Sim 5.1 运行时；现有 Newton 不替代它。')
    add('platform','仿真运行平台','passed' if sys.platform.startswith('linux') else 'missing','官方 Linux / Ubuntu 运行栈；本机仍可准备计划和导入结果。')
    for name,expected in SUBMODULES.items():
        actual=git_revision(source/name)
        add(name,name,'passed' if actual==expected else 'missing',actual or f'缺少固定提交 {expected}')
    for name in ('Robots','Object','Material','Eval_Layout'):
        path=source/'Assets'/name
        present=path.is_dir() and next(path.iterdir(),None) is not None
        add('assets_'+name,'资产 '+name,'passed' if present else 'missing','已找到目录；首次原生加载仍需验证所选任务引用。' if present else '未下载必要仿真资产；无需下载训练数据。')
    if config and sys.platform.startswith('linux'):
        python=Path(config['python'])
        if python.is_file():
            probe='import json,sys;from importlib.metadata import version,PackageNotFoundError\nd={}\nfor p in ("isaacsim","torch"):\n try:d[p]=version(p)\n except PackageNotFoundError:d[p]=None\nprint(json.dumps(d))'
            try:
                result=subprocess.run([str(python),'-c',probe],capture_output=True,text=True,timeout=15)
                packages=json.loads(result.stdout) if result.returncode==0 else {}
                valid=str(packages.get('isaacsim','')).startswith('5.1.0') and str(packages.get('torch','')).startswith('2.7.0')
                add('packages','Isaac Sim / Torch 版本','passed' if valid else 'missing',json.dumps(packages) if packages else result.stderr[-1000:])
            except (OSError,ValueError,subprocess.TimeoutExpired) as exc: add('packages','SDK 元数据','missing',str(exc))
        else: add('packages','SDK Python','missing','独立 Python 不存在。')
    else: add('packages','SDK 元数据','missing','待在独立 Linux 运行环境核验，无 SDK 导入或许可自动接受。')
    ready=all(item['status']=='passed' for item in checks)
    return {'ready':ready,'checks':checks,'message':'静态依赖检查通过，原生加载与任务结果以实际运行日志为准。' if ready else '等待完整 RoboDojo Linux GPU 环境；任务目录、计划和结果导入可用。',
            'source_root':str(source),'revision':revision,'runtime':{'configured':bool(config),'kind':config.get('kind') if config else None}}


def plan(spec):
    spec=spec if isinstance(spec,EvaluationPlan) else EvaluationPlan.model_validate(spec)
    status=readiness(); source=Path(status['source_root'])
    commands=[]
    for task in spec.task_ids:
        for seed in spec.seeds:
            args=['bash',str(source/'scripts/robodojo.sh'),'client','--task',task,'--env-cfg',spec.embodiment,
                  '--policy-name',spec.policy_name,'--policy-host',spec.policy_host,'--policy-port',str(spec.policy_port),
                  '--ckpt',spec.checkpoint,'--action-type',spec.action_type,'--seed',str(seed),'--eval-num',str(spec.episodes)]
            commands.append({'task_id':task,'seed':seed,'argv':args,'cwd':str(source)})
    data={'benchmark_id':'robodojo','revision':REVISION,'settings':spec.model_dump(),'commands':commands,
          'ready':status['ready'],'warnings':['smoke 通过仅表示执行与产物有效，不表示机器人完成任务。',
          '本地结果不等于经过官方云端反作弊的榜单成绩。','策略使用 XPolicyLab 原生 WebSocket；工作台 HTTP/Codex 配置不会自动转换。'],
          'official_protocol':{'base_tasks':42,'runnable_configurations':54,'seeds':[0,1,2],
              'generalization':'25 standard + 25 random per seed','other_tasks':50,
              'overall':'五维等权平均，跨三个种子汇总；单任务不报告官方总分。'}}
    data['plan_id']='robodojo-plan-'+hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:20]
    return data


def _external_log(kind, metadata):
    log=RunLog(kind,{**metadata,'external_telemetry':True})
    # An external policy was not instrumented by our inference wrapper. Empty
    # local call events cannot establish zero tokens or zero financial cost.
    log.metadata['telemetry_schema_version']='robodojo-external-policy-v1'
    log.metadata['external_telemetry']=True
    log._call_costs.append(UNREPORTED.copy())
    return log


def _metrics(result):
    summary=result['summary']
    return {'success':None,'episodes':summary['eval_time'],'success_rate':summary['success_rate'],
            'score':summary['score'],'sim_time_s':None,'inference_s':None,'inference_count':None,
            'benchmark_scope':result['coverage']['scope']}


def import_result(request):
    request=request if isinstance(request,ResultImport) else ResultImport.model_validate(request)
    result=parse_result(request.payload,request.task_id,request.seed,request.policy,request.embodiment)
    raw=json.dumps(request.payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(raw)>10*1024*1024: raise ValueError('单份结果 JSON 超过 10 MiB，请按任务和种子分别导入。')
    directory=None; videos=[]
    if request.artifact_dir:
        directory=Path(request.artifact_dir).expanduser().resolve(strict=True)
        if not directory.is_dir(): raise ValueError('原生结果路径须为目录。')
        original=directory/'_result.json'
        if not original.is_file() or original.is_symlink(): raise ValueError('结果目录须包含原始 _result.json。')
        if json.loads(original.read_text(encoding='utf-8')) != request.payload:
            raise ValueError('结果目录中的 _result.json 与本次导入不一致。')
        if request.copy_videos:
            for path in sorted(directory.glob('episode_*')):
                if path.suffix.lower() not in ('.mp4','.avi','.mkv','.webm'): continue
                if path.is_symlink() or not path.resolve().is_relative_to(directory): raise ValueError('原生视频不得使用越界链接。')
                if path.stat().st_size>2*1024**3: raise ValueError('单个视频超过 2 GiB，请先单独归档或压缩。')
                videos.append(path)
            if sum(path.stat().st_size for path in videos)>8*1024**3: raise ValueError('本次视频超过 8 GiB，请分批导入。')
            if shutil.disk_usage(ROOT).free < sum(path.stat().st_size for path in videos)+256*1024**2:
                raise ValueError('磁盘空间不足以保留所选原生视频。')
    elif request.copy_videos: raise ValueError('复制视频需要提供对应的本机原生结果目录。')
    log=_external_log('benchmark',{'model':request.policy,'task':request.task_id,'robot':request.embodiment,
        'seed':request.seed,'backend':{'engine_id':'isaacsim','engine_version':'5.1.0',
            'evaluator_source':'robodojo_imported_result','evaluator_version':REVISION},
        'benchmark':{k:v for k,v in result.items() if k!='raw'},'record_origin':'external_import',
        'source_sha256':hashlib.sha256(raw).hexdigest()})
    try:
        folder=log.folder/'benchmark';folder.mkdir()
        (folder/'_result.json').write_bytes(raw)
        native=[]
        for video in videos:
            target=folder/'videos'/video.name;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(video,target)
            camera=next((name for name in ('head','left_wrist','right_wrist') if '_cam_'+name in video.name),'combined')
            native.append({'name':video.name,'camera':camera,'path':target.relative_to(log.folder).as_posix(),
                'url':f'/api/runs/{log.id}/artifact/'+target.relative_to(log.folder).as_posix()})
        log.artifacts['native_videos']=native
        log.write('benchmark_result',result)
        log.finish({'status':'imported','message':'已导入官方格式的本地结果；未认证为官方榜单成绩。','metrics':_metrics(result)})
        return {'run_id':log.id,'benchmark':log.metadata['benchmark'],'metrics':_metrics(result),'native_videos':native}
    except Exception as exc:
        log.write('error',{'message':str(exc)})
        log.finish({'status':'error','message':str(exc),'metrics':{'success':None}})
        raise


class RoboDojoController:
    def __init__(self):
        self.state={'active':False,'status':'idle','message':'准备官方评测计划，或导入已完成的原生结果。'}
        self.task=None;self.process=None;self.log=None;self.start_generation=0

    def snapshot(self):
        timing=self.log.timing_snapshot().get('timing') if self.log else None
        return {**self.state,**({'timing':timing} if timing else {})}

    def info(self):
        history=[]
        for row in list_runs(limit=None):
            if row.get('kind')!='benchmark': continue
            history.append({key:row.get(key) for key in ('run_id','display_name','status','model','task','seed','metrics','benchmark','message')})
            history[-1]['native_videos']=(row.get('artifacts') or {}).get('native_videos',[])
            if len(history)>=100: break
        return {**catalog(),'readiness':readiness(),'state':self.snapshot(),'history':history}

    async def start(self,spec):
        if self.state['active']: raise ValueError('已有 RoboDojo 评测正在运行。')
        specification=EvaluationPlan.model_validate(spec)
        self.start_generation+=1;generation=self.start_generation
        self.log=None;self.task=None
        self.state={'active':True,'status':'preparing','message':'正在检查官方运行环境，尚未启动客户端。'}
        try:
            proposed=await asyncio.to_thread(plan,specification)
            if generation!=self.start_generation: return self.snapshot()
            if not proposed['ready']: raise ValueError('RoboDojo 原生运行环境尚未就绪。请先配置独立 Linux GPU 环境；当前仅准备计划和导入结果。')
        except BaseException as exc:
            if generation==self.start_generation:
                self.state.update(active=False,status='error',message=str(exc) or '启动检查已取消。')
            raise
        try:
            config=runtime_config();source=Path(config['source_root'])
            if not (source/'XPolicyLab/policy'/specification.policy_name/'deploy.py').is_file():
                raise ValueError('所选策略缺少官方 XPolicyLab deploy.py 适配器。')
            self.log=_external_log('benchmark',{'model':specification.policy_name,'task':','.join(specification.task_ids),
                'settings':specification.model_dump(),'benchmark':{'benchmark_id':'robodojo','revision':REVISION,'mode':specification.mode},
                'record_origin':'native_job','backend':{'engine_id':'isaacsim','engine_version':'5.1.0',
                    'evaluator_source':'robodojo_native_result','evaluator_version':REVISION}})
            (self.log.folder/'benchmark').mkdir()
            (self.log.folder/'benchmark/plan.json').write_text(json.dumps(proposed,ensure_ascii=False,indent=2),encoding='utf-8')
        except Exception as exc:
            self.state.update(active=False,status='error',message=str(exc))
            if self.log: self.log.finish({'status':'error','message':str(exc)})
            raise
        self.state={'active':True,'status':'running','message':'正在运行官方 RoboDojo 评测客户端。',
                    'run_id':self.log.id,'rows':[]}
        self.task=asyncio.create_task(self.run(specification,config))
        return self.snapshot()

    async def _terminate(self):
        process=self.process
        if process is None or process.returncode is not None: return
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError: return
        try: await asyncio.wait_for(process.wait(),10)
        except TimeoutError:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            await process.wait()

    async def stop(self):
        self.start_generation+=1
        if self.state.get('status')=='preparing':
            self.state.update(active=False,status='stopped',message='启动检查已取消，未创建原生评测进程。')
        if self.task and not self.task.done():
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
        self.task=None

    async def run(self,spec,config):
        source=Path(config['source_root']);log=self.log;started=time.monotonic()
        try:
            for task in spec.task_ids:
                for seed in spec.seeds:
                    native_id=datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')+'_'+uuid4().hex[:10]
                    additional=f'ckpt_name={spec.checkpoint},action_type={spec.action_type}'
                    host=f'[{spec.policy_host}]' if ':' in spec.policy_host else spec.policy_host
                    args=[config['python'],'-u',str(source/'src/eval_client/main.py'),'--task_name',task,
                          '--env_cfg_type',spec.embodiment,'--num_envs','1','--enable_cameras','--headless',
                          '--kit_args','--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera',
                          '--device_id','0','--policy_name',spec.policy_name,'--port',str(spec.policy_port),
                          '--host',spec.policy_host,'--protocol','ws','--policy_server_url',f'ws://{host}:{spec.policy_port}',
                          '--additional_info',additional,'--seed',str(seed)]
                    env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONPATH':os.pathsep.join([str(source),str(source/'XPolicyLab')]),
                         'ROBODOJO_RUN_ID':native_id,'EVAL_NUM':str(spec.episodes),'CUDA_VISIBLE_DEVICES':'0'}
                    # No license-acceptance variables are introduced here.
                    output=log.folder/'benchmark'/f'{task}-{seed}.log'
                    with output.open('wb') as stream:
                        self.process=await asyncio.create_subprocess_exec(*args,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        await asyncio.wait_for(self.process.wait(),max(1,spec.max_wall_time_s-(time.monotonic()-started)))
                    if self.process.returncode:
                        raise RuntimeError(f'官方客户端退出 {self.process.returncode}；完整诊断见 benchmark/{output.name}。')
                    result_path=source/'eval_result/RoboDojo'/task/spec.policy_name/spec.embodiment/f'{seed}_{additional}'/native_id/'_result.json'
                    if not result_path.is_file(): raise RuntimeError('官方客户端未生成本次调用的 _result.json；不能记为成功。')
                    imported=await asyncio.to_thread(import_result,ResultImport(task_id=task,seed=seed,policy=spec.policy_name,
                        embodiment=spec.embodiment,payload=json.loads(result_path.read_text(encoding='utf-8')),artifact_dir=str(result_path.parent),copy_videos=spec.copy_videos))
                    self.state['rows'].append(imported);log.write('benchmark_trial',imported)
            self.state.update(status='complete',message='官方客户端执行结束；策略 SR、过程得分与覆盖率请查看逐任务结果。')
        except asyncio.CancelledError:
            self.state.update(status='stopped',message='评测已停止；已归档结果和原生恢复文件保留。');raise
        except TimeoutError:
            message=f'评测超过 {spec.max_wall_time_s} 秒墙钟预算；客户端将停止，已完成结果和原生日志保留。'
            self.state.update(status='error',message=message);log.write('error',{'message':message})
        except Exception as exc:
            self.state.update(status='error',message=str(exc));log.write('error',{'message':str(exc)})
        finally:
            await self._terminate();self.process=None;self.state['active']=False
            log.finish({'status':self.state['status'],'message':self.state['message'],'rows':self.state['rows'],
                        'metrics':{'success':None,'wall_time_s':round(time.monotonic()-started,3),'inference_count':None,'inference_s':None}})
