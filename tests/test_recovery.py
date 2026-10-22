"""Live-run recovery with real local HTTP; no external model or credentials."""
import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
from threading import Lock, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from server.agent import AgentConfig, api_action
from server.recovery import RecoveryCoordinator, invoke_with_recovery
from server.repair import EDITABLE, RepairConfig, RepairController
from server.runlog import CURRENT_LOG, RunLog


ACTION={'status':'done','frame':'world','position_m':[.4,0.,.5],
        'quaternion_wxyz':[1.,0.,0.,0.],'gripper_open':1.,'duration_s':1.,
        'explanation':'本地协议验证','hand_joint_positions_rad':None,'robot_id':None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append((self.path,body))
        if self.path=='/repair':
            response={'protocol':'embodied-agent-v1','model':'fixture-coding-model',
                'output':{'diagnosis':'本轮返回空白 JSON，尚未执行动作。','category':'provider',
                    'evidence':['finish_reason=stop, content=whitespace'],'recommendations':['纠正输出契约后重试。'],
                    'edits':[], 'recovery':self.server.plan},'usage':{'input_tokens':10,'output_tokens':5},
                'debug_echo':getattr(self.server,'secret_echo','')}
        else:
            self.server.agent_calls+=1
            invalid=self.server.always_bad or self.server.agent_calls==1
            response={'id':'fixture','object':'chat.completion','created':0,'model':body['model'],
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant',
                    'content':'   ' if invalid else json.dumps(ACTION)}}],
                'usage':{'prompt_tokens':20,'completion_tokens':8,'total_tokens':28}}
        content=json.dumps(response).encode()
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)


class Worker:
    def __init__(self):
        self.lock=Lock();self.commands=[]
        self.snapshot={'ready':True,'generation':1,'time':3.,'active_robot':'arm1','paused':True}
    async def execute(self,command):
        self.commands.append(command)
        if command['action']=='pause':self.snapshot['paused']=command['value']
        else:raise AssertionError('Inference recovery must not submit physical commands.')


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='recovery-contract-');self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        for path in EDITABLE:
            target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('# fixture source\n')
        mock=patch('server.runlog.RUNS',root/'runs');mock.start();self.addCleanup(mock.stop)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.requests=[];self.server.agent_calls=0;self.server.always_bad=False
        self.server.plan={'action':'retry','instruction':'Return the JSON action instance only, without whitespace-only output.', 'settings':None}
        self.thread=Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.server.server_close);self.addCleanup(lambda:self.thread.join(2));self.addCleanup(self.server.shutdown)
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.repair=RepairController(root)
        self.repair.configure(RepairConfig(provider='custom',model='fixture-coding-model',base_url=self.base+'/repair',auto_analyze=True,max_recovery_attempts=2))
        self.log=RunLog('agent',{'model':'fixture-policy-model','provider':'compatible','task':'recovery fixture'})
        self.owner=SimpleNamespace(worker=Worker(),state={'active':True,'status':'thinking','run_id':self.log.id,'round':7},run_log=self.log,
            config=AgentConfig(provider='compatible',model='fixture-policy-model',base_url=self.base+'/v1',api_key='fixture-main-key'))
        self.coordinator=RecoveryCoordinator(self.repair,secrets=lambda:['fixture-main-key'])
        self.owner.recovery=self.coordinator
        self.observation={'generation':1,'time':3.,'active_robot':'arm1'}
        image=io.BytesIO();Image.new('RGB',(8,8),'#667788').save(image,'JPEG')
        self.cameras=[{'name':name,'image_base64':base64.b64encode(image.getvalue()).decode()} for name in ('front','side','wrist')]
        self.prompts=[]

    async def operation(self,feedback,config):
        self.prompts.append((feedback,config))
        return await api_action(config,json.dumps({'task':'test','recovery_feedback':feedback}),self.cameras)

    async def call(self,operation=None):
        token=CURRENT_LOG.set(self.log)
        self.log.current_round=7
        try:return await invoke_with_recovery(self.owner,'agent_inference',operation or self.operation,self.observation,self.owner.config)
        finally:CURRENT_LOG.reset(token)

    def events(self,kind):
        return [row['data'] for row in map(json.loads,self.log.path.read_text(encoding='utf-8').splitlines()) if row['event']==kind]

    async def wait_paused(self,task):
        for _ in range(250):
            if self.owner.state['status']=='repair_paused':return
            if task.done():await task
            await asyncio.sleep(.01)
        self.fail('Experiment did not enter the recovery pause.')

    async def test_empty_body_is_repaired_in_same_run_round_and_call_accounting(self):
        action,usage=await self.call()
        self.assertEqual(action.status,'done');self.assertEqual(usage['completion_tokens'],8)
        self.assertTrue(self.owner.state['active']);self.assertEqual(self.owner.state['round'],7)
        self.assertEqual(self.owner.state['run_id'],self.log.id);self.assertFalse(self.log._finished)
        self.assertEqual(self.events('finish'),[])
        self.assertEqual(len(self.events('recoverable_error')),1)
        self.assertEqual(len(self.events('recovery_complete')),1)
        calls=self.events('inference')
        self.assertEqual([row['status'] for row in calls],['error','complete','complete'])
        self.assertEqual([row['phase'] for row in calls],['agent_inference','repair_diagnosis','agent_inference'])
        self.assertEqual([row['round'] for row in calls],[7,7,7])
        self.assertEqual([row['attempt'] for row in calls],[0,1,1])
        self.assertEqual(calls[1]['recovery_id'],calls[2]['recovery_id'])
        self.assertEqual(self.log._inference_count,3)
        self.assertEqual(sum(row['output_tokens'] for row in calls),21)
        main_requests=[body for path,body in self.server.requests if path.endswith('completions')]
        self.assertEqual(len(main_requests),2)
        for request in main_requests:
            self.assertEqual(len([part for part in request['messages'][-1]['content'] if part['type']=='image_url']),3)
        self.assertIn('JSON action instance',self.prompts[1][0])
        self.assertTrue(self.coordinator.view['status']=='recovered')
        self.log.finish({'status':'model_done'})
        from server.analysis import analyze_run
        analysis=analyze_run(self.log.id)
        self.assertTrue(analysis['recovery_assisted']);self.assertEqual(analysis['recovery_attempts'],1)
        self.assertEqual(analysis['total']['calls'],3)
        self.assertEqual(analysis['total']['output_tokens'],21)

    async def test_repeated_failures_exhaust_budget_without_infinite_paid_calls(self):
        self.server.always_bad=True
        with self.assertRaisesRegex(RuntimeError,'恢复预算已耗尽（2/2）'):await self.call()
        self.assertEqual(self.server.agent_calls,3)
        self.assertEqual(len([path for path,_ in self.server.requests if path=='/repair']),2)
        self.assertEqual(len(self.events('inference')),5)
        self.assertEqual(len(self.events('recovery_exhausted')),1)
        self.assertEqual(self.coordinator.view['status'],'exhausted')
        self.assertIsNone(self.coordinator.current)
        self.log.finish({'status':'error'})

    async def test_manual_resume_is_single_use_and_picks_up_user_configuration(self):
        self.repair.config.auto_recover=False
        task=asyncio.create_task(self.call())
        try:
            await self.wait_paused(task)
            failure_id=self.coordinator.view['failure_id']
            with self.assertRaises(ValueError):self.coordinator.resume('stale-id')
            self.owner.config=self.owner.config.model_copy(update={'max_output_tokens':32768})
            result=self.coordinator.resume(failure_id)
            self.assertFalse(result['recovery']['can_resume'])
            with self.assertRaises(ValueError):self.coordinator.resume(failure_id)
            await asyncio.wait_for(task,3)
            self.assertEqual(self.prompts[1][1].max_output_tokens,32768)
            self.assertEqual(len(self.events('recovery_manual_settings')),1)
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.log.finish({'status':'complete'})

    async def test_cancelling_paused_recovery_cannot_execute_a_late_result(self):
        self.repair.config.auto_recover=False
        task=asyncio.create_task(self.call());await self.wait_paused(task)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(self.server.agent_calls,1);self.assertIsNone(self.coordinator.current)
        self.assertEqual(self.coordinator.view['status'],'stopped')
        self.assertFalse(self.coordinator.view['can_resume'])
        self.log.finish({'status':'stopped'})

    async def test_adapter_swallowing_cancel_cannot_return_late_action_or_start_repair(self):
        entered=asyncio.Event()
        async def late_adapter(feedback,config):
            entered.set()
            try:await asyncio.sleep(20)
            except asyncio.CancelledError:return ACTION, None
        task=asyncio.create_task(self.call(late_adapter));await entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(self.server.requests,[])
        self.log.finish({'status':'stopped'})

    async def test_repair_credentials_are_redacted_before_writing_parent_log(self):
        key='fixture-repair-credential-echo'
        from pydantic import SecretStr
        self.repair.config.api_key=SecretStr(key);self.server.secret_echo=key
        await self.call()
        self.assertNotIn(key,self.log.path.read_text(encoding='utf-8'))
        self.assertIn('[已隐藏密钥]',self.log.path.read_text(encoding='utf-8'))
        self.log.finish({'status':'complete'})

    async def test_settings_fix_is_used_on_retry_and_following_calls(self):
        self.server.plan['settings']={'max_output_tokens':32768}
        await self.call();await self.call()
        requests=[body for path,body in self.server.requests if path.endswith('completions')]
        self.assertEqual([body['max_tokens'] for body in requests],[16384,32768,32768])
        self.assertEqual(self.owner.config.max_output_tokens,16384)
        self.log.finish({'status':'complete'})

    async def test_manual_approval_keeps_validated_suggested_settings_without_user_edit(self):
        self.repair.config.auto_recover=False
        self.server.plan['settings']={'max_output_tokens':32768}
        task=asyncio.create_task(self.call())
        try:
            await self.wait_paused(task)
            self.coordinator.resume(self.coordinator.view['failure_id'])
            await asyncio.wait_for(task,3)
            requests=[body for path,body in self.server.requests if path.endswith('completions')]
            self.assertEqual([body['max_tokens'] for body in requests],[16384,32768])
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.log.finish({'status':'complete'})

    async def test_parallel_failed_stages_are_diagnosed_serially_and_keep_their_owner(self):
        entered=asyncio.Event();release=asyncio.Event();counts={'a':0,'b':0}
        async def stage(name,feedback,config):
            counts[name]+=1
            if counts[name]==1:raise ValueError(name+' provider failed before execution')
            if name=='a':entered.set();await release.wait()
            return name
        async def call_stage(name):
            token=CURRENT_LOG.set(self.log)
            try:return await invoke_with_recovery(self.owner,'decision:'+name,
                lambda feedback,config:stage(name,feedback,config),self.observation,None)
            finally:CURRENT_LOG.reset(token)
        first=asyncio.create_task(call_stage('a'));second=asyncio.create_task(call_stage('b'))
        try:
            await asyncio.wait_for(entered.wait(),3)
            self.assertEqual(len([path for path,_ in self.server.requests if path=='/repair']),1)
            self.assertEqual(self.coordinator.view['phase'],'decision:a')
            self.assertIs(self.coordinator.current['task'],first)
            release.set()
            self.assertEqual(await asyncio.wait_for(asyncio.gather(first,second),3),['a','b'])
            self.assertEqual(counts,{'a':2,'b':2})
            self.assertEqual(len([path for path,_ in self.server.requests if path=='/repair']),2)
            self.assertIsNone(self.coordinator.current)
        finally:
            for task in (first,second):
                if not task.done():task.cancel()
            await asyncio.gather(first,second,return_exceptions=True)
            self.log.finish({'status':'complete'})

    async def test_recovery_ownership_is_held_until_retried_request_finishes(self):
        retried=asyncio.Event();release=asyncio.Event()
        async def operation(feedback,config):
            if feedback:retried.set();await release.wait()
            return await self.operation(feedback,config)
        task=asyncio.create_task(self.call(operation))
        try:
            await asyncio.wait_for(retried.wait(),3)
            self.assertIsNotNone(self.coordinator.current)
            self.assertEqual(self.coordinator.current['run_id'],self.log.id)
            self.assertEqual(self.coordinator.view['status'],'retrying')
            release.set();await asyncio.wait_for(task,3)
            self.assertIsNone(self.coordinator.current)
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.log.finish({'status':'complete'})

    async def test_scene_change_prevents_manual_resume(self):
        self.repair.config.auto_recover=False
        task=asyncio.create_task(self.call())
        try:
            await self.wait_paused(task)
            self.owner.worker.snapshot['generation']=2
            with self.assertRaisesRegex(RuntimeError,'状态已改变'):self.coordinator.resume(self.coordinator.view['failure_id'])
            self.assertEqual(self.server.agent_calls,1)
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.log.finish({'status':'stopped'})

    async def test_error_after_manual_key_change_is_redacted_when_budget_expires(self):
        self.repair.config.auto_recover=False;self.repair.config.max_recovery_attempts=1
        new_key='new-runtime-credential-echo'
        async def operation(feedback,config):
            if feedback:raise ValueError('provider echoed '+config.api_key.get_secret_value())
            return await self.operation(feedback,config)
        task=asyncio.create_task(self.call(operation))
        try:
            await self.wait_paused(task)
            from pydantic import SecretStr
            self.owner.config=self.owner.config.model_copy(update={'api_key':SecretStr(new_key)})
            self.coordinator.resume(self.coordinator.view['failure_id'])
            with self.assertRaisesRegex(RuntimeError,'恢复预算已耗尽') as raised:await asyncio.wait_for(task,3)
            self.assertNotIn(new_key,str(raised.exception))
            self.assertNotIn(new_key,self.log.path.read_text(encoding='utf-8'))
            self.assertIn(new_key,self.log.secrets)
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.log.finish({'status':'error'})

    async def test_resume_api_rejects_remote_origin_and_deduplicates_live_diagnosis(self):
        from server.app import app
        self.repair.config.auto_recover=False
        task=asyncio.create_task(self.call())
        try:
            await self.wait_paused(task)
            with patch('server.app.repair',self.repair),patch('server.app.recovery',self.coordinator):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                    response=await client.post('/api/repair/resume',json={'failure_id':self.coordinator.view['failure_id']},headers={'Origin':'https://example.org'})
                    self.assertEqual(response.status_code,403)
                    for _ in range(2):
                        response=await client.post('/api/repair/start',json={'source_run_id':self.log.id})
                        self.assertEqual(response.status_code,200)
                    self.assertEqual(len([path for path,_ in self.server.requests if path=='/repair']),1)
                    response=await client.post('/api/repair/resume',json={'failure_id':'stale'})
                    self.assertEqual(response.status_code,409)
                    response=await client.post('/api/repair/resume',json={'failure_id':self.coordinator.view['failure_id']})
                    self.assertEqual(response.status_code,200)
            await asyncio.wait_for(task,3)
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.log.finish({'status':'complete'})
