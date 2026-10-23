"""Repair workflow on disposable source copies and local HTTP; no paid inference."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import sys
from threading import Thread
import unittest
from unittest.mock import patch

import httpx

from server import repair, runlog
from server.agent import api_action
from pydantic import ValidationError

REGRESSION = """import unittest
from server.agent import response_value
class Regression(unittest.TestCase):
    def test_reported_value(self): self.assertEqual(response_value(), 'correct')
"""


def proposal():
    return repair.RepairProposal(diagnosis='响应解析逻辑错误。', category='code', evidence=['原实现返回 wrong。'],
        recommendations=['将该分支改为正确值并执行回归。'], edits=[
        {'path':'server/agent.py','before':"return 'wrong'",'after':"return 'correct'"},
        {'path':repair.REGRESSION,'before':'','after':REGRESSION}])


class RepairTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='workbench-repair-')
        self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        for name in ('server','tests'): (self.root/name).mkdir()
        (self.root/'server/__init__.py').write_text('')
        for name in repair.EDITABLE: (self.root/name).write_text('# fixture source\n', encoding='utf-8')
        (self.root/'server/agent.py').write_bytes(b"def response_value():\r\n    return 'wrong'\r\n")
        (self.root/'tests/test_provider_contracts.py').write_text("import unittest\nfrom server.agent import response_value\nclass Contract(unittest.TestCase):\n    def test_type(self): self.assertIsInstance(response_value(), str)\n")
        self.controller=repair.RepairController(self.root)
        mocked=patch('server.runlog.RUNS',self.root/'runs');mocked.start();self.addCleanup(mocked.stop)

    async def propose(self, result=None):
        async def boundary(config,prompt,cameras,**kwargs):
            self.assertEqual(cameras,[]);self.assertEqual(kwargs['schema_name'],'workbench_repair')
            for name in ('provider-error-triage','provider-contract-repair','minimal-fix-validation'):
                self.assertIn(name,kwargs['system_prompt'])
            self.assertNotIn('sensitive-example-key', prompt)
            return result or proposal(), {'input_tokens':1,'output_tokens':1}
        with patch('server.repair.codex_action',boundary),patch('server.repair.validate_agent_access'):
            await self.controller.start(repair.RepairStart(description='bad response sensitive-example-key'),secrets=['sensitive-example-key'])
            await self.controller.task
        self.assertEqual(self.controller.state['status'],'proposed')

    async def test_full_propose_reproduce_validate_apply_and_restore_after_restart(self):
        original=(self.root/'server/agent.py').read_bytes()
        await self.propose();self.assertEqual((self.root/'server/agent.py').read_bytes(),original)
        await self.controller.validate(allow_local_code_execution=True);await self.controller.task
        self.assertEqual(self.controller.state['status'],'validated',self.controller.state)
        self.assertIn('FAILED',self.controller.state['validation']['output'])
        self.assertEqual((self.root/'server/agent.py').read_bytes(),original)
        self.controller.apply()
        self.assertIn(b"return 'correct'\r\n",(self.root/'server/agent.py').read_bytes())
        restarted=repair.RepairController(self.root)
        self.assertEqual(restarted.state['status'],'applied')
        restarted.rollback()
        self.assertEqual((self.root/'server/agent.py').read_bytes(),original)
        self.assertFalse((self.root/repair.REGRESSION).exists())

    async def test_apply_preserves_user_changes_and_requires_validation(self):
        await self.propose()
        with self.assertRaisesRegex(ValueError,'验证'):self.controller.apply()
        await self.controller.validate(allow_local_code_execution=True);await self.controller.task
        path=self.root/'server/agent.py';path.write_text('# user change\n')
        with self.assertRaisesRegex(ValueError,'其他操作'):self.controller.apply()
        self.assertEqual(path.read_text(),'# user change\n')

    async def test_long_coding_timeout_is_configurable_and_elapsed_time_survives_failure(self):
        self.controller.configure(repair.RepairConfig(timeout_s=900))
        with self.assertRaises(ValidationError): repair.RepairConfig(timeout_s=1)
        # Main robot call limits are independent from the dedicated coding task.
        from server.agent import AgentConfig
        with self.assertRaises(ValidationError): AgentConfig(timeout_s=900)
        entered=asyncio.Event();release=asyncio.Event()
        async def boundary(config,*args,**kwargs):
            self.assertEqual(config.timeout_s,900)
            entered.set();await release.wait();raise TimeoutError()
        with patch('server.repair.codex_action',boundary),patch('server.repair.validate_agent_access'):
            await self.controller.start(repair.RepairStart(description='slow diagnosis'))
            await entered.wait()
            before=self.controller.snapshot();await asyncio.sleep(.02)
            during=self.controller.snapshot()
            self.assertTrue(during['active']);self.assertEqual(during['timeout_s'],900)
            self.assertGreater(during['elapsed_s'],before['elapsed_s'])
            release.set();await self.controller.task
        finished=self.controller.snapshot()
        self.assertFalse(finished['active']);self.assertEqual(finished['status'],'error')
        self.assertEqual(finished['message'],'TimeoutError')
        self.assertGreaterEqual(finished['elapsed_s'],during['elapsed_s'])
        self.assertEqual(finished['elapsed_s'],self.controller.snapshot()['elapsed_s'])

    async def test_diagnostic_context_keeps_actual_error_and_output_without_duplicated_reasoning(self):
        source=runlog.RunLog('agent',{'model':'fixture','provider':'compatible'})
        response={'status':'move','type':'json_object','explanation':'bad response fixture-key'}
        for i in range(4):
            source.write('model_output',{'choices':[{'finish_reason':'stop','content':json.dumps(response),
                'returned_reasoning':'long-reasoning '*10000}], 'usage':{'completion_tokens':100}})
            source.write('inference',{'status':'error','reasoning':{'text':'long-reasoning '*10000},
                'public_text':json.dumps(response),'usage':{'completion_tokens':100},'error':'extra forbidden type'})
        message='error-head '+('x'*15000)+' error-tail fixture-key'
        source.write('error',{'message':message});source.finish({'status':'error','message':'extra forbidden type'})
        original=source.path.read_bytes()
        context=repair.source_context(source.id,'',['fixture-key'])
        encoded=json.dumps(context,ensure_ascii=False)
        self.assertEqual(len(context['recent_events']),6)
        self.assertLess(len(encoded),13000)
        self.assertNotIn('long-reasoning',encoded);self.assertNotIn('fixture-key',encoded)
        self.assertIn('json_object',encoded);self.assertIn('error-head',encoded);self.assertIn('error-tail',encoded)
        for event in context['recent_events']:json.loads(event['diagnostic_excerpt'])
        self.assertEqual(source.path.read_bytes(),original)
        self.assertEqual(repair.source_context(None,'manual diagnosis')['recent_events'],[])

    async def test_regression_that_passes_original_is_not_accepted_as_a_fix(self):
        result=proposal();result.edits[-1].after=REGRESSION.replace("'correct'","'wrong'")
        await self.propose(result)
        await self.controller.validate(allow_local_code_execution=True);await self.controller.task
        self.assertEqual(self.controller.state['status'],'validation_failed')
        self.assertIn('未在原始实现上复现',self.controller.state['validation']['output'])

    async def test_partial_apply_and_interrupted_restore_remain_recoverable(self):
        await self.propose();await self.controller.validate(allow_local_code_execution=True);await self.controller.task
        original=(self.root/'server/agent.py').read_bytes();native=repair.atomic_bytes
        def fail_candidate(path,content):
            if path==self.root/repair.REGRESSION:raise OSError('fixture write failure')
            return native(path,content)
        with patch('server.repair.atomic_bytes',side_effect=fail_candidate),self.assertRaises(OSError):self.controller.apply()
        restored=repair.RepairController(self.root)
        self.assertEqual(restored.state['status'],'apply_failed');restored.rollback()
        self.assertEqual((self.root/'server/agent.py').read_bytes(),original)
        self.assertFalse((self.root/repair.REGRESSION).exists())

    async def test_import_errors_and_skips_are_not_a_valid_reproduction(self):
        result=proposal();result.edits[-1].after=REGRESSION.replace('response_value','unknown_function')
        await self.propose(result)
        with self.assertRaisesRegex(ValueError,'允许'):await self.controller.validate()
        await self.controller.validate(allow_local_code_execution=True);await self.controller.task
        self.assertEqual(self.controller.state['status'],'validation_failed')
        self.assertIn('导入错误',self.controller.state['validation']['output'])

    async def test_cancelled_validation_keeps_evidence_and_requires_a_new_candidate(self):
        result=proposal()
        result.edits[-1].after=REGRESSION.replace('import unittest', 'import unittest\nimport time\nfrom pathlib import Path').replace(
            "def test_reported_value(self): self.assertEqual(response_value(), 'correct')",
            "def test_reported_value(self):\n        Path('validation-started').write_text('entered')\n        time.sleep(30)\n        self.assertEqual(response_value(), 'correct')")
        original=(self.root/'server/agent.py').read_bytes()
        await self.propose(result)
        await self.controller.validate(allow_local_code_execution=True)
        candidate=self.controller.current['folder']/'candidate'
        try:
            async with asyncio.timeout(5):
                while not (candidate/'validation-started').exists():
                    if self.controller.task.done():
                        await self.controller.task
                        self.fail('Validation ended before the regression started.')
                    await asyncio.sleep(.02)
            state=await asyncio.wait_for(self.controller.stop(),12)
            self.assertFalse(state['active'])
            self.assertEqual(state['status'],'validation_failed')
            self.assertIn('已取消',state['message'])
            self.assertIn('重新分析生成新候选',state['message'])
            self.assertEqual((candidate/repair.REGRESSION).read_text(encoding='utf-8'),result.edits[-1].after)
            self.assertTrue((candidate/'validation-started').is_file())
            self.assertEqual((self.root/'server/agent.py').read_bytes(),original)
            self.assertFalse((self.root/repair.REGRESSION).exists())
            with self.assertRaisesRegex(ValueError,'重新分析'):
                await self.controller.validate(allow_local_code_execution=True)
            with self.assertRaisesRegex(ValueError,'验证'):
                self.controller.apply()
        finally:
            if not self.controller.task.done():
                await asyncio.wait_for(self.controller.stop(),12)

    @unittest.skipUnless(os.name=='nt','Windows process-tree termination')
    async def test_stop_validation_terminates_child_process_and_closes_inherited_pipe(self):
        child_code='import time; time.sleep(30)'
        parent_code='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",'+repr(child_code)+']); print(p.pid,flush=True); time.sleep(30)'
        process=await asyncio.create_subprocess_exec(sys.executable,'-c',parent_code,stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,creationflags=repair.subprocess.CREATE_NO_WINDOW)
        child_pid=int((await asyncio.wait_for(process.stdout.readline(),5)).decode().strip())
        await asyncio.wait_for(self.controller._terminate(process),12)
        self.assertIsNotNone(process.returncode)
        check=await asyncio.create_subprocess_exec('tasklist','/FI',f'PID eq {child_pid}','/FO','CSV','/NH',stdout=asyncio.subprocess.PIPE,
            creationflags=repair.subprocess.CREATE_NO_WINDOW)
        output,_=await check.communicate();self.assertNotIn(f'"{child_pid}"',output.decode('utf-8','replace'))

    async def test_paths_syntax_missing_regression_and_credentials(self):
        for path in ('../outside.py','server/control.py','C:/Windows/test.py'):
            with self.subTest(path=path),self.assertRaises(ValueError):
                repair.prepare_edits(self.root,[repair.RepairEdit(path=path,before='',after='x=1')])
        with self.assertRaisesRegex(ValueError,'回归'):
            repair.prepare_edits(self.root,proposal().edits[:1])
        with self.assertRaises(SyntaxError):
            repair.prepare_edits(self.root,[repair.RepairEdit(path=repair.REGRESSION,before='',after='def broken(')])
        config=repair.RepairConfig(provider='compatible',model='code-model',base_url='https://api.deepseek.com',api_key='private-key')
        self.controller.configure(config)
        self.assertNotIn('private-key',json.dumps(self.controller.public_config()))
        self.controller.configure(config.model_copy(update={'api_key':repair.SecretStr('')}))
        self.assertEqual(self.controller.config.api_key.get_secret_value(),'private-key')
        self.assertNotIn('private-key',json.dumps(repair.redact({'api_key':'private-key','error':'Bearer private-key'},['private-key'])))

    async def test_local_compatible_model_accepts_text_only_repair_contract(self):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));captured.append(body)
                payload={'id':'fixture','object':'chat.completion','created':0,'model':body['model'],
                    'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':proposal().model_dump_json()}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}}
                content=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            config=repair.RepairConfig(provider='compatible',model='fixture-coding-model',base_url=f'http://127.0.0.1:{server.server_port}/v1',api_key='fixture')
            output,usage=await api_action(config,'diagnostic data',[],response_type=repair.RepairProposal,system_prompt=repair.system_contract(),schema_name='workbench_repair')
            self.assertEqual(output.category,'code');self.assertEqual(usage['completion_tokens'],20)
            self.assertEqual([item['type'] for item in captured[0]['messages'][-1]['content']],['text'])
        finally:server.shutdown();server.server_close();thread.join(2)

    async def test_api_rejects_remote_mutation_and_apply_during_robot_experiment(self):
        from server.app import app,agent
        with patch('server.app.repair',self.controller):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                blocked=await client.post('/api/repair/start',json={'description':'fixture'},headers={'Origin':'https://example.org'})
                self.assertEqual(blocked.status_code,403)
                with patch.dict(agent.state,{'active':True}):
                    self.assertEqual((await client.post('/api/repair/apply')).status_code,409)
                data=(await client.get('/api/repair/config')).json();self.assertEqual(len(data['skills']),3)


if __name__=='__main__':unittest.main()
