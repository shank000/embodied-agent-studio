import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.harnesspai import (HarnessPAIController, ProgramValidation, ProgramEdit,
    ProgramFile, EXAMPLE, PYTHON, UPSTREAM_REVISION, digest, retrieve_program)
from server.harnesspai_primitives import HarnessTool, primitive_contract, call_primitive
from server.policy import PolicyConfig


class AutomaticContractTests(unittest.TestCase):
    def test_auto_needs_no_code_but_requires_separate_validation(self):
        self.assertIsNone(ProgramValidation(mode='auto').source)
        for args in ({'seeds':[42]}, {'seeds':[0,42]},
                     {'action_model':{'kind':'smolvla'}},
                     {'tools':[{'name':'vision','model':'v','description':'locate','endpoint':'http://127.0.0.1:9010'}]*2}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ProgramValidation(mode='auto', **args)

    def test_retrieval_matches_task_and_interface_and_checks_program(self):
        with tempfile.TemporaryDirectory() as directory, patch('server.harnesspai.LIBRARY',Path(directory)):
            folder=Path(directory)/'version'; folder.mkdir()
            source=EXAMPLE.read_text(encoding='utf-8')
            (folder/'program.py').write_text(source,encoding='utf-8',newline='')
            request=ProgramValidation(mode='auto')
            manifest={'run_id':'version','scene_id':request.scene_id,'task_id':request.task_id,
                      'source_sha256':digest(source),'upstream_revision':UPSTREAM_REVISION,
                      'primitive_contract':primitive_contract()}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(retrieve_program(request)[0],source)
            self.assertIsNone(retrieve_program(ProgramValidation(mode='auto',task_id='lift_red'))[0])
            request.action_model=PolicyConfig(kind='vla',endpoint='http://127.0.0.1:9010')
            self.assertIsNone(retrieve_program(request)[0])
            manifest['primitive_contract']=primitive_contract(request.action_model)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            request.action_model.model='a-different-compatible-model'
            self.assertEqual(retrieve_program(request)[0],source)
            (folder/'program.py').write_text('def run(ctx): pass')
            self.assertIsNone(retrieve_program(request)[0])


class PrimitiveHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests=[]
        async def respond(reader,writer):
            head=(await reader.readuntil(b'\r\n\r\n')).decode()
            headers=dict(line.split(': ',1) for line in head.split('\r\n')[1:] if ': ' in line)
            body=json.loads(await reader.readexactly(int(headers['Content-Length'])))
            self.requests.append((head,body))
            if head.startswith('POST /failure '):
                status='503 Service Unavailable'; reply={'detail':'model worker unavailable'}
            elif body['protocol']=='embodied-tool-v1':
                status='200 OK'; reply={'protocol':'embodied-tool-v1','result':{'pixel':[30,40],'confidence':.9}}
            else:
                status='200 OK'; reply={'protocol':'embodied-policy-v1','action_type':'actuator_targets',
                    'joint_names':['actuator1'],'actions':[[.2]],'dt_s':.05}
            payload=json.dumps(reply).encode()
            writer.write(f'HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n'.encode()+payload)
            await writer.drain(); writer.close(); await writer.wait_closed()
        self.server=await asyncio.start_server(respond,'127.0.0.1',0)
        self.url=f'http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}'

    async def asyncTearDown(self):
        self.server.close(); await self.server.wait_closed()

    async def test_public_payload_authentication_and_both_protocols(self):
        request={'kind':'policy','instruction':'raise end effector','observation':{'cameras':[]},'seed':42,'reset':True}
        policy=PolicyConfig(kind='vla',model='test-policy',endpoint=self.url+'/predict',api_key='test-secret')
        result=await call_primitive(request,policy,[])
        self.assertEqual(result['actions'],[[.2]])
        self.assertIn('Authorization: Bearer test-secret',self.requests[0][0])
        self.assertEqual(self.requests[0][1]['model'],'test-policy')
        tool=HarnessTool(name='vision',model='test-vision',description='pixel coordinates',endpoint=self.url+'/tool')
        result=await call_primitive({**request,'kind':'tool','name':'vision','input':{'target':'red'}},None,[tool])
        self.assertEqual(result['pixel'],[30,40])
        self.assertEqual(self.requests[-1][1]['input'],{'target':'red'})
        policy.endpoint=self.url+'/failure'
        with self.assertRaisesRegex(ValueError,'503.*model worker unavailable'):
            await call_primitive(request,policy,[])


@unittest.skipUnless(os.getenv('HARNESSPAI_INTEGRATION')=='1' and PYTHON.is_file(),'Native physics and upstream runtime required')
class NativeAutomaticTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_archive_then_reuse_without_model_or_duplicate_version(self):
        """Real upstream scheduling and physics; deterministic writer checks orchestration only."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); calls=[]
            async def writer(config,prompt,cameras,**kwargs):
                calls.append(prompt)
                allowed=prompt.rsplit('ALLOWED REPLACEMENTS: ',1)[-1].strip()
                files=[] if allowed=='NONE; files must be []' or '/**/' in allowed else [ProgramFile(path=allowed,content=EXAMPLE.read_text(encoding='utf-8'))]
                return ProgramEdit(explanation='Use public TCP control.',files=files),None
            with patch('server.runlog.RUNS',root/'runs'), patch('server.harnesspai.LIBRARY',root/'methods/harnesspai'), patch('server.harnesspai.invoke_agent_action',writer), patch('server.harnesspai.validate_agent_access'):
                request=ProgramValidation(mode='auto',record_video=False,evolution_seeds=[0],allow_local_code_execution=True)
                first=HarnessPAIController(); await first.start(request)
                await asyncio.wait_for(first.task,90)
                self.assertEqual(first.state['status'],'verified',first.state)
                self.assertTrue(first.state['archive']['ok'],first.state)
                self.assertFalse(first.state['archive']['reused'])
                self.assertGreater(len(calls),0)
                previous=len(calls)
                second=HarnessPAIController(); await second.start(request)
                await asyncio.wait_for(second.task,45)
                self.assertEqual(second.state['status'],'verified',second.state)
                self.assertTrue(second.state['archive']['reused'])
                self.assertEqual(len(calls),previous)
                self.assertEqual(len(list((root/'methods/harnesspai').glob('*/manifest.json'))),1)
                self.assertEqual(second.state['source_origin']['run_id'],first.log.id)
                self.assertEqual(second.state['model_calls'],0)

    async def test_policy_and_tool_bridge_execute_in_native_episode_and_redact_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); seen=[]; secret='not-a-real-key-native-test'
            async def primitive(request,policy,tools):
                seen.append(request)
                self.assertNotIn('evaluation',request['observation'])
                self.assertEqual(len(request['observation']['cameras']),3)
                if request['kind']=='tool': return {'z_offset':.025}
                pose=dict(request['observation']['tcp'])
                pose['position_m']=list(pose['position_m']);pose['position_m'][2]+=.025
                pose.update(status='move',gripper_open=1.,duration_s=1.,explanation='Test bridge')
                return {'protocol':'embodied-policy-v1','action_type':'tcp_pose','pose':pose}
            source='def run(ctx):\n    ctx.tool("vision","find the target")\n    ctx.execute_policy(ctx.policy("raise the end effector"))\n'
            tool=HarnessTool(name='vision',model='test',description='location',endpoint='http://127.0.0.1:9010',api_key=secret)
            request=ProgramValidation(source=source,seeds=[42],record_video=False,allow_local_code_execution=True,
                action_model=PolicyConfig(kind='vla',endpoint='http://127.0.0.1:9010',api_key=secret),tools=[tool])
            with patch('server.runlog.RUNS',root),patch('server.harnesspai.call_primitive',primitive):
                controller=HarnessPAIController();await controller.start(request)
                await asyncio.wait_for(controller.task,30)
                self.assertEqual(controller.state['status'],'verified',controller.state)
                self.assertEqual(controller.state['primitive_calls'],2)
                self.assertEqual(controller.state['rows'][0]['policy_executions'],1)
                self.assertTrue(seen[1]['reset'])
                async def failure(*args): raise RuntimeError('worker failed '+secret)
                with patch('server.harnesspai.call_primitive',failure):
                    failed=HarnessPAIController();await failed.start(request)
                    await asyncio.wait_for(failed.task,30)
                    self.assertEqual(failed.state['status'],'error')
                    self.assertIn('worker failed',failed.state['rows'][0]['message'])
            for path in root.rglob('*'):
                if path.suffix in ('.json','.jsonl','.log','.py'):
                    self.assertNotIn(secret,path.read_text(encoding='utf-8',errors='replace'),str(path))
